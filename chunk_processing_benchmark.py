"""Replay the same four Pika latents under alternative chunk schedules.

Capture runs the unmodified four sequential samplers once and persists their
outputs. Decode variants consume those identical latents, so decoder and save
timings do not include a fresh denoising run.
"""

import argparse
import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import benchmark as base

RESULTS = base.RESULTS / "chunk_processing"
LATENTS = RESULTS / "latents"
POOL = ThreadPoolExecutor(max_workers=2)


class CaptureLatent:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"samples": ("LATENT",), "index": ("INT",)}}

    RETURN_TYPES = ()
    FUNCTION = "capture"
    OUTPUT_NODE = True
    CATEGORY = "benchmark"

    def capture(self, samples, index):
        import torch

        LATENTS.mkdir(parents=True, exist_ok=True)
        parts = samples["samples"].unbind()
        path = LATENTS / f"chunk_{index}.pt"
        torch.save({"video": parts[0].cpu(), "audio": parts[1].cpu()}, path)
        print("captured", index, tuple(parts[0].shape), tuple(parts[1].shape), flush=True)
        return ()


class LoadLatent:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"index": ("INT",)}}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "load"
    CATEGORY = "benchmark"

    def load(self, index):
        import comfy.nested_tensor
        import torch

        data = torch.load(LATENTS / f"chunk_{index}.pt", map_location="cpu", weights_only=True)
        return ({"samples": comfy.nested_tensor.NestedTensor((data["video"], data["audio"]))},)


class PairVAEDecode:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"samples_a": ("LATENT",), "samples_b": ("LATENT",), "vae": ("VAE",)}}

    RETURN_TYPES = ("IMAGE", "IMAGE")
    FUNCTION = "decode"
    CATEGORY = "benchmark"

    def decode(self, samples_a, samples_b, vae):
        import torch

        a = samples_a["samples"].unbind()[0]
        b = samples_b["samples"].unbind()[0]
        if a.shape != b.shape or a.shape[0] != 1:
            raise ValueError(f"Pair requires matching singleton video shapes: {a.shape}, {b.shape}")
        both = torch.cat((a, b), dim=0)
        print("pair decode input", tuple(both.shape), "memory estimate", vae.memory_used_decode(both.shape, vae.vae_dtype), flush=True)
        images = vae.decode(both)
        if images.ndim != 5 or images.shape[0] != 2:
            raise ValueError(f"Unexpected VAE batch output: {images.shape}")
        return (images[0], images[1])


class AsyncSave:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"video": ("VIDEO",), "index": ("INT",), "variant": ("STRING",)}}

    RETURN_TYPES = ("ASYNC_SAVE",)
    FUNCTION = "dispatch"
    CATEGORY = "benchmark"

    def dispatch(self, video, index, variant):
        from comfy_api.latest._util.video_types import VideoContainer
        path = base.OUTPUT / variant / f"chunk_{index}" / "output_00001_.mp4"
        path.parent.mkdir(parents=True, exist_ok=True)

        def write():
            started = time.perf_counter()
            video.save_to(str(path), format=VideoContainer.AUTO, codec="auto")
            return {"index": index, "path": str(path), "seconds": time.perf_counter() - started}

        return (POOL.submit(write),)


class WaitSaves:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {f"future_{i}": ("ASYNC_SAVE",) for i in range(4)}}

    RETURN_TYPES = ()
    FUNCTION = "wait"
    OUTPUT_NODE = True
    CATEGORY = "benchmark"

    def wait(self, **kwargs):
        results = [kwargs[f"future_{i}"].result() for i in range(4)]
        print("async saves", json.dumps(results), flush=True)
        return ()


class DeferredParallelSaves:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {f"video_{i}": ("VIDEO",) for i in range(4)}}

    RETURN_TYPES = ()
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "benchmark"

    def save(self, **kwargs):
        from comfy_api.latest._util.video_types import VideoContainer

        started = time.perf_counter()

        def write(index):
            path = base.OUTPUT / "full_deferred_parallel" / f"chunk_{index + 1}" / "output_00001_.mp4"
            path.parent.mkdir(parents=True, exist_ok=True)
            begun = time.perf_counter()
            kwargs[f"video_{index}"].save_to(str(path), format=VideoContainer.AUTO, codec="auto")
            return {"chunk": index + 1, "seconds": time.perf_counter() - begun, "path": str(path)}

        with ThreadPoolExecutor(max_workers=4) as pool:
            jobs = [pool.submit(write, index) for index in range(4)]
            saved = [job.result() for job in jobs]
        details = {"save_wall_seconds": time.perf_counter() - started, "chunks": saved}
        (RESULTS / "full_deferred_parallel_save.json").write_text(json.dumps(details, indent=2) + "\n")
        print("deferred parallel saves", json.dumps(details), flush=True)
        return ()


class SingleNVENCSave:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {f"video_{i}": ("VIDEO",) for i in range(4)}}

    RETURN_TYPES = ()
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "benchmark"

    def save(self, **kwargs):
        return self._save(
            output_name="raccoon_full_nvenc_single_assembled.mp4",
            details_name="full_nvenc_single_save.json",
            encoder_args=["-preset", "p4", "-cq", "20", "-b:v", "0"],
            **kwargs,
        )

    def _save(self, output_name, details_name, encoder_args, **kwargs):
        import torch

        videos = [kwargs[f"video_{i}"].get_components().images for i in range(4)]
        audios = [base.INPUT / f"audio_{i}.wav" for i in range(1, 5)]
        frame_counts = []
        for audio in audios:
            duration = float(subprocess.check_output([
                "ffprobe", "-v", "error", "-show_entries", "format=duration",
                "-of", "csv=p=0", str(audio),
            ], timeout=30))
            frame_counts.append(round(duration * 24))
        shapes = [tuple(images.shape) for images in videos]
        if any(len(shape) != 4 or shape[1:] != shapes[0][1:] for shape in shapes):
            raise ValueError(f"Mismatched decoded frame shapes: {shapes}")
        if any(shape[0] < count for shape, count in zip(shapes, frame_counts)):
            raise ValueError(f"Too few decoded frames: {shapes}, need {frame_counts}")
        _, height, width, channels = shapes[0]
        if channels != 3:
            raise ValueError(f"Expected RGB frames, got {shapes}")

        output = base.RESULTS / output_name
        command = [
            "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s:v", f"{width}x{height}",
            "-r", "24", "-i", "pipe:0",
        ]
        for audio in audios:
            command += ["-i", str(audio)]
        command += [
            "-filter_complex", "[0:v]scale=1080:1920[v];[1:a][2:a][3:a][4:a]concat=n=4:v=0:a=1[a]",
            "-map", "[v]", "-map", "[a]", "-c:v", "h264_nvenc", *encoder_args,
            "-pix_fmt", "yuv420p", "-c:a", "aac",
            "-b:a", "192k", "-movflags", "+faststart", str(output),
        ]
        started = time.perf_counter()
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                   stderr=subprocess.PIPE, bufsize=8 * 1024 * 1024)
        frames_written = 0
        try:
            for images, count in zip(videos, frame_counts):
                for frame in images[:count]:
                    rgb = (frame * 255).clamp(0, 255).to(dtype=torch.uint8, device="cpu")
                    process.stdin.write(rgb.contiguous().numpy().tobytes())
                    frames_written += 1
            process.stdin.close()
            error_log = process.stderr.read().decode(errors="replace")
            return_code = process.wait(timeout=600)
            if return_code:
                raise RuntimeError(f"NVENC exited {return_code}: {error_log[-2000:]}")
        except Exception:
            process.kill()
            process.wait()
            raise
        finally:
            process.stderr.close()
        details = {"save_wall_seconds": time.perf_counter() - started,
                   "frames": frames_written, "frames_per_chunk": frame_counts,
                   "input_shapes": shapes, "output": str(output)}
        (RESULTS / details_name).write_text(json.dumps(details, indent=2) + "\n")
        print("single NVENC save", json.dumps(details), flush=True)
        return ()


class SizeMatchedNVENCSave(SingleNVENCSave):
    def save(self, **kwargs):
        return self._save(
            output_name="raccoon_full_nvenc_matched_assembled.mp4",
            details_name="full_nvenc_matched_save.json",
            encoder_args=[
                "-preset", "p5", "-rc", "vbr", "-b:v", "3700k",
                "-maxrate", "4800k", "-bufsize", "7400k",
                "-multipass", "fullres", "-spatial-aq", "1", "-temporal-aq", "1",
            ],
            **kwargs,
        )


def register(runtime):
    runtime["main"].nodes.NODE_CLASS_MAPPINGS.update({
        "H3PrepareBarrier": base.PrepareBarrier,
        "H3SampleBarrier": base.SampleBarrier,
        "H3CaptureLatent": CaptureLatent,
        "H3LoadLatent": LoadLatent,
        "H3PairVAEDecode": PairVAEDecode,
        "H3AsyncSave": AsyncSave,
        "H3WaitSaves": WaitSaves,
        "H3DeferredParallelSaves": DeferredParallelSaves,
        "H3SingleNVENCSave": SingleNVENCSave,
        "H3SizeMatchedNVENCSave": SizeMatchedNVENCSave,
    })


def capture_graph():
    sources = [base.graph_for(c, f"unused/chunk_{c['index']}/output")[0] for c in base.CHUNKS]
    graph = base.grouped_graph(sources, staged=True)
    for i in range(4):
        graph.pop(f"c{i}_17")
        graph[f"capture_{i}"] = {"class_type": "H3CaptureLatent", "inputs": {
            "samples": ["sample_barrier", i], "index": i + 1}}
    return graph


def decode_graph(variant):
    if variant not in {"decode_serial", "decode_pairs", "decode_async", "decode_pairs_async"}:
        raise ValueError(variant)
    pair_decode = variant in {"decode_pairs", "decode_pairs_async"}
    async_save = variant in {"decode_async", "decode_pairs_async"}
    source = base.graph_for(base.CHUNKS[0], "unused/output")[0]
    graph = {"vae": source["3"]}
    for i in range(4):
        index = i + 1
        src = base.graph_for(base.CHUNKS[i], f"{variant}/chunk_{index}/output")[0]
        graph[f"load_{i}"] = {"class_type": "H3LoadLatent", "inputs": {"index": index}}
        graph[f"audio_{i}"] = src["22"]
        if pair_decode:
            pair_name = "pair_0" if i in (0, 2) else "pair_1"
            image_ref = [pair_name, 0 if i < 2 else 1]
        else:
            graph[f"decode_{i}"] = {"class_type": "VAEDecode", "inputs": {
                "samples": [f"load_{i}", 0], "vae": ["vae", 0]}}
            image_ref = [f"decode_{i}", 0]
        if "15" in src:
            crop = json.loads(json.dumps(src["15"]))
            crop["inputs"]["image"] = image_ref
            graph[f"crop_{i}"] = crop
            image_ref = [f"crop_{i}", 0]
        create = json.loads(json.dumps(src["16"]))
        create["inputs"]["images"] = image_ref
        create["inputs"]["audio"] = [f"audio_{i}", 0]
        graph[f"create_{i}"] = create
        if async_save:
            graph[f"save_{i}"] = {"class_type": "H3AsyncSave", "inputs": {
                "video": [f"create_{i}", 0], "index": index, "variant": variant}}
        else:
            save = json.loads(json.dumps(src["17"]))
            save["inputs"]["video"] = [f"create_{i}", 0]
            graph[f"save_{i}"] = save
    if pair_decode:
        for n, (a, b) in enumerate(((0, 2), (1, 3))):
            graph[f"pair_{n}"] = {"class_type": "H3PairVAEDecode", "inputs": {
                "samples_a": [f"load_{a}", 0], "samples_b": [f"load_{b}", 0], "vae": ["vae", 0]}}
    if async_save:
        graph["wait_saves"] = {"class_type": "H3WaitSaves", "inputs": {
            f"future_{i}": [f"save_{i}", 0] for i in range(4)}}
    return graph


def full_graph(variant):
    if variant not in {"full_async", "full_pairs", "full_pairs_async", "full_deferred_parallel", "full_nvenc_single", "full_nvenc_matched"}:
        raise ValueError(variant)
    sources = [base.graph_for(c, f"{variant}/chunk_{c['index']}/output")[0] for c in base.CHUNKS]
    graph = base.grouped_graph(sources, staged=True)
    if variant in {"full_pairs", "full_pairs_async"}:
        for n, (a, b) in enumerate(((0, 2), (1, 3))):
            graph[f"pair_{n}"] = {"class_type": "H3PairVAEDecode", "inputs": {
                "samples_a": ["sample_barrier", a], "samples_b": ["sample_barrier", b],
                "vae": ["3", 0]}}
        for i in range(4):
            graph.pop(f"c{i}_11")
            pair_name = "pair_0" if i in (0, 2) else "pair_1"
            graph[f"c{i}_15"]["inputs"]["image"] = [pair_name, 0 if i < 2 else 1]
    if variant in {"full_async", "full_pairs_async"}:
        for i in range(4):
            graph[f"c{i}_17"] = {"class_type": "H3AsyncSave", "inputs": {
                "video": [f"c{i}_16", 0], "index": i + 1, "variant": variant}}
        graph["wait_saves"] = {"class_type": "H3WaitSaves", "inputs": {
            f"future_{i}": [f"c{i}_17", 0] for i in range(4)}}
    if variant in {"full_deferred_parallel", "full_nvenc_single", "full_nvenc_matched"}:
        for i in range(4):
            graph.pop(f"c{i}_17")
        class_type = ("H3DeferredParallelSaves" if variant == "full_deferred_parallel"
                      else "H3SizeMatchedNVENCSave" if variant == "full_nvenc_matched"
                      else "H3SingleNVENCSave")
        graph["save_all"] = {"class_type": class_type, "inputs": {
            f"video_{i}": [f"c{i}_16", 0] for i in range(4)}}
    return graph


def run(action):
    RESULTS.mkdir(parents=True, exist_ok=True)
    runtime = base.load_models(base.ROOT / "assets", base.RUNTIME)
    register(runtime)
    actions = ["decode_serial", "decode_pairs", "decode_async", "decode_serial_repeat"] if action == "all_decodes" else [action]
    for item in actions:
        if item == "capture":
            graph = capture_graph()
        elif item in {"full_async", "full_pairs", "full_pairs_async", "full_deferred_parallel", "full_nvenc_single", "full_nvenc_matched"}:
            graph = full_graph(item)
        else:
            if any(not (LATENTS / f"chunk_{i}.pt").is_file() for i in range(1, 5)):
                raise FileNotFoundError("Capture all four latents first")
            graph = decode_graph("decode_serial" if item == "decode_serial_repeat" else item)
        (RESULTS / f"{item}_graph.json").write_text(json.dumps(graph, indent=2) + "\n")
        started = time.perf_counter()
        result = base.infer(runtime, graph, prompt_id=item, timeout_seconds=3600)
        elapsed = time.perf_counter() - started
        (RESULTS / f"{item}.json").write_text(json.dumps({"seconds": elapsed, **result["telemetry"]}, indent=2) + "\n")
        print(item, f"{elapsed:.3f}s", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["capture", "decode_serial", "decode_pairs", "decode_async", "decode_pairs_async", "full_async", "full_pairs", "full_pairs_async", "full_deferred_parallel", "full_nvenc_single", "full_nvenc_matched", "all_decodes"])
    run(parser.parse_args().action)
