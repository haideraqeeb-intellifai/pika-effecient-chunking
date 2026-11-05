"""Single-RTX-5090 comparison of serial and stage-grouped Pika H3 chunks.

Run from this directory with .venv/bin/python benchmark.py prepare, then
.venv/bin/python benchmark.py run. The source repository is read only.
"""

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / "script2video"
sys.path.insert(0, str(SOURCE))

import requests

from ml_modules.minimax_h3_turbo.infer import infer, load_models, validate_assets
from ml_modules.minimax_h3_turbo.workflow import build_workflow
from workers.image2video_h3_turbo.schema import schema_image2video_h3_turbo

REFERENCE = json.loads((ROOT / "raccoon_50s_reference.json").read_text())
CHUNKS = REFERENCE["chunks"]
RUNTIME = ROOT / "runtime"
INPUT = RUNTIME / "input" / "raccoon"
OUTPUT = RUNTIME / "output"
RESULTS = ROOT / "results"


def download(url, path):
    if path.is_file():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    with requests.get(url, stream=True, timeout=(20, 120)) as response:
        response.raise_for_status()
        with temporary.open("wb") as stream:
            for block in response.iter_content(1024 * 1024):
                stream.write(block)
    temporary.replace(path)


def prepare():
    for chunk in CHUNKS:
        index = chunk["index"]
        original = chunk["submitted_generation_input"]
        download(original["image_path"], INPUT / "headshot.jpg")
        download(original["audio_path"], INPUT / f"audio_{index}.wav")
    for path in sorted(INPUT.iterdir()):
        digest = hashlib.file_digest(path.open("rb"), "sha256").hexdigest()
        print(path.name, path.stat().st_size, digest, flush=True)


def graph_for(chunk, prefix):
    index = chunk["index"]
    original = chunk["submitted_generation_input"]
    audio = INPUT / f"audio_{index}.wav"
    duration = float(subprocess.check_output([
        "ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(audio),
    ], timeout=60))
    payload = schema_image2video_h3_turbo(
        **{**original, "webhook_url": "", "num_frames": max(5, math.ceil(duration * 24))}
    ).model_dump()
    return build_workflow(
        payload, first_frame_name="raccoon/headshot.jpg", last_frame_name="raccoon/headshot.jpg",
        audio_name=f"raccoon/audio_{index}.wav", output_prefix=prefix, seed=payload["seed"],
    )


class PrepareBarrier:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {f"{kind}_{i}": (type_name,) for i in range(4)
                             for kind, type_name in (("cond", "CONDITIONING"), ("latent", "LATENT"))}}

    RETURN_TYPES = tuple(t for _ in range(4) for t in ("CONDITIONING", "LATENT"))
    FUNCTION = "pass_through"
    CATEGORY = "benchmark"

    def pass_through(self, **values):
        return tuple(values[f"{kind}_{i}"] for i in range(4) for kind in ("cond", "latent"))


class SampleBarrier:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {f"latent_{i}": ("LATENT",) for i in range(4)}}

    RETURN_TYPES = ("LATENT",) * 4
    FUNCTION = "pass_through"
    CATEGORY = "benchmark"

    def pass_through(self, **values):
        # Keep the five-minute aggregate's outputs off VRAM while the decoder loads.
        return tuple({**values[f"latent_{i}"], "samples": values[f"latent_{i}"]["samples"].cpu()}
                     for i in range(4))


SHARED = {"1", "2", "3", "4", "pika", "sage", "8", "9"}


def grouped_graph(graphs, staged=True):
    if len(graphs) != 4:
        raise ValueError("This benchmark expects the reference file's four chunks")
    graph = {key: value for key, value in graphs[0].items() if key in SHARED}
    prep_inputs, sample_inputs = {}, {}
    for index, source in enumerate(graphs):
        prefix = f"c{index}_"

        def remap(value):
            if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str) and value[0] in source:
                return [value[0] if value[0] in SHARED else prefix + value[0], value[1]]
            if isinstance(value, dict):
                return {key: remap(item) for key, item in value.items()}
            return value

        for node_id, node in source.items():
            if node_id not in SHARED:
                graph[prefix + node_id] = remap(node)
        if staged:
            prep_inputs[f"cond_{index}"] = [prefix + "5", 0]
            prep_inputs[f"latent_{index}"] = [prefix + "5", 1]
            graph[prefix + "7"]["inputs"]["conditioning"] = ["prep_barrier", 2 * index]
            graph[prefix + "10"]["inputs"]["latent_image"] = ["prep_barrier", 2 * index + 1]
            sample_inputs[f"latent_{index}"] = [prefix + "10", 0]
            for node_id in ("11", "12"):
                graph[prefix + node_id]["inputs"]["samples"] = ["sample_barrier", index]
    if staged:
        graph["prep_barrier"] = {"class_type": "H3PrepareBarrier", "inputs": prep_inputs}
        graph["sample_barrier"] = {"class_type": "H3SampleBarrier", "inputs": sample_inputs}
    return graph


def one_run(runtime, graph, label):
    (RESULTS / "graphs").mkdir(parents=True, exist_ok=True)
    (RESULTS / "graphs" / f"{label}.json").write_text(json.dumps(graph, indent=2) + "\n")
    started = time.perf_counter()
    result = infer(runtime, graph, prompt_id=label, timeout_seconds=3600)
    elapsed = time.perf_counter() - started
    (RESULTS / f"{label}.json").write_text(json.dumps({"seconds": elapsed, **result["telemetry"]}, indent=2) + "\n")
    print(label, f"{elapsed:.3f}s", flush=True)
    return elapsed


def run():
    validate_assets(ROOT / "assets")
    for index in range(1, 5):
        if not (INPUT / f"audio_{index}.wav").is_file():
            raise FileNotFoundError("Run `benchmark.py prepare` first")
    runtime = load_models(ROOT / "assets", RUNTIME)
    runtime["main"].nodes.NODE_CLASS_MAPPINGS.update({
        "H3PrepareBarrier": PrepareBarrier, "H3SampleBarrier": SampleBarrier,
    })
    graphs = [graph_for(chunk, f"baseline/chunk_{chunk['index']}/output")[0] for chunk in CHUNKS]
    totals = {}
    started = time.perf_counter()
    totals["baseline_chunks"] = [one_run(runtime, graph, f"baseline_{i}") for i, graph in enumerate(graphs, 1)]
    totals["baseline_seconds"] = time.perf_counter() - started
    grouped = [graph_for(chunk, f"grouped/chunk_{chunk['index']}/output")[0] for chunk in CHUNKS]
    totals["grouped_seconds"] = one_run(runtime, grouped_graph(grouped), "grouped")
    (RESULTS / "summary.json").write_text(json.dumps(totals, indent=2) + "\n")
    print(json.dumps(totals, indent=2), flush=True)


def repeat_one():
    validate_assets(ROOT / "assets")
    runtime = load_models(ROOT / "assets", RUNTIME)
    graph = graph_for(CHUNKS[0], "repeat/chunk_1/output")[0]
    one_run(runtime, graph, "repeat_1")


def shared_only():
    validate_assets(ROOT / "assets")
    runtime = load_models(ROOT / "assets", RUNTIME)
    graphs = [graph_for(chunk, f"shared/chunk_{chunk['index']}/output")[0] for chunk in CHUNKS]
    one_run(runtime, grouped_graph(graphs, staged=False), "shared")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare", "run", "repeat_one", "shared_only"])
    action = parser.parse_args().action
    {"prepare": prepare, "run": run, "repeat_one": repeat_one, "shared_only": shared_only}[action]()
