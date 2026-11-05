"""Build the local, media-rich report for the H3 chunk scheduling experiment."""

from __future__ import annotations

import hashlib
import html
import json
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
DEST = RESULTS / "chunk_processing"
FINDINGS = json.loads((RESULTS / "findings.json").read_text())
SUMMARY = json.loads((DEST / "summary.json").read_text())
STITCH = json.loads((DEST / "stitch_timings.json").read_text())
DIRECT = {name: json.loads((DEST / f"{name}.json").read_text())["seconds"]
          for name in ("full_pairs", "full_async", "full_deferred_parallel", "full_nvenc_single", "full_nvenc_matched")}
SAVE_METRICS = json.loads((DEST / "save_strategy_metrics.json").read_text())
SAME_SOURCE = json.loads((DEST / "nvenc_same_source_comparison.json").read_text())

DARK_CSS = """
:root { color-scheme: dark; --ink:#edf3ff; --muted:#afbdd4; --bg:#0a1120; --line:#2b3a55; --blue:#a29dff; --teal:#56d7d0; --orange:#ffae7d; }
body { background:radial-gradient(circle at 85% -10%,#1c2d4a 0,#0a1120 42%); background-attachment:fixed; }
a { color:#b3b8ff; }
.hero { background:linear-gradient(125deg,#20365a,#172b49 64%,#292956); border:1px solid #3a4d71; }
.hero p { color:#d8e3f5; }
.hero-pills span { background:#ffffff14; border-color:#ffffff30; }
.stat,.panel,.approach { background:#131f33; border-color:var(--line); }
.bar-track { background:#263650; }
.bar { background:linear-gradient(90deg,#a49dff,#6d81f4); }
.bar.gradient { background:linear-gradient(90deg,#41d3c7,#4c9dcc); }
tbody tr:hover { background:#1a2a43; }
button { background:#1b2b43; color:var(--ink); border-color:#3a4c69; }
button:hover { background:#273b59; }
button.active { background:#35365e; border-color:#aaa6ff; color:#f1eeff; }
.metric-strip>div { background:#1b2b43; }
.pill.measured { color:#91f5dc; background:#134b46; }
.pill.estimated { color:#ffdda2; background:#654b25; }
.pill.blocked { color:#ffc6bc; background:#60332e; }
.diagram { background:#eef2f8; border-color:#556885; box-shadow:0 12px 30px #060a1499; }
.note { background:#20314a; color:#dbe8fa; border-color:#a3a0ff; }
.note code { color:#ffddb4; }
footer { color:#a8b7cf; }
a:focus-visible,button:focus-visible,video:focus-visible { outline:3px solid #c2c0ff; outline-offset:3px; }
::selection { background:#625eba; color:white; }
"""

APPROACHES = [
    ("baseline", "Separate requests", "01-separate-requests.png", FINDINGS["baseline_seconds"], "Direct run", "Four complete chunk requests, one after the other."),
    ("shared", "Shared loader graph", "02-shared-loader-graph.png", FINDINGS["shared"]["seconds"], "Direct run", "One graph loads the components and processes all four chunks."),
    ("grouped", "Stage grouped", "03-stage-grouped.png", FINDINGS["grouped"]["seconds"], "Direct run", "Prepare all conditions, denoise all chunks, then decode and save each."),
    ("full_pairs", "Paired VAE decode", "04-paired-vae.png", DIRECT["full_pairs"], "Direct run", "Denoise all four, decode matching chunks in pairs, then save."),
    ("full_async", "Asynchronous save", "05-async-save.png", DIRECT["full_async"], "Direct run", "Denoise all four; save a decoded chunk while the next decodes."),
    ("full_pairs_async", "Paired VAE + asynchronous save", "06-paired-vae-async-save.png", SUMMARY["full_end_to_end_pairs_async_seconds"], "Direct run", "Denoise all four; decode in pairs and overlap saving. Full workflow rerun."),
    ("full_deferred_parallel", "Deferred parallel CPU saves", "07-deferred-parallel-save.png", DIRECT["full_deferred_parallel"], "Direct run", "Decode every chunk first, encode four chunk MP4s concurrently, then stitch."),
    ("full_nvenc_single", "Single final NVENC encode", "08-single-nvenc-save.png", DIRECT["full_nvenc_single"], "Direct run", "Decode every chunk, then make one final H.264 MP4 with NVENC and no stitch."),
    ("full_nvenc_matched", "Size-matched final NVENC", "09-size-matched-nvenc.png", DIRECT["full_nvenc_matched"], "Direct run", "Encode one MP4 directly with NVENC VBR tuned below the CPU output size."),
]


def run(command: list[str], *, binary: bool = False):
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, timeout=180)
    return result.stdout if binary else result.stderr.decode(errors="replace")


def probe(path: Path) -> dict:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height,nb_frames,r_frame_rate:format=duration", "-of", "json", str(path)],
        capture_output=True, text=True, check=True, timeout=30,
    )
    data = json.loads(result.stdout)
    video = data["streams"][0]
    return {"width": int(video["width"]), "height": int(video["height"]), "frames": int(video["nb_frames"]), "fps": video["r_frame_rate"], "seconds": float(data["format"]["duration"])}


def audio_hash(path: Path) -> str:
    pcm = run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(path), "-map", "0:a:0", "-f", "s16le", "-acodec", "pcm_s16le", "-"], binary=True)
    return hashlib.sha256(pcm).hexdigest()


def quality(reference: Path, candidate: Path, metric: str) -> float:
    log = run(["ffmpeg", "-nostdin", "-hide_banner", "-i", str(reference), "-i", str(candidate), "-lavfi", metric, "-an", "-f", "null", "-"])
    pattern = r"average:([\d.]+|inf)" if metric == "psnr" else r"All:([\d.]+)"
    match = re.search(pattern, log)
    if not match:
        raise ValueError(f"Could not parse {metric} for {candidate}: {log[-900:]}")
    return float(match.group(1))


def collect_metrics() -> dict:
    reference = RESULTS / "raccoon_baseline_assembled.mp4"
    ref_audio_hash = audio_hash(reference)
    records = {}
    for key, *_ in APPROACHES:
        video = RESULTS / f"raccoon_{key}_assembled.mp4"
        info = probe(video)
        if info["frames"] != 1186 or info["width"] != 1080 or info["height"] != 1920 or info["fps"] != "24/1":
            raise ValueError(f"Unexpected video dimensions or length for {video}: {info}")
        records[key] = {
            **info,
            "video": video.name,
            "bytes": video.stat().st_size,
            "psnr_db_vs_baseline": None if key == "baseline" else quality(reference, video, "psnr"),
            "ssim_vs_baseline": 1.0 if key == "baseline" else quality(reference, video, "ssim"),
            "audio_pcm_exact_match": audio_hash(video) == ref_audio_hash,
        }
        print(f"{key}: PSNR {records[key]['psnr_db_vs_baseline']}, SSIM {records[key]['ssim_vs_baseline']}, audio exact {records[key]['audio_pcm_exact_match']}", flush=True)
    output = {"reference": reference.name, "comparison": "Frame-aligned decoded stitched videos", "records": records}
    (DEST / "report_metrics.json").write_text(json.dumps(output, indent=2) + "\n")
    return output


def e(value: object) -> str:
    return html.escape(str(value), quote=True)


def build_html(metrics: dict) -> str:
    baseline = FINDINGS["baseline_seconds"]
    cards = []
    rows = []
    for index, (key, name, diagram, seconds, measurement, description) in enumerate(APPROACHES, 1):
        record = metrics["records"][key]
        saved = baseline - seconds
        percent = saved / baseline * 100
        ready = seconds + STITCH[key]
        psnr = "Reference" if key == "baseline" else f"{record['psnr_db_vs_baseline']:.3f} dB"
        ssim = f"{record['ssim_vs_baseline']:.6f}"
        video_url = "../" + record["video"]
        diagram_url = "../diagrams/compact/" + diagram
        badge = "measured" if measurement == "Direct run" else "estimated"
        output_description = ("One final encode · 49.42 s · 1,186 frames · 1080 × 1920 · 24 fps"
                              if key in {"full_nvenc_single", "full_nvenc_matched"} else
                              "Four chunks stitched into one video · 49.42 s · 1,186 frames · 1080 × 1920 · 24 fps")
        rows.append(f'''<tr><td><a href="#a-{e(key)}">{e(name)}</a></td><td><strong>{seconds:.3f} s</strong></td><td>{ready:.3f} s</td><td>{saved:.3f} s</td><td>{percent:.2f}%</td><td>{record["bytes"]/1_000_000:.2f} MB</td><td>{e(psnr)}</td><td>{ssim}</td><td><span class="pill {badge}">{e(measurement)}</span></td></tr>''')
        cards.append(f'''
        <article class="approach" id="a-{e(key)}">
          <div class="card-head"><div><div class="eyebrow">APPROACH {index:02d}</div><h3>{e(name)}</h3><p>{e(description)}</p></div><span class="pill {badge}">{e(measurement)}</span></div>
          <img class="diagram" src="{e(diagram_url)}" alt="Architecture diagram for {e(name)}" loading="lazy">
          <div class="metric-strip"><div><small>Generation</small><b>{seconds:.3f} s</b></div><div><small>Stitch</small><b>{STITCH[key]:.3f} s</b></div><div><small>Ready video</small><b>{ready:.3f} s</b></div><div><small>Saved vs separate requests</small><b>{saved:.3f} s <em>({percent:.2f}%)</em></b></div><div><small>PSNR / SSIM</small><b>{e(psnr)} / {ssim}</b></div></div>
          <div class="media"><video controls playsinline preload="metadata" src="{e(video_url)}" aria-label="Complete 49-second video generated with {e(name)}"></video><div class="media-caption"><span>{output_description}</span><a href="{e(video_url)}" download>Download MP4 ↗</a></div></div>
        </article>''')

    bar_rows = []
    max_seconds = baseline
    for key, name, _, seconds, measurement, _ in APPROACHES:
        width = seconds / max_seconds * 100
        variant_class = "gradient" if key == "full_async" else ""
        bar_rows.append(f'<div class="bar-row"><span>{e(name)}</span><div class="bar-track"><div class="bar {variant_class}" style="width:{width:.3f}%"></div></div><b>{seconds:.1f} s</b><small>{"●" if measurement == "Direct run" else "○"}</small></div>')

    fastest = min(APPROACHES, key=lambda approach: approach[3])
    fastest_name, fastest_seconds = fastest[1], fastest[3]
    fastest_ready = min(APPROACHES, key=lambda approach: approach[3] + STITCH[approach[0]])
    fastest_ready_name = fastest_ready[1]
    fastest_ready_seconds = fastest_ready[3] + STITCH[fastest_ready[0]]
    async_ready = DIRECT["full_async"] + STITCH["full_async"]
    deferred_ready = DIRECT["full_deferred_parallel"] + STITCH["full_deferred_parallel"]
    nvenc_ready = DIRECT["full_nvenc_single"]
    matched_ready = DIRECT["full_nvenc_matched"]
    deferred_save = json.loads((DEST / "full_deferred_parallel_save.json").read_text())["save_wall_seconds"]
    nvenc_save = json.loads((DEST / "full_nvenc_single_save.json").read_text())["save_wall_seconds"]
    matched_save = json.loads((DEST / "full_nvenc_matched_save.json").read_text())["save_wall_seconds"]
    async_wait = next(node["duration_seconds"] for node in json.loads((DEST / "full_async.json").read_text())["stages"]["nodes"]
                      if node["node_id"] == "wait_saves")
    size_ratio = SAVE_METRICS["full_nvenc_single"]["bytes"] / SAVE_METRICS["full_async"]["bytes"]
    async_size_mb = SAVE_METRICS["full_async"]["bytes"] / 1_000_000
    nvenc_size_mb = SAVE_METRICS["full_nvenc_single"]["bytes"] / 1_000_000
    matched_size_mb = SAVE_METRICS["full_nvenc_matched"]["bytes"] / 1_000_000
    same_x264 = SAME_SOURCE["x264_veryfast_crf20"]
    same_nvenc = SAME_SOURCE["nvenc_p5_vbr_3500"]
    script = """const buttons=document.querySelectorAll('[data-sort]');buttons.forEach(b=>b.addEventListener('click',()=>{const key=b.dataset.sort;const tr=[...document.querySelectorAll('#results tbody tr')];const idx={time:1,ready:2,saved:3,size:5,psnr:6,ssim:7}[key];tr.sort((a,b)=>{const aa=parseFloat(a.children[idx].textContent),bb=parseFloat(b.children[idx].textContent);return key==='time'||key==='ready'||key==='size'?aa-bb:bb-aa});document.querySelector('#results tbody').replaceChildren(...tr);buttons.forEach(x=>x.classList.toggle('active',x===b))}));document.querySelector('[data-sort="time"]').click();"""
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>H3 Flex · Chunk Scheduling Benchmark</title>
<style>
:root{{--ink:#18253e;--muted:#61708c;--bg:#f5f7fb;--line:#dce3ef;--blue:#6562f4;--teal:#13a1a4;--orange:#f08756}}
*{{box-sizing:border-box}}html{{scroll-behavior:smooth}}body{{margin:0;background:var(--bg);color:var(--ink);font:16px/1.5 system-ui,-apple-system,Segoe UI,sans-serif}}a{{color:#5557da;text-decoration:none}}a:hover{{text-decoration:underline}}.shell{{max-width:1320px;margin:auto;padding:36px 24px 84px}}.hero{{background:#172948;color:white;border-radius:24px;padding:44px 48px;position:relative;overflow:hidden}}.hero:after{{content:'';position:absolute;width:480px;height:480px;right:-130px;top:-270px;background:radial-gradient(circle,#6b66ea99,transparent 68%);pointer-events:none}}.eyebrow{{font-size:12px;letter-spacing:.14em;font-weight:800;color:#8193be}}.hero .eyebrow{{color:#aab9df}}h1{{font-size:clamp(32px,4vw,54px);line-height:1.1;letter-spacing:-.04em;margin:12px 0 18px}}.hero p{{max-width:830px;color:#d6e0f2;margin:0;font-size:18px}}.hero-pills{{display:flex;flex-wrap:wrap;gap:10px;margin-top:24px}}.hero-pills span{{background:#ffffff18;border:1px solid #ffffff30;padding:7px 12px;border-radius:99px;font-size:13px}}.section{{margin-top:40px}}h2{{font-size:29px;letter-spacing:-.03em;margin:0 0 12px}}.intro{{color:var(--muted);margin:0 0 24px}}.stats{{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-top:24px}}.stat{{background:white;border:1px solid var(--line);border-radius:16px;padding:20px}}.stat small,.metric-strip small{{display:block;color:var(--muted);font-size:12px;font-weight:700;text-transform:uppercase;letter-spacing:.05em}}.stat b{{font-size:28px;line-height:1.2;display:block;margin-top:5px}}.stat span{{color:var(--muted);font-size:13px}}.panel{{background:white;border:1px solid var(--line);border-radius:18px;padding:28px}}.bar-row{{display:grid;grid-template-columns:220px 1fr 76px 18px;align-items:center;gap:14px;margin:12px 0;font-size:13px}}.bar-row b{{font-variant-numeric:tabular-nums;text-align:right}}.bar-row small{{color:var(--muted)}}.bar-track{{height:17px;border-radius:20px;background:#eef1fa;overflow:hidden}}.bar{{height:100%;background:linear-gradient(90deg,#7b78f7,#4f61dc);border-radius:20px}}.bar.gradient{{background:linear-gradient(90deg,#10b3aa,#268aab)}}.legend{{color:var(--muted);font-size:12px;margin-top:14px}}.table-scroll{{overflow:auto}}table{{width:100%;border-collapse:collapse;min-width:1050px;font-size:14px}}th{{text-align:left;color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.06em;white-space:nowrap}}td,th{{padding:14px 12px;border-bottom:1px solid var(--line)}}td:not(:first-child){{font-variant-numeric:tabular-nums;white-space:nowrap}}tr:last-child td{{border:0}}.controls{{display:flex;gap:8px;flex-wrap:wrap;margin:0 0 14px}}button{{font:inherit;background:white;color:var(--ink);border:1px solid var(--line);padding:7px 12px;border-radius:9px;cursor:pointer}}button.active{{background:#e9e9ff;border-color:#7775f0;color:#4441ba}}.approach{{background:white;border:1px solid var(--line);border-radius:20px;padding:28px;margin-top:24px;scroll-margin-top:24px}}.card-head{{display:flex;align-items:start;justify-content:space-between;gap:24px}}h3{{font-size:26px;letter-spacing:-.025em;margin:3px 0 5px}}.card-head p{{margin:0;color:var(--muted)}}.pill{{display:inline-block;white-space:nowrap;font-size:11px;font-weight:800;text-transform:uppercase;letter-spacing:.05em;border-radius:99px;padding:6px 10px}}.pill.measured{{color:#087e77;background:#def6f1}}.pill.estimated{{color:#865b19;background:#fff1d7}}.pill.blocked{{color:#a0443b;background:#fee6df}}.diagram{{width:100%;height:auto;margin:22px 0;border-radius:12px;border:1px solid #e4e8f1;display:block}}.metric-strip{{display:grid;grid-template-columns:repeat(5,1fr);gap:12px;margin:0 0 20px}}.metric-strip>div{{background:#f6f8fc;padding:12px 14px;border-radius:12px}}.metric-strip b{{display:block;font-size:18px;white-space:nowrap;font-variant-numeric:tabular-nums}}.metric-strip em{{font-style:normal;color:var(--muted);font-size:14px}}.media{{display:grid;grid-template-columns:minmax(220px,390px) 1fr;align-items:end;gap:24px}}video{{width:100%;background:#101828;max-height:700px;border-radius:12px;display:block}}.media-caption{{color:var(--muted);font-size:13px;display:grid;gap:10px;align-self:center}}.media-caption a{{font-weight:700}}.note{{padding:18px 22px;background:#eef3fb;border-left:4px solid #7675ec;border-radius:8px;color:#344666}}.note p{{margin:0 0 8px}}.note p:last-child{{margin:0}}.method{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}.method .panel p{{margin:8px 0 0;color:var(--muted)}}.method h3{{font-size:18px}}.rejected{{border-left:5px solid #ed907e}}.rejected .diagram{{max-width:100%}}footer{{margin-top:30px;color:var(--muted);font-size:12px}}
@media(max-width:850px){{.stats{{grid-template-columns:repeat(2,1fr)}}.metric-strip{{grid-template-columns:repeat(2,1fr)}}.media{{grid-template-columns:1fr}}.media video{{max-width:420px}}.bar-row{{grid-template-columns:150px 1fr 66px 12px;gap:8px}}.method{{grid-template-columns:1fr}}}}@media(max-width:540px){{.shell{{padding:18px 12px 60px}}.hero{{padding:30px 22px}}.stats{{grid-template-columns:1fr 1fr}}.stat{{padding:15px}}.stat b{{font-size:22px}}.approach{{padding:18px}}.card-head{{display:block}}.card-head .pill{{margin-top:12px}}.bar-row{{grid-template-columns:1fr 78px;gap:2px}}.bar-track{{grid-column:1}}.bar-row b{{grid-column:2;grid-row:1}}.bar-row small{{display:none}}.metric-strip{{grid-template-columns:1fr 1fr}}.metric-strip b{{font-size:15px}}}}
{DARK_CSS}
</style></head><body><main class="shell">
<header class="hero"><div class="eyebrow">EXPERIMENT REPORT · RTX 5090 · PIKA FLEX / H3</div><h1>Chunk processing benchmark</h1><p>Four fixed raccoon reference chunks produce one 49.42-second video for each successful architecture. Generation times use the same model, prompts, audio, and two-step Pika Flex settings.</p><div class="hero-pills"><span>1 GPU</span><span>4 input chunks</span><span>{len(APPROACHES)} playable architectures</span><span>1 rejected batch experiment</span></div></header>
<div class="stats"><div class="stat"><small>Separate requests</small><b>{baseline:.3f} s</b><span>Measured generation time</span></div><div class="stat"><small>Fastest generation</small><b>{fastest_seconds:.3f} s</b><span>{e(fastest_name)}</span></div><div class="stat"><small>Fastest ready video</small><b>{fastest_ready_seconds:.3f} s</b><span>{e(fastest_ready_name)}</span></div><div class="stat"><small>Output length</small><b>49.42 s</b><span>1,186 frames, 24 fps</span></div></div>
<section class="section"><h2>Generation time</h2><p class="intro">Lower is better. Every bar is a direct full-workflow run on the single RTX 5090.</p><div class="panel">{''.join(bar_rows)}<div class="legend">● Directly measured full run. Final stitching is excluded from every bar and shown separately in the table; the single-file NVENC paths need no stitch.</div></div></section>
<section class="section"><h2>Saving paths compared</h2><p class="intro">The CPU path overlaps chunk saves with VAE decoding and stitches four MP4s. The size-matched NVENC path streams decoded frames into one MP4 after all VAE calls.</p><img class="diagram" src="../diagrams/compact/10-save-strategies-compared.png" alt="Architecture comparison of asynchronous CPU saves with final stitch versus size-matched single-file NVENC encoding" loading="lazy"></section>
<section class="section"><h2>Save-stage follow-up</h2><p class="intro">These paths use the same four chunk inputs and sequential H3/VAE stages. They differ in when and how encoding starts. Each timing is one direct run.</p><div class="method"><div class="panel"><h3>Async saves versus deferred saves</h3><p>Overlapping saves produced a ready video in <strong>{async_ready:.3f} s</strong>. Waiting until all decodes finished, then starting four CPU encodes together, took <strong>{deferred_ready:.3f} s</strong> including the same final stitch: <strong>{deferred_ready-async_ready:.3f} s slower</strong>. The deferred encodes occupied {deferred_save:.3f} s after decoding; the async workflow waited {async_wait:.3f} s for saves at the end.</p></div><div class="panel"><h3>Original NVENC, unconstrained size</h3><p>One NVENC H.264 save took {nvenc_save:.3f} s after decoding and produced the finished video in <strong>{nvenc_ready:.3f} s</strong>, with no merge. Its CQ20 output was {nvenc_size_mb:.1f} MB versus {async_size_mb:.1f} MB for the CPU path ({size_ratio:.2f}× larger), so this setting does not meet the file-size constraint.</p></div><div class="panel"><h3>NVENC tuned to CPU file size</h3><p>NVENC p5 with VBR at 3.7 Mb/s, full-resolution multipass, and adaptive quantization took {matched_save:.3f} s to save one MP4 after decoding. The direct run finished in <strong>{matched_ready:.3f} s</strong> with a <strong>{matched_size_mb:.2f} MB</strong> file, below the CPU path's {async_size_mb:.2f} MB. It finished {async_ready-matched_ready:.3f} s sooner than async saving in these single runs. A few seconds of difference needs repeated runs before treating it as a stable speed win.</p></div><div class="panel"><h3>Same-source codec check</h3><p>To remove H3 run variation, the same decoded CPU-path MP4 was re-encoded once with each encoder. x264 veryfast CRF20 produced {same_x264['bytes']/1_000_000:.2f} MB, {same_x264['psnr_db']:.3f} dB PSNR, and {same_x264['ssim']:.6f} SSIM versus that common source. NVENC p5 at 3.5 Mb/s produced {same_nvenc['bytes']/1_000_000:.2f} MB, {same_nvenc['psnr_db']:.3f} dB, and {same_nvenc['ssim']:.6f}. Both results include a second lossy encode, so this check compares encoders under the same input rather than measuring raw VAE output quality.</p></div></div></section>
<section class="section"><h2>Results at a glance</h2><p class="intro">Ready-video time adds a separately measured final stitch to generation. PSNR and SSIM compare corresponding decoded frames in the complete output against the separate-request baseline. Higher means closer to that reference render.</p><div class="controls"><button type="button" class="active" data-sort="time">Fastest generation</button><button type="button" data-sort="ready">Fastest ready video</button><button type="button" data-sort="saved">Most time saved</button><button type="button" data-sort="size">Smallest MP4</button><button type="button" data-sort="psnr">Highest PSNR</button><button type="button" data-sort="ssim">Highest SSIM</button></div><div class="panel table-scroll"><table id="results"><thead><tr><th>Architecture</th><th>Generation</th><th>Ready video</th><th>Time saved</th><th>Percent saved</th><th>MP4 size</th><th>PSNR</th><th>SSIM</th><th>Timing basis</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div></section>
<section class="section"><h2>Architectures and videos</h2><p class="intro">Each player contains the complete 49.42-second output. Most paths stitch four chunk files; both NVENC paths encode one complete MP4 directly.</p>{''.join(cards)}</section>
<section class="section"><h2>Batch size 2 test</h2><article class="approach rejected"><div class="card-head"><div><div class="eyebrow">APPROACH 10</div><h3>Parallel H3 denoising batch</h3><p>Attempted to denoise two chunks together on the single RTX 5090.</p></div><span class="pill blocked">Rejected by model</span></div><img class="diagram" src="../diagrams/compact/10-h3-batch-rejected.png" alt="Architecture diagram showing unsupported H3 batch-size-two denoising" loading="lazy"><div class="note"><p><strong>No generation time or video.</strong> The H3 forward pass raises: <code>{e(SUMMARY['denoise_batch_2'])}</code>.</p><p>There is therefore no PSNR or SSIM result for this approach.</p></div></article></section>
<section class="section"><h2>How to read these numbers</h2><div class="method"><div class="panel"><h3>Timing basis</h3><p>All {len(APPROACHES)} table values are timed full generation runs on the same single RTX 5090. The async-save graph waits for every H3 sampler before VAE decoding and moves the four latents to CPU at that barrier. In its direct run, the chunk execution order was 4 → 3 → 2 → 1. ComfyUI loads VAE weights for decoding and dynamically offloads H3 weights as needed; the graph has no explicit complete-unload step.</p></div><div class="panel"><h3>Image comparison</h3><p>FFmpeg PSNR and SSIM filters compare aligned, decoded video frames against the separate-request baseline. All {len(APPROACHES)} complete outputs contain 1,186 frames at 1080 × 1920 and 24 fps. Baseline is the reference (SSIM 1.0; PSNR would be infinite). The decoded audio stream matches the reference exactly in all outputs.</p></div><div class="panel"><h3>What PSNR and SSIM mean here</h3><p>They measure similarity to one baseline render, not artistic quality. A repeated unchanged serial run also differs from that baseline, so small metric differences can reflect generation or encoding variation. The timings are single-run observations; differences of a few seconds need repeated runs to establish a stable ranking.</p></div><div class="panel"><h3>Stitch and source</h3><p>Final stitching is timed separately and added for paths with chunk MP4s; both NVENC paths have zero stitch time. The four chunk durations yield 293, 295, 290 and 308 output frames. Inputs: <a href="../../raccoon_50s_reference.json">raccoon_50s_reference.json</a>. Raw timing: <a href="../findings.json">findings.json</a>, <a href="summary.json">summary.json</a>, <a href="full_pairs.json">full_pairs.json</a>, <a href="full_async.json">full_async.json</a>, <a href="full_deferred_parallel.json">full_deferred_parallel.json</a>, <a href="full_nvenc_single.json">full_nvenc_single.json</a>, <a href="full_nvenc_matched.json">full_nvenc_matched.json</a>, and <a href="stitch_timings.json">stitch_timings.json</a>. Frame comparison: <a href="report_metrics.json">report_metrics.json</a>.</p></div></div></section>
<footer>Generated locally from the benchmark outputs. Every embedded video is an assembled complete output for its architecture.</footer>
</main><script>{script}</script></body></html>'''


if __name__ == "__main__":
    metrics = collect_metrics()
    output = DEST / "report.html"
    output.write_text(build_html(metrics))
    print(f"Report: {output}")
