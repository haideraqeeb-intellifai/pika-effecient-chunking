# Pika H3 chunk processing benchmark

This project compares scheduling, decoding, and video save strategies for four Pika H3 video chunks on an RTX 5090. The Python scripts run the experiments and build the report. The `results/` directory contains measured JSON, graphs, reports, videos, and captured latents; `runtime/` contains the experiment inputs and chunk videos.

Open `results/h3-chunk-processing-report-2026-11-05/index.html` to view the media-rich report. `results/chunk_processing/report.html` is the earlier local report.

## Reproducing the benchmark

1. Clone with submodules (`git clone --recurse-submodules ...`) to obtain SageAttention at the tested revision.
2. Provide the sibling `script2video` repository at `../script2video`. `benchmark.py` imports its `ml_modules.minimax_h3_turbo` and `workers.image2video_h3_turbo` modules. The version and access instructions for that sibling repository were not recorded in this experiment.
3. Install the Python dependencies required by `script2video` and SageAttention, along with `requests`, PyTorch, FFmpeg, and `ffprobe`. The local `.venv` is deliberately omitted because it is machine-specific; no frozen environment manifest was recorded.
4. Obtain the model files listed in [MODEL_REFERENCES.md](MODEL_REFERENCES.md) and place them under `assets/` with the listed names. The model files are not part of this public repository.
5. Run `python benchmark.py prepare` to fetch the source image and audio from the URLs in `raccoon_50s_reference.json`, then run `python benchmark.py run`. Run `python chunk_processing_benchmark.py --help` for the additional variants.

The saved reports and media can be inspected without rerunning inference. Reproducing inference requires access to the external code and model files above.
