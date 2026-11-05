"""Trim H3's frame-grid padding and assemble the raccoon reference timeline."""

import glob
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REFERENCE = json.loads((ROOT / "raccoon_50s_reference.json").read_text())
variant = sys.argv[1] if len(sys.argv) > 1 else "grouped"
videos = [Path(glob.glob(str(ROOT / "runtime/output" / variant / f"chunk_{i}" / "*.mp4"))[0])
          for i in range(1, 5)]
audios = [ROOT / "runtime/input/raccoon" / f"audio_{i}.wav" for i in range(1, 5)]
frames = []
for audio in audios:
    seconds = float(subprocess.check_output([
        "ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(audio)
    ], timeout=60))
    frames.append(round(seconds * 24))

filters = []
for index, count in enumerate(frames):
    filters.append(f"[{index}:v]trim=end_frame={count},setpts=PTS-STARTPTS,scale=1080:1920[v{index}]")
    filters.append(f"[{index+4}:a]asetpts=PTS-STARTPTS[a{index}]")
filters.append("".join(f"[v{i}]" for i in range(4)) + "concat=n=4:v=1:a=0[v]")
filters.append("".join(f"[a{i}]" for i in range(4)) + "concat=n=4:v=0:a=1[a]")
output = ROOT / "results" / f"raccoon_{variant}_assembled.mp4"
command = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y"]
for path in videos + audios:
    command += ["-i", str(path)]
command += ["-filter_complex", ";".join(filters), "-map", "[v]", "-map", "[a]",
            "-r", "24", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(output)]
started = time.perf_counter()
subprocess.run(command, check=True, timeout=600)
elapsed = time.perf_counter() - started
print(json.dumps({"variant": variant, "frames_per_chunk": frames, "total_frames": sum(frames),
                  "stitch_seconds": elapsed, "output": str(output)}, indent=2))
