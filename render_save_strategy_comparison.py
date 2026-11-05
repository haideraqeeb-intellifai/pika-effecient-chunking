"""Render the tested asynchronous CPU-save and size-matched NVENC paths together."""

import json
from pathlib import Path

from PIL import Image, ImageDraw

from render_chunk_diagrams import (
    BACKGROUND, BORDER, BLUE_FILL, GRAY, MUTED, NAVY, ORANGE, ORANGE_FILL,
    PURPLE, PURPLE_FILL, TEAL, TEAL_FILL, arrow, card, centered, text,
)


WIDTH, HEIGHT = 1800, 760
ROOT = Path(__file__).resolve().parent / "results"
OUT = ROOT / "diagrams/compact/10-save-strategies-compared.png"


def result(variant):
    seconds = json.loads((ROOT / "chunk_processing" / f"{variant}.json").read_text())["seconds"]
    stitch = json.loads((ROOT / "chunk_processing/stitch_timings.json").read_text())[variant]
    size_mb = (ROOT / f"raccoon_{variant}_assembled.mp4").stat().st_size / 1_000_000
    return seconds + stitch, size_mb


def label(draw, x, y, value, color):
    draw.rounded_rectangle((x, y, x + 170, y + 36), radius=18, fill=color)
    centered(draw, (x, y, x + 170, y + 36), value, 18, "white", True)


def small_box(draw, box, title, color, fill, size=20):
    draw.rounded_rectangle(box, radius=12, fill=fill, outline=color, width=3)
    centered(draw, box, title, size, NAVY, True)


def render():
    cpu_seconds, cpu_size = result("full_async")
    nvenc_seconds, nvenc_size = result("full_nvenc_matched")
    image = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
    draw = ImageDraw.Draw(image)
    text(draw, (55, 28), "Two ways to save the same four decoded chunks", 40, NAVY, True)
    text(draw, (57, 87), "Both paths first prepare all inputs and denoise chunks 1–4 one at a time with H3.", 22, MUTED)

    draw.rounded_rectangle((50, 137, 1750, 420), radius=20, fill="white", outline=BORDER, width=2)
    label(draw, 80, 159, "CPU + merge", TEAL)
    text(draw, (270, 165), f"{cpu_seconds:.2f} s to final video", 22, NAVY, True)
    text(draw, (1300, 165), f"{cpu_size:.2f} MB  ·  4 → 1 MP4", 21, MUTED)
    text(draw, (88, 231), "GPU", 18, ORANGE, True)
    vae_x = [300, 560, 820, 1080]
    for position, x in enumerate(vae_x):
        chunk = 4 - position
        small_box(draw, (x, 211, x + 185, 279), f"VAE chunk {chunk}", ORANGE, ORANGE_FILL)
        if position < 3:
            arrow(draw, (x + 192, 245), (vae_x[position + 1] - 8, 245), GRAY, 4)
    text(draw, (88, 338), "CPU", 18, TEAL, True)
    save_x = [515, 775, 1035, 1295]
    for position, x in enumerate(save_x):
        chunk = 4 - position
        small_box(draw, (x, 316, x + 170, 374), f"Save {chunk}", TEAL, TEAL_FILL)
        source_x = vae_x[position] + 125
        draw.line((source_x, 279, source_x, 294, x + 50, 294, x + 50, 314), fill=TEAL, width=3)
        draw.polygon(((x + 50, 315), (x + 42, 302), (x + 58, 302)), fill=TEAL)
    small_box(draw, (1525, 316, 1695, 374), "Final stitch", NAVY, BLUE_FILL, 19)
    arrow(draw, (1472, 345), (1515, 345), GRAY, 4)
    text(draw, (300, 391), "A save overlaps the next VAE decode. Stitching waits for all four saves to finish.", 18, MUTED)

    draw.rounded_rectangle((50, 445, 1750, 710), radius=20, fill="white", outline=BORDER, width=2)
    label(draw, 80, 467, "NVENC · sized", PURPLE)
    text(draw, (270, 473), f"{nvenc_seconds:.2f} s to final video", 22, NAVY, True)
    text(draw, (1300, 473), f"{nvenc_size:.2f} MB  ·  0 → 1 MP4", 21, MUTED)
    card(draw, (300, 540, 600, 635), "VAE chunks 4–1", "Decode one at a time", ORANGE, ORANGE_FILL,
         icon="V", title_size=23, sub_size=17)
    arrow(draw, (610, 587), (683, 587), GRAY, 4)
    card(draw, (695, 540, 1045, 635), "Frames 1 + 2 + 3 + 4", "One continuous frame stream", PURPLE,
         PURPLE_FILL, icon="F", title_size=22, sub_size=17)
    arrow(draw, (1055, 587), (1128, 587), GRAY, 4)
    card(draw, (1140, 540, 1450, 635), "FFmpeg + NVENC", "p5 · VBR 3.7 Mb/s", TEAL,
         TEAL_FILL, icon="N", title_size=22, sub_size=17)
    arrow(draw, (1460, 587), (1515, 587), GRAY, 4)
    small_box(draw, (1525, 540, 1695, 635), "Final MP4", NAVY, BLUE_FILL, 21)
    text(draw, (300, 669), "Frames stream in output order; FFmpeg joins audio 1–4 and encodes one MP4.", 18, MUTED)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    image.save(OUT, optimize=True)
    print(OUT)


if __name__ == "__main__":
    render()
