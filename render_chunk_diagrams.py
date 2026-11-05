"""Render compact, image-first architecture diagrams for the tested schedules."""

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


OUT = Path(__file__).resolve().parent / "results" / "diagrams" / "compact"
OUT.mkdir(parents=True, exist_ok=True)


def direct_seconds(variant):
    path = OUT.parent.parent / "chunk_processing" / f"{variant}.json"
    return json.loads(path.read_text())["seconds"]

W, H = 1600, 430
NAVY = "#17233A"
MUTED = "#546178"
GRAY = "#748097"
BACKGROUND = "#F7F8FB"
BORDER = "#DBE0EA"
PURPLE = "#7058FF"
PURPLE_FILL = "#F0ECFF"
ORANGE = "#F68B55"
ORANGE_FILL = "#FFF0E8"
TEAL = "#1FA6A7"
TEAL_FILL = "#E7F7F7"
RED = "#D25E60"
RED_FILL = "#FFF0F0"
BLUE_FILL = "#EAF0FA"

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def f(size, bold=False):
    return ImageFont.truetype(BOLD if bold else FONT, size)


def text(draw, xy, value, size=24, color=NAVY, bold=False):
    draw.text(xy, value, font=f(size, bold), fill=color)


def centered(draw, box, value, size=24, color=NAVY, bold=False):
    x0, y0, x1, y1 = box
    bounds = draw.textbbox((0, 0), value, font=f(size, bold))
    tw, th = bounds[2] - bounds[0], bounds[3] - bounds[1]
    draw.text(((x0 + x1 - tw) / 2, (y0 + y1 - th) / 2 - bounds[1]),
              value, font=f(size, bold), fill=color)


def base(title, subtitle, badge):
    im = Image.new("RGB", (W, H), BACKGROUND)
    d = ImageDraw.Draw(im)
    text(d, (54, 30), title, 43, NAVY, True)
    text(d, (56, 92), subtitle, 23, MUTED)
    d.rounded_rectangle((50, 146, 1550, 385), radius=20, fill="white", outline=BORDER, width=2)
    d.rounded_rectangle((78, 165, 78 + max(270, len(badge) * 17), 203), radius=19, fill="#EFF1F5")
    text(d, (100, 174), badge.upper(), 19, MUTED, True)
    return im, d


def card(d, box, title, sub, color, fill, dark=False, icon=None, title_size=26, sub_size=19):
    x0, y0, x1, y1 = box
    d.rounded_rectangle(box, radius=14, fill=fill, outline=color, width=3)
    foreground = "white" if dark else NAVY
    secondary = "#CED9EE" if dark else MUTED
    if icon:
        cx, cy, r = x0 + 42, (y0 + y1) / 2, 22
        d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=color if not dark else "#425980")
        centered(d, (cx - r, cy - r, cx + r, cy + r), icon, 21, "white", True)
        tx = x0 + 78
    else:
        tx = x0 + 23
    if sub:
        text(d, (tx, y0 + 22), title, title_size, foreground, True)
        text(d, (tx, y0 + 62), sub, sub_size, secondary)
    else:
        text(d, (tx, y0 + (y1 - y0 - title_size) / 2 - 5), title, title_size, foreground, True)


def arrow(d, start, end, color=GRAY, width=5):
    x0, y0 = start
    x1, y1 = end
    d.line((x0, y0, x1, y1), fill=color, width=width)
    if abs(x1 - x0) >= abs(y1 - y0):
        s = 1 if x1 > x0 else -1
        d.polygon(((x1, y1), (x1 - s * 14, y1 - 9), (x1 - s * 14, y1 + 9)), fill=color)
    else:
        s = 1 if y1 > y0 else -1
        d.polygon(((x1, y1), (x1 - 9, y1 - s * 14), (x1 + 9, y1 - s * 14)), fill=color)


def poly_arrow(d, points, color=GRAY, width=4):
    d.line(points, fill=color, width=width, joint="curve")
    arrow(d, points[-2], points[-1], color, width)


def lane(d, x, y, label, color):
    d.rounded_rectangle((x, y, x + 74, y + 34), radius=17, fill=color)
    centered(d, (x, y, x + 74, y + 34), label, 17, "white", True)


def save(im, name):
    path = OUT / f"{name}.png"
    im.save(path, optimize=True)
    return path


def separate():
    im, d = base("1  Separate chunk requests", "432.5 s measured  ·  A full chunk finishes before the next starts", "one GPU · serial requests")
    xs = [78, 456, 834, 1212]
    for i, x in enumerate(xs, 1):
        card(d, (x, 232, x + 310, 340), f"Chunk {i}", "Prepare · H3 · VAE · save", PURPLE, PURPLE_FILL, icon=str(i), title_size=26, sub_size=17)
        if i < 4:
            arrow(d, (x + 320, 286), (xs[i] - 14, 286))
    return save(im, "01-separate-requests")


def shared():
    im, d = base("2  Shared loader graph", "416.6 s measured  ·  Four independent branches use the same loader nodes", "one graph · no stage barriers")
    card(d, (100, 211, 1500, 269), "Shared text encoder  +  H3 / Pika  +  video VAE", None, PURPLE, PURPLE_FILL, title_size=23)
    xs = [100, 460, 820, 1180]
    for i, x in enumerate(xs, 1):
        arrow(d, (x + 160, 269), (x + 160, 289), PURPLE, 4)
        card(d, (x, 295, x + 320, 358), f"Chunk {i}: P → H3 → VAE", None, TEAL, TEAL_FILL, title_size=21)
    return save(im, "02-shared-loader-graph")


def staged():
    im, d = base("3  Group by model stage", "415.4 s measured  ·  One stage completes across all chunks before switching", "prepare all · denoise all · decode all")
    specs = [
        (80, "Prepare 1–4", "Text + inputs", PURPLE, PURPLE_FILL, "P"),
        (380, "Denoise 1–4", "H3 / Pika", NAVY, NAVY, "D"),
        (680, "CPU latents", "Four tensors", GRAY, BLUE_FILL, "L"),
        (980, "Decode 1–4", "Video VAE", ORANGE, ORANGE_FILL, "V"),
        (1280, "Save 1–4", "Then stitch", TEAL, TEAL_FILL, "S"),
    ]
    for idx, (x, title, sub, color, fill, icon) in enumerate(specs):
        card(d, (x, 229, x + 240, 344), title, sub, color, fill,
             dark=color == NAVY, icon=icon, title_size=21, sub_size=17)
        if idx < len(specs) - 1:
            arrow(d, (x + 247, 287), (specs[idx + 1][0] - 8, 287))
    return save(im, "03-stage-grouped")


def pair_vae():
    im, d = base("4  Pair VAE decoding", f"{direct_seconds('full_pairs'):.1f} s direct run  ·  Match chunks by latent length", "same denoising · batch two during decode")
    card(d, (82, 238, 405, 331), "H3: chunks 1–4", "Serial denoising", NAVY, NAVY, dark=True, icon="D", title_size=23)
    card(d, (555, 211, 956, 275), "VAE batch: 1 + 3", None, ORANGE, ORANGE_FILL, icon="V", title_size=23)
    card(d, (555, 296, 956, 360), "VAE batch: 2 + 4", None, ORANGE, ORANGE_FILL, icon="V", title_size=23)
    card(d, (1100, 238, 1517, 331), "Split → save 1–4", "Two outputs per VAE call", TEAL, TEAL_FILL, icon="S", title_size=23)
    poly_arrow(d, [(405, 284), (470, 284), (470, 243), (544, 243)])
    poly_arrow(d, [(405, 284), (470, 284), (470, 328), (544, 328)])
    poly_arrow(d, [(956, 243), (1022, 243), (1022, 284), (1088, 284)])
    poly_arrow(d, [(956, 328), (1022, 328), (1022, 284), (1088, 284)])
    return save(im, "04-paired-vae")


def async_save():
    im, d = base("5  Save while the next chunk decodes", f"{direct_seconds('full_async'):.1f} s direct run  ·  Observed decode order: 4 → 3 → 2 → 1", "GPU decode · CPU video save")
    lane(d, 78, 237, "GPU", ORANGE)
    lane(d, 78, 318, "CPU", TEAL)
    xs = [180, 505, 830, 1155]
    observed_order = [4, 3, 2, 1]
    for position, (i, x) in enumerate(zip(observed_order, xs)):
        card(d, (x, 218, x + 245, 284), f"VAE chunk {i}", None, ORANGE, ORANGE_FILL, title_size=21)
        if position < len(xs) - 1:
            arrow(d, (x + 253, 251), (xs[position + 1] - 9, 251), GRAY, 4)
    # Saving starts after a decode and runs alongside the next GPU card.
    for i, x in zip(observed_order, [425, 750, 1075, 1350]):
        end = min(x + 190, 1520)
        d.rounded_rectangle((x, 314, end, 356), radius=11, fill=TEAL_FILL, outline=TEAL, width=3)
        centered(d, (x, 314, end, 356), f"Save {i}", 19, NAVY, True)
    poly_arrow(d, [(425, 285), (425, 303), (520, 303), (520, 313)], TEAL, 3)
    poly_arrow(d, [(750, 285), (750, 303), (845, 303), (845, 313)], TEAL, 3)
    poly_arrow(d, [(1075, 285), (1075, 303), (1170, 303), (1170, 313)], TEAL, 3)
    poly_arrow(d, [(1400, 285), (1400, 303), (1430, 303), (1430, 313)], TEAL, 3)
    text(d, (820, 171), "A save overlaps the next VAE call", 20, MUTED)
    return save(im, "05-async-save")


def pair_async():
    im, d = base("6  Pair VAE + asynchronous save", "406.0 s direct run  ·  Two VAE calls with overlapping saves", "two VAE calls · overlapped saves")
    lane(d, 78, 227, "GPU", ORANGE)
    lane(d, 78, 323, "CPU", TEAL)
    card(d, (190, 215, 635, 281), "VAE batch: 1 + 3", None, ORANGE, ORANGE_FILL, title_size=24)
    card(d, (785, 215, 1230, 281), "VAE batch: 2 + 4", None, ORANGE, ORANGE_FILL, title_size=24)
    arrow(d, (648, 248), (773, 248), GRAY, 4)
    card(d, (685, 312, 1000, 365), "Save 1 + 3", None, TEAL, TEAL_FILL, title_size=20)
    card(d, (1170, 312, 1500, 365), "Save 2 + 4", None, TEAL, TEAL_FILL, title_size=20)
    poly_arrow(d, [(635, 281), (635, 338), (674, 338)], TEAL, 3)
    poly_arrow(d, [(1230, 281), (1230, 300), (1335, 300), (1335, 311)], TEAL, 3)
    return save(im, "06-paired-vae-async-save")


def denied_batch():
    im, d = base("10  Two-chunk H3 denoising batch", "Runtime test blocked  ·  The current H3 forward pass accepts batch size 1 only", "tested · unsupported")
    card(d, (80, 235, 380, 339), "Chunks 1 + 3", "Independent inputs", PURPLE, PURPLE_FILL, icon="2", title_size=22)
    arrow(d, (391, 287), (480, 287))
    card(d, (490, 235, 790, 339), "Stack latents", "Batch size = 2", GRAY, BLUE_FILL, icon="B", title_size=22)
    arrow(d, (801, 287), (890, 287))
    card(d, (900, 235, 1200, 339), "H3 denoiser", "Forward pass", NAVY, NAVY, dark=True, icon="D", title_size=22)
    arrow(d, (1211, 287), (1300, 287), RED)
    card(d, (1310, 235, 1520, 339), "Rejected", "batch size 1 only", RED, RED_FILL, title_size=22, sub_size=17)
    return save(im, "10-h3-batch-rejected")


def deferred_parallel():
    im, d = base("7  Save all chunks after decoding", f"{direct_seconds('full_deferred_parallel'):.1f} s generation  ·  Four CPU encodes begin after all VAE calls", "deferred · four parallel saves")
    card(d, (80, 226, 350, 340), "VAE chunks 4–1", "GPU, one at a time", ORANGE, ORANGE_FILL, icon="V", title_size=21, sub_size=17)
    arrow(d, (361, 283), (440, 283))
    card(d, (450, 226, 680, 340), "All decoded", "Wait at barrier", GRAY, BLUE_FILL, icon="B", title_size=21, sub_size=17)
    arrow(d, (691, 283), (770, 283))
    card(d, (780, 226, 1190, 340), "CPU encode 1–4", "Four saves run together", TEAL, TEAL_FILL, icon="4", title_size=21, sub_size=17)
    arrow(d, (1201, 283), (1280, 283))
    card(d, (1290, 226, 1520, 340), "Merge", "Final MP4", NAVY, NAVY, dark=True, title_size=22, sub_size=17)
    return save(im, "07-deferred-parallel-save")


def single_nvenc():
    im, d = base("8  One final NVENC save", f"{direct_seconds('full_nvenc_single'):.1f} s to finished video  ·  No intermediate MP4s or merge", "decode all · encode once")
    card(d, (80, 226, 360, 340), "VAE chunks 4–1", "GPU, one at a time", ORANGE, ORANGE_FILL, icon="V", title_size=21, sub_size=17)
    arrow(d, (371, 283), (450, 283))
    card(d, (460, 226, 740, 340), "Frames 1→4", "Trim + scale · 1,186", PURPLE, PURPLE_FILL, icon="F", title_size=21, sub_size=17)
    arrow(d, (751, 283), (830, 283))
    card(d, (840, 226, 1190, 340), "FFmpeg NVENC", "H.264 + four WAVs", TEAL, TEAL_FILL, icon="N", title_size=21, sub_size=17)
    arrow(d, (1201, 283), (1280, 283))
    card(d, (1290, 226, 1520, 340), "One MP4", "49.42 seconds", NAVY, NAVY, dark=True, title_size=22, sub_size=17)
    return save(im, "08-single-nvenc-save")


def matched_nvenc():
    size_mb = (OUT.parent.parent / "raccoon_full_nvenc_matched_assembled.mp4").stat().st_size / 1_000_000
    im, d = base("9  Size-matched final NVENC", f"{direct_seconds('full_nvenc_matched'):.1f} s to finished video  ·  {size_mb:.2f} MB output", "decode all · encode once")
    card(d, (80, 226, 360, 340), "VAE chunks 4–1", "GPU, one at a time", ORANGE, ORANGE_FILL, icon="V", title_size=21, sub_size=17)
    arrow(d, (371, 283), (450, 283))
    card(d, (460, 226, 740, 340), "Frames 1→4", "Trim + scale · 1,186", PURPLE, PURPLE_FILL, icon="F", title_size=21, sub_size=17)
    arrow(d, (751, 283), (830, 283))
    card(d, (840, 226, 1190, 340), "FFmpeg NVENC", "H.264 VBR · 3.7 Mb/s", TEAL, TEAL_FILL, icon="N", title_size=21, sub_size=17)
    arrow(d, (1201, 283), (1280, 283))
    card(d, (1290, 226, 1520, 340), "One MP4", "No merge", NAVY, NAVY, dark=True, title_size=22, sub_size=17)
    return save(im, "09-size-matched-nvenc")


if __name__ == "__main__":
    for render in (separate, shared, staged, pair_vae, async_save, pair_async,
                   deferred_parallel, single_nvenc, matched_nvenc, denied_batch):
        print(render())
