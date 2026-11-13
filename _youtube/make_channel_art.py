#!/usr/bin/env python3
"""Generate channel art: banner (2048x1152), avatar (800x800), first thumbnail (1280x720).
Run from anywhere: python3 _youtube/make_channel_art.py
"""
import pathlib
from PIL import Image, ImageDraw, ImageFont, ImageFilter

OUT = pathlib.Path(__file__).resolve().parent.parent / "assets" / "images" / "youtube"
OUT.mkdir(parents=True, exist_ok=True)
F = "/usr/share/fonts/truetype/dejavu/"
BOLD, REG, MONO = F + "DejaVuSans-Bold.ttf", F + "DejaVuSans.ttf", F + "DejaVuSansMono-Bold.ttf"

BG0, BG1 = (10, 16, 32), (18, 30, 58)      # navy gradient
CYAN, ORANGE, WHITE, GREY = (56, 189, 248), (251, 146, 60), (240, 244, 250), (148, 163, 184)
SHARDS = [(56, 189, 248), (34, 197, 94), (251, 146, 60), (232, 121, 249)]


def gradient(w, h):
    im = Image.new("RGB", (w, h), BG0)
    px = im.load()
    for y in range(h):
        t = y / max(h - 1, 1)
        c = tuple(int(BG0[i] * (1 - t) + BG1[i] * t) for i in range(3))
        for x in range(w):
            px[x, y] = c
    return im


def grid(draw, w, h, step, color=(30, 45, 80)):
    for x in range(0, w, step):
        draw.line([(x, 0), (x, h)], fill=color, width=1)
    for y in range(0, h, step):
        draw.line([(0, y), (w, y)], fill=color, width=1)


def font(path, size):
    return ImageFont.truetype(path, size)


def text_w(draw, s, f):
    b = draw.textbbox((0, 0), s, font=f)
    return b[2] - b[0], b[3] - b[1]


def glow_text(im, xy, s, f, fill, glow=CYAN, radius=18):
    layer = Image.new("RGBA", im.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    d.text(xy, s, font=f, fill=glow + (160,))
    layer = layer.filter(ImageFilter.GaussianBlur(radius))
    im.paste(layer, (0, 0), layer)
    ImageDraw.Draw(im).text(xy, s, font=f, fill=fill)


def matrix(draw, x, y, cols, rows, cw, ch, colors, gap=3):
    """Draw a matrix as colored cells; colors[r] colors row r."""
    for r in range(rows):
        for c in range(cols):
            x0, y0 = x + c * (cw + gap), y + r * (ch + gap)
            draw.rectangle([x0, y0, x0 + cw, y0 + ch], fill=colors[r], outline=None)


def banner():
    W, H = 2048, 1152
    im = gradient(W, H)
    d = ImageDraw.Draw(im)
    grid(d, W, H, 64)
    # decorative matrices outside the safe area (desktop shows 2048x423 band = y 364..788)
    for i, (x, y) in enumerate([(120, 420), (1760, 420)]):
        cols = [SHARDS[(r // 2 + i) % 4] for r in range(8)]
        matrix(d, x, y, 6, 8, 22, 22, cols)
    # safe area: 1235x338 centered -> x 406..1641, y 407..745
    f1, f2, f3 = font(BOLD, 112), font(REG, 44), font(MONO, 34)
    title = "Aleksey Vlasenko"
    tw, th = text_w(d, title, f1)
    glow_text(im, ((W - tw) // 2, 430), title, f1, WHITE)
    d = ImageDraw.Draw(im)
    sub = "PyTorch on TPU  |  training performance  |  autoresearch"
    sw, sh = text_w(d, sub, f2)
    d.text(((W - sw) // 2, 585), sub, font=f2, fill=CYAN)
    url = "vlasenkoalexey.github.io"
    uw, uh = text_w(d, url, f3)
    d.text(((W - uw) // 2, 665), url, font=f3, fill=GREY)
    im.save(OUT / "banner.png", optimize=True)


def avatar():
    S = 800
    im = gradient(S, S)
    d = ImageDraw.Draw(im)
    grid(d, S, S, 50)
    # ring
    d.ellipse([40, 40, S - 40, S - 40], outline=CYAN, width=14)
    f = font(BOLD, 330)
    s = "AV"
    tw, th = text_w(d, s, f)
    b = d.textbbox((0, 0), s, font=f)
    glow_text(im, ((S - tw) // 2 - b[0], (S - th) // 2 - b[1] - 10), s, f, WHITE, radius=22)
    im.save(OUT / "avatar.png", optimize=True)
    # alt: upscaled GitHub photo if present next to this script
    photo = pathlib.Path(__file__).resolve().parent / "gh_avatar_src.png"
    if photo.exists():
        p = Image.open(photo).convert("RGB").resize((S, S), Image.LANCZOS)
        p.save(OUT / "avatar-photo.png", optimize=True)


def thumbnail():
    W, H = 1280, 720
    im = gradient(W, H)
    d = ImageDraw.Draw(im)
    grid(d, W, H, 40)
    # left: title
    f1, f2, f3 = font(BOLD, 84), font(BOLD, 46), font(REG, 32)
    glow_text(im, (56, 70), "Muon", f1, WHITE)
    glow_text(im, (56, 168), "+ FlexShard", f1, CYAN)
    d = ImageDraw.Draw(im)
    d.text((60, 310), "why the optimizer", font=f2, fill=GREY)
    d.text((60, 366), "needs WHOLE matrices", font=f2, fill=ORANGE)
    d.text((60, 610), "PyTorch distributed training", font=f3, fill=GREY)
    # right: FSDP row shards -> whole matrix
    x0, y0 = 830, 130
    rows, cols, cw, ch, gap = 8, 6, 18, 18, 3
    blk_h = 2 * (ch + gap)
    for s in range(4):
        y = y0 + s * (blk_h + 20)
        matrix(d, x0, y, cols, 2, cw, ch, [SHARDS[s], SHARDS[s]], gap)
        d.text((x0 - 108, y + 6), f"rank {s}", font=font(MONO, 24), fill=SHARDS[s])
    d.text((x0 - 40, y0 - 50), "FSDP Shard(0)", font=font(MONO, 28), fill=GREY)
    mw = cols * (cw + gap)
    ax = x0 + mw + 30
    d.line([(ax, 300), (ax + 60, 300)], fill=WHITE, width=7)
    d.polygon([(ax + 60, 284), (ax + 86, 300), (ax + 60, 316)], fill=WHITE)
    wx, wy = ax + 110, 215
    matrix(d, wx, wy, cols, rows, cw, ch, [WHITE] * rows, gap)
    mh = rows * (ch + gap)
    d.rectangle([wx - 8, wy - 8, wx + mw + 5, wy + mh + 5], outline=ORANGE, width=4)
    d.text((wx - 20, wy + mh + 22), "X X^T", font=font(MONO, 28), fill=ORANGE)
    d.text((wx - 40, wy + mh + 58), "Newton-Schulz", font=font(MONO, 22), fill=GREY)
    im.save(OUT / "thumb-muon-flexshard.png", optimize=True)


if __name__ == "__main__":
    banner(); avatar(); thumbnail()
    for p in sorted(OUT.glob("*.png")):
        print(p, Image.open(p).size, f"{p.stat().st_size/1024:.0f} KB")
