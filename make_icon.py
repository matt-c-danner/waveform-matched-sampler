#!/usr/bin/env python3
"""Generate AppIcon.icns for Waveform Matched Sampler."""
import math
import shutil
import subprocess
import sys
from pathlib import Path

try:
    from PIL import Image, ImageDraw
except ImportError:
    print("pip install Pillow")
    sys.exit(1)

BG    = (17,  17,  17, 255)   # near-black
GREEN = ( 0, 229,   0, 255)   # #00e500

SIZES = [
    ("icon_16x16.png",        16),
    ("icon_16x16@2x.png",     32),
    ("icon_32x32.png",        32),
    ("icon_32x32@2x.png",     64),
    ("icon_128x128.png",     128),
    ("icon_128x128@2x.png",  256),
    ("icon_256x256.png",     256),
    ("icon_256x256@2x.png",  512),
    ("icon_512x512.png",     512),
    ("icon_512x512@2x.png", 1024),
]


def waveform_polygon(cx, cy, w, h, steps=512):
    """Filled audio-clip waveform (mirrored above and below centre line)."""
    half   = h / 2
    upper  = []
    lower  = []

    for i in range(steps + 1):
        t = i / steps
        x = cx - w / 2 + t * w

        # Envelope: fast attack, sustain, slow decay
        if t < 0.04:
            env = t / 0.04
        elif t < 0.72:
            env = 1.0
        else:
            env = (1.0 - t) / 0.28

        # Layered harmonics give it an organic, audio-clip look
        wave = (math.sin(t * math.pi * 11) * 0.55
              + math.sin(t * math.pi * 17) * 0.28
              + math.sin(t * math.pi * 29) * 0.17)

        amp = env * abs(wave) * half
        amp = max(amp, half * 0.035)   # always a thin sliver

        upper.append((x, cy - amp))
        lower.append((x, cy + amp))

    return upper + list(reversed(lower))


def draw_icon(size: int) -> Image.Image:
    img  = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    pad = max(2, int(size * 0.04))
    r   = int(size * 0.20)
    draw.rounded_rectangle(
        [pad, pad, size - pad, size - pad],
        radius=r, fill=BG,
    )

    mx = size * 0.08
    my = size * 0.20
    cx = size / 2
    cy = size / 2

    poly = waveform_polygon(cx, cy, size - 2 * mx, size - 2 * my)
    draw.polygon(poly, fill=GREEN)

    # Subtle centre line
    lw = max(1, int(size * 0.007))
    draw.line([(size * 0.08, cy), (size * 0.92, cy)],
              fill=(0, 60, 0, 180), width=lw)

    return img


def main():
    iconset = Path("AppIcon.iconset")
    iconset.mkdir(exist_ok=True)

    master = draw_icon(1024)

    for filename, px in SIZES:
        img = master if px == 1024 else master.resize((px, px), Image.LANCZOS)
        img.save(iconset / filename)
        print(f"  wrote {filename}")

    icns = Path("AppIcon.icns")
    subprocess.run(
        ["iconutil", "-c", "icns", str(iconset), "-o", str(icns)],
        check=True,
    )
    shutil.rmtree(iconset)
    print(f"\nDone → {icns}")


if __name__ == "__main__":
    main()
