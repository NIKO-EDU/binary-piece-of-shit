"""
sobel_demo.py — Phase 2 demo: Sobel edge map from a VisDrone image

Picks the first image in data/VisDrone2019-DET-train/images/, runs the C
Sobel kernel, and saves a normalised edge map to data/sobel_demo.png.

Usage:
    python scripts/sobel_demo.py
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT    = Path(__file__).resolve().parent.parent
IMG_DIR = ROOT / "data" / "VisDrone2019-DET-train" / "images"
OUT_PNG = ROOT / "data" / "sobel_demo.png"

sys.path.insert(0, str(ROOT / "src"))
import kernel_wrapper as kw


def main() -> None:
    images = sorted(IMG_DIR.glob("*.jpg"))
    if not images:
        print(f"ERROR: no images found in {IMG_DIR}", file=sys.stderr)
        sys.exit(1)

    img_path = images[0]
    print(f"Input image : {img_path.name}")

    # Load and resize to 640×640 grayscale
    gray = np.array(
        Image.open(img_path).convert("RGB").resize((640, 640), Image.LANCZOS).convert("L"),
        dtype=np.uint8,
    )
    print(f"Grayscale   : {gray.shape}  dtype={gray.dtype}")

    # Run C Sobel kernel
    gradient = kw.sobel_conv(gray)
    print(f"Gradient    : min={gradient.min():.1f}  max={gradient.max():.1f}  "
          f"mean={gradient.mean():.1f}")

    # Normalise to [0, 255] uint8
    g_min, g_max = float(gradient.min()), float(gradient.max())
    if g_max > g_min:
        normed = ((gradient - g_min) / (g_max - g_min) * 255.0).astype(np.uint8)
    else:
        normed = np.zeros_like(gradient, dtype=np.uint8)

    Image.fromarray(normed).save(OUT_PNG)
    print(f"Edge map saved to {OUT_PNG}")


if __name__ == "__main__":
    main()
