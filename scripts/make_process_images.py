"""
make_process_images.py — Generate step-by-step processing visualisations.

Outputs to data/presentation_images/:
  00_original_640x640_rgb.png      — image after resize (RGB)
  r_bitplane_0_lsb.png … r_bitplane_7_msb.png   — 8 planes for Red channel
  g_bitplane_0_lsb.png … g_bitplane_7_msb.png   — 8 planes for Green channel
  b_bitplane_0_lsb.png … b_bitplane_7_msb.png   — 8 planes for Blue channel
  sobel_edge_map.png               — Sobel gradient magnitude (float32 reference)

Usage:
    python scripts/make_process_images.py [path/to/image.jpg]
    (defaults to first image in data/VisDrone2019-DET-train/images/)
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT    = Path(__file__).resolve().parent.parent
IMG_DIR = ROOT / "data" / "VisDrone2019-DET-train" / "images"
OUT_DIR = ROOT / "data" / "presentation_images"

sys.path.insert(0, str(ROOT / "src"))
import kernel_wrapper as kw


def save(arr: np.ndarray, path: Path, label: str) -> None:
    Image.fromarray(arr).save(path)
    print(f"  saved: {path.name}  ({arr.shape[1]}×{arr.shape[0]})")


def main() -> None:
    # Pick image
    if len(sys.argv) > 1:
        img_path = Path(sys.argv[1])
    else:
        images = sorted(IMG_DIR.glob("*.jpg"))
        if not images:
            print(f"ERROR: no images in {IMG_DIR}", file=sys.stderr)
            sys.exit(1)
        img_path = images[0]

    print(f"Source image : {img_path.name}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Step 0: resize to 640×640, keep RGB ─────────────────────────────
    pil_rgb = Image.open(img_path).convert("RGB").resize((640, 640), Image.LANCZOS)
    rgb     = np.array(pil_rgb, dtype=np.uint8)   # [640, 640, 3]

    save(rgb, OUT_DIR / "00_original_640x640_rgb.png", "RGB")

    # ── Steps 1–24: 8 bit-planes for each of R, G, B ────────────────────
    plane_labels = ["lsb", "", "", "", "", "", "", "msb"]
    channel_names = ["r", "g", "b"]

    for ch_idx, ch_name in enumerate(channel_names):
        channel = rgb[:, :, ch_idx]   # [640, 640] uint8
        for i in range(8):
            plane_bits = ((channel >> i) & 1).astype(np.uint8)   # 0 or 1
            plane_img  = (plane_bits * 255).astype(np.uint8)      # 0 or 255

            suffix = f"_{plane_labels[i]}" if plane_labels[i] else ""
            fname  = f"{ch_name}_bitplane_{i}{suffix}.png"
            save(plane_img, OUT_DIR / fname, f"{ch_name.upper()} bit-plane {i}")

    # ── Sobel edge map (float32 reference, on green channel) ────────────
    green    = rgb[:, :, 1]
    gradient = kw.sobel_conv(green)
    g_min, g_max = float(gradient.min()), float(gradient.max())
    if g_max > g_min:
        normed = ((gradient - g_min) / (g_max - g_min) * 255.0).astype(np.uint8)
    else:
        normed = np.zeros_like(gradient, dtype=np.uint8)

    save(normed, OUT_DIR / "sobel_edge_map.png", "Sobel (float32, green channel)")

    print(f"\nAll images saved to {OUT_DIR}/")
    print(f"  24 binary planes: 8 per channel (R, G, B).")
    print(f"  LSB planes (bit 0) will look like noise — expected.")
    print(f"  MSB planes (bit 7) will show the coarse shape of the scene.")


if __name__ == "__main__":
    main()
