"""
run_first_layer.py — Run the first BNN conv layer on a VisDrone image.

Decomposes an RGB image into 24 binary planes (R×8 + G×8 + B×8),
applies 64 configurable XNOR-Popcount filters (3×3 kernel, 24 channels),
and saves the 64 integer score maps and 64 binary feature maps.

Usage:
    python3 scripts/run_first_layer.py [path/to/image.jpg]
    (defaults to first image in data/VisDrone2019-DET-train/images/)

Outputs in data/first_layer_output/:
    score_filter_00..07.png     — normalised int32 score maps, first 8 filters
    binary_filter_00..07.png    — thresholded binary output, first 8 filters
    all_64_filters_grid.png     — 8×8 grid of all 64 binary feature maps
"""

import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT    = Path(__file__).resolve().parent.parent
IMG_DIR = ROOT / "data" / "VisDrone2019-DET-train" / "images"
OUT_DIR = ROOT / "data" / "first_layer_output"

sys.path.insert(0, str(ROOT / "src"))
from binary_layer import BinaryConvLayer


def load_rgb_planes(img_path: Path) -> np.ndarray:
    """
    Load one image and return 24 binary planes [24, 640, 640] uint8.

    Channel order:
        planes[0..7]   = R bits 0-7 (LSB to MSB)
        planes[8..15]  = G bits 0-7
        planes[16..23] = B bits 0-7
    """
    pil = Image.open(img_path).convert("RGB").resize((640, 640), Image.LANCZOS)
    rgb = np.array(pil, dtype=np.uint8)   # [640, 640, 3]

    planes = []
    for ch in range(3):        # R=0, G=1, B=2
        for bit in range(8):   # LSB→MSB
            planes.append(((rgb[:, :, ch] >> bit) & 1).astype(np.uint8))

    return np.stack(planes, axis=0)   # [24, 640, 640]


def save_score_map(arr: np.ndarray, path: Path) -> None:
    """Normalise an int32 score map to [0, 255] uint8 and save as PNG."""
    a_min, a_max = float(arr.min()), float(arr.max())
    if a_max > a_min:
        normed = ((arr - a_min) / (a_max - a_min) * 255.0).astype(np.uint8)
    else:
        normed = np.zeros_like(arr, dtype=np.uint8)
    Image.fromarray(normed).save(path)


def main() -> None:
    # ── Select image ────────────────────────────────────────────────────
    if len(sys.argv) > 1:
        img_path = Path(sys.argv[1])
    else:
        images = sorted(IMG_DIR.glob("*.jpg"))
        if not images:
            print(f"ERROR: no images in {IMG_DIR}", file=sys.stderr)
            sys.exit(1)
        img_path = images[0]

    print(f"Image  : {img_path.name}")

    # ── Decompose into 24 binary planes ─────────────────────────────────
    planes = load_rgb_planes(img_path)
    print(f"Input  : {planes.shape}  (24 channels = R×8 + G×8 + B×8)")

    # ── Build layer ──────────────────────────────────────────────────────
    # 64 filters, 24 input channels (fits in uint64), 3×3 kernel
    layer = BinaryConvLayer(n_filters=64, n_ch=24, kH=3, kW=3)
    print(f"Layer  : {layer.n_filters} filters × {layer.n_ch} channels × {layer.kH}×{layer.kW} kernel")
    print(f"         Score range per filter per pixel: [{-layer.max_score}, +{layer.max_score}]")
    print()

    # ── Swap in custom kernels here if needed ────────────────────────────
    # Example: set filter 0 to a horizontal-edge detector across all channels
    # w = np.full((layer.n_ch, layer.kH, layer.kW), -1, dtype=np.int8)
    # w[:, 0, :] = +1   # top row = +1, middle and bottom = -1
    # layer.set_filter(0, w)
    #
    # Or replace all weights at once:
    # new_weights = np.ones((64, 24, 3, 3), dtype=np.int8)
    # layer.set_weights(new_weights)

    # ── Forward pass: 64 integer score maps ─────────────────────────────
    t0 = time.perf_counter()
    scores = layer.forward(planes)   # [64, 640, 640] int32
    elapsed_ms = (time.perf_counter() - t0) * 1000

    print(f"Scores : {scores.shape}  dtype={scores.dtype}  ({elapsed_ms:.1f} ms)")
    print(f"         actual range : [{scores.min()}, {scores.max()}]")
    print(f"         mean         : {scores.mean():.2f}")
    print()

    # Per-filter statistics
    print("  filter   min    max   mean   std")
    print("  ------  ----   ----  -----  ----")
    for f in range(min(8, layer.n_filters)):
        s = scores[f]
        print(f"  {f:02d}      {s.min():4d}   {s.max():4d}  {s.mean():5.1f}  {s.std():4.1f}")
    if layer.n_filters > 8:
        print(f"  ... ({layer.n_filters - 8} more filters)")
    print()

    # ── Apply threshold → 64 binary feature maps ────────────────────────
    threshold = 0   # change this to require stronger matches
    binary = layer.apply_threshold(scores, threshold=threshold)
    fire_rate = binary.mean() * 100
    print(f"Binary : {binary.shape}  threshold={threshold}  fire_rate={fire_rate:.1f}%")
    print()

    # ── Save outputs ─────────────────────────────────────────────────────
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # First 8 score maps (normalised to [0,255] for viewing)
    for f in range(8):
        save_score_map(scores[f], OUT_DIR / f"score_filter_{f:02d}.png")

    # First 8 binary maps
    for f in range(8):
        Image.fromarray(binary[f] * 255).save(OUT_DIR / f"binary_filter_{f:02d}.png")

    # 8×8 grid of all 64 binary maps, each thumbnail 80×80
    cell = 80
    grid = np.zeros((8 * cell, 8 * cell), dtype=np.uint8)
    for f in range(64):
        r, c = divmod(f, 8)
        thumb = np.array(
            Image.fromarray(binary[f] * 255).resize((cell, cell), Image.NEAREST)
        )
        grid[r * cell:(r + 1) * cell, c * cell:(c + 1) * cell] = thumb
    Image.fromarray(grid).save(OUT_DIR / "all_64_filters_grid.png")

    print(f"Saved to {OUT_DIR}/")
    print(f"  score_filter_00..07.png   — int32 score maps (normalised), first 8 filters")
    print(f"  binary_filter_00..07.png  — binary feature maps, first 8 filters")
    print(f"  all_64_filters_grid.png   — 8×8 grid of all 64 binary feature maps")
    print()
    print("To use custom kernels, edit the 'Swap in custom kernels' block above.")
    print(f"  layer.set_filter(idx, weights)   # weights shape: [{layer.n_ch}, {layer.kH}, {layer.kW}], values ±1")
    print(f"  layer.set_weights(weights)        # weights shape: [{layer.n_filters}, {layer.n_ch}, {layer.kH}, {layer.kW}], values ±1")


if __name__ == "__main__":
    main()
