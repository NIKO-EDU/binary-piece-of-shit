"""
pack_data.py — Phase 1: VisDrone image → packed uint64 bit-plane tensors

For each image in data/VisDrone2019-DET-train/images/:
  1. Resize to 640×640 (LANCZOS, simple stretch — no letterboxing)
  2. Keep as RGB uint8 — decompose each channel (R, G, B) independently
  3. For each channel: 8 bit-planes: plane[i] = (channel >> i) & 1
  4. Pack each plane's rows into uint64 integers (640 px → 10 × uint64)
  5. Save as data/packed/<stem>.npy   shape: [3, 8, 640, 10] uint64
     axis 0: channel (0=R, 1=G, 2=B)
     axis 1: bit-plane (0=LSB … 7=MSB)
     axis 2: row
     axis 3: packed uint64 column (10 × 64 = 640 pixels)

Also writes data/packed/index.json: {stem: [orig_width, orig_height]}

Usage:
    python scripts/pack_data.py
"""

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT     = Path(__file__).resolve().parent.parent
IMG_DIR  = ROOT / "data" / "VisDrone2019-DET-train" / "images"
OUT_DIR  = ROOT / "data" / "packed"
INDEX_F  = OUT_DIR / "index.json"

TARGET_SIZE = (640, 640)
N_PLANES    = 8
N_CHANNELS  = 3  # R, G, B


def pack_image(img_path: Path) -> tuple[np.ndarray, tuple[int, int]]:
    """
    Load, resize, and pack one image into a [3, 8, 640, 10] uint64 array.
    Returns (packed_array, (orig_width, orig_height)).

    Axis layout:
        [channel, bit_plane, row, packed_col]
        channel  : 0=R, 1=G, 2=B
        bit_plane: 0=LSB … 7=MSB
    """
    img = Image.open(img_path).convert("RGB")
    orig_size = img.size  # (width, height)

    # Resize to 640×640 (LANCZOS — high-quality downsampling)
    img = img.resize(TARGET_SIZE, Image.LANCZOS)

    # RGB uint8: shape [640, 640, 3]
    rgb = np.array(img, dtype=np.uint8)
    assert rgb.shape == (TARGET_SIZE[1], TARGET_SIZE[0], 3), f"Unexpected shape {rgb.shape}"

    channel_planes = []
    for ch in range(N_CHANNELS):          # 0=R, 1=G, 2=B
        channel = rgb[:, :, ch]           # [640, 640] uint8
        planes = []
        for i in range(N_PLANES):
            # Extract bit i: values are 0 or 1
            plane_bits = ((channel >> i) & 1).astype(np.uint8)  # [640, 640]

            # Pack 640-wide rows into bytes, then reinterpret as uint64
            # np.packbits with bitorder='big': MSB first in each byte
            packed_bytes = np.packbits(plane_bits, axis=1, bitorder="big")  # [640, 80]
            # 80 bytes / 8 = 10 uint64 per row
            packed_u64 = packed_bytes.view(np.uint64)  # [640, 10]
            planes.append(packed_u64)

        # Stack planes for this channel: [8, 640, 10]
        channel_planes.append(np.stack(planes, axis=0))

    # Stack channels: [3, 8, 640, 10] uint64
    tensor = np.stack(channel_planes, axis=0)
    assert tensor.shape == (N_CHANNELS, N_PLANES, 640, 10), f"Unexpected tensor shape {tensor.shape}"
    assert tensor.dtype == np.uint64

    return tensor, orig_size


def main() -> None:
    if not IMG_DIR.exists():
        print(f"ERROR: image directory not found: {IMG_DIR}", file=sys.stderr)
        sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(IMG_DIR.glob("*.jpg"))
    total = len(image_paths)
    if total == 0:
        print(f"ERROR: no .jpg files found in {IMG_DIR}", file=sys.stderr)
        sys.exit(1)

    print(f"Packing {total} images to {OUT_DIR} ...")

    index: dict[str, list[int]] = {}
    errors: list[str] = []

    for idx, img_path in enumerate(image_paths, start=1):
        stem = img_path.stem
        out_path = OUT_DIR / f"{stem}.npy"

        try:
            tensor, orig_size = pack_image(img_path)
            np.save(out_path, tensor)
            index[stem] = list(orig_size)
        except Exception as exc:
            errors.append(f"{img_path.name}: {exc}")
            print(f"  ERROR {img_path.name}: {exc}", file=sys.stderr)
            continue

        if idx % 500 == 0 or idx == total:
            print(f"  [{idx}/{total}] done")

    # Write index
    with open(INDEX_F, "w") as f:
        json.dump(index, f, indent=2)
    print(f"Index written to {INDEX_F}")

    if errors:
        print(f"\n{len(errors)} error(s) encountered:", file=sys.stderr)
        for e in errors:
            print(f"  {e}", file=sys.stderr)
        sys.exit(1)

    print(f"\nDone. {len(index)} files packed successfully.")


if __name__ == "__main__":
    main()
