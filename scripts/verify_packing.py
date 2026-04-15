"""
verify_packing.py — Phase 1 test

Loads 10 random packed .npy files from data/packed/, reconstructs the
RGB image from the 3×8 bit-planes, and asserts pixel-perfect equality
with the original image.

Usage:
    python scripts/verify_packing.py
"""

import json
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT     = Path(__file__).resolve().parent.parent
IMG_DIR  = ROOT / "data" / "VisDrone2019-DET-train" / "images"
PACK_DIR = ROOT / "data" / "packed"
INDEX_F  = PACK_DIR / "index.json"

TARGET_SIZE = (640, 640)
N_PLANES    = 8
N_CHANNELS  = 3  # R, G, B
N_SAMPLES   = 10


def unpack_plane(packed: np.ndarray) -> np.ndarray:
    """
    Unpack one bit-plane from uint64 storage back to uint8 binary pixels.

    Parameters
    ----------
    packed : [640, 10] uint64

    Returns
    -------
    bits : [640, 640] uint8 (values 0 or 1)
    """
    # Reinterpret as bytes (same memory, no copy)
    as_bytes = packed.view(np.uint8)          # [640, 80]
    bits = np.unpackbits(as_bytes, axis=1, bitorder="big")  # [640, 640]
    return bits[:, :TARGET_SIZE[0]]            # trim to exactly 640


def reconstruct_rgb(tensor: np.ndarray) -> np.ndarray:
    """
    Reconstruct uint8 RGB image from packed bit-plane tensor.

    Parameters
    ----------
    tensor : [3, 8, 640, 10] uint64
             axis 0: channel (0=R, 1=G, 2=B)
             axis 1: bit-plane (0=LSB … 7=MSB)

    Returns
    -------
    rgb : [640, 640, 3] uint8
    """
    assert tensor.shape == (N_CHANNELS, N_PLANES, 640, 10), f"Unexpected shape {tensor.shape}"
    channels = []
    for ch in range(N_CHANNELS):
        accum = np.zeros((640, 640), dtype=np.uint16)
        for i in range(N_PLANES):
            plane_bits = unpack_plane(tensor[ch, i])  # [640, 640] uint8
            accum += plane_bits.astype(np.uint16) << i
        assert accum.max() <= 255, f"Reconstruction overflow ch={ch}: max={accum.max()}"
        channels.append(accum.astype(np.uint8))
    return np.stack(channels, axis=-1)  # [640, 640, 3]


def load_original_rgb(stem: str) -> np.ndarray:
    """Load and resize to 640×640 RGB uint8."""
    img_path = IMG_DIR / f"{stem}.jpg"
    img = Image.open(img_path).convert("RGB")
    img = img.resize(TARGET_SIZE, Image.LANCZOS)
    return np.array(img, dtype=np.uint8)


def main() -> None:
    if not PACK_DIR.exists():
        print(f"ERROR: packed dir not found: {PACK_DIR}", file=sys.stderr)
        print("Run `python scripts/pack_data.py` first.", file=sys.stderr)
        sys.exit(1)

    npy_files = sorted(PACK_DIR.glob("*.npy"))
    if len(npy_files) == 0:
        print("ERROR: no .npy files found in data/packed/", file=sys.stderr)
        sys.exit(1)

    samples = random.sample(npy_files, min(N_SAMPLES, len(npy_files)))
    print(f"Verifying {len(samples)} random packed files ...\n")

    all_passed = True
    for npy_path in samples:
        stem = npy_path.stem
        tensor = np.load(npy_path)

        # Reconstruct from packed representation
        recon = reconstruct_rgb(tensor)

        # Load original (apply same pipeline as pack_data.py)
        original = load_original_rgb(stem)

        if np.array_equal(recon, original):
            print(f"  PASS  {stem}")
        else:
            diff = np.abs(recon.astype(int) - original.astype(int))
            print(f"  FAIL  {stem}  max_diff={diff.max()}  n_mismatch={np.sum(diff > 0)}")
            all_passed = False

    print()
    if all_passed:
        print(f"ALL {len(samples)} samples PASSED — packing is lossless.")
        sys.exit(0)
    else:
        print("SOME samples FAILED.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
