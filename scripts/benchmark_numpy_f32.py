"""
benchmark_numpy_f32.py — Float32 Sobel baseline using scipy.ndimage

Runs the identical 3×3 Sobel operation as the C kernel but in pure float32
via scipy.ndimage.convolve. Same 100-image sample (seed=42) as benchmark_kernel.py.

This is the fair comparison: same math, different compute paradigm.

Usage:
    python scripts/benchmark_numpy_f32.py
"""

import json
import random
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import convolve

ROOT     = Path(__file__).resolve().parent.parent
IMG_DIR  = ROOT / "data" / "VisDrone2019-DET-train" / "images"
OUT_JSON = ROOT / "data" / "results_numpy_f32.json"

N_IMAGES = 100
WARMUP   = 3
SEED     = 42

# Identical Sobel kernels to those in kernel.c
GX = np.array([[-1, 0, 1],
               [-2, 0, 2],
               [-1, 0, 1]], dtype=np.float32)

GY = np.array([[-1, -2, -1],
               [ 0,  0,  0],
               [ 1,  2,  1]], dtype=np.float32)


def sobel_f32(gray: np.ndarray) -> np.ndarray:
    """float32 Sobel gradient magnitude via scipy.ndimage (zero-padding = 'constant')."""
    f = gray.astype(np.float32)
    gx = convolve(f, GX, mode="constant", cval=0.0)
    gy = convolve(f, GY, mode="constant", cval=0.0)
    return np.sqrt(gx * gx + gy * gy)


def load_gray(img_path: Path) -> np.ndarray:
    return np.array(
        Image.open(img_path).convert("RGB").resize((640, 640), Image.LANCZOS).convert("L"),
        dtype=np.uint8,
    )


def percentile(values: list[float], p: float) -> float:
    s = sorted(values)
    idx = (len(s) - 1) * p / 100.0
    lo, hi = int(idx), min(int(idx) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (idx - lo)


def main() -> None:
    images = sorted(IMG_DIR.glob("*.jpg"))
    rng = random.Random(SEED)
    sample = rng.sample(images, N_IMAGES)

    warmup_gray = load_gray(sample[0])
    print(f"Warming up ({WARMUP} iters) ...")
    for _ in range(WARMUP):
        sobel_f32(warmup_gray)

    latencies_ms: list[float] = []
    print(f"Benchmarking {N_IMAGES} images (float32 NumPy Sobel) ...")
    for i, img_path in enumerate(sample, start=1):
        gray = load_gray(img_path)
        t0 = time.perf_counter()
        sobel_f32(gray)
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)
        if i % 25 == 0:
            print(f"  [{i}/{N_IMAGES}]  last={latencies_ms[-1]:.2f} ms")

    mean_ms = float(np.mean(latencies_ms))
    std_ms  = float(np.std(latencies_ms))
    p50     = percentile(latencies_ms, 50)
    p95     = percentile(latencies_ms, 95)
    fps     = 1000.0 / mean_ms

    result = {
        "model": "numpy-f32-sobel",
        "n_images": N_IMAGES,
        "weight_size_bytes": 2 * 9 * 4,   # 2 kernels × 9 values × 4 bytes (float32)
        "latency_ms": {
            "mean": round(mean_ms, 3),
            "std":  round(std_ms,  3),
            "p50":  round(p50,     3),
            "p95":  round(p95,     3),
            "min":  round(min(latencies_ms), 3),
            "max":  round(max(latencies_ms), 3),
        },
        "fps": round(fps, 2),
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n{'='*42}")
    print(f"  Model        : {result['model']}")
    print(f"  Weight size  : {result['weight_size_bytes']} bytes (float32)")
    print(f"  Mean latency : {mean_ms:.2f} ms  (±{std_ms:.2f})")
    print(f"  p50 / p95    : {p50:.2f} / {p95:.2f} ms")
    print(f"  FPS          : {fps:.1f}")
    print(f"{'='*42}")
    print(f"Results saved to {OUT_JSON}")

    # Print head-to-head if C kernel results exist
    c_json = ROOT / "data" / "results_kernel.json"
    if c_json.exists():
        c = json.loads(c_json.read_text())
        c_fps  = c["fps"]
        c_mean = c["latency_ms"]["mean"]
        speedup = fps / c_fps  # positive = f32 faster, negative = C faster
        print(f"\n--- Head-to-head (same operation, same images) ---")
        print(f"  NumPy float32 : {mean_ms:.2f} ms  →  {fps:.1f} FPS")
        print(f"  C int kernel  : {c_mean:.2f} ms  →  {c_fps:.1f} FPS")
        if c_fps > fps:
            print(f"  C kernel is {c_fps/fps:.1f}x faster than float32")
        else:
            print(f"  float32 is {fps/c_fps:.1f}x faster than C kernel")


if __name__ == "__main__":
    main()
