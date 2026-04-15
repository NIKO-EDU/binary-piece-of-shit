"""
benchmark_kernel.py — Phase 4: C Sobel kernel throughput benchmark

Runs the C Sobel kernel on 100 VisDrone images (same sample as benchmark_yolo.py)
and records latency/FPS for a direct head-to-head comparison.

Usage:
    python scripts/benchmark_kernel.py
    (requires `make` to produce src/kernel.so)
"""

import json
import random
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT    = Path(__file__).resolve().parent.parent
IMG_DIR = ROOT / "data" / "VisDrone2019-DET-train" / "images"
OUT_JSON = ROOT / "data" / "results_kernel.json"

sys.path.insert(0, str(ROOT / "src"))
import kernel_wrapper as kw

N_IMAGES = 100
WARMUP   = 3
SEED     = 42   # same seed as benchmark_yolo.py → identical image sample


def load_gray(img_path: Path) -> np.ndarray:
    """Load, resize to 640×640 and convert to grayscale uint8."""
    return np.array(
        Image.open(img_path).convert("RGB").resize((640, 640), Image.LANCZOS).convert("L"),
        dtype=np.uint8,
    )


def percentile(values: list[float], p: float) -> float:
    sorted_v = sorted(values)
    idx = (len(sorted_v) - 1) * p / 100.0
    lo, hi = int(idx), min(int(idx) + 1, len(sorted_v) - 1)
    return sorted_v[lo] + (sorted_v[hi] - sorted_v[lo]) * (idx - lo)


def main() -> None:
    images = sorted(IMG_DIR.glob("*.jpg"))
    if len(images) < N_IMAGES:
        print(f"ERROR: only {len(images)} images available.", file=sys.stderr)
        sys.exit(1)

    rng = random.Random(SEED)
    sample = rng.sample(images, N_IMAGES)

    # Warmup
    warmup_gray = load_gray(sample[0])
    print(f"Warming up ({WARMUP} iters) ...")
    for _ in range(WARMUP):
        kw.sobel_conv(warmup_gray)

    # Benchmark
    latencies_ms: list[float] = []
    print(f"Benchmarking {N_IMAGES} images ...")
    for i, img_path in enumerate(sample, start=1):
        gray = load_gray(img_path)
        t0 = time.perf_counter()
        kw.sobel_conv(gray)
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)
        if i % 25 == 0:
            print(f"  [{i}/{N_IMAGES}]  last={latencies_ms[-1]:.1f} ms")

    mean_ms = float(np.mean(latencies_ms))
    std_ms  = float(np.std(latencies_ms))
    p50     = percentile(latencies_ms, 50)
    p95     = percentile(latencies_ms, 95)
    fps     = 1000.0 / mean_ms

    # Sobel kernel: 2 kernels × 9 int8 weights = 18 bytes
    weight_bytes = 18

    result = {
        "model": "c-sobel-kernel",
        "n_images": N_IMAGES,
        "weight_size_bytes": weight_bytes,
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
    print(f"  Weight size  : {weight_bytes} bytes")
    print(f"  Mean latency : {mean_ms:.1f} ms  (±{std_ms:.1f})")
    print(f"  p50 / p95    : {p50:.1f} / {p95:.1f} ms")
    print(f"  FPS          : {fps:.1f}")
    print(f"{'='*42}")
    print(f"Results saved to {OUT_JSON}")


if __name__ == "__main__":
    main()
