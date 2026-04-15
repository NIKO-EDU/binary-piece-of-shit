"""
benchmark_packed.py — The fair XNOR-Popcount vs float32 benchmark

Compares the SAME 3×3 convolution at two channel depths:
  - 8 channels  (first layer: our 8 bit-planes packed into uint8)
  - 64 channels (hidden layers: 64 binary channels packed into uint64)

For each depth, two variants run on identical inputs and weights:
  - XNOR-Popcount: N channels processed by one XNOR + POPCNT instruction
  - float32:       N channels processed by N FP multiply-adds (unpacked from bits)

This isolates the effect of bit-packing — all other costs are identical.

Usage:
    python scripts/benchmark_packed.py
    (requires `make` to produce src/kernel.so)
"""

import json
import random
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT     = Path(__file__).resolve().parent.parent
IMG_DIR  = ROOT / "data" / "VisDrone2019-DET-train" / "images"
OUT_JSON = ROOT / "data" / "results_packed.json"

sys.path.insert(0, str(ROOT / "src"))
import kernel_wrapper as kw

N_IMAGES = 100
WARMUP   = 3
SEED     = 42
KH, KW   = 3, 3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def bench(fn, args_list: list, warmup_args) -> dict:
    """Run warmup then timed loop. Returns stats dict."""
    for _ in range(WARMUP):
        fn(*warmup_args)
    times = []
    for args in args_list:
        t0 = time.perf_counter()
        fn(*args)
        times.append((time.perf_counter() - t0) * 1000.0)
    mean = float(np.mean(times))
    return {
        "mean_ms": round(mean, 3),
        "std_ms":  round(float(np.std(times)), 3),
        "p50_ms":  round(percentile(times, 50), 3),
        "p95_ms":  round(percentile(times, 95), 3),
        "fps":     round(1000.0 / mean, 1),
    }


# ---------------------------------------------------------------------------
# Weight preparation
# ---------------------------------------------------------------------------

def make_weights_8ch(rng: np.random.Generator):
    """9 random uint8 values — each bit is a channel weight (1=+1, 0=−1)."""
    wt_u8 = rng.integers(0, 256, size=(KH * KW,), dtype=np.uint8)
    # Float32 version: unpack bits to ±1.0, shape [8, KH*KW]
    wt_f32 = np.empty((8, KH * KW), dtype=np.float32)
    for ch in range(8):
        wt_f32[ch] = np.where((wt_u8 >> ch) & 1, 1.0, -1.0).astype(np.float32)
    return wt_u8.reshape(KH, KW), wt_f32.flatten()


def make_weights_64ch(rng: np.random.Generator):
    """9 random uint64 values — each bit is a channel weight (1=+1, 0=−1)."""
    wt_u64 = rng.integers(0, np.iinfo(np.int64).max, size=(KH * KW,), dtype=np.int64).view(np.uint64)
    # Float32 version: unpack 64 bits to ±1.0, shape [64, KH*KW]
    wt_f32 = np.empty((64, KH * KW), dtype=np.float32)
    for ch in range(64):
        wt_f32[ch] = np.where((wt_u64 >> ch) & 1, 1.0, -1.0).astype(np.float32)
    return wt_u64.reshape(KH, KW), wt_f32.flatten()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    images = sorted(IMG_DIR.glob("*.jpg"))
    rng_sample = random.Random(SEED)
    sample = rng_sample.sample(images, N_IMAGES)

    # Pre-load all images (avoid disk I/O in the timed loop)
    print("Pre-loading images ...")
    grays_u8  = [load_gray(p) for p in sample]
    # 64ch input: replicate uint8 across all 8 bytes of uint64
    # byte 0..7 all equal the grayscale value → 64 channels (8 copies of 8 planes)
    grays_u64 = [g.astype(np.uint64) * np.uint64(0x0101010101010101) for g in grays_u8]
    print(f"  {N_IMAGES} images loaded. Shape: {grays_u8[0].shape}")

    np_rng = np.random.default_rng(SEED)
    wt_u8,  wt_f32_8ch  = make_weights_8ch(np_rng)
    wt_u64, wt_f32_64ch = make_weights_64ch(np_rng)

    results = {}

    # --- 8-channel XNOR ---
    print("\nBenchmarking: XNOR u8 (8 channels) ...")
    stats = bench(
        lambda g: kw.xnor_packed_u8_conv(g, wt_u8, KH, KW, 8),
        [(g,) for g in grays_u8],
        (grays_u8[0],),
    )
    results["xnor_u8"] = stats
    print(f"  {stats['fps']:.1f} FPS  (mean {stats['mean_ms']:.2f} ms)")

    # --- 8-channel float32 ---
    print("Benchmarking: float32 u8 (8 channels) ...")
    stats = bench(
        lambda g: kw.float32_conv_nch_u8(g, wt_f32_8ch, KH, KW, 8),
        [(g,) for g in grays_u8],
        (grays_u8[0],),
    )
    results["float32_u8"] = stats
    print(f"  {stats['fps']:.1f} FPS  (mean {stats['mean_ms']:.2f} ms)")

    # --- 64-channel XNOR ---
    print("Benchmarking: XNOR u64 (64 channels) ...")
    stats = bench(
        lambda g: kw.xnor_packed_u64_conv(g, wt_u64, KH, KW, 64),
        [(g,) for g in grays_u64],
        (grays_u64[0],),
    )
    results["xnor_u64"] = stats
    print(f"  {stats['fps']:.1f} FPS  (mean {stats['mean_ms']:.2f} ms)")

    # --- 64-channel float32 ---
    print("Benchmarking: float32 u64 (64 channels) ...")
    stats = bench(
        lambda g: kw.float32_conv_nch_u64(g, wt_f32_64ch, KH, KW, 64),
        [(g,) for g in grays_u64],
        (grays_u64[0],),
    )
    results["float32_u64"] = stats
    print(f"  {stats['fps']:.1f} FPS  (mean {stats['mean_ms']:.2f} ms)")

    # --- Print table ---
    speedup_8  = results["xnor_u8"]["fps"]  / results["float32_u8"]["fps"]
    speedup_64 = results["xnor_u64"]["fps"] / results["float32_u64"]["fps"]

    print(f"\n{'='*60}")
    print(f"  {'Variant':<28} {'Channels':>8}  {'FPS':>8}  {'Speedup':>8}")
    print(f"  {'-'*56}")
    print(f"  {'XNOR-Popcount  (binary)':<28} {'8':>8}  {results['xnor_u8']['fps']:>8.1f}  {speedup_8:>7.1f}x")
    print(f"  {'float32        (reference)':<28} {'8':>8}  {results['float32_u8']['fps']:>8.1f}  {'1.0x':>8}")
    print(f"  {'-'*56}")
    print(f"  {'XNOR-Popcount  (binary)':<28} {'64':>8}  {results['xnor_u64']['fps']:>8.1f}  {speedup_64:>7.1f}x")
    print(f"  {'float32        (reference)':<28} {'64':>8}  {results['float32_u64']['fps']:>8.1f}  {'1.0x':>8}")
    print(f"{'='*60}")
    print(f"\nKey insight: at 64 channels (hidden layers), XNOR-Popcount is {speedup_64:.1f}x faster.")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {OUT_JSON}")


if __name__ == "__main__":
    main()
