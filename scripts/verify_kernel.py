"""
verify_kernel.py — Phase 2 test

Compares the output of the C xnor_popcount_conv kernel against a pure-NumPy
reference implementation on randomised inputs.

Runs multiple trials with different sizes and kernel shapes.

Usage:
    python scripts/verify_kernel.py
    (requires `make` to have been run to produce src/kernel.so)
"""

import sys
from pathlib import Path

import numpy as np

# Ensure the src package is importable
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import kernel_wrapper as kw


# ---------------------------------------------------------------------------
# Pure-NumPy reference implementation
# ---------------------------------------------------------------------------

def numpy_xnor_popcount(
    input_arr: np.ndarray,
    weights: np.ndarray,
    kH: int,
    kW: int,
) -> np.ndarray:
    """
    Reference XNOR-Popcount convolution in pure NumPy.

    Matches the semantics of the C kernel exactly:
      agree = ((input_bit > 0) == (weight > 0))
      output = 2 * popcount - kH*kW
    Zero-padding at borders.
    """
    rows, cols = input_arr.shape
    padH, padW = kH // 2, kW // 2
    output = np.zeros((rows, cols), dtype=np.int32)

    for r in range(rows):
        for c in range(cols):
            popcount = 0
            for kr in range(kH):
                for kc in range(kW):
                    ir = r + kr - padH
                    ic = c + kc - padW
                    if 0 <= ir < rows and 0 <= ic < cols:
                        ib = int(input_arr[ir, ic]) > 0
                    else:
                        ib = False           # zero-pad
                    wb = int(weights[kr, kc]) > 0
                    popcount += 1 if (ib == wb) else 0
            output[r, c] = 2 * popcount - kH * kW
    return output


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------

TRIALS = [
    # (rows, cols, kH, kW, seed)
    (8,   8,  3, 3,  0),
    (64, 64,  3, 3,  1),
    (16, 32,  3, 3,  2),
    (10, 10,  5, 5,  3),
    (4,   4,  1, 1,  4),
    (7,   7,  7, 7,  5),   # kernel same size as image
    (5,   9,  3, 5,  6),   # rectangular kernel
]


def run_trial(rows: int, cols: int, kH: int, kW: int, seed: int) -> bool:
    rng = np.random.default_rng(seed)

    # Binary input (0 or 1)
    input_arr = rng.integers(0, 2, size=(rows, cols), dtype=np.uint8)

    # Binary weights (+1 or -1)
    weights = rng.choice(np.array([-1, 1], dtype=np.int8), size=(kH, kW))

    # C result
    c_out = kw.xnor_popcount_conv(input_arr, weights, kH, kW)

    # NumPy reference
    np_out = numpy_xnor_popcount(input_arr, weights, kH, kW)

    if np.array_equal(c_out, np_out):
        return True
    else:
        diff = c_out.astype(int) - np_out.astype(int)
        n_mismatch = int(np.sum(diff != 0))
        print(f"    MISMATCH: {n_mismatch} pixels differ; "
              f"max_diff={int(np.abs(diff).max())}")
        print(f"    First mismatch: C={c_out.flat[np.argmax(diff != 0)]}  "
              f"NumPy={np_out.flat[np.argmax(diff != 0)]}")
        return False


def main() -> None:
    print("=== verify_kernel.py — C kernel vs NumPy reference ===\n")
    all_passed = True

    for rows, cols, kH, kW, seed in TRIALS:
        label = f"rows={rows} cols={cols} kH={kH} kW={kW} seed={seed}"
        ok = run_trial(rows, cols, kH, kW, seed)
        status = "PASS" if ok else "FAIL"
        print(f"  {status}  {label}")
        if not ok:
            all_passed = False

    print()
    if all_passed:
        print(f"ALL {len(TRIALS)} trials PASSED — C kernel matches NumPy reference.")
        sys.exit(0)
    else:
        print("SOME trials FAILED.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
