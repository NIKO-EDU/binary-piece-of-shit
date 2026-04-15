# Detailed Implementation Plan: Bit-Plane XNOR-Popcount SOD

Reference the `highlevelplan.md` for the project vision and presentation goals.

---

## Phase 1 — Data Infrastructure

**Goal:** Produce packed `uint64` bit-plane tensors from raw VisDrone images.

### Tasks
1. **Resize pipeline**
   - Load each `.jpg` from `data/VisDrone2019-DET-train/images/`.
   - Resize to **640×640** (letterbox or stretch — decide and document the choice).
   - Convert RGB → Grayscale (`uint8`).

2. **Bit-plane decomposition**
   - For each pixel value `v`, extract 8 binary planes: `plane[i] = (v >> i) & 1` for `i` in 0–7.
   - Output shape per image: `[8, 640, 640]` boolean/uint8.

3. **Bit-packing into `uint64`**
   - Pack each 640-pixel row into **10 × `uint64`** using `numpy.packbits` + view cast or manual shifting.
   - Final tensor shape per image: `[8, 640, 10]` of dtype `uint64`.
   - Save as `.npy` files alongside originals or in a `data/packed/` directory.

4. **Sanity check (required metric)**
   - Reconstruct the grayscale image from the 8 planes: `sum(plane[i] << i for i in range(8))`.
   - Assert pixel-perfect equality with the original grayscale frame.
   - Write a test script `scripts/verify_packing.py` that runs this check on a sample of 10 images.

---

## Phase 2 — C Kernel (XNOR-Popcount Engine)

**Goal:** Implement a correct, fast 3×3 convolution using XNOR + POPCNT on packed `uint64` inputs.

### Tasks
1. **Implement `xnor_conv3x3` in C** (`src/kernel.c`)
   - Signature: `void xnor_conv3x3(uint64_t *input, uint64_t *weights, int32_t *output, int rows, int cols_u64)`
   - Inner loop: `result += __builtin_popcountll(~(input_window ^ weight))` for each of the 9 positions.
   - Use `__builtin_popcountll` (GCC) with a compile-time fallback for non-GCC.

2. **Bias/BN fusion**
   - After the popcount sum, add `int16` bias.
   - Threshold: output bit = `(sum + bias) > 0 ? 1 : 0`.

3. **Sobel validation (Engine Validation asset)**
   - Hardcode 3×3 Sobel Gx and Gy weights as `uint64` packed values.
   - Run kernel on one VisDrone image.
   - Output a 640×640 edge map (PNG) via Python + ctypes.
   - Visually confirm edges align with image structure.

4. **Math parity test (required metric)**
   - Write `scripts/verify_kernel.py`.
   - Run `xnor_conv3x3` on a random input via ctypes.
   - Compare result to equivalent NumPy convolution (unpack → float → convolve → threshold).
   - Assert outputs are identical.

---

## Phase 3 — YOLO-tiny Baseline

**Goal:** Establish concrete latency/memory numbers to beat.

### Tasks
1. **Setup**
   - Install `ultralytics` and `onnxruntime`.
   - Export YOLOv8-tiny to ONNX: `yolo export model=yolov8n.pt format=onnx`.

2. **Benchmark script** (`scripts/benchmark_yolo.py`)
   - Load 100 random VisDrone images, resize to 640×640.
   - Run inference with ONNX Runtime (CPU provider).
   - Record: mean/std inference time (ms), p95 latency, weight file size (MB).

3. **Metrics to capture**
   - Average FPS (=1000/mean_ms).
   - Weight memory footprint: `os.path.getsize("yolov8n.onnx") / 1e6` MB.
   - IoU on a small VisDrone validation subset (optional stretch goal).

---

## Phase 4 — Integration (ctypes Bridge)

**Goal:** Pass real VisDrone bit-plane data through the C kernel from Python.

### Tasks
1. **Compile the kernel as a shared library**
   - `gcc -O3 -march=native -shared -fPIC -o src/kernel.so src/kernel.c`
   - Add a `Makefile` target.

2. **Python wrapper** (`src/kernel_wrapper.py`)
   - Load `kernel.so` via `ctypes.CDLL`.
   - Define `argtypes` and `restype` for `xnor_conv3x3`.
   - Expose a `convolve(image_uint64: np.ndarray, weights_uint64: np.ndarray) -> np.ndarray` function.

3. **End-to-end pipeline script** (`scripts/run_pipeline.py`)
   - Load a VisDrone image → pack to `uint64` planes → pass through kernel → save output.

4. **Benchmark script** (`scripts/benchmark_kernel.py`)
   - Same 100-image set as Phase 3.
   - Record mean FPS and peak RSS memory (`resource.getrusage`).

---

## Phase 5 — Presentation Assets

**Goal:** Produce the four visual/data assets described in the high-level plan.

### Asset A — Detail Preservation Visual
- Script: `scripts/make_bitplane_visual.py`
- Output: side-by-side figure (original | 1-bit binarized | 8 bit-planes | reconstructed).
- Pick a VisDrone image with a small (~15px) vehicle.

### Asset B — Edge Map (Engine Validation)
- Reuse Sobel output from Phase 2, Task 3.
- Clean up into a presentation-quality PNG with a caption overlay.

### Asset C — Speed Showdown
- Pull numbers from Phase 3 (YOLO) and Phase 4 (C kernel) benchmarks.
- Plot a bar chart: FPS comparison.
- Script: `scripts/make_speed_chart.py`.

### Asset D — Memory Efficiency Table
- Static table (no script needed):
  | Model | Weight Size |
  |-------|-------------|
  | YOLOv8-tiny (ONNX) | ~12–20 MB |
  | NAS-BNN (packed) | ~400–800 KB |
- Note the L2 cache fit argument.

---

## Deliverable Checklist

- [ ] Phase 1: `scripts/verify_packing.py` passes on 10 images
- [ ] Phase 2: `scripts/verify_kernel.py` passes (math parity)
- [ ] Phase 2: Sobel edge map PNG generated
- [ ] Phase 3: YOLO baseline numbers recorded
- [ ] Phase 4: `scripts/benchmark_kernel.py` produces FPS number > 100
- [ ] Phase 5: All four presentation assets generated

---

## Suggested File Layout

```
SOD-OPT/
├── CLAUDE.md
├── Makefile
├── plan/
│   ├── highlevelplan.md
│   └── detailed_plan.md
├── data/
│   └── VisDrone2019-DET-train/
│       ├── images/
│       └── annotations/
├── src/
│   ├── kernel.c
│   └── kernel_wrapper.py
└── scripts/
    ├── verify_packing.py
    ├── verify_kernel.py
    ├── benchmark_yolo.py
    ├── benchmark_kernel.py
    ├── run_pipeline.py
    ├── make_bitplane_visual.py
    └── make_speed_chart.py
```
