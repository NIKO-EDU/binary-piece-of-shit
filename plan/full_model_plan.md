# Full Model Plan: BNN Backbone + YOLO Detection Head

## The Core Argument

YOLOv8n on CPU is slow not because it was designed for GPU — it is slow because every layer
is float32 matrix multiplication, which is bandwidth-limited on CPU. XNOR-Popcount replaces
those multiplications with bitwise ops + popcount, which are:

- 1 clock cycle on modern CPUs (vs 4–8 for float multiply)
- 32× smaller weight memory → entire model fits in L2/L3 cache
- No FPU pressure → runs on drone ARM cores at full speed

The detection head (box coordinates, class scores) must stay float32 because it outputs
continuous values. The detection head is tiny — on YOLOv8n it is ~5% of total parameters.
The backbone + neck is the compute bottleneck.

---

## Target Architecture

```
Input (640×640 uint8)
    ↓
[Stem conv — float32, keep]     ← first layer, raw pixels, not binarized
    ↓
[Binary backbone layers × N]    ← XNOR-Popcount C kernel
    ↓
[Binary neck/FPN layers × M]    ← same kernel, different weights
    ↓
[Detection head — float32]      ← YOLOv8 Detect module, unchanged
    ↓
Box predictions + class scores
```

YOLOv8's `Detect` module (DFL loss, anchor-free boxes) is bolted directly onto the feature
maps produced by the binary backbone. No changes to the detection head.

---

## What Needs to Be Built

### 1. `BinaryConv2d` PyTorch module (training only)

```python
class BinaryConv2d(nn.Module):
    def forward(self, x):
        # Straight-through estimator: binarize weights during forward
        w_bin = torch.sign(self.weight)
        x_bin = torch.sign(x)
        return F.conv2d(x_bin, w_bin, ...)
    # Gradient passes through sign() unchanged (STE)
```

Standard approach used in XNOR-Net, BiRealNet, BENN.

### 2. Small backbone config

4–6 binary conv layers with channel counts matching what the detection head expects
(e.g. 64 / 128 / 256). Does not need to mirror YOLOv8's exact CSP structure.

### 3. Training loop

Fine-tune on VisDrone using the YOLO detection loss (box + class + DFL).
Even 20 epochs on a subset produces a defensible benchmark number.

### 4. Weight export

After training: `torch.sign(layer.weight)` → ±1 tensors → pack to `uint64` →
pass to the existing C kernel at inference time.

### 5. Inference bridge

Binary backbone runs in C. Output feature maps are unpacked to float32 and passed
to the YOLO Detect head. The Detect head runs via ONNX Runtime or PyTorch.

---

## Missing Pieces in the Current C Kernel

`xnor_packed_u64_conv` currently outputs raw `int32` popcount scores. To chain layers
and feed into the detection head, three things need to be added (~50 lines of C):

1. **BN-fusion threshold** — after popcount: `output_bit = (sum + bias) > threshold ? 1 : 0`
   BN parameters from training are folded into a single integer threshold per channel.
2. **Binary activation packing** — threshold output is repacked to `uint64` for the next
   binary layer's input.
3. **Float32 unpack for final layer** — the last binary layer's int32 scores are cast to
   float32 before entering the detection head.

---

## Success Metric

The comparison is CPU inference time vs mAP trade-off on VisDrone:

| Model                   | CPU FPS    | mAP@50 (VisDrone) |
|-------------------------|------------|-------------------|
| YOLOv8n (float32)       | ~15 FPS    | ~28%              |
| BNN backbone + YOLO head| target 60+ | expect ~20–24%    |

A 4× speedup at ~80% of the accuracy is a strong result for drone hardware.
Drones prioritize latency (detect before you crash) over peak accuracy.

---

## Suggested Implementation Order

1. Add BN-fusion threshold + binary activation packing to `kernel.c`
2. Add float32 unpack function to `kernel.c` for the final layer output
3. Implement `BinaryConv2d` in PyTorch and build a 5-layer binary backbone
4. Attach YOLOv8's Detect head to the backbone output
5. Train on a VisDrone subset (start small: 500 images, 20 epochs)
6. Export binary weights → pack to uint64 → benchmark on CPU
7. Compare FPS and mAP against the YOLOv8n ONNX baseline from Phase 3
