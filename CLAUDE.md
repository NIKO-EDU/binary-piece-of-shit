# SOD-OPT — Claude Code Context

## Project Overview
This project implements a **Bit-Plane NAS-BNN** (Neural Architecture Search Binary Neural Network) for Salient Object Detection (SOD) on drone hardware, benchmarked against YOLOv8-tiny. The goal is to prove that XNOR-Popcount logic on bit-decomposed images outperforms YOLO-tiny in speed and memory on CPU-bound hardware.

## Key Directories

| Path | Purpose |
|------|---------|
| `data/VisDrone2019-DET-train/images/` | Raw training images (VisDrone dataset) |
| `data/VisDrone2019-DET-train/annotations/` | Corresponding annotations for training |
| `plan/highlevelplan.md` | High-level project roadmap — read this first for context |
| `plan/detailed_plan.md` | Step-by-step implementation plan with phases, tasks, and metrics |

## Important Notes
- Images should be standardized to **640x640** to match YOLOv8-tiny's default input size.
- The core representation is 8 bit-planes packed into `uint64_t` integers (640px row → 10 × `uint64`).
- The C backend uses `__builtin_popcountll` / `_mm_popcnt_u64` — no floating-point units.
- The Python/C bridge is done via `ctypes`.
- Do **not** modify anything under `data/` — treat it as read-only input.
