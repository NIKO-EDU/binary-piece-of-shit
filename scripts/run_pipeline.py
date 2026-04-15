"""
run_pipeline.py — Phase 4: end-to-end demo

Passes one VisDrone image through both pipelines:
  1. YOLOv8n ONNX  → detection count
  2. C Sobel kernel → edge map saved to data/pipeline_demo_sobel.png

Usage:
    python scripts/run_pipeline.py
"""

import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

ROOT      = Path(__file__).resolve().parent.parent
IMG_DIR   = ROOT / "data" / "VisDrone2019-DET-train" / "images"
ONNX_PATH = ROOT / "yolov8n.onnx"
OUT_PNG   = ROOT / "data" / "pipeline_demo_sobel.png"

sys.path.insert(0, str(ROOT / "src"))
import kernel_wrapper as kw

CONF_THRESHOLD = 0.25


def main() -> None:
    images = sorted(IMG_DIR.glob("*.jpg"))
    if not images:
        print("ERROR: no images found.", file=sys.stderr)
        sys.exit(1)

    img_path = images[0]
    print(f"Image: {img_path.name}")

    # --- Load and resize ---
    pil_img = Image.open(img_path).convert("RGB").resize((640, 640), Image.LANCZOS)
    gray = np.array(pil_img.convert("L"), dtype=np.uint8)

    # --- YOLO inference ---
    session = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    tensor = (np.array(pil_img, dtype=np.float32) / 255.0).transpose(2, 0, 1)[np.newaxis]
    outputs = session.run(None, {input_name: np.ascontiguousarray(tensor)})

    # YOLOv8 ONNX output: [1, 84, 8400] — first 4 rows = box coords, rest = class scores
    preds = outputs[0][0]           # [84, 8400]
    scores = preds[4:, :].max(axis=0)  # max class score per anchor
    n_detections = int((scores >= CONF_THRESHOLD).sum())
    print(f"YOLO detections (conf≥{CONF_THRESHOLD}): {n_detections}")

    # --- C Sobel kernel ---
    gradient = kw.sobel_conv(gray)
    g_min, g_max = float(gradient.min()), float(gradient.max())
    normed = ((gradient - g_min) / (g_max - g_min) * 255.0).astype(np.uint8) \
             if g_max > g_min else np.zeros_like(gradient, dtype=np.uint8)
    Image.fromarray(normed).save(OUT_PNG)
    print(f"Sobel output  : shape={gradient.shape}  min={g_min:.1f}  max={g_max:.1f}")
    print(f"Edge map saved: {OUT_PNG}")


if __name__ == "__main__":
    main()
