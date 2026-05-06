"""
demo_heatmap.py — Visualise HeatmapScreener inference on a VisDrone image.

Usage
-----
    python3 scripts/demo_heatmap.py [--checkpoint PATH] [--image-idx N] [--out PATH]

Output
------
A side-by-side PNG with four panels:
  1. Original grayscale image (640×640)
  2. Top-4 MSB planes as a composite (bits 7–4 stacked to uint8)
  3. Raw heatmap (0–1 float, rendered as colour map)
  4. Heatmap overlaid on original image (red channel = heatmap intensity)
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from heatmap_model import HeatmapScreener, extract_msb_planes

ROOT    = Path(__file__).resolve().parent.parent
IMG_DIR = ROOT / "data" / "VisDrone2019-DET-train" / "images"
RUN_DIR = ROOT / "runs" / "heatmap_train"

N_MSB    = 4
IMG_SIZE = 640


def _latest_checkpoint() -> Path:
    ckpts = sorted(RUN_DIR.glob("epoch_*.pt"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found in {RUN_DIR}")
    return ckpts[-1]


def _colourmap(arr: np.ndarray) -> np.ndarray:
    """Map [H, W] float32 in [0,1] to [H, W, 3] uint8 using a hot colourmap."""
    r = np.clip(arr * 2.0,       0.0, 1.0)
    g = np.clip(arr * 2.0 - 0.5, 0.0, 1.0)
    b = np.clip(arr * 2.0 - 1.0, 0.0, 1.0)
    rgb = np.stack([r, g, b], axis=-1)
    return (rgb * 255).astype(np.uint8)


def _msb_composite(planes: np.ndarray) -> np.ndarray:
    """Reconstruct a grayscale image from the top-4 MSB planes for display."""
    # Each plane i (0-indexed from MSB) contributes 2^(7-i)
    result = np.zeros(planes.shape[1:], dtype=np.float32)
    for i in range(planes.shape[0]):
        result += planes[i] * (1 << (7 - i))
    result = np.clip(result, 0, 255).astype(np.uint8)
    return np.stack([result, result, result], axis=-1)


def run(args):
    device = torch.device("cpu")

    ckpt_path = Path(args.checkpoint) if args.checkpoint else _latest_checkpoint()
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt  = torch.load(ckpt_path, map_location=device)
    n_msb = ckpt.get("n_msb", N_MSB)

    model = HeatmapScreener(n_msb=n_msb).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"  epoch={ckpt['epoch']}, n_msb={n_msb}")

    # --- Load image ---
    imgs = sorted(IMG_DIR.glob("*.jpg"))
    if not imgs:
        raise FileNotFoundError(f"No images found in {IMG_DIR}")
    img_path = imgs[args.image_idx % len(imgs)]
    print(f"Image: {img_path.name}")

    pil  = Image.open(img_path).convert("L")
    pil  = pil.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    gray = np.array(pil, dtype=np.uint8)

    planes = extract_msb_planes(gray, n_msb)                    # [4, 640, 640]
    inp    = torch.from_numpy(planes).unsqueeze(0).to(device)   # [1, 4, 640, 640]

    with torch.no_grad():
        heatmap = model(inp)[0, 0].cpu().numpy()                 # [80, 80]

    # --- Build panels ---
    H, W = IMG_SIZE, IMG_SIZE

    panel_orig    = np.stack([gray, gray, gray], axis=-1)        # [H, W, 3]
    panel_msb     = _msb_composite(planes)                       # [H, W, 3]
    heat_up       = np.array(
        Image.fromarray(heatmap).resize((W, H), Image.NEAREST)
    )
    panel_heatmap = _colourmap(heat_up)                          # [H, W, 3]
    panel_overlay = panel_orig.copy().astype(np.float32)
    panel_overlay[:, :, 0] = np.clip(
        panel_overlay[:, :, 0] + heat_up * 200, 0, 255
    )
    panel_overlay = panel_overlay.astype(np.uint8)

    composite = np.concatenate(
        [panel_orig, panel_msb, panel_heatmap, panel_overlay], axis=1
    )                                                             # [H, 4*W, 3]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(composite).save(out_path)
    print(f"Saved: {out_path}")


def parse_args():
    p = argparse.ArgumentParser(description="HeatmapScreener inference demo")
    p.add_argument("--checkpoint", type=str,  default="",
                   help="path to .pt checkpoint (default: latest in runs/heatmap_train/)")
    p.add_argument("--image-idx",  type=int,  default=0,
                   help="index of image in VisDrone train set to visualise")
    p.add_argument("--out",        type=str,
                   default="data/heatmap_demo.png",
                   help="output PNG path")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
