"""
train_heatmap.py — Train the Phase 1 HeatmapScreener on VisDrone.

Usage
-----
    python3 scripts/train_heatmap.py [--epochs N] [--batch B] [--lr LR]
                                     [--max-samples N] [--device cpu|cuda]
                                     [--img-dir PATH] [--ann-dir PATH] [--run-dir PATH]

What this does
--------------
1. Loads VisDrone images, converts to grayscale, extracts top-4 MSB planes.
2. Builds Gaussian GT heatmaps from bounding-box annotations.
3. Trains HeatmapScreener with weighted BCE loss.
4. Logs per-epoch loss and recall (fraction of annotated centers in heatmap > 0.5).
5. Saves checkpoints to runs/heatmap_train/.

VisDrone annotation format (per line)
--------------------------------------
  x1, y1, w, h, score, category_id, truncation, occlusion
  - score=0 → ignored region, skip
  - category_id: 0=ignore, 1-10=valid, 11=others (skip)

GT heatmap generation
---------------------
For each valid bbox, a Gaussian blob is stamped onto an 80×80 float32 target map.
Sigma = max(1.0, sqrt(w80 * h80) / 4) where w80, h80 are bbox dimensions at 80×80 scale.
"""

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from heatmap_model import HeatmapScreener, extract_msb_planes

ROOT         = Path(__file__).resolve().parent.parent
_DEFAULT_IMG = ROOT / "data" / "VisDrone2019-DET-train" / "images"
_DEFAULT_ANN = ROOT / "data" / "VisDrone2019-DET-train" / "annotations"
_DEFAULT_RUN = ROOT / "runs" / "heatmap_train"

VALID_CATS  = set(range(1, 11))
IMG_SIZE    = 640
HEATMAP_OUT = 80          # output spatial resolution
N_MSB       = 4           # top-4 bit-planes: bits 7,6,5,4


# ---------------------------------------------------------------------------
# Gaussian blob helper
# ---------------------------------------------------------------------------

def _gaussian_blob(h: int, w: int, cy: float, cx: float, sigma: float) -> np.ndarray:
    """Return a [h, w] float32 array with a Gaussian centred at (cy, cx)."""
    ys = np.arange(h, dtype=np.float32)
    xs = np.arange(w, dtype=np.float32)
    gy = np.exp(-0.5 * ((ys - cy) / sigma) ** 2)
    gx = np.exp(-0.5 * ((xs - cx) / sigma) ** 2)
    return np.outer(gy, gx)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class HeatmapDataset(Dataset):
    """
    Each item:
      msb_planes : [N_MSB, 640, 640] float32, values {0, 1}
      heatmap    : [1, 80, 80]       float32, values in [0, 1]

    label_format options
    --------------------
    'visdrone' : comma-separated  x1,y1,w,h,score,cat,...  (pixel coords)
    'yolo'     : space-separated  class cx cy w h           (normalised 0-1)
    """

    def __init__(self, img_dir: Path, ann_dir: Path,
                 max_samples: int = 0, label_format: str = "visdrone"):
        self.img_dir      = img_dir
        self.ann_dir      = ann_dir
        self.label_format = label_format
        stems = [p.stem for p in sorted(img_dir.glob("*.jpg"))]
        if max_samples:
            stems = stems[:max_samples]
        self.stems = stems

    def __len__(self) -> int:
        return len(self.stems)

    def _parse_boxes(self, ann_path: Path, orig_w: int, orig_h: int):
        """Yield (cx80, cy80, w80, h80) for every valid box in the label file."""
        if not ann_path.exists():
            return

        with open(ann_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                if self.label_format == "yolo":
                    # class cx cy w h  — already normalised to [0, 1]
                    parts = line.split()
                    if len(parts) < 5:
                        continue
                    cx_n, cy_n, w_n, h_n = float(parts[1]), float(parts[2]), \
                                            float(parts[3]), float(parts[4])
                    if w_n <= 0 or h_n <= 0:
                        continue
                    yield (cx_n * HEATMAP_OUT,
                           cy_n * HEATMAP_OUT,
                           w_n  * HEATMAP_OUT,
                           h_n  * HEATMAP_OUT)

                else:  # visdrone
                    # x1,y1,w,h,score,cat,...  — pixel coords in original image
                    parts = line.split(",")
                    if len(parts) < 6:
                        continue
                    x1, y1, bw, bh = float(parts[0]), float(parts[1]), \
                                      float(parts[2]), float(parts[3])
                    score = int(parts[4])
                    cat   = int(parts[5])
                    if score == 0 or cat not in VALID_CATS or bw <= 0 or bh <= 0:
                        continue
                    scale_x = HEATMAP_OUT / orig_w
                    scale_y = HEATMAP_OUT / orig_h
                    yield ((x1 + bw / 2) * scale_x,
                           (y1 + bh / 2) * scale_y,
                           bw * scale_x,
                           bh * scale_y)

    def __getitem__(self, idx: int) -> dict:
        stem = self.stems[idx]

        pil = Image.open(self.img_dir / f"{stem}.jpg").convert("L")
        orig_w, orig_h = pil.size
        pil  = pil.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
        gray = np.array(pil, dtype=np.uint8)

        msb     = extract_msb_planes(gray, N_MSB)
        heatmap = np.zeros((HEATMAP_OUT, HEATMAP_OUT), dtype=np.float32)

        for cx80, cy80, w80, h80 in self._parse_boxes(
                self.ann_dir / f"{stem}.txt", orig_w, orig_h):
            sigma = max(1.0, math.sqrt(w80 * h80) / 4.0)
            blob  = _gaussian_blob(HEATMAP_OUT, HEATMAP_OUT, cy80, cx80, sigma)
            heatmap = np.maximum(heatmap, blob)

        return {
            "msb_planes": torch.from_numpy(msb),
            "heatmap":    torch.from_numpy(heatmap).unsqueeze(0),
        }


# ---------------------------------------------------------------------------
# Recall metric
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_recall(
    model: HeatmapScreener,
    loader: DataLoader,
    device: torch.device,
    threshold: float = 0.5,
) -> float:
    """
    Fraction of annotated Gaussian centers that land in a predicted cell > threshold.
    A center is "covered" if the heatmap value at its GT peak position exceeds threshold.
    """
    model.eval()
    hits = total = 0
    for batch in loader:
        planes = batch["msb_planes"].to(device)
        gt     = batch["heatmap"].to(device)          # [B, 1, 80, 80]
        pred   = model(planes)                        # [B, 1, 80, 80]
        pred_bin = (pred > threshold).float()
        gt_bin   = (gt  > threshold).float()
        # count pixels where GT fires and pred also fires
        hits  += (pred_bin * gt_bin).sum().item()
        total += gt_bin.sum().item()
    model.train()
    return hits / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args):
    device  = torch.device(args.device)
    img_dir = Path(args.img_dir)
    ann_dir = Path(args.ann_dir)
    run_dir = Path(args.run_dir)

    model = HeatmapScreener(n_msb=N_MSB).to(device)
    print(f"HeatmapScreener parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Weighted BCE: ~50× upweight positives to compensate class imbalance
    pos_weight = torch.tensor([100.0], device=device)

    optimizer = torch.optim.Adam([
        {"params": model.backbone.parameters(), "lr": args.lr * 10},
        {"params": model.head.parameters(),     "lr": args.lr},
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    dataset = HeatmapDataset(img_dir, ann_dir,
                             max_samples=args.max_samples,
                             label_format=args.label_format)
    loader  = DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=True,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
    )
    print(f"Dataset: {len(dataset)} images, {len(loader)} batches/epoch")

    run_dir.mkdir(parents=True, exist_ok=True)

    model.train()
    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0.0
        t0 = time.perf_counter()

        for batch in loader:
            planes = batch["msb_planes"].to(device)
            target = batch["heatmap"].to(device)

            optimizer.zero_grad()
            pred = model(planes)

            # pos_weight applied only to positive pixels; background gets weight 1.0
            weight = pos_weight * target + (1.0 - target)
            loss   = nn.functional.binary_cross_entropy(pred, target, weight=weight)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.backbone.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()

        scheduler.step()

        elapsed = time.perf_counter() - t0
        avg_loss = epoch_loss / len(loader)

        recall_str = ""
        if epoch % 5 == 0 or epoch == args.epochs:
            recall = compute_recall(model, loader, device)
            recall_str = f" | recall={recall:.3f}"
            ckpt_path = run_dir / f"epoch_{epoch:03d}.pt"
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "n_msb":       N_MSB,
            }, ckpt_path)
            print(f"  → saved {ckpt_path}")

        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"loss={avg_loss:.4f}{recall_str} | {elapsed:.0f}s"
        )

    print(f"\nDone. Checkpoints in {run_dir}/")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train HeatmapScreener on VisDrone")
    p.add_argument("--epochs",      type=int,   default=30)
    p.add_argument("--batch",       type=int,   default=8)
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--device",      type=str,   default="cpu")
    p.add_argument("--max-samples", type=int,   default=0,
                   help="limit dataset size (0 = all)")
    p.add_argument("--img-dir",     type=str,   default=str(_DEFAULT_IMG),
                   help="path to images folder")
    p.add_argument("--ann-dir",     type=str,   default=str(_DEFAULT_ANN),
                   help="path to annotations folder")
    p.add_argument("--run-dir",      type=str,   default=str(_DEFAULT_RUN),
                   help="where to save checkpoints")
    p.add_argument("--label-format", type=str,   default="visdrone",
                   choices=["visdrone", "yolo"],
                   help="visdrone=comma x1,y1,w,h,score,cat  yolo=space class,cx,cy,w,h normalised")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
