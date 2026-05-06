"""
train_heatmap.py — Train the Phase 1 HeatmapScreener on VisDrone.

Usage
-----
    python3 scripts/train_heatmap.py [--epochs N] [--batch B] [--lr LR]
                                     [--max-samples N] [--device cpu|cuda]

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

ROOT    = Path(__file__).resolve().parent.parent
IMG_DIR = ROOT / "data" / "VisDrone2019-DET-train" / "images"
ANN_DIR = ROOT / "data" / "VisDrone2019-DET-train" / "annotations"
RUN_DIR = ROOT / "runs" / "heatmap_train"

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
    """

    def __init__(self, img_dir: Path, ann_dir: Path, max_samples: int = 0):
        self.img_dir = img_dir
        self.ann_dir = ann_dir
        stems = [p.stem for p in sorted(img_dir.glob("*.jpg"))]
        if max_samples:
            stems = stems[:max_samples]
        self.stems = stems

    def __len__(self) -> int:
        return len(self.stems)

    def __getitem__(self, idx: int) -> dict:
        stem = self.stems[idx]

        # --- Load and resize to 640×640 grayscale ---
        pil = Image.open(self.img_dir / f"{stem}.jpg").convert("L")
        orig_w, orig_h = pil.size
        pil = pil.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
        gray = np.array(pil, dtype=np.uint8)   # [640, 640]

        msb = extract_msb_planes(gray, N_MSB)  # [4, 640, 640] float32

        # --- Build GT heatmap ---
        heatmap = np.zeros((HEATMAP_OUT, HEATMAP_OUT), dtype=np.float32)

        ann_path = self.ann_dir / f"{stem}.txt"
        if ann_path.exists():
            scale_x = HEATMAP_OUT / orig_w
            scale_y = HEATMAP_OUT / orig_h
            with open(ann_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split(",")
                    if len(parts) < 6:
                        continue
                    x1, y1, bw, bh = float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])
                    score = int(parts[4])
                    cat   = int(parts[5])

                    if score == 0 or cat not in VALID_CATS or bw <= 0 or bh <= 0:
                        continue

                    cx80 = (x1 + bw / 2) * scale_x
                    cy80 = (y1 + bh / 2) * scale_y
                    w80  = bw * scale_x
                    h80  = bh * scale_y
                    sigma = max(1.0, math.sqrt(w80 * h80) / 4.0)

                    blob = _gaussian_blob(HEATMAP_OUT, HEATMAP_OUT, cy80, cx80, sigma)
                    heatmap = np.maximum(heatmap, blob)

        return {
            "msb_planes": torch.from_numpy(msb),
            "heatmap":    torch.from_numpy(heatmap).unsqueeze(0),  # [1, 80, 80]
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
    device = torch.device(args.device)

    model = HeatmapScreener(n_msb=N_MSB).to(device)
    print(f"HeatmapScreener parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Weighted BCE: ~50× upweight positives to compensate class imbalance
    pos_weight = torch.tensor([50.0], device=device)
    criterion  = nn.BCELoss(weight=pos_weight.expand(args.batch, 1, HEATMAP_OUT, HEATMAP_OUT))

    optimizer = torch.optim.Adam([
        {"params": model.backbone.parameters(), "lr": args.lr * 10},
        {"params": model.head.parameters(),     "lr": args.lr},
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    dataset = HeatmapDataset(IMG_DIR, ANN_DIR, max_samples=args.max_samples)
    loader  = DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=True,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
    )
    print(f"Dataset: {len(dataset)} images, {len(loader)} batches/epoch")

    RUN_DIR.mkdir(parents=True, exist_ok=True)

    model.train()
    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0.0
        t0 = time.perf_counter()

        for batch in loader:
            planes = batch["msb_planes"].to(device)
            target = batch["heatmap"].to(device)

            optimizer.zero_grad()
            pred = model(planes)

            # Trim pos_weight tensor to actual batch size (last batch may be smaller)
            bsz = pred.shape[0]
            pw  = pos_weight.expand(bsz, 1, HEATMAP_OUT, HEATMAP_OUT)
            loss = nn.functional.binary_cross_entropy(pred, target, weight=pw)

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
            ckpt_path = RUN_DIR / f"epoch_{epoch:03d}.pt"
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

    print(f"\nDone. Checkpoints in {RUN_DIR}/")


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
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
