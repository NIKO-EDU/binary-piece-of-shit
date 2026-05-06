"""
train_heatmap.py — Train the Phase 1 HeatmapScreener on VisDrone.

Usage
-----
    python3 scripts/train_heatmap.py [--epochs N] [--batch B] [--lr LR]
                                     [--max-samples N] [--device cpu|cuda]
                                     [--img-dir PATH] [--ann-dir PATH]
                                     [--run-dir PATH] [--cache-dir PATH]
                                     [--label-format visdrone|yolo]

GT heatmap
----------
Each bounding box is drawn as a filled rectangle (1.0) on the 80×80 target map.
Any pixel inside the box is a positive target — covering 80% of a box earns 80%
of the maximum reward. Overlapping boxes are merged with np.maximum.

If --cache-dir is given, grayscale images and heatmaps are saved to .npy on first
pass and loaded from disk on all subsequent epochs. Epoch 1 is slower; epochs 2-30
skip all JPEG/annotation processing.

Metrics
-------
soft_cov  : sum(pred * gt) / sum(gt)  — primary; avg predicted prob inside GT boxes
recall@0.5: fraction of GT-positive pixels where pred > 0.5  — secondary
Both computed in a single forward pass every 10 epochs.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from heatmap_model import HeatmapScreener, extract_msb_planes

ROOT         = Path(__file__).resolve().parent.parent
_DEFAULT_IMG = ROOT / "data" / "VisDrone2019-DET-train" / "images"
_DEFAULT_ANN = ROOT / "data" / "VisDrone2019-DET-train" / "annotations"
_DEFAULT_RUN = ROOT / "runs" / "heatmap_train"

VALID_CATS  = set(range(1, 11))
IMG_SIZE    = 640
HEATMAP_OUT = 80
N_MSB       = 4


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class HeatmapDataset(Dataset):
    """
    Returns:
      msb_planes : [N_MSB, 640, 640] float32, values {0, 1}
      heatmap    : [1, 80, 80]       float32, values in {0, 1}
                   1.0 inside any GT bounding box, 0.0 everywhere else.

    If cache_dir is set, saves gray .npy and heatmap .npy on first access
    and loads from disk on all subsequent calls.
    Delete the cache directory if GT format changes.
    """

    def __init__(self, img_dir: Path, ann_dir: Path,
                 max_samples: int = 0,
                 label_format: str = "visdrone",
                 cache_dir: Optional[Path] = None):
        self.img_dir      = img_dir
        self.ann_dir      = ann_dir
        self.label_format = label_format
        self.cache_dir    = cache_dir

        if cache_dir is not None:
            (cache_dir / "gray").mkdir(parents=True, exist_ok=True)
            (cache_dir / "heatmap").mkdir(parents=True, exist_ok=True)

        stems = [p.stem for p in sorted(img_dir.glob("*.jpg"))]
        if max_samples:
            stems = stems[:max_samples]
        self.stems = stems

    def __len__(self) -> int:
        return len(self.stems)

    def _parse_boxes(self, ann_path: Path, orig_w: int, orig_h: int):
        """Yield integer (x1, y1, x2, y2) in 80×80 space for each valid box."""
        if not ann_path.exists():
            return

        with open(ann_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                if self.label_format == "yolo":
                    parts = line.split()
                    if len(parts) < 5:
                        continue
                    cx_n, cy_n, w_n, h_n = (float(parts[1]), float(parts[2]),
                                             float(parts[3]), float(parts[4]))
                    if w_n <= 0 or h_n <= 0:
                        continue
                    cx80, cy80 = cx_n * HEATMAP_OUT, cy_n * HEATMAP_OUT
                    w80,  h80  = w_n  * HEATMAP_OUT, h_n  * HEATMAP_OUT

                else:  # visdrone: x1,y1,w,h,score,cat,...  pixel coords
                    parts = line.split(",")
                    if len(parts) < 6:
                        continue
                    x1_, y1_, bw, bh = (float(parts[0]), float(parts[1]),
                                        float(parts[2]), float(parts[3]))
                    if int(parts[4]) == 0 or int(parts[5]) not in VALID_CATS:
                        continue
                    if bw <= 0 or bh <= 0:
                        continue
                    sx, sy = HEATMAP_OUT / orig_w, HEATMAP_OUT / orig_h
                    cx80, cy80 = (x1_ + bw / 2) * sx, (y1_ + bh / 2) * sy
                    w80,  h80  = bw * sx, bh * sy

                x1 = max(0,           int(cx80 - w80 / 2))
                y1 = max(0,           int(cy80 - h80 / 2))
                x2 = min(HEATMAP_OUT, max(x1 + 1, round(cx80 + w80 / 2)))
                y2 = min(HEATMAP_OUT, max(y1 + 1, round(cy80 + h80 / 2)))
                yield x1, y1, x2, y2

    def _build(self, stem: str):
        """Compute grayscale [640,640] uint8 and heatmap [80,80] float32 from scratch."""
        pil = Image.open(self.img_dir / f"{stem}.jpg").convert("L")
        orig_w, orig_h = pil.size
        pil  = pil.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
        gray = np.array(pil, dtype=np.uint8)

        heatmap = np.zeros((HEATMAP_OUT, HEATMAP_OUT), dtype=np.float32)
        for x1, y1, x2, y2 in self._parse_boxes(
                self.ann_dir / f"{stem}.txt", orig_w, orig_h):
            heatmap[y1:y2, x1:x2] = 1.0

        return gray, heatmap

    def __getitem__(self, idx: int) -> dict:
        stem = self.stems[idx]

        if self.cache_dir is not None:
            gray_path    = self.cache_dir / "gray"    / f"{stem}.npy"
            heatmap_path = self.cache_dir / "heatmap" / f"{stem}.npy"
            if gray_path.exists() and heatmap_path.exists():
                gray    = np.load(gray_path)
                heatmap = np.load(heatmap_path)
            else:
                gray, heatmap = self._build(stem)
                np.save(gray_path,    gray)
                np.save(heatmap_path, heatmap)
        else:
            gray, heatmap = self._build(stem)

        return {
            "msb_planes": torch.from_numpy(extract_msb_planes(gray, N_MSB)),
            "heatmap":    torch.from_numpy(heatmap).unsqueeze(0),
        }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_metrics(
    model: HeatmapScreener,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    """
    Single forward pass returning:
      soft_cov  : sum(pred * gt) / sum(gt)   — partial coverage, primary metric
      recall@0.5: GT-positive pixels where pred > 0.5 / total GT-positive pixels
    """
    model.eval()
    soft_num = hard_num = denom = 0.0
    for batch in loader:
        planes = batch["msb_planes"].to(device)
        gt     = batch["heatmap"].to(device)        # [B, 1, 80, 80], values {0,1}
        pred   = model(planes)                      # [B, 1, 80, 80], values (0,1)
        soft_num += (pred * gt).sum().item()
        hard_num += ((pred > 0.5) * gt).sum().item()
        denom    += gt.sum().item()
    model.train()
    if denom == 0:
        return 0.0, 0.0
    return soft_num / denom, hard_num / denom


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args):
    device    = torch.device(args.device)
    img_dir   = Path(args.img_dir)
    ann_dir   = Path(args.ann_dir)
    run_dir   = Path(args.run_dir)
    cache_dir = Path(args.cache_dir) if args.cache_dir else None

    model = HeatmapScreener(n_msb=N_MSB).to(device)
    print(f"HeatmapScreener parameters: {sum(p.numel() for p in model.parameters()):,}")
    sys.stdout.flush()

    pos_weight = torch.tensor([100.0], device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    dataset = HeatmapDataset(img_dir, ann_dir,
                             max_samples=args.max_samples,
                             label_format=args.label_format,
                             cache_dir=cache_dir)
    loader  = DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=True,
        num_workers=4,
        prefetch_factor=2,
        pin_memory=(device.type == "cuda"),
    )
    print(f"Dataset: {len(dataset)} images, {len(loader)} batches/epoch")
    if cache_dir:
        print(f"Cache dir: {cache_dir}  (epoch 1 builds cache, later epochs load from disk)")
    sys.stdout.flush()

    run_dir.mkdir(parents=True, exist_ok=True)

    model.train()
    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0.0
        t0 = time.perf_counter()

        for batch in loader:
            planes = batch["msb_planes"].to(device)
            target = batch["heatmap"].to(device)

            optimizer.zero_grad()
            pred   = model(planes)
            weight = pos_weight * target + (1.0 - target)
            loss   = nn.functional.binary_cross_entropy(pred, target, weight=weight)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        scheduler.step()

        elapsed  = time.perf_counter() - t0
        avg_loss = epoch_loss / len(loader)

        metric_str = ""
        if epoch % 10 == 0 or epoch == args.epochs:
            soft_cov, r50 = compute_metrics(model, loader, device)
            metric_str = f" | soft_cov={soft_cov:.3f} recall@0.5={r50:.3f}"

        if epoch % 5 == 0 or epoch == args.epochs:
            ckpt_path = run_dir / f"epoch_{epoch:03d}.pt"
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "n_msb":       N_MSB,
            }, ckpt_path)
            print(f"  → saved {ckpt_path}")
            sys.stdout.flush()

        print(f"Epoch {epoch:3d}/{args.epochs} | loss={avg_loss:.4f}{metric_str} | {elapsed:.0f}s")
        sys.stdout.flush()

    print(f"\nDone. Checkpoints in {run_dir}/")
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train HeatmapScreener on VisDrone")
    p.add_argument("--epochs",       type=int,   default=30)
    p.add_argument("--batch",        type=int,   default=8)
    p.add_argument("--lr",           type=float, default=3e-4)
    p.add_argument("--device",       type=str,   default="cpu")
    p.add_argument("--max-samples",  type=int,   default=0,
                   help="limit dataset size (0 = all)")
    p.add_argument("--img-dir",      type=str,   default=str(_DEFAULT_IMG))
    p.add_argument("--ann-dir",      type=str,   default=str(_DEFAULT_ANN))
    p.add_argument("--run-dir",      type=str,   default=str(_DEFAULT_RUN))
    p.add_argument("--cache-dir",    type=str,   default="",
                   help="directory for preprocessed .npy cache (empty = no cache)")
    p.add_argument("--label-format", type=str,   default="visdrone",
                   choices=["visdrone", "yolo"])
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
