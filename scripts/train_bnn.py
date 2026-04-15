"""
train_bnn.py — Train the BNN backbone + YOLOv8 detection head on VisDrone.

Usage
-----
    python3 scripts/train_bnn.py [--epochs N] [--batch B] [--lr LR] [--device cpu|cuda]

What this does
--------------
1. Loads VisDrone images + annotations.
2. Converts VisDrone box format (x,y,w,h pixels) to normalised xywh for the loss.
3. Runs the BNNDetector forward pass.
4. Computes v8DetectionLoss (box + cls + DFL).
5. Backpropagates through the full model — binary weights updated via STE.
6. Saves checkpoints to runs/bnn_train/.

VisDrone annotation format (per line)
--------------------------------------
  x1, y1, w, h, score, category_id, truncation, occlusion
  - score=0 → ignored region, skip
  - category_id: 0=ignore, 1-10=valid classes (mapped to 0-9), 11=others (skip)

Label format fed to the loss
-----------------------------
  batch["batch_idx"] : [N]    long  — which image in the batch each box belongs to
  batch["cls"]       : [N, 1] float — class index 0-9
  batch["bboxes"]    : [N, 4] float — normalised xywh (cx, cy, w, h) in [0, 1]
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from bnn_model import BNNDetector

from ultralytics.utils.loss import v8DetectionLoss

ROOT    = Path(__file__).resolve().parent.parent
IMG_DIR = ROOT / "data" / "VisDrone2019-DET-train" / "images"
ANN_DIR = ROOT / "data" / "VisDrone2019-DET-train" / "annotations"
RUN_DIR = ROOT / "runs" / "bnn_train"

# VisDrone class mapping: category_id 1-10 → class index 0-9
VALID_CATS = set(range(1, 11))
IMG_SIZE   = 640


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class VisDroneDataset(Dataset):
    """
    Loads VisDrone images and converts annotations to normalised xywh format.

    Each item is a dict:
      image  : [3, 640, 640] float32, values in [-1, 1]
      boxes  : [N, 4] float32, normalised xywh (cx, cy, w, h) in [0, 1]
      labels : [N]    int64,   class index 0–9
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

        # --- Load and resize image ---
        pil = Image.open(self.img_dir / f"{stem}.jpg").convert("RGB")
        orig_w, orig_h = pil.size
        pil = pil.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
        img = torch.from_numpy(
            np.array(pil, dtype=np.float32).transpose(2, 0, 1)  # [3, H, W]
        ) / 127.5 - 1.0   # normalise to [-1, 1]

        # --- Load and convert annotations ---
        boxes, labels = [], []
        ann_path = self.ann_dir / f"{stem}.txt"
        if ann_path.exists():
            with open(ann_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split(",")
                    if len(parts) < 6:
                        continue
                    x1, y1, w, h = float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])
                    score      = int(parts[4])
                    cat        = int(parts[5])

                    if score == 0 or cat not in VALID_CATS or w <= 0 or h <= 0:
                        continue

                    # Convert pixel coords to normalised cx, cy, w, h
                    cx = (x1 + w / 2) / orig_w
                    cy = (y1 + h / 2) / orig_h
                    nw = w / orig_w
                    nh = h / orig_h
                    boxes.append([cx, cy, nw, nh])
                    labels.append(cat - 1)   # 1-indexed → 0-indexed

        if boxes:
            boxes_t  = torch.tensor(boxes,  dtype=torch.float32)
            labels_t = torch.tensor(labels, dtype=torch.int64)
        else:
            boxes_t  = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros((0,),   dtype=torch.int64)

        return {"image": img, "boxes": boxes_t, "labels": labels_t}


def collate_fn(batch: list[dict]) -> dict:
    """
    Stack images and concatenate variable-length boxes, adding batch_idx.
    """
    images     = torch.stack([b["image"]  for b in batch], dim=0)  # [B, 3, H, W]
    batch_idx  = []
    all_boxes  = []
    all_labels = []

    for i, b in enumerate(batch):
        n = b["boxes"].shape[0]
        batch_idx.append(torch.full((n,), i, dtype=torch.long))
        all_boxes.append(b["boxes"])
        all_labels.append(b["labels"])

    return {
        "image":     images,
        "batch_idx": torch.cat(batch_idx),                          # [N]
        "bboxes":    torch.cat(all_boxes),                          # [N, 4]
        "cls":       torch.cat(all_labels).float().unsqueeze(1),    # [N, 1]
    }


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args):
    device = torch.device(args.device)

    # --- Model ---
    model = BNNDetector(nc=10).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  backbone : {sum(p.numel() for p in model.backbone.parameters()):,}")
    print(f"  head     : {sum(p.numel() for p in model.head.parameters()):,}")

    # --- Loss ---
    criterion = v8DetectionLoss(model)

    # --- Optimiser ---
    # Separate LR: binary backbone needs a larger LR to overcome STE noise.
    optimizer = torch.optim.Adam([
        {"params": model.backbone.parameters(), "lr": args.lr * 10},
        {"params": model.head.parameters(),     "lr": args.lr},
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    # --- Data ---
    dataset = VisDroneDataset(IMG_DIR, ANN_DIR, max_samples=args.max_samples)
    loader  = DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=True,
        num_workers=2,
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
    )
    print(f"Dataset: {len(dataset)} images, {len(loader)} batches/epoch")

    # --- Output dir ---
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    # --- Train ---
    model.train()
    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0.0
        epoch_box = epoch_cls = epoch_dfl = 0.0
        t0 = time.perf_counter()

        for batch in loader:
            # Move data to device
            batch["image"]     = batch["image"].to(device)
            batch["batch_idx"] = batch["batch_idx"].to(device)
            batch["bboxes"]    = batch["bboxes"].to(device)
            batch["cls"]       = batch["cls"].to(device)

            optimizer.zero_grad()

            preds = model(batch["image"])
            # criterion returns ([box, cls, dfl] * batch_size, detached_items)
            loss_vec, loss_items = criterion(preds, batch)
            loss = loss_vec.sum()   # scalar for backward

            loss.backward()
            # Clip gradients on the backbone to stabilise STE training
            torch.nn.utils.clip_grad_norm_(model.backbone.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_box  += loss_items[0].item()
            epoch_cls  += loss_items[1].item()
            epoch_dfl  += loss_items[2].item()

        scheduler.step()

        n = len(loader)
        elapsed = time.perf_counter() - t0
        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"loss={epoch_loss/n:.3f} "
            f"(box={epoch_box/n:.3f} cls={epoch_cls/n:.3f} dfl={epoch_dfl/n:.3f}) | "
            f"{elapsed:.0f}s"
        )

        # Save checkpoint every 5 epochs
        if epoch % 5 == 0 or epoch == args.epochs:
            ckpt_path = RUN_DIR / f"epoch_{epoch:03d}.pt"
            torch.save({
                "epoch":          epoch,
                "model_state":    model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "binary_weights": model.export_binary_weights(),
                "bn_threshold":   model.export_bn_threshold(),
            }, ckpt_path)
            print(f"  → saved {ckpt_path}")

    print(f"\nTraining complete. Checkpoints in {RUN_DIR}/")
    print("To load and run inference:")
    print("  ckpt = torch.load('runs/bnn_train/epoch_XXX.pt')")
    print("  model.load_state_dict(ckpt['model_state'])")
    print("  # binary_weights for C kernel: ckpt['binary_weights']")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train BNN backbone + YOLOv8 head on VisDrone")
    p.add_argument("--epochs",      type=int,   default=20)
    p.add_argument("--batch",       type=int,   default=4)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--device",      type=str,   default="cpu")
    p.add_argument("--max-samples", type=int,   default=0,
                   help="limit dataset size for quick tests (0 = use all)")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
