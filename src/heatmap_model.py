"""
heatmap_model.py — Phase 1 screener: MSB bit-planes → object density heatmap.

Architecture
------------
Input  : [B, 4, 640, 640]  float32  — top-4 MSB planes (bits 7,6,5,4), values {0,1}
Layer 1: BinaryConv2d(4→32)  + MaxPool2d(2)  → [B, 32, 320, 320]
Layer 2: BinaryConv2d(32→64) + MaxPool2d(2)  → [B, 64, 160, 160]
Layer 3: BinaryConv2d(64→64) + MaxPool2d(2)  → [B, 64,  80,  80]
Output : Conv2d(64→1, 1×1)  + Sigmoid        → [B,  1,  80,  80]

Each output cell covers an 8×8-pixel region of the 640×640 input.
Output values are in [0, 1]: 0 = empty sky, 1 = dense object cluster.

BinaryConv2d and the STE helpers are imported from bnn_model.py — not duplicated.
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from binary_ops import BinaryConv2d


# ---------------------------------------------------------------------------
# MSB plane extractor
# ---------------------------------------------------------------------------

def extract_msb_planes(img_uint8: np.ndarray, n_bits: int = 4) -> np.ndarray:
    """
    Extract the top-N MSB planes from a grayscale uint8 image.

    Parameters
    ----------
    img_uint8 : [H, W] uint8
    n_bits    : how many planes to extract, starting from bit 7 (default 4)

    Returns
    -------
    planes : [n_bits, H, W] float32, values in {0.0, 1.0}
             planes[0] = bit 7 (MSB), planes[-1] = bit (8 - n_bits)
    """
    planes = []
    for i in range(7, 7 - n_bits, -1):
        planes.append(((img_uint8 >> i) & 1).astype(np.float32))
    return np.stack(planes, axis=0)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class HeatmapScreener(nn.Module):
    """
    Lightweight binary screener that maps MSB planes to an object density heatmap.

    Parameters
    ----------
    n_msb : number of MSB planes used as input channels (default 4)
    """

    def __init__(self, n_msb: int = 4):
        super().__init__()

        self.backbone = nn.Sequential(
            BinaryConv2d(n_msb, 32, ksize=3),
            nn.MaxPool2d(2),
            BinaryConv2d(32, 64, ksize=3),
            nn.MaxPool2d(2),
            BinaryConv2d(64, 64, ksize=3),
            nn.MaxPool2d(2),
        )

        # Float32 1×1 projection → single-channel heatmap
        self.head = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : [B, n_msb, H, W] float32, values {0, 1}

        Returns
        -------
        heatmap : [B, 1, H/8, W/8] float32, values in (0, 1)
        """
        feat = self.backbone(x)
        return torch.sigmoid(self.head(feat))
