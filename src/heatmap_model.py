"""
heatmap_model.py — Phase 1 screener: MSB bit-planes → object density heatmap.

Architecture
------------
Input  : [B, 4, 640, 640]  float32  — top-4 MSB planes (bits 7,6,5,4), values {0,1}
Layer 1: Conv2d(4→32,  3×3) + BN + ReLU + MaxPool2d(2)  → [B, 32, 320, 320]
Layer 2: Conv2d(32→64, 3×3) + BN + ReLU + MaxPool2d(2)  → [B, 64, 160, 160]
Layer 3: Conv2d(64→64, 3×3) + BN + ReLU + MaxPool2d(2)  → [B, 64,  80,  80]
Output : Conv2d(64→1, 1×1) + Sigmoid                    → [B,  1,  80,  80]

Training uses standard float32 weights. At export, sign(weight) produces the
±1 binary weights for XNOR-Popcount deployment in the C kernel.

Each output cell covers an 8×8-pixel region of the 640×640 input.
Output values are in [0, 1]: 0 = empty sky, 1 = dense object cluster.
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


sys.path.insert(0, str(Path(__file__).resolve().parent))


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


def _conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class HeatmapScreener(nn.Module):
    """
    Lightweight float32 screener trained to produce object density heatmaps.
    Weights are exported as ±1 via sign() for binary deployment.

    Parameters
    ----------
    n_msb : number of MSB planes used as input channels (default 4)
    """

    def __init__(self, n_msb: int = 4):
        super().__init__()

        self.backbone = nn.Sequential(
            _conv_block(n_msb, 32),
            _conv_block(32, 64),
            _conv_block(64, 64),
        )
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
        return torch.sigmoid(self.head(self.backbone(x)))

    def export_binary_weights(self) -> list[torch.Tensor]:
        """Return ±1 int8 weights for each Conv2d layer in the backbone."""
        return [
            torch.sign(m.weight).to(torch.int8).detach().cpu()
            for m in self.backbone.modules()
            if isinstance(m, nn.Conv2d)
        ]
