"""
bnn_model.py — BNN backbone + YOLOv8 detection head.

Architecture
------------
Input  : [B, 3, 640, 640] float32  (standard RGB, values roughly in [-1, 1])
Layer 1: BinaryConv2d(3 → 64, 3×3)  + BN   ← single binary conv, STE-trained
Pyramid: avg_pool × 3, no parameters       ← P3=80px, P4=40px, P5=20px
Head   : Detect(nc, ch=(64,64,64))         ← stolen from ultralytics YOLOv8

The binary weights are ±1 (stored as float internally, binarised at each forward
via torch.sign() with the Straight-Through Estimator for gradients).
After training, call model.export_binary_weights() to get int8 tensors ready for
the C inference kernel.

The detection head is the exact Detect module from the ultralytics package,
initialised fresh (no pretrained weights) but compatible with v8DetectionLoss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from types import SimpleNamespace
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from binary_ops import BinaryConv2d, binary, _STESign  # noqa: F401 — re-exported

from ultralytics.nn.modules.head import Detect


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class BNNDetector(nn.Module):
    """
    Binary backbone + YOLOv8-style detection head.

    Backbone
    --------
    One BinaryConv2d(3, 64, 3×3) layer with BN.
    Input is standard RGB normalised to [-1, 1].

    Multi-scale pyramid (no learnable parameters)
    ----------------------------------------------
    avg_pool2d(stride=8)  → P3: [B, 64, 80, 80]   (for 640-px input)
    avg_pool2d(stride=16) → P4: [B, 64, 40, 40]
    avg_pool2d(stride=32) → P5: [B, 64, 20, 20]

    Detection head
    --------------
    ultralytics Detect(nc=nc, ch=(64, 64, 64)) — anchor-free DFL head.
    Directly accepts the three feature maps and outputs box + class predictions.

    Training compatibility
    ----------------------
    self.model[-1] is the Detect module, as expected by ultralytics v8DetectionLoss.
    self.args holds the loss hyperparameters (box/cls/dfl gains).

    Parameters
    ----------
    nc    : number of detection classes (10 for VisDrone)
    """

    # Downsampling strides for P3 / P4 / P5
    STRIDES = [8, 16, 32]

    def __init__(self, nc: int = 10):
        super().__init__()

        # --- Binary backbone ---
        self.backbone = BinaryConv2d(3, 64, ksize=3)

        # --- Detection head (borrowed from ultralytics) ---
        head = Detect(nc=nc, ch=(64, 64, 64))
        head.stride = torch.tensor(self.STRIDES, dtype=torch.float)
        head.bias_init()

        # self.model is a ModuleList whose last element is the Detect module —
        # this matches the layout that v8DetectionLoss introspects via model.model[-1].
        self.model = nn.ModuleList([self.backbone, head])

        # Loss hyperparameters (box / cls / dfl gain). Override before training if needed.
        self.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5)

    # Convenience accessor
    @property
    def head(self) -> Detect:
        return self.model[-1]

    def forward(self, x: torch.Tensor) -> dict:
        """
        Parameters
        ----------
        x : [B, 3, H, W] float32, RGB normalised to [-1, 1]

        Returns
        -------
        dict with keys 'boxes', 'scores', 'feats'  (training mode)
        """
        feat = self.backbone(x)                           # [B, 64, H, W]

        # Avg-pool downsampling — no learned parameters
        p3 = F.avg_pool2d(feat, kernel_size=8)            # [B, 64, H/8,  W/8 ]
        p4 = F.avg_pool2d(feat, kernel_size=16)           # [B, 64, H/16, W/16]
        p5 = F.avg_pool2d(feat, kernel_size=32)           # [B, 64, H/32, W/32]

        return self.head([p3, p4, p5])

    # ------------------------------------------------------------------
    # Inference export helpers
    # ------------------------------------------------------------------

    def export_binary_weights(self) -> torch.Tensor:
        """
        Return backbone weights as int8 ±1 tensor for the C inference kernel.

        Returns
        -------
        weights : [64, 3, 3, 3] int8, values ±1
        """
        return torch.sign(self.backbone.weight).to(torch.int8).detach().cpu()

    def export_bn_threshold(self) -> torch.Tensor:
        """
        Compute the integer threshold that replaces BN at inference.

        After BN, the activation is: BN(sum) = gamma*(sum - mean)/std + beta
        We want BN(sum) > 0, which rearranges to:
            sum > mean - beta * std/gamma    (when gamma > 0)

        Returns a per-output-channel threshold tensor [64] float32.
        The C kernel compares the popcount sum against this threshold.
        """
        bn = self.backbone.bn
        mean  = bn.running_mean
        var   = bn.running_var
        gamma = bn.weight
        beta  = bn.bias
        eps   = bn.eps
        std   = (var + eps).sqrt()
        # threshold t: popcount_sum > t → activate
        return (mean - beta * std / gamma).detach().cpu()
