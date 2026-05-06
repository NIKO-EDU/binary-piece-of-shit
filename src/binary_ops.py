"""
binary_ops.py — Shared binary convolution primitives.

Imported by both bnn_model.py and heatmap_model.py so neither pulls in the
other's heavy dependencies (ultralytics, etc.).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class _STESign(torch.autograd.Function):
    """
    Binary activation via sign().

    Forward : y = sign(x)
    Backward: dy/dx = 1 where |x| <= 1, else 0  (hard STE clipping)
    """
    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(x)
        return x.sign()

    @staticmethod
    def backward(ctx, grad: torch.Tensor) -> torch.Tensor:
        (x,) = ctx.saved_tensors
        return grad * (x.abs() <= 1.0).float()


def binary(x: torch.Tensor) -> torch.Tensor:
    return _STESign.apply(x)


class BinaryConv2d(nn.Module):
    """
    2-D convolution with ±1 weights and ±1 activations (STE training).

    Parameters
    ----------
    in_ch  : input channels
    out_ch : output channels
    ksize  : kernel size (must be odd, default 3)
    """

    def __init__(self, in_ch: int, out_ch: int, ksize: int = 3):
        super().__init__()
        assert ksize % 2 == 1, "kernel size must be odd"
        self.weight = nn.Parameter(torch.empty(out_ch, in_ch, ksize, ksize))
        self.bn     = nn.BatchNorm2d(out_ch)
        nn.init.kaiming_uniform_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = binary(self.weight)
        x = binary(x)
        return self.bn(F.conv2d(x, w, padding=self.weight.shape[-1] // 2))
