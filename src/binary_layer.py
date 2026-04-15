"""
binary_layer.py — First binary conv layer: 24-channel input → N integer score maps.

Takes the 24 binary planes produced by the RGB bit-plane decomposition
(R bits 0-7, G bits 0-7, B bits 0-7) and applies N XNOR-Popcount filters,
each with shape [n_ch, kH, kW] of ±1 weights.

The output before thresholding is an integer score map per filter:
  score range = [-n_ch * kH * kW, +n_ch * kH * kW]
  positive → filter pattern matches; negative → anti-matches

After thresholding (score > threshold → 1, else 0) the output is binary,
ready to be packed into uint64 and fed into the next layer.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kernel_wrapper as kw


class BinaryConvLayer:
    """
    XNOR-Popcount convolution layer with configurable ±1 weights.

    Parameters
    ----------
    n_filters : number of output feature maps (e.g. 64)
    n_ch      : number of input channels packed per pixel (max 64, e.g. 24 for RGB×8)
    kH, kW    : kernel height and width (must be odd)
    seed      : RNG seed for weight initialisation (default 42)
    """

    def __init__(self, n_filters: int, n_ch: int, kH: int, kW: int, seed: int = 42):
        if n_ch > 64:
            raise ValueError(f"n_ch={n_ch} exceeds uint64 capacity of 64")
        if kH % 2 == 0 or kW % 2 == 0:
            raise ValueError("kH and kW must be odd")

        self.n_filters = n_filters
        self.n_ch = n_ch
        self.kH = kH
        self.kW = kW
        self.max_score = n_ch * kH * kW  # theoretical max per filter per pixel

        # Initialise with random ±1 weights
        rng = np.random.default_rng(seed)
        raw = rng.integers(0, 2, size=(n_filters, n_ch, kH, kW), dtype=np.int8)
        self.weights = np.where(raw == 0, np.int8(-1), np.int8(1))

    # ------------------------------------------------------------------
    # Weight management
    # ------------------------------------------------------------------

    def set_weights(self, weights: np.ndarray) -> None:
        """
        Replace all filter weights.

        Parameters
        ----------
        weights : [n_filters, n_ch, kH, kW] int8 or int, values must be +1 or -1
        """
        weights = np.asarray(weights, dtype=np.int8)
        expected = (self.n_filters, self.n_ch, self.kH, self.kW)
        if weights.shape != expected:
            raise ValueError(f"Expected shape {expected}, got {weights.shape}")
        if not np.all((weights == 1) | (weights == -1)):
            raise ValueError("All weight values must be +1 or -1")
        self.weights = weights.copy()

    def set_filter(self, filter_idx: int, weights: np.ndarray) -> None:
        """
        Replace the weights of a single filter.

        Parameters
        ----------
        filter_idx : which filter to update (0-indexed)
        weights    : [n_ch, kH, kW] int8, values ±1
        """
        weights = np.asarray(weights, dtype=np.int8)
        expected = (self.n_ch, self.kH, self.kW)
        if weights.shape != expected:
            raise ValueError(f"Expected shape {expected}, got {weights.shape}")
        if not np.all((weights == 1) | (weights == -1)):
            raise ValueError("All weight values must be +1 or -1")
        self.weights[filter_idx] = weights

    # ------------------------------------------------------------------
    # Packing helpers (binary → uint64)
    # ------------------------------------------------------------------

    def pack_input(self, planes: np.ndarray) -> np.ndarray:
        """
        Pack [n_ch, H, W] binary (0/1) into [H, W] uint64.

        Bit i of each uint64 = channel i's activation at that pixel position.

        Parameters
        ----------
        planes : [n_ch, H, W] uint8, values 0 or 1

        Returns
        -------
        packed : [H, W] uint64
        """
        if planes.shape[0] != self.n_ch:
            raise ValueError(f"Expected {self.n_ch} channels, got {planes.shape[0]}")
        # Delegate to C: auto-vectorised loop under -O3 -march=native.
        return kw.pack_channels_to_u64(planes, self.n_ch)

    def pack_all_weights(self) -> np.ndarray:
        """
        Pack all filter weights into [n_filters, kH, kW] uint64 in one shot.

        Bit i at position (f, kr, kc) = 1 if weights[f, i, kr, kc] > 0.

        Returns
        -------
        packed : [n_filters, kH, kW] uint64
        """
        # weights: [n_filters, n_ch, kH, kW] with values ±1
        # Convert to 0/1 bits then reduce over the channel axis.
        bits   = (self.weights > 0).astype(np.uint64)                   # [n_filters, n_ch, kH, kW]
        shifts = np.arange(self.n_ch, dtype=np.uint64).reshape(1, -1, 1, 1)  # [1, n_ch, 1, 1]
        return np.bitwise_or.reduce(bits << shifts, axis=1)             # [n_filters, kH, kW]

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, planes: np.ndarray) -> np.ndarray:
        """
        Apply all n_filters XNOR-Popcount convolutions to the input planes.

        Uses a single C call (xnor_multi_filter_conv) — no Python loop over
        filters, no repeated ctypes overhead.

        Parameters
        ----------
        planes : [n_ch, H, W] uint8, values 0 or 1

        Returns
        -------
        scores : [n_filters, H, W] int32
                 Range: [-n_ch*kH*kW, +n_ch*kH*kW]
                 Larger positive score = stronger match to that filter pattern.
        """
        if planes.shape[0] != self.n_ch:
            raise ValueError(f"Expected {self.n_ch} input channels, got {planes.shape[0]}")

        input_packed   = self.pack_input(planes)       # [H, W] uint64
        weights_packed = self.pack_all_weights()       # [n_filters, kH, kW] uint64

        return kw.xnor_multi_filter_conv(
            input_packed, weights_packed,
            self.kH, self.kW,
            self.n_ch, self.n_filters,
        )

    def apply_threshold(self, scores: np.ndarray, threshold: int = 0) -> np.ndarray:
        """
        Binarize integer score maps with a per-layer threshold.

        Parameters
        ----------
        scores    : [n_filters, H, W] int32 — output of forward()
        threshold : fire if score > threshold (default 0)
                    Increase to require stronger pattern matches before firing.

        Returns
        -------
        binary : [n_filters, H, W] uint8, values 0 or 1
                 Ready to be packed into uint64 for the next layer.
        """
        # Delegate to C: avoids creating the large boolean intermediate array
        # that numpy's comparison operator would allocate.
        return kw.threshold_i32_to_u8(scores, threshold)
