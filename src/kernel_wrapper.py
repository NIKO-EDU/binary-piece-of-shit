"""
kernel_wrapper.py — ctypes bridge to src/kernel.so

Exposes:
    xnor_popcount_conv(input_u8, weights_i8, rows, cols, kH, kW) -> np.ndarray int32
    sobel_conv(input_u8, rows, cols) -> np.ndarray float32
"""

import ctypes
import os
import sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_LIB_NAME = "kernel.dll" if sys.platform == "win32" else "kernel.so"
_LIB_PATH = os.path.join(_HERE, _LIB_NAME)

_lib = None


def _load():
    global _lib
    if _lib is None:
        if not os.path.exists(_LIB_PATH):
            raise FileNotFoundError(
                f"{_LIB_NAME} not found at {_LIB_PATH}. Run `make` first."
            )
        _lib = ctypes.CDLL(_LIB_PATH)

        # int xnor_popcount_conv(uint8*, int8*, int32*, int, int, int, int)
        _lib.xnor_popcount_conv.restype = ctypes.c_int
        _lib.xnor_popcount_conv.argtypes = [
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.POINTER(ctypes.c_int8),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]

        # int sobel_conv(uint8*, float*, int, int)
        _lib.sobel_conv.restype = ctypes.c_int
        _lib.sobel_conv.argtypes = [
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_int,
        ]

        # int xnor_packed_u8_conv(uint8*, uint8*, int32*, int, int, int, int, int)
        _lib.xnor_packed_u8_conv.restype = ctypes.c_int
        _lib.xnor_packed_u8_conv.argtypes = [
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int,
            ctypes.c_int,
        ]

        # int xnor_packed_u64_conv(uint64*, uint64*, int32*, int, int, int, int, int)
        _lib.xnor_packed_u64_conv.restype = ctypes.c_int
        _lib.xnor_packed_u64_conv.argtypes = [
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int,
            ctypes.c_int,
        ]

        # int pack_channels_to_u64(uint8*, uint64*, int, int)
        _lib.pack_channels_to_u64.restype = ctypes.c_int
        _lib.pack_channels_to_u64.argtypes = [
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.c_int, ctypes.c_int,
        ]

        # int threshold_i32_to_u8(int32*, uint8*, int, int, int)
        _lib.threshold_i32_to_u8.restype = ctypes.c_int
        _lib.threshold_i32_to_u8.argtypes = [
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ]

        # int xnor_multi_filter_conv(uint64*, uint64*, int32*, int, int, int, int, int, int)
        _lib.xnor_multi_filter_conv.restype = ctypes.c_int
        _lib.xnor_multi_filter_conv.argtypes = [
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int,   # n_ch, n_filters
        ]

        # int float32_conv_nch_u8(uint8*, float*, float*, int, int, int, int, int)
        _lib.float32_conv_nch_u8.restype = ctypes.c_int
        _lib.float32_conv_nch_u8.argtypes = [
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int,
            ctypes.c_int,
        ]

        # int float32_conv_nch_u64(uint64*, float*, float*, int, int, int, int, int)
        _lib.float32_conv_nch_u64.restype = ctypes.c_int
        _lib.float32_conv_nch_u64.argtypes = [
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int,
            ctypes.c_int,
        ]
    return _lib


def xnor_popcount_conv(
    input_arr: np.ndarray,
    weights_arr: np.ndarray,
    kH: int,
    kW: int,
) -> np.ndarray:
    """
    Run XNOR-Popcount convolution.

    Parameters
    ----------
    input_arr   : (rows, cols) uint8 — values must be 0 or 1
    weights_arr : (kH, kW)     int8  — values must be +1 or -1
    kH, kW      : kernel dimensions (must be odd)

    Returns
    -------
    output : (rows, cols) int32
    """
    lib = _load()

    input_arr = np.ascontiguousarray(input_arr, dtype=np.uint8)
    weights_arr = np.ascontiguousarray(weights_arr, dtype=np.int8)

    rows, cols = input_arr.shape
    output = np.empty((rows, cols), dtype=np.int32)

    rc = lib.xnor_popcount_conv(
        input_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        weights_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
        output.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        ctypes.c_int(rows),
        ctypes.c_int(cols),
        ctypes.c_int(kH),
        ctypes.c_int(kW),
    )
    if rc != 0:
        raise ValueError(f"xnor_popcount_conv returned error code {rc}")
    return output


def xnor_packed_u8_conv(
    input_arr: np.ndarray,
    weights_arr: np.ndarray,
    kH: int, kW: int,
    n_ch: int,
) -> np.ndarray:
    """XNOR-Popcount convolution, channels packed in uint8 (n_ch ≤ 8)."""
    lib = _load()
    input_arr   = np.ascontiguousarray(input_arr,   dtype=np.uint8)
    weights_arr = np.ascontiguousarray(weights_arr, dtype=np.uint8)
    rows, cols  = input_arr.shape
    output      = np.empty((rows, cols), dtype=np.int32)
    rc = lib.xnor_packed_u8_conv(
        input_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        weights_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        output.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        ctypes.c_int(rows), ctypes.c_int(cols),
        ctypes.c_int(kH),   ctypes.c_int(kW),
        ctypes.c_int(n_ch),
    )
    if rc != 0:
        raise ValueError(f"xnor_packed_u8_conv returned {rc}")
    return output


def pack_channels_to_u64(planes: np.ndarray, n_ch: int) -> np.ndarray:
    """
    Pack [n_ch, H, W] binary (0/1) uint8 into [H, W] uint64.
    Bit i = planes[i] at each pixel. Vectorised in C with -O3.
    """
    lib = _load()
    planes = np.ascontiguousarray(planes, dtype=np.uint8)
    _, H, W = planes.shape
    output = np.empty((H, W), dtype=np.uint64)
    rc = lib.pack_channels_to_u64(
        planes.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        output.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
        ctypes.c_int(n_ch),
        ctypes.c_int(H * W),
    )
    if rc != 0:
        raise ValueError(f"pack_channels_to_u64 returned {rc}")
    return output


def threshold_i32_to_u8(scores: np.ndarray, threshold: int = 0) -> np.ndarray:
    """
    Binarize [n, H, W] int32 score maps: 1 if score > threshold else 0.
    Vectorised in C with -O3 (avoids large numpy temporaries).
    """
    lib = _load()
    scores = np.ascontiguousarray(scores, dtype=np.int32)
    n = scores.shape[0]
    npix = scores.shape[1] * scores.shape[2]
    output = np.empty_like(scores, dtype=np.uint8)
    rc = lib.threshold_i32_to_u8(
        scores.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        output.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        ctypes.c_int(n),
        ctypes.c_int(npix),
        ctypes.c_int(threshold),
    )
    if rc != 0:
        raise ValueError(f"threshold_i32_to_u8 returned {rc}")
    return output


def xnor_multi_filter_conv(
    input_arr: np.ndarray,
    weights_arr: np.ndarray,
    kH: int, kW: int,
    n_ch: int,
    n_filters: int,
) -> np.ndarray:
    """
    Apply n_filters XNOR-Popcount filters in a single C call.

    Parameters
    ----------
    input_arr   : (rows, cols) uint64 — n_ch channels packed per pixel
    weights_arr : (n_filters, kH, kW) uint64 — n_ch weight bits per position
    kH, kW      : kernel dimensions (must be odd)
    n_ch        : number of active channels (1–64)
    n_filters   : number of filters

    Returns
    -------
    output : (n_filters, rows, cols) int32
    """
    lib = _load()
    input_arr   = np.ascontiguousarray(input_arr,   dtype=np.uint64)
    weights_arr = np.ascontiguousarray(weights_arr, dtype=np.uint64)
    rows, cols  = input_arr.shape
    output      = np.empty((n_filters, rows, cols), dtype=np.int32)
    rc = lib.xnor_multi_filter_conv(
        input_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
        weights_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
        output.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        ctypes.c_int(rows), ctypes.c_int(cols),
        ctypes.c_int(kH),   ctypes.c_int(kW),
        ctypes.c_int(n_ch), ctypes.c_int(n_filters),
    )
    if rc != 0:
        raise ValueError(f"xnor_multi_filter_conv returned {rc}")
    return output


def xnor_packed_u64_conv(
    input_arr: np.ndarray,
    weights_arr: np.ndarray,
    kH: int, kW: int,
    n_ch: int,
) -> np.ndarray:
    """XNOR-Popcount convolution, channels packed in uint64 (n_ch ≤ 64)."""
    lib = _load()
    input_arr   = np.ascontiguousarray(input_arr,   dtype=np.uint64)
    weights_arr = np.ascontiguousarray(weights_arr, dtype=np.uint64)
    rows, cols  = input_arr.shape
    output      = np.empty((rows, cols), dtype=np.int32)
    rc = lib.xnor_packed_u64_conv(
        input_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
        weights_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
        output.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        ctypes.c_int(rows), ctypes.c_int(cols),
        ctypes.c_int(kH),   ctypes.c_int(kW),
        ctypes.c_int(n_ch),
    )
    if rc != 0:
        raise ValueError(f"xnor_packed_u64_conv returned {rc}")
    return output


def float32_conv_nch_u8(
    input_arr: np.ndarray,
    weights_arr: np.ndarray,
    kH: int, kW: int,
    n_ch: int,
) -> np.ndarray:
    """Float32 reference convolution from packed uint8 input (n_ch ≤ 8)."""
    lib = _load()
    input_arr   = np.ascontiguousarray(input_arr,   dtype=np.uint8)
    weights_arr = np.ascontiguousarray(weights_arr, dtype=np.float32)
    rows, cols  = input_arr.shape
    output      = np.empty((rows, cols), dtype=np.float32)
    rc = lib.float32_conv_nch_u8(
        input_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        weights_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_int(rows), ctypes.c_int(cols),
        ctypes.c_int(kH),   ctypes.c_int(kW),
        ctypes.c_int(n_ch),
    )
    if rc != 0:
        raise ValueError(f"float32_conv_nch_u8 returned {rc}")
    return output


def float32_conv_nch_u64(
    input_arr: np.ndarray,
    weights_arr: np.ndarray,
    kH: int, kW: int,
    n_ch: int,
) -> np.ndarray:
    """Float32 reference convolution from packed uint64 input (n_ch ≤ 64)."""
    lib = _load()
    input_arr   = np.ascontiguousarray(input_arr,   dtype=np.uint64)
    weights_arr = np.ascontiguousarray(weights_arr, dtype=np.float32)
    rows, cols  = input_arr.shape
    output      = np.empty((rows, cols), dtype=np.float32)
    rc = lib.float32_conv_nch_u64(
        input_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
        weights_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_int(rows), ctypes.c_int(cols),
        ctypes.c_int(kH),   ctypes.c_int(kW),
        ctypes.c_int(n_ch),
    )
    if rc != 0:
        raise ValueError(f"float32_conv_nch_u64 returned {rc}")
    return output


def sobel_conv(input_arr: np.ndarray) -> np.ndarray:
    """
    Run Sobel edge detection.

    Parameters
    ----------
    input_arr : (rows, cols) uint8 grayscale image

    Returns
    -------
    output : (rows, cols) float32 gradient magnitude
    """
    lib = _load()

    input_arr = np.ascontiguousarray(input_arr, dtype=np.uint8)
    rows, cols = input_arr.shape
    output = np.empty((rows, cols), dtype=np.float32)

    rc = lib.sobel_conv(
        input_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_int(rows),
        ctypes.c_int(cols),
    )
    if rc != 0:
        raise ValueError(f"sobel_conv returned error code {rc}")
    return output
