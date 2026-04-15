/*
 * kernel.c — XNOR-Popcount and Sobel convolution kernels
 *
 * Compile as shared library:
 *   Linux:   gcc -O3 -march=native -fopenmp -shared -fPIC -lm -o kernel.so kernel.c
 *   Windows: gcc -O3 -march=native -fopenmp -shared -lm -o kernel.dll kernel.c
 *
 * Compile for tests:
 *   Linux:   gcc -O1 -g -fsanitize=address,undefined -Wall -Wextra -lm \
 *                -o kernel_test kernel.c -DKERNEL_TEST_MAIN
 *   Windows: gcc -O1 -g -Wall -Wextra -lm -o kernel_test.exe kernel.c -DKERNEL_TEST_MAIN
 */

#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef _WIN32
#  define KERNEL_EXPORT __declspec(dllexport)
#else
#  define KERNEL_EXPORT
#endif

/* --------------------------------------------------------------------------
 * xnor_popcount_conv
 *
 * Binary-weight XNOR-Popcount 2-D convolution.
 *
 * input   : [rows × cols] uint8, each value must be 0 or 1 (a single bit-plane)
 * weights : [kH × kW]  int8,  each value must be +1 or -1
 * output  : [rows × cols] int32, caller-allocated, filled on success
 * rows    : image height  (>= 1)
 * cols    : image width   (>= 1)
 * kH      : kernel height (>= 1, odd)
 * kW      : kernel width  (>= 1, odd)
 *
 * Returns 0 on success, -1 on invalid arguments.
 *
 * Zero-padding is used for out-of-bounds positions.
 *
 * XNOR semantics (bit-level equivalent of +1/-1 multiplication):
 *   agree = ((input_bit > 0) == (weight > 0)) ? 1 : 0
 *   output = 2 * popcount - kH*kW   maps to [-kH*kW, +kH*kW]
 * -------------------------------------------------------------------------- */
KERNEL_EXPORT int xnor_popcount_conv(
    const uint8_t *input,
    const int8_t  *weights,
    int32_t       *output,
    int rows, int cols,
    int kH, int kW)
{
    /* Validate pointers */
    if (!input || !weights || !output) return -1;

    /* Validate dimensions */
    if (rows < 1 || cols < 1)   return -1;
    if (kH < 1  || kW < 1)      return -1;
    if (kH % 2 == 0 || kW % 2 == 0) return -1;  /* kernels must be odd-sized */

    int padH = kH / 2;
    int padW = kW / 2;
    int kernel_area = kH * kW;

    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            int32_t popcount = 0;

            for (int kr = 0; kr < kH; kr++) {
                for (int kc = 0; kc < kW; kc++) {
                    int ir = r + kr - padH;
                    int ic = c + kc - padW;

                    /* Zero-pad: out-of-bounds positions treated as bit 0 */
                    uint8_t input_bit = 0;
                    if (ir >= 0 && ir < rows && ic >= 0 && ic < cols) {
                        /* Clamp to {0,1} — accept any nonzero as 1 */
                        input_bit = (input[(size_t)ir * (size_t)cols + (size_t)ic]) ? 1u : 0u;
                    }

                    int8_t w = weights[(size_t)kr * (size_t)kW + (size_t)kc];

                    /*
                     * XNOR: agree when both "positive" or both "non-positive".
                     * input_bit is 0 or 1; weight is +1 or -1.
                     */
                    int agree = ((input_bit > 0) == (w > 0)) ? 1 : 0;
                    popcount += agree;
                }
            }

            /* Map popcount in [0, kernel_area] to result in [-kernel_area, +kernel_area] */
            output[(size_t)r * (size_t)cols + (size_t)c] = 2 * popcount - kernel_area;
        }
    }
    return 0;
}


/* --------------------------------------------------------------------------
 * sobel_conv
 *
 * Standard 3×3 Sobel edge-detection convolution.
 *
 * input  : [rows × cols] uint8 grayscale image
 * output : [rows × cols] float gradient magnitude (caller-allocated)
 * rows   : image height (>= 1)
 * cols   : image width  (>= 1)
 *
 * Returns 0 on success, -1 on invalid arguments.
 * Zero-padding used for border pixels.
 * -------------------------------------------------------------------------- */
KERNEL_EXPORT int sobel_conv(
    const uint8_t *input,
    float         *output,
    int rows, int cols)
{
    if (!input || !output) return -1;
    if (rows < 1 || cols < 1) return -1;

    /* Standard Sobel kernels */
    static const int8_t Gx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    static const int8_t Gy[3][3] = {{-1,-2,-1}, { 0, 0, 0}, { 1, 2, 1}};

    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            float gx = 0.0f;
            float gy = 0.0f;

            for (int kr = 0; kr < 3; kr++) {
                for (int kc = 0; kc < 3; kc++) {
                    int ir = r + kr - 1;
                    int ic = c + kc - 1;

                    float pixel = 0.0f;
                    if (ir >= 0 && ir < rows && ic >= 0 && ic < cols) {
                        pixel = (float)input[(size_t)ir * (size_t)cols + (size_t)ic];
                    }

                    gx += pixel * (float)Gx[kr][kc];
                    gy += pixel * (float)Gy[kr][kc];
                }
            }

            output[(size_t)r * (size_t)cols + (size_t)c] = sqrtf(gx * gx + gy * gy);
        }
    }
    return 0;
}


/* --------------------------------------------------------------------------
 * xnor_packed_u8_conv
 *
 * N-channel (≤ 8) XNOR-Popcount convolution with channels packed in uint8.
 *
 * input   : [rows × cols] uint8 — bit i is channel i's binary activation
 * weights : [kH × kW]    uint8 — bit i is channel i's binary weight (1=+1, 0=−1)
 * output  : [rows × cols] int32
 * n_ch    : number of active channels (1–8); only the lowest n_ch bits are used
 *
 * Inputs and weights are treated as ±1 (not 0/1):
 *   bit=1 → +1,  bit=0 → −1
 * XNOR agreement: both bits identical → agree (+1 × +1 = +1 or −1 × −1 = +1)
 * output = 2 * popcount − n_ch * kH * kW   maps to [−N, +N]
 *
 * Returns 0 on success, −1 on invalid arguments.
 * Zero-padding for out-of-bounds positions.
 * -------------------------------------------------------------------------- */
KERNEL_EXPORT int xnor_packed_u8_conv(
    const uint8_t *input,
    const uint8_t *weights,
    int32_t       *output,
    int rows, int cols,
    int kH, int kW,
    int n_ch)
{
    if (!input || !weights || !output)       return -1;
    if (rows < 1 || cols < 1)               return -1;
    if (kH < 1   || kW < 1)                 return -1;
    if (kH % 2 == 0 || kW % 2 == 0)         return -1;
    if (n_ch < 1 || n_ch > 8)               return -1;

    unsigned int mask = (n_ch == 8) ? 0xFFu : (1u << n_ch) - 1u;
    int kernel_area = kH * kW;
    int padH = kH / 2, padW = kW / 2;

    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            int32_t popcount = 0;

            for (int kr = 0; kr < kH; kr++) {
                for (int kc = 0; kc < kW; kc++) {
                    int ir = r + kr - padH;
                    int ic = c + kc - padW;

                    unsigned int in_packed = 0u;
                    if (ir >= 0 && ir < rows && ic >= 0 && ic < cols)
                        in_packed = (unsigned int)input[(size_t)ir * (size_t)cols + (size_t)ic] & mask;

                    unsigned int w_packed =
                        (unsigned int)weights[(size_t)kr * (size_t)kW + (size_t)kc] & mask;

                    /* XNOR: bit agreement = ~(in XOR w); count agreements across n_ch channels */
                    popcount += __builtin_popcount(~(in_packed ^ w_packed) & mask);
                }
            }

            output[(size_t)r * (size_t)cols + (size_t)c] = 2 * popcount - n_ch * kernel_area;
        }
    }
    return 0;
}


/* --------------------------------------------------------------------------
 * xnor_packed_u64_conv
 *
 * N-channel (≤ 64) XNOR-Popcount convolution with channels packed in uint64.
 * Identical semantics to xnor_packed_u8_conv; one __builtin_popcountll call
 * processes 64 channels in a single instruction.
 * -------------------------------------------------------------------------- */
KERNEL_EXPORT int xnor_packed_u64_conv(
    const uint64_t *input,
    const uint64_t *weights,
    int32_t        *output,
    int rows, int cols,
    int kH, int kW,
    int n_ch)
{
    if (!input || !weights || !output)       return -1;
    if (rows < 1 || cols < 1)               return -1;
    if (kH < 1   || kW < 1)                 return -1;
    if (kH % 2 == 0 || kW % 2 == 0)         return -1;
    if (n_ch < 1 || n_ch > 64)              return -1;

    uint64_t mask = (n_ch == 64) ? UINT64_MAX : ((uint64_t)1 << n_ch) - 1;
    int kernel_area = kH * kW;
    int padH = kH / 2, padW = kW / 2;

    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            int32_t popcount = 0;

            for (int kr = 0; kr < kH; kr++) {
                for (int kc = 0; kc < kW; kc++) {
                    int ir = r + kr - padH;
                    int ic = c + kc - padW;

                    uint64_t in_packed = 0;
                    if (ir >= 0 && ir < rows && ic >= 0 && ic < cols)
                        in_packed = input[(size_t)ir * (size_t)cols + (size_t)ic] & mask;

                    uint64_t w_packed =
                        weights[(size_t)kr * (size_t)kW + (size_t)kc] & mask;

                    popcount += __builtin_popcountll(~(in_packed ^ w_packed) & mask);
                }
            }

            output[(size_t)r * (size_t)cols + (size_t)c] = 2 * popcount - n_ch * kernel_area;
        }
    }
    return 0;
}


/* --------------------------------------------------------------------------
 * xnor_multi_filter_conv
 *
 * Apply n_filters XNOR-Popcount filters in a single C call, eliminating
 * Python→ctypes round-trip overhead for every filter.
 *
 * input   : [rows × cols] uint64 — n_ch channels packed per pixel
 * weights : [n_filters × kH × kW] uint64 — n_ch weight bits per kernel position
 * output  : [n_filters × rows × cols] int32 — caller-allocated
 *
 * Loop order: pixel → filters → kernel positions.
 * The kH×kW input patches for each pixel are precomputed into a small stack
 * array (9 × uint64 for a 3×3 kernel) and reused across all n_filters.
 * At 3×3 × 8 bytes = 72 bytes this fits in a single cache line, so each pixel
 * costs one set of gather loads shared across all filters rather than
 * n_filters separate gather sequences.
 *
 * Returns 0 on success, -1 on invalid arguments.
 * -------------------------------------------------------------------------- */
KERNEL_EXPORT int xnor_multi_filter_conv(
    const uint64_t *input,
    const uint64_t *weights,
    int32_t        *output,
    int rows, int cols,
    int kH, int kW,
    int n_ch,
    int n_filters)
{
    if (!input || !weights || !output)       return -1;
    if (rows < 1 || cols < 1)               return -1;
    if (kH < 1   || kW < 1)                 return -1;
    if (kH % 2 == 0 || kW % 2 == 0)         return -1;
    if (n_ch < 1 || n_ch > 64)              return -1;
    if (n_filters < 1)                       return -1;

    uint64_t mask       = (n_ch == 64) ? UINT64_MAX : ((uint64_t)1 << n_ch) - 1;
    int kernel_area     = kH * kW;
    int padH            = kH / 2;
    int padW            = kW / 2;
    size_t frame_size   = (size_t)rows * (size_t)cols;
    size_t kernel_size  = (size_t)kH * (size_t)kW;

    if (kernel_area > 128) return -1;  /* supports up to 11×11 kernel */

    /* Parallelise over rows. For each row-iteration each thread:
     *   1. Gathers the kH×kW patches ONCE per pixel (9 uint64 = 72 bytes, one cache line).
     *   2. Applies all n_filters to those cached patches — no re-reading input.
     * This minimises input reads while keeping output writes sequential within
     * each filter's row range for a given thread. */
    #pragma omp parallel for schedule(static)
    for (int r = 0; r < rows; r++) {
        uint64_t patches[128];
        for (int c = 0; c < cols; c++) {

            /* Gather the kH×kW neighbourhood patches once for this pixel. */
            int k = 0;
            for (int kr = 0; kr < kH; kr++) {
                for (int kc = 0; kc < kW; kc++) {
                    int ir = r + kr - padH;
                    int ic = c + kc - padW;
                    patches[k++] = (ir >= 0 && ir < rows && ic >= 0 && ic < cols)
                        ? input[(size_t)ir * (size_t)cols + (size_t)ic] & mask
                        : (uint64_t)0;
                }
            }

            /* Apply all filters using the cached patches. */
            for (int f = 0; f < n_filters; f++) {
                const uint64_t *w = weights + (size_t)f * kernel_size;
                int32_t popcount = 0;
                for (int kk = 0; kk < kernel_area; kk++)
                    popcount += __builtin_popcountll(~(patches[kk] ^ w[kk]) & mask);
                output[(size_t)f * frame_size + (size_t)r * (size_t)cols + (size_t)c] =
                    2 * popcount - n_ch * kernel_area;
            }
        }
    }
    return 0;
}


/* --------------------------------------------------------------------------
 * float32_conv_nch_u8
 *
 * Float32 reference implementation for xnor_packed_u8_conv.
 * Same packed uint8 input; unpacks each bit to ±1 float then does FP multiply-add.
 * Exists purely so we can measure the FP cost of the identical computation.
 *
 * weights : [n_ch × kH × kW] float32, values +1.0 or −1.0
 *           weights[ch * kH*kW + kr*kW + kc]
 * output  : [rows × cols] float32
 * -------------------------------------------------------------------------- */
KERNEL_EXPORT int float32_conv_nch_u8(
    const uint8_t *input,
    const float   *weights,
    float         *output,
    int rows, int cols,
    int kH, int kW,
    int n_ch)
{
    if (!input || !weights || !output)       return -1;
    if (rows < 1 || cols < 1)               return -1;
    if (kH < 1   || kW < 1)                 return -1;
    if (kH % 2 == 0 || kW % 2 == 0)         return -1;
    if (n_ch < 1 || n_ch > 8)               return -1;

    int padH = kH / 2, padW = kW / 2;

    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            float acc = 0.0f;

            for (int ch = 0; ch < n_ch; ch++) {
                for (int kr = 0; kr < kH; kr++) {
                    for (int kc = 0; kc < kW; kc++) {
                        int ir = r + kr - padH;
                        int ic = c + kc - padW;

                        /* Unpack bit ch from packed byte → ±1 float */
                        float in_bit = -1.0f;
                        if (ir >= 0 && ir < rows && ic >= 0 && ic < cols)
                            in_bit = (float)(2 * (int)((input[(size_t)ir * (size_t)cols + (size_t)ic] >> ch) & 1u) - 1);

                        float w = weights[(size_t)ch * (size_t)(kH * kW) +
                                          (size_t)kr * (size_t)kW + (size_t)kc];
                        acc += in_bit * w;
                    }
                }
            }

            output[(size_t)r * (size_t)cols + (size_t)c] = acc;
        }
    }
    return 0;
}


/* --------------------------------------------------------------------------
 * float32_conv_nch_u64
 *
 * Float32 reference for xnor_packed_u64_conv.
 * Same packed uint64 input; unpacks each of the n_ch bits to ±1 float.
 * -------------------------------------------------------------------------- */
KERNEL_EXPORT int float32_conv_nch_u64(
    const uint64_t *input,
    const float    *weights,
    float          *output,
    int rows, int cols,
    int kH, int kW,
    int n_ch)
{
    if (!input || !weights || !output)       return -1;
    if (rows < 1 || cols < 1)               return -1;
    if (kH < 1   || kW < 1)                 return -1;
    if (kH % 2 == 0 || kW % 2 == 0)         return -1;
    if (n_ch < 1 || n_ch > 64)              return -1;

    int padH = kH / 2, padW = kW / 2;

    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            float acc = 0.0f;

            for (int ch = 0; ch < n_ch; ch++) {
                for (int kr = 0; kr < kH; kr++) {
                    for (int kc = 0; kc < kW; kc++) {
                        int ir = r + kr - padH;
                        int ic = c + kc - padW;

                        float in_bit = -1.0f;
                        if (ir >= 0 && ir < rows && ic >= 0 && ic < cols)
                            in_bit = (float)(2 * ((int)((input[(size_t)ir * (size_t)cols + (size_t)ic] >> ch) & 1)) - 1);

                        float w = weights[(size_t)ch * (size_t)(kH * kW) +
                                          (size_t)kr * (size_t)kW + (size_t)kc];
                        acc += in_bit * w;
                    }
                }
            }

            output[(size_t)r * (size_t)cols + (size_t)c] = acc;
        }
    }
    return 0;
}


/* --------------------------------------------------------------------------
 * pack_channels_to_u64
 *
 * Pack [n_ch, H, W] binary (0/1) uint8 planes into [H, W] uint64,
 * where bit i of each output word = planes[i] at that pixel.
 *
 * Written as a simple loop so -O3 -march=native auto-vectorises with AVX2.
 *
 * planes : [n_ch × H × W] uint8, values 0 or 1, row-major (C order)
 * output : [H × W] uint64, caller-allocated
 * n_ch   : number of input channels (1–64)
 * npix   : H * W
 *
 * Returns 0 on success, -1 on invalid arguments.
 * -------------------------------------------------------------------------- */
KERNEL_EXPORT int pack_channels_to_u64(
    const uint8_t *planes,
    uint64_t      *output,
    int n_ch,
    int npix)
{
    if (!planes || !output)     return -1;
    if (n_ch < 1 || n_ch > 64) return -1;
    if (npix < 1)               return -1;

    memset(output, 0, (size_t)npix * sizeof(uint64_t));
    for (int ch = 0; ch < n_ch; ch++) {
        const uint8_t *src = planes + (size_t)ch * (size_t)npix;
        uint64_t shift = (uint64_t)ch;
        for (int i = 0; i < npix; i++)
            output[i] |= (uint64_t)src[i] << shift;
    }
    return 0;
}


/* --------------------------------------------------------------------------
 * threshold_i32_to_u8
 *
 * Apply a scalar threshold to [n, npix] int32 score maps, producing
 * [n, npix] uint8 binary maps (1 if score > threshold, else 0).
 *
 * Written as a simple loop so -O3 auto-vectorises with SSE/AVX comparisons.
 *
 * scores    : [n × npix] int32, row-major
 * output    : [n × npix] uint8, caller-allocated
 * n         : number of feature maps (e.g. 64)
 * npix      : H * W
 * threshold : fire if score > threshold
 *
 * Returns 0 on success, -1 on invalid arguments.
 * -------------------------------------------------------------------------- */
KERNEL_EXPORT int threshold_i32_to_u8(
    const int32_t *scores,
    uint8_t       *output,
    int n,
    int npix,
    int threshold)
{
    if (!scores || !output) return -1;
    if (n < 1 || npix < 1)  return -1;

    size_t total = (size_t)n * (size_t)npix;
    for (size_t i = 0; i < total; i++)
        output[i] = (scores[i] > threshold) ? 1u : 0u;
    return 0;
}


/* ============================================================
 * Unit tests — compiled only when -DKERNEL_TEST_MAIN is set
 * ============================================================ */
#ifdef KERNEL_TEST_MAIN

static int tests_run    = 0;
static int tests_passed = 0;

#define CHECK(cond, name) do {                                      \
    tests_run++;                                                    \
    if (cond) {                                                     \
        printf("  PASS: %s\n", name);                              \
        tests_passed++;                                             \
    } else {                                                        \
        printf("  FAIL: %s  (line %d)\n", name, __LINE__);        \
    }                                                               \
} while (0)

/* Helper: fill array with a constant int32 value */
static void fill_i32(int32_t *arr, size_t n, int32_t v) {
    for (size_t i = 0; i < n; i++) arr[i] = v;
}

/* Helper: check that every element equals v */
static int all_equal_i32(const int32_t *arr, size_t n, int32_t v) {
    for (size_t i = 0; i < n; i++)
        if (arr[i] != v) { printf("    mismatch at %zu: got %d expected %d\n", i, arr[i], v); return 0; }
    return 1;
}

/* ------------------------------------------------------------------
 * Test 1: NULL pointer arguments → must return -1, no crash
 * ------------------------------------------------------------------ */
static void test_null_pointers(void) {
    printf("\n[Test 1] NULL pointer arguments\n");
    uint8_t  inp[1] = {0};
    int8_t   wts[1] = {1};
    int32_t  out[1] = {0};
    float    fout[1] = {0};

    CHECK(xnor_popcount_conv(NULL, wts,  out, 1, 1, 1, 1) == -1, "xnor: NULL input");
    CHECK(xnor_popcount_conv(inp,  NULL, out, 1, 1, 1, 1) == -1, "xnor: NULL weights");
    CHECK(xnor_popcount_conv(inp,  wts, NULL, 1, 1, 1, 1) == -1, "xnor: NULL output");
    CHECK(sobel_conv(NULL, fout,  1, 1) == -1, "sobel: NULL input");
    CHECK(sobel_conv(inp,  NULL,  1, 1) == -1, "sobel: NULL output");
}

/* ------------------------------------------------------------------
 * Test 2: Bad dimensions → must return -1
 * ------------------------------------------------------------------ */
static void test_bad_dimensions(void) {
    printf("\n[Test 2] Bad dimensions\n");
    uint8_t inp[1] = {0};
    int8_t  wts[9];
    int32_t out[1] = {0};
    float   fout[1] = {0};
    memset(wts, 1, sizeof wts);

    CHECK(xnor_popcount_conv(inp, wts, out, 0, 1, 3, 3) == -1, "xnor: rows=0");
    CHECK(xnor_popcount_conv(inp, wts, out, 1, 0, 3, 3) == -1, "xnor: cols=0");
    CHECK(xnor_popcount_conv(inp, wts, out, 1, 1, 0, 3) == -1, "xnor: kH=0");
    CHECK(xnor_popcount_conv(inp, wts, out, 1, 1, 3, 0) == -1, "xnor: kW=0");
    CHECK(sobel_conv(inp, fout, 0, 1) == -1, "sobel: rows=0");
    CHECK(sobel_conv(inp, fout, 1, 0) == -1, "sobel: cols=0");
}

/* ------------------------------------------------------------------
 * Test 3: Even kernel size → must return -1
 * ------------------------------------------------------------------ */
static void test_even_kernel(void) {
    printf("\n[Test 3] Even kernel size rejection\n");
    uint8_t inp[4] = {0, 1, 0, 1};
    int8_t  wts[4];
    int32_t out[4];
    memset(wts, 1, sizeof wts);

    CHECK(xnor_popcount_conv(inp, wts, out, 2, 2, 2, 3) == -1, "xnor: even kH=2");
    CHECK(xnor_popcount_conv(inp, wts, out, 2, 2, 3, 2) == -1, "xnor: even kW=2");
}

/* ------------------------------------------------------------------
 * Test 4: All-zero input, all +1 weights → output = -kH*kW everywhere
 *
 * Every input bit is 0, every weight is +1.
 * XNOR: (0 > 0) == (1 > 0) → F == T → disagree for every position.
 * popcount = 0, result = 2*0 - 9 = -9.
 * Holds for ALL pixels including borders (zero-padding also gives 0 bits).
 * ------------------------------------------------------------------ */
static void test_all_zero_positive_weights(void) {
    printf("\n[Test 4] All-zero input, all +1 weights → output = -9 everywhere\n");
    const int R = 4, C = 4;
    uint8_t  inp[16];
    int8_t   wts[9];
    int32_t  out[16];
    memset(inp, 0, sizeof inp);
    memset(wts, 1, sizeof wts);   /* +1 everywhere */
    fill_i32(out, 16, 99);        /* sentinel */

    int rc = xnor_popcount_conv(inp, wts, out, R, C, 3, 3);
    CHECK(rc == 0, "returns 0");
    CHECK(all_equal_i32(out, 16, -9), "all outputs == -9");
}

/* ------------------------------------------------------------------
 * Test 5: All-zero input, all -1 weights → output = +9 everywhere
 *
 * Every input bit is 0, every weight is -1.
 * XNOR: (0 > 0) == (-1 > 0) → F == F → agree for every position.
 * popcount = 9, result = 2*9 - 9 = 9.
 * ------------------------------------------------------------------ */
static void test_all_zero_negative_weights(void) {
    printf("\n[Test 5] All-zero input, all -1 weights → output = +9 everywhere\n");
    const int R = 4, C = 4;
    uint8_t inp[16];
    int8_t  wts[9];
    int32_t out[16];
    memset(inp, 0, sizeof inp);
    memset(wts, (unsigned char)-1, sizeof wts);  /* -1 everywhere */
    fill_i32(out, 16, 99);

    int rc = xnor_popcount_conv(inp, wts, out, R, C, 3, 3);
    CHECK(rc == 0, "returns 0");
    CHECK(all_equal_i32(out, 16, 9), "all outputs == +9");
}

/* ------------------------------------------------------------------
 * Test 6: Sobel on flat image (all same value) → interior gradient = 0
 *
 * Border pixels neighbour zero-padding, so they correctly have non-zero
 * gradients.  Only interior pixels (row/col >= 1 and < R/C - 1) must be ~0.
 * ------------------------------------------------------------------ */
static void test_sobel_flat(void) {
    printf("\n[Test 6] Sobel on flat image → interior gradient = 0\n");
    const int R = 8, C = 8;
    uint8_t inp[64];
    float   out[64];
    memset(inp, 128, sizeof inp);

    int rc = sobel_conv(inp, out, R, C);
    CHECK(rc == 0, "returns 0");

    /* Check only interior pixels — they see no zero-padding */
    int interior_ok = 1;
    for (int r = 1; r < R - 1; r++) {
        for (int c = 1; c < C - 1; c++) {
            if (fabsf(out[r * C + c]) > 1e-3f) {
                printf("    interior mismatch at (%d,%d): got %f\n", r, c, out[r * C + c]);
                interior_ok = 0;
            }
        }
    }
    CHECK(interior_ok, "interior gradients ≈ 0");
}

/* ------------------------------------------------------------------
 * Test 7: Sobel on vertical step edge → large Gx at boundary, near-zero Gy
 *
 * Left half = 0, right half = 255.
 * The vertical edge column (col = C/2 - 1 to C/2) should have large Gx.
 * ------------------------------------------------------------------ */
static void test_sobel_vertical_edge(void) {
    printf("\n[Test 7] Sobel on vertical step edge → large gradient at boundary\n");
    const int R = 8, C = 8;
    uint8_t inp[64];
    float   out[64];
    for (int r = 0; r < R; r++)
        for (int c = 0; c < C; c++)
            inp[r * C + c] = (c < C / 2) ? 0 : 255;

    int rc = sobel_conv(inp, out, R, C);
    CHECK(rc == 0, "returns 0");

    /* Check that at least one pixel on the edge column has gradient > 200 */
    int found_large = 0;
    for (int r = 1; r < R - 1; r++) {   /* skip border rows to avoid padding effects */
        float g = out[r * C + (C / 2)];
        if (g > 200.0f) { found_large = 1; break; }
    }
    CHECK(found_large, "large gradient found at step edge");
}

/* ------------------------------------------------------------------
 * Test 8: Border safety — 1×1 input, must not crash
 * ------------------------------------------------------------------ */
static void test_border_1x1(void) {
    printf("\n[Test 8] 1×1 input — border safety\n");
    uint8_t inp[1] = {1};
    int8_t  wts[9];
    int32_t xout[1];
    float   sout[1];
    memset(wts, 1, sizeof wts);  /* all +1 */

    /*
     * 1×1 input, all padding: only the center (input=1, weight=+1) agrees.
     * All 8 surrounding positions are padding (bit=0) vs weight=+1 → disagree.
     * popcount = 1, result = 2*1 - 9 = -7
     */
    int rc = xnor_popcount_conv(inp, wts, xout, 1, 1, 3, 3);
    CHECK(rc == 0, "xnor: returns 0 for 1×1");
    CHECK(xout[0] == -7, "xnor: 1×1 correct value (-7)");

    rc = sobel_conv(inp, sout, 1, 1);
    CHECK(rc == 0, "sobel: returns 0 for 1×1");
    /* All neighbours are padding=0, centre=1; Gx = Gy = 0 for a 1×1 image */
    CHECK(fabsf(sout[0]) < 1e-3f, "sobel: 1×1 gradient ≈ 0");
}

/* ------------------------------------------------------------------
 * Test 9: 1×1 kernel (kH=kW=1) — degenerate but valid
 * ------------------------------------------------------------------ */
static void test_1x1_kernel(void) {
    printf("\n[Test 9] 1×1 kernel — pointwise XNOR\n");
    const int R = 2, C = 2;
    uint8_t inp[4] = {1, 0, 1, 0};
    int8_t  wts[1] = {1};   /* single +1 weight */
    int32_t out[4];

    /*
     * 1×1 kernel, no padding needed.
     * For input=1, weight=+1: agree → popcount=1, result=2*1-1=+1
     * For input=0, weight=+1: disagree → popcount=0, result=2*0-1=-1
     */
    int rc = xnor_popcount_conv(inp, wts, out, R, C, 1, 1);
    CHECK(rc == 0, "returns 0");
    CHECK(out[0] == 1  && out[1] == -1 &&
          out[2] == 1  && out[3] == -1, "pointwise values correct");
}

/* ------------------------------------------------------------------
 * Test 10: NULL / bad args for packed functions
 * ------------------------------------------------------------------ */
static void test_packed_bad_args(void) {
    printf("\n[Test 10] NULL / bad args for packed functions\n");
    uint8_t  u8in[1]  = {0};
    uint8_t  u8wt[1]  = {0};
    uint64_t u64in[1] = {0};
    uint64_t u64wt[1] = {0};
    float    fwt[1]   = {1.0f};
    float    fout[1]  = {0};
    int32_t  out[1]   = {0};

    /* NULL checks — u8 */
    CHECK(xnor_packed_u8_conv(NULL,  u8wt, out,  1,1,1,1,1) == -1, "u8 xnor: NULL input");
    CHECK(xnor_packed_u8_conv(u8in,  NULL, out,  1,1,1,1,1) == -1, "u8 xnor: NULL weights");
    CHECK(xnor_packed_u8_conv(u8in,  u8wt, NULL, 1,1,1,1,1) == -1, "u8 xnor: NULL output");
    CHECK(float32_conv_nch_u8(NULL,  fwt,  fout, 1,1,1,1,1) == -1, "u8 f32:  NULL input");
    CHECK(float32_conv_nch_u8(u8in,  NULL, fout, 1,1,1,1,1) == -1, "u8 f32:  NULL weights");
    CHECK(float32_conv_nch_u8(u8in,  fwt,  NULL, 1,1,1,1,1) == -1, "u8 f32:  NULL output");

    /* NULL checks — u64 */
    CHECK(xnor_packed_u64_conv(NULL,  u64wt, out,  1,1,1,1,1) == -1, "u64 xnor: NULL input");
    CHECK(xnor_packed_u64_conv(u64in, NULL,  out,  1,1,1,1,1) == -1, "u64 xnor: NULL weights");
    CHECK(xnor_packed_u64_conv(u64in, u64wt, NULL, 1,1,1,1,1) == -1, "u64 xnor: NULL output");
    CHECK(float32_conv_nch_u64(NULL,  fwt,   fout, 1,1,1,1,1) == -1, "u64 f32:  NULL input");
    CHECK(float32_conv_nch_u64(u64in, NULL,  fout, 1,1,1,1,1) == -1, "u64 f32:  NULL weights");
    CHECK(float32_conv_nch_u64(u64in, fwt,   NULL, 1,1,1,1,1) == -1, "u64 f32:  NULL output");

    /* Bad n_ch */
    CHECK(xnor_packed_u8_conv(u8in, u8wt, out,   1,1,1,1, 0) == -1, "u8 xnor: n_ch=0");
    CHECK(xnor_packed_u8_conv(u8in, u8wt, out,   1,1,1,1, 9) == -1, "u8 xnor: n_ch=9");
    CHECK(xnor_packed_u64_conv(u64in, u64wt, out, 1,1,1,1, 0) == -1, "u64 xnor: n_ch=0");
    CHECK(xnor_packed_u64_conv(u64in, u64wt, out, 1,1,1,1,65) == -1, "u64 xnor: n_ch=65");

    /* Even kernel */
    CHECK(xnor_packed_u8_conv(u8in, u8wt, out, 1,1,2,3,1) == -1, "u8 xnor: even kH");
    CHECK(xnor_packed_u64_conv(u64in, u64wt, out, 1,1,3,2,1) == -1, "u64 xnor: even kW");
}

/* ------------------------------------------------------------------
 * Test 11: 8ch, all-zero input, all-zero weight byte (all channels -1/-1)
 *
 * Both input and weight bits are 0 (interpreted as -1).
 * XNOR(0x00, 0x00) = ~(0x00 ^ 0x00) & 0xFF = 0xFF → 8 agreements/position.
 * Over 9 positions: 72 agreements.
 * result = 2*72 - 8*9 = 144 - 72 = +72.
 * ------------------------------------------------------------------ */
static void test_packed_u8_all_zero_input_weights(void) {
    printf("\n[Test 11] 8ch, input=0x00, weights=0x00 → output = +72 everywhere\n");
    const int R = 4, C = 4;
    uint8_t inp[16], wts[9];
    int32_t out[16];
    memset(inp, 0x00, sizeof inp);
    memset(wts, 0x00, sizeof wts);
    fill_i32(out, 16, 99);

    int rc = xnor_packed_u8_conv(inp, wts, out, R, C, 3, 3, 8);
    CHECK(rc == 0, "returns 0");
    CHECK(all_equal_i32(out, 16, 72), "all outputs == +72");
}

/* ------------------------------------------------------------------
 * Test 12: 8ch, all-0xFF input, all-0xFF weight byte (all channels +1/+1)
 *
 * Interior pixels (no padding neighbours): all 9 positions × 8 agreements = 72.
 * result = 2*72 - 72 = +72.
 * Border pixels see zero-padding, which disagrees with 0xFF weights — they
 * will have lower values; we only assert the interior.
 * ------------------------------------------------------------------ */
static void test_packed_u8_all_one_input_weights(void) {
    printf("\n[Test 12] 8ch, input=0xFF, weights=0xFF → interior output = +72\n");
    const int R = 6, C = 6;    /* 6×6 so (1..4, 1..4) interior pixels exist */
    uint8_t inp[36], wts[9];
    int32_t out[36];
    memset(inp, 0xFF, sizeof inp);
    memset(wts, 0xFF, sizeof wts);
    fill_i32(out, 36, 99);

    int rc = xnor_packed_u8_conv(inp, wts, out, R, C, 3, 3, 8);
    CHECK(rc == 0, "returns 0");

    /* Check only interior pixels (r ∈ [1,R-2], c ∈ [1,C-2]) */
    int interior_ok = 1;
    for (int r = 1; r < R - 1; r++) {
        for (int c = 1; c < C - 1; c++) {
            if (out[r * C + c] != 72) {
                printf("    interior mismatch at (%d,%d): got %d\n", r, c, out[r * C + c]);
                interior_ok = 0;
            }
        }
    }
    CHECK(interior_ok, "interior outputs == +72");
}

/* ------------------------------------------------------------------
 * Test 13: 8ch parity — XNOR u8 result == float32 result (known input)
 *
 * Uses a 4×4 input with a checkerboard pattern and a 1×1 kernel (no
 * neighbour access, so border effects are irrelevant).
 * ------------------------------------------------------------------ */
static void test_packed_u8_parity_f32(void) {
    printf("\n[Test 13] 8ch XNOR u8 parity with float32 reference\n");
    const int R = 4, C = 4, NCH = 4;

    /* Checkerboard of 0xAA / 0x55 — alternating bits per pixel */
    uint8_t inp[16];
    for (int i = 0; i < 16; i++) inp[i] = (i % 2 == 0) ? 0xAAu : 0x55u;

    /* 1×1 kernel, weight = 0x0F (channels 0-3 positive, 4-7 don't matter) */
    uint8_t  xnor_wt[1] = {0x0Fu};
    int32_t  xnor_out[16];

    /* Float32 weights: unpack the same 4 bits to ±1.0 */
    /* weight byte 0x0F: bits 0,1,2,3 = 1 (+1.0), bits 4,5,6,7 = 0 (-1.0) */
    /* We only use NCH=4 channels, so weights[ch] = (0x0F >> ch) & 1 ? +1 : -1 */
    float f32_wt[4 * 1];  /* [n_ch × kH*kW] = [4 × 1] */
    for (int ch = 0; ch < NCH; ch++)
        f32_wt[ch] = ((0x0Fu >> ch) & 1u) ? 1.0f : -1.0f;
    float f32_out[16];

    int rc1 = xnor_packed_u8_conv(inp, xnor_wt, xnor_out, R, C, 1, 1, NCH);
    int rc2 = float32_conv_nch_u8(inp, f32_wt,  f32_out,  R, C, 1, 1, NCH);
    CHECK(rc1 == 0 && rc2 == 0, "both return 0");

    int parity_ok = 1;
    for (int i = 0; i < R * C; i++) {
        int32_t f32_as_int = (int32_t)roundf(f32_out[i]);
        if (xnor_out[i] != f32_as_int) {
            printf("    mismatch at %d: xnor=%d f32=%d\n", i, xnor_out[i], f32_as_int);
            parity_ok = 0;
        }
    }
    CHECK(parity_ok, "XNOR u8 output == float32 output for all pixels");
}

/* ------------------------------------------------------------------
 * Test 14: 64ch, all-zero uint64 input, all-zero uint64 weight
 *
 * XNOR(0, 0) = ~0 = UINT64_MAX → 64 agreements/position.
 * Over 9 positions: 576 agreements.
 * result = 2*576 - 64*9 = 1152 - 576 = +576.
 * ------------------------------------------------------------------ */
static void test_packed_u64_all_zero(void) {
    printf("\n[Test 14] 64ch, input=0, weights=0 → output = +576 everywhere\n");
    const int R = 4, C = 4;
    uint64_t inp[16], wts[9];
    int32_t  out[16];
    memset(inp, 0, sizeof inp);
    memset(wts, 0, sizeof wts);
    fill_i32(out, 16, 99);

    int rc = xnor_packed_u64_conv(inp, wts, out, R, C, 3, 3, 64);
    CHECK(rc == 0, "returns 0");
    CHECK(all_equal_i32(out, 16, 576), "all outputs == +576");
}

/* ------------------------------------------------------------------
 * Test 15: 64ch parity — XNOR u64 result == float32 result
 *
 * Uses 1×1 kernel on a 2×2 image to avoid border effects.
 * ------------------------------------------------------------------ */
static void test_packed_u64_parity_f32(void) {
    printf("\n[Test 15] 64ch XNOR u64 parity with float32 reference\n");
    const int R = 2, C = 2, NCH = 64;

    /* Alternating uint64 patterns */
    uint64_t inp[4] = {0xAAAAAAAAAAAAAAAAULL, 0x5555555555555555ULL,
                       0x5555555555555555ULL, 0xAAAAAAAAAAAAAAAAULL};
    uint64_t xnor_wt[1] = {0xF0F0F0F0F0F0F0F0ULL};
    int32_t  xnor_out[4];

    float f32_wt[NCH * 1];
    for (int ch = 0; ch < NCH; ch++)
        f32_wt[ch] = ((xnor_wt[0] >> ch) & 1ULL) ? 1.0f : -1.0f;
    float f32_out[4];

    int rc1 = xnor_packed_u64_conv(inp, xnor_wt, xnor_out, R, C, 1, 1, NCH);
    int rc2 = float32_conv_nch_u64(inp, f32_wt,  f32_out,  R, C, 1, 1, NCH);
    CHECK(rc1 == 0 && rc2 == 0, "both return 0");

    int parity_ok = 1;
    for (int i = 0; i < R * C; i++) {
        int32_t f32_as_int = (int32_t)roundf(f32_out[i]);
        if (xnor_out[i] != f32_as_int) {
            printf("    mismatch at %d: xnor=%d f32=%d\n", i, xnor_out[i], f32_as_int);
            parity_ok = 0;
        }
    }
    CHECK(parity_ok, "XNOR u64 output == float32 output for all pixels");
}

/* ------------------------------------------------------------------
 * main
 * ------------------------------------------------------------------ */
int main(void) {
    printf("=== kernel.c unit tests ===\n");

    test_null_pointers();
    test_bad_dimensions();
    test_even_kernel();
    test_all_zero_positive_weights();
    test_all_zero_negative_weights();
    test_sobel_flat();
    test_sobel_vertical_edge();
    test_border_1x1();
    test_1x1_kernel();
    test_packed_bad_args();
    test_packed_u8_all_zero_input_weights();
    test_packed_u8_all_one_input_weights();
    test_packed_u8_parity_f32();
    test_packed_u64_all_zero();
    test_packed_u64_parity_f32();

    printf("\n=== Results: %d / %d passed ===\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}

#endif /* KERNEL_TEST_MAIN */
