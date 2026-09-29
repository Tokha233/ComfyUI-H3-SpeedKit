/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "utils.cuh"
#include "dtype_dispatch.cuh"
#include "input_act_codes.h"

#include <cmath>
#include <cfloat>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace comfy {

namespace {

constexpr int kInt8Threads = 256;

template<typename T>
__device__ __forceinline__ float to_float(T val);
template<> __device__ __forceinline__ float to_float<float>(float val) { return val; }
template<> __device__ __forceinline__ float to_float<half>(half val) { return __half2float(val); }
template<> __device__ __forceinline__ float to_float<nv_bfloat16>(nv_bfloat16 val) { return __bfloat162float(val); }

template<typename T>
__device__ __forceinline__ float finite_max_for_dtype();
template<> __device__ __forceinline__ float finite_max_for_dtype<float>() { return FLT_MAX; }
template<> __device__ __forceinline__ float finite_max_for_dtype<half>() { return 65504.0f; }
template<> __device__ __forceinline__ float finite_max_for_dtype<nv_bfloat16>() { return 3.38953139e38f; }

template<typename T>
__device__ __forceinline__ float finite_absmax_for_int8_scale(float abs_max) {
    return fminf(abs_max, finite_max_for_dtype<T>());
}

template<typename T>
__device__ __forceinline__ T from_float(float val);
template<> __device__ __forceinline__ float from_float<float>(float val) { return val; }
template<> __device__ __forceinline__ half from_float<half>(float val) { return __float2half_rn(val); }
template<> __device__ __forceinline__ nv_bfloat16 from_float<nv_bfloat16>(float val) { return __float2bfloat16_rn(val); }

template<typename T>
__device__ __forceinline__ float quant_div_to_float(T val, float scale) {
    const float scale_t = to_float(from_float<T>(scale));
    return to_float(from_float<T>(to_float(val) / scale_t));
}

template<typename T>
__device__ __forceinline__ float quant_div_float_to_float(float val, float scale) {
    const float scale_t = to_float(from_float<T>(scale));
    return to_float(from_float<T>(to_float(from_float<T>(val)) / scale_t));
}

template<typename T>
__device__ __forceinline__ float stochastic_sum_to_float(float scaled, T rng) {
    return to_float(from_float<T>(scaled + to_float(rng)));
}

template<>
__device__ __forceinline__ float stochastic_sum_to_float<float>(float scaled, float rng) {
    return scaled + rng;
}

template<>
__device__ __forceinline__ float stochastic_sum_to_float<half>(float scaled, half rng) {
    return __half2float(__hadd(__float2half_rn(scaled), rng));
}

template<>
__device__ __forceinline__ float stochastic_sum_to_float<nv_bfloat16>(float scaled, nv_bfloat16 rng) {
    return __bfloat162float(__hadd(__float2bfloat16_rn(scaled), rng));
}

__device__ __forceinline__ uint32_t pcg_hash(uint32_t x) {
    const uint32_t state = x * 747796405u + 2891336453u;
    const uint32_t word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

template<typename T>
__device__ __forceinline__ T stochastic_rng_value(int64_t idx, uint64_t seed) {
    const uint64_t key = static_cast<uint64_t>(idx) + seed;
    const uint32_t folded = static_cast<uint32_t>(key) ^ static_cast<uint32_t>(key >> 32);
    const float value = static_cast<float>(pcg_hash(folded) >> 8) * 0x1.0p-24f;
    return from_float<T>(value);
}

template<typename T>
__device__ __forceinline__ void store4_contiguous(T* out, int64_t idx, float x, float y, float z, float w) {
    out[idx] = from_float<T>(x);
    out[idx + 1] = from_float<T>(y);
    out[idx + 2] = from_float<T>(z);
    out[idx + 3] = from_float<T>(w);
}

template<>
__device__ __forceinline__ void store4_contiguous<float>(float* out, int64_t idx, float x, float y, float z, float w) {
    reinterpret_cast<float4*>(out)[idx / 4] = make_float4(x, y, z, w);
}

template<>
__device__ __forceinline__ void store4_contiguous<half>(half* out, int64_t idx, float x, float y, float z, float w) {
    reinterpret_cast<half2*>(out)[idx / 2] = __floats2half2_rn(x, y);
    reinterpret_cast<half2*>(out)[idx / 2 + 1] = __floats2half2_rn(z, w);
}

template<>
__device__ __forceinline__ void store4_contiguous<nv_bfloat16>(
    nv_bfloat16* out, int64_t idx, float x, float y, float z, float w)
{
    reinterpret_cast<nv_bfloat162*>(out)[idx / 2] = __floats2bfloat162_rn(x, y);
    reinterpret_cast<nv_bfloat162*>(out)[idx / 2 + 1] = __floats2bfloat162_rn(z, w);
}

__device__ __forceinline__ float warp_reduce_max(float v) {
    for (int offset = kThreadsPerWarp / 2; offset > 0; offset >>= 1) {
        v = fmaxf(v, __shfl_down_sync(0xffffffff, v, offset));
    }
    return v;
}

__device__ __forceinline__ int warp_reduce_sum_i32(int v) {
    for (int offset = kThreadsPerWarp / 2; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffff, v, offset);
    }
    return v;
}

template<int NUM_WARPS>
__device__ __forceinline__ float block_reduce_max_t(float v, float* warp_smem, float* block_smem);

template<int NUM_WARPS>
__device__ __forceinline__ int block_reduce_sum_i32_t(int v, int* warp_smem, int* block_smem) {
    const int lane = threadIdx.x & (kThreadsPerWarp - 1);
    const int wid = threadIdx.x >> 5;
    v = warp_reduce_sum_i32(v);
    if (lane == 0) {
        warp_smem[wid] = v;
    }
    __syncthreads();
    if (wid == 0) {
        int total = lane < NUM_WARPS ? warp_smem[lane] : 0;
        total = warp_reduce_sum_i32(total);
        if (lane == 0) {
            *block_smem = total;
        }
    }
    __syncthreads();
    return *block_smem;
}

template<typename InputType, int BLOCK_THREADS, bool STOCHASTIC>
__global__ void quantize_int8_rowwise_kernel(
    const InputType* __restrict__ x,
    int8_t* __restrict__ q,
    float* __restrict__ scales,
    int K,
    uint64_t seed)
{
    constexpr int kWarps = BLOCK_THREADS / kThreadsPerWarp;
    __shared__ float warp_smem[kWarps];
    __shared__ float block_smem;

    const int row = static_cast<int>(blockIdx.x);
    const int tid = threadIdx.x;
    const int64_t row_offset = static_cast<int64_t>(row) * K;

    float abs_max = 0.0f;
    for (int col = tid; col < K; col += blockDim.x) {
        abs_max = fmaxf(abs_max, fabsf(to_float(x[row_offset + col])));
    }

    abs_max = block_reduce_max_t<kWarps>(abs_max, warp_smem, &block_smem);
    const float scale = fmaxf(
        finite_absmax_for_int8_scale<InputType>(abs_max) * (1.0f / 127.0f),
        1.0e-30f);

    if (tid == 0) {
        scales[row] = scale;
    }

    for (int col = tid; col < K; col += blockDim.x) {
        const int64_t idx = row_offset + col;
        const float scaled = quant_div_to_float<InputType>(x[idx], scale);
        float quantized;
        if constexpr (STOCHASTIC) {
            const InputType noise = stochastic_rng_value<InputType>(idx, seed);
            quantized = floorf(stochastic_sum_to_float<InputType>(scaled, noise));
        } else {
            quantized = nearbyintf(scaled);
        }
        quantized = fminf(127.0f, fmaxf(-128.0f, quantized));
        q[idx] = static_cast<int8_t>(quantized);
    }
}

// Fused ConvRot (online Hadamard rotation) + row-wise INT8 quantization.
//
// The rotation group is fixed to 256 = 4^4 and uses the symmetric "regular"
// Hadamard block H4:
//     [  1  1  1 -1 ]
//     [  1  1 -1  1 ]
//     [  1 -1  1  1 ]
//     [ -1  1  1  1 ]
// The full normalized matrix is H256 = (1/16) * (H4 (x) H4 (x) H4 (x) H4).
// Because it is a Kronecker power, the matvec H256 @ x factors into 4 radix-4
// "butterfly" stages (strides 1, 4, 16, 64) with a 1/2 scale per stage, i.e. a
// Fast Hadamard Transform: O(256*4) work per group instead of O(256*256).
//
// The whole row is rotated in shared memory and quantized in place, so the
// rotated bf16 activation is never written to / read back from global memory
// (the unfused path's main cost).
//
// One block per row. The block holds BLOCK_THREADS / 256 groups "in flight" at
// once: for large K the row's shared-memory footprint forces a single block per
// SM, so we want a wide block (many warps) to hide latency rather than a narrow
// 256-thread block. Each thread owns local element `i = tid % 256` of group
// slot `sub = tid / 256`.
constexpr int kConvRotGroup = 256;

__device__ __forceinline__ float h4_row_dot(int d, float x0, float x1, float x2, float x3) {
    // Row d of H4 dotted with (x0, x1, x2, x3).
    switch (d) {
        case 0:  return  x0 + x1 + x2 - x3;
        case 1:  return  x0 + x1 - x2 + x3;
        case 2:  return  x0 - x1 + x2 + x3;
        default: return -x0 + x1 + x2 + x3;
    }
}

template<int NUM_WARPS>
__device__ __forceinline__ float block_reduce_max_t(float v, float* warp_smem, float* block_smem) {
    const int lane = threadIdx.x & (kThreadsPerWarp - 1);
    const int wid = threadIdx.x >> 5;
    v = warp_reduce_max(v);
    if (lane == 0) {
        warp_smem[wid] = v;
    }
    __syncthreads();
    if (wid == 0) {
        float total = lane < NUM_WARPS ? warp_smem[lane] : 0.0f;
        total = warp_reduce_max(total);
        if (lane == 0) {
            *block_smem = total;
        }
    }
    __syncthreads();
    return *block_smem;
}

template<typename InputType, int BLOCK_THREADS, bool STOCHASTIC>
__global__ void quantize_int8_rowwise_convrot_kernel(
    const InputType* __restrict__ x,
    int8_t* __restrict__ q,
    float* __restrict__ scales,
    int K,
    uint64_t seed)
{
    constexpr int kGroupsInFlight = BLOCK_THREADS / kConvRotGroup;
    constexpr int kWarps = BLOCK_THREADS / kThreadsPerWarp;

    extern __shared__ float smem[];
    float* row_buf = smem;                         // K floats: rotated row, in place
    float* tmp = smem + K;                          // kGroupsInFlight * 2 * 256 floats

    __shared__ float warp_smem[kWarps];
    __shared__ float block_smem;

    const int row = static_cast<int>(blockIdx.x);
    const int tid = threadIdx.x;
    const int64_t row_offset = static_cast<int64_t>(row) * K;

    // Load the row into shared memory as float.
    for (int col = tid; col < K; col += BLOCK_THREADS) {
        row_buf[col] = to_float(x[row_offset + col]);
    }
    __syncthreads();

    // Fast Hadamard transform, kGroupsInFlight groups at a time.
    const int n_groups = K / kConvRotGroup;
    const int sub = tid / kConvRotGroup;
    const int i = tid % kConvRotGroup;
    // Each slot gets a private double buffer so inactive lanes (when n_groups is
    // not a multiple of kGroupsInFlight) can keep hitting __syncthreads without
    // touching the live row data.
    float* buf0 = tmp + sub * (2 * kConvRotGroup);
    float* buf1 = buf0 + kConvRotGroup;
    const int iters = (n_groups + kGroupsInFlight - 1) / kGroupsInFlight;

    for (int it = 0; it < iters; ++it) {
        const int g = it * kGroupsInFlight + sub;
        const bool active = (g < n_groups);
        // active: ping-pong between the row region and buf0, ending in the row
        // region after 4 swaps. inactive: ping-pong privately in buf0/buf1.
        float* src = active ? (row_buf + g * kConvRotGroup) : buf0;
        float* dst = active ? buf0 : buf1;
        #pragma unroll
        for (int stage = 0; stage < 4; ++stage) {
            const int s = (stage == 0) ? 1 : (stage == 1) ? 4 : (stage == 2) ? 16 : 64;
            const int d = (i / s) & 3;
            const int base = i - d * s;
            const float v = 0.5f * h4_row_dot(
                d, src[base], src[base + s], src[base + 2 * s], src[base + 3 * s]);
            dst[i] = v;
            __syncthreads();
            float* t = src; src = dst; dst = t;
        }
    }

    // Row absmax over the rotated values -> per-row scale.
    float abs_max = 0.0f;
    for (int col = tid; col < K; col += BLOCK_THREADS) {
        abs_max = fmaxf(abs_max, fabsf(row_buf[col]));
    }
    abs_max = block_reduce_max_t<kWarps>(abs_max, warp_smem, &block_smem);
    const float scale = fmaxf(
        finite_absmax_for_int8_scale<InputType>(abs_max) * (1.0f / 127.0f),
        1.0e-30f);
    if (tid == 0) {
        scales[row] = scale;
    }

    for (int col = tid; col < K; col += BLOCK_THREADS) {
        const int64_t idx = row_offset + col;
        const float scaled = quant_div_float_to_float<InputType>(row_buf[col], scale);
        float quantized;
        if constexpr (STOCHASTIC) {
            const InputType noise = stochastic_rng_value<InputType>(idx, seed);
            quantized = floorf(stochastic_sum_to_float<InputType>(scaled, noise));
        } else {
            quantized = nearbyintf(scaled);
        }
        quantized = fminf(127.0f, fmaxf(-128.0f, quantized));
        q[idx] = static_cast<int8_t>(quantized);
    }
}

template<typename OutputType, typename BiasType>
__global__ void dequantize_int8_linear_kernel(
    const int32_t* __restrict__ input,
    const float* __restrict__ x_scales,
    const float* __restrict__ weight_scales,
    const BiasType* __restrict__ bias,
    OutputType* __restrict__ output,
    int64_t total,
    int N,
    int weight_scale_size,
    bool has_bias)
{
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }

    const int col = static_cast<int>(idx % N);
    const int row = static_cast<int>(idx / N);
    const float weight_scale = weight_scales[weight_scale_size == 1 ? 0 : col];
    float value = static_cast<float>(input[idx]) * x_scales[row] * weight_scale;
    if (has_bias) {
        value += to_float(bias[col]);
    }
    output[idx] = from_float<OutputType>(value);
}

template<typename OutputType>
__global__ void dequantize_int8_linear_vec4_kernel(
    const int32_t* __restrict__ input,
    const float* __restrict__ x_scales,
    const float* __restrict__ weight_scales,
    OutputType* __restrict__ output,
    int64_t total_vec4,
    int N,
    int weight_scale_size)
{
    const int64_t idx4 = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx4 >= total_vec4) {
        return;
    }

    const int64_t idx = idx4 * 4;
    const int col = static_cast<int>(idx % N);
    const int row = static_cast<int>(idx / N);
    const float x_scale = x_scales[row];
    const int4 acc = reinterpret_cast<const int4*>(input)[idx4];

    if (weight_scale_size == 1) {
        const float scale = x_scale * weight_scales[0];
        store4_contiguous(
            output, idx,
            static_cast<float>(acc.x) * scale,
            static_cast<float>(acc.y) * scale,
            static_cast<float>(acc.z) * scale,
            static_cast<float>(acc.w) * scale);
        return;
    }

    const float4 ws = reinterpret_cast<const float4*>(weight_scales)[col / 4];
    store4_contiguous(
        output, idx,
        static_cast<float>(acc.x) * x_scale * ws.x,
        static_cast<float>(acc.y) * x_scale * ws.y,
        static_cast<float>(acc.z) * x_scale * ws.z,
        static_cast<float>(acc.w) * x_scale * ws.w);
}

template<int BLOCK_THREADS, typename OutputType, typename BiasType>
__global__ void int8_gemv_dequant_kernel(
    const int8_t* __restrict__ x,
    const int8_t* __restrict__ weight,
    const float* __restrict__ x_scales,
    const float* __restrict__ weight_scales,
    const BiasType* __restrict__ bias,
    OutputType* __restrict__ output,
    int N,
    int K,
    int weight_scale_size,
    bool has_bias)
{
    constexpr int kWarps = BLOCK_THREADS / kThreadsPerWarp;
    __shared__ int warp_smem[kWarps];
    __shared__ int block_smem;

    const int n = static_cast<int>(blockIdx.x);
    const int tid = threadIdx.x;
    const int8_t* __restrict__ w_row = weight + static_cast<int64_t>(n) * K;

    int acc = 0;
    const int K4 = K >> 2;
    const int* __restrict__ x4 = reinterpret_cast<const int*>(x);
    const int* __restrict__ w4 = reinterpret_cast<const int*>(w_row);
    for (int k4 = tid; k4 < K4; k4 += BLOCK_THREADS) {
        acc = __dp4a(x4[k4], w4[k4], acc);
    }
    for (int k = (K4 << 2) + tid; k < K; k += BLOCK_THREADS) {
        acc += static_cast<int>(x[k]) * static_cast<int>(w_row[k]);
    }

    acc = block_reduce_sum_i32_t<kWarps>(acc, warp_smem, &block_smem);
    if (tid == 0) {
        const float weight_scale = weight_scales[weight_scale_size == 1 ? 0 : n];
        float value = static_cast<float>(acc) * x_scales[0] * weight_scale;
        if (has_bias) {
            value += to_float(bias[n]);
        }
        output[n] = from_float<OutputType>(value);
    }
}

template<int WARPS_PER_BLOCK, typename OutputType, typename BiasType>
__global__ void int8_gemv_dequant_warp_kernel(
    const int8_t* __restrict__ x,
    const int8_t* __restrict__ weight,
    const float* __restrict__ x_scales,
    const float* __restrict__ weight_scales,
    const BiasType* __restrict__ bias,
    OutputType* __restrict__ output,
    int N,
    int K,
    int weight_scale_size,
    bool has_bias)
{
    const int lane = threadIdx.x & (kThreadsPerWarp - 1);
    const int warp = threadIdx.x >> 5;
    const int n = static_cast<int>(blockIdx.x) * WARPS_PER_BLOCK + warp;
    if (n >= N) {
        return;
    }

    const int K4 = K >> 2;
    const int* __restrict__ x4 = reinterpret_cast<const int*>(x);
    const int* __restrict__ w4 = reinterpret_cast<const int*>(weight + static_cast<int64_t>(n) * K);

    int acc = 0;
    for (int k4 = lane; k4 < K4; k4 += kThreadsPerWarp) {
        acc = __dp4a(x4[k4], w4[k4], acc);
    }
    acc = warp_reduce_sum_i32(acc);

    if (lane == 0) {
        const float weight_scale = weight_scales[weight_scale_size == 1 ? 0 : n];
        float value = static_cast<float>(acc) * x_scales[0] * weight_scale;
        if (has_bias) {
            value += to_float(bias[n]);
        }
        output[n] = from_float<OutputType>(value);
    }
}

template<typename OutputType>
__global__ void dequantize_int8_simple_kernel(
    const int8_t* __restrict__ input,
    const float* __restrict__ scales,
    OutputType* __restrict__ output,
    int64_t total,
    int inner_dim,
    int scale_mode)
{
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }

    int64_t scale_idx = 0;
    if (scale_mode == 1) {
        scale_idx = idx;
    } else if (scale_mode == 2) {
        scale_idx = idx / inner_dim;
    }
    output[idx] = from_float<OutputType>(static_cast<float>(input[idx]) * scales[scale_idx]);
}

template<typename OutputType>
__global__ void dequantize_int8_simple_vec4_kernel(
    const int8_t* __restrict__ input,
    const float* __restrict__ scales,
    OutputType* __restrict__ output,
    int64_t total_vec4,
    int inner_dim_vec4,
    int scale_mode)
{
    const int64_t idx4 = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx4 >= total_vec4) {
        return;
    }

    const char4 q4 = reinterpret_cast<const char4*>(input)[idx4];
    const float scale = scales[scale_mode == 0 ? 0 : idx4 / inner_dim_vec4];
    const int64_t idx = idx4 * 4;
    store4_contiguous(output, idx,
        static_cast<float>(q4.x) * scale,
        static_cast<float>(q4.y) * scale,
        static_cast<float>(q4.z) * scale,
        static_cast<float>(q4.w) * scale);
}

template<typename OutputType>
__global__ void dequantize_int8_rowwise_vec4_2d_kernel(
    const int8_t* __restrict__ input,
    const float* __restrict__ scales,
    OutputType* __restrict__ output,
    int rows,
    int inner_dim_vec4)
{
    const int row = static_cast<int>(blockIdx.x);
    const int col4 = static_cast<int>(blockIdx.y) * blockDim.x + threadIdx.x;
    if (row >= rows || col4 >= inner_dim_vec4) {
        return;
    }

    const int64_t idx4 = static_cast<int64_t>(row) * inner_dim_vec4 + col4;
    const char4 q4 = reinterpret_cast<const char4*>(input)[idx4];
    const float scale = scales[row];
    const int64_t idx = idx4 * 4;
    store4_contiguous(output, idx,
        static_cast<float>(q4.x) * scale,
        static_cast<float>(q4.y) * scale,
        static_cast<float>(q4.z) * scale,
        static_cast<float>(q4.w) * scale);
}

template<int BLOCK_THREADS, typename OutputType>
__global__ void dequantize_int8_convrot_kernel(
    const int8_t* __restrict__ q,
    const float* __restrict__ scales,
    OutputType* __restrict__ output,
    int K,
    int scale_size)
{
    constexpr int kGroupsInFlight = BLOCK_THREADS / kConvRotGroup;

    extern __shared__ float smem[];
    float* row_buf = smem;
    float* tmp = smem + K;

    const int row = static_cast<int>(blockIdx.x);
    const int tid = threadIdx.x;
    const int64_t row_offset = static_cast<int64_t>(row) * K;
    const float scale = scales[scale_size == 1 ? 0 : row];

    for (int col = tid; col < K; col += BLOCK_THREADS) {
        row_buf[col] = static_cast<float>(q[row_offset + col]) * scale;
    }
    __syncthreads();

    const int n_groups = K / kConvRotGroup;
    const int sub = tid / kConvRotGroup;
    const int i = tid % kConvRotGroup;
    float* buf0 = tmp + sub * (2 * kConvRotGroup);
    float* buf1 = buf0 + kConvRotGroup;
    const int iters = (n_groups + kGroupsInFlight - 1) / kGroupsInFlight;

    for (int it = 0; it < iters; ++it) {
        const int g = it * kGroupsInFlight + sub;
        const bool active = (g < n_groups);
        float* src = active ? (row_buf + g * kConvRotGroup) : buf0;
        float* dst = active ? buf0 : buf1;
        #pragma unroll
        for (int stage = 0; stage < 4; ++stage) {
            const int s = (stage == 0) ? 1 : (stage == 1) ? 4 : (stage == 2) ? 16 : 64;
            const int d = (i / s) & 3;
            const int base = i - d * s;
            const float v = 0.5f * h4_row_dot(
                d, src[base], src[base + s], src[base + 2 * s], src[base + 3 * s]);
            dst[i] = v;
            __syncthreads();
            float* t = src; src = dst; dst = t;
        }
    }

    for (int col = tid; col < K; col += BLOCK_THREADS) {
        output[row_offset + col] = from_float<OutputType>(row_buf[col]);
    }
}

template<int BLOCK_THREADS, typename OutputType>
__global__ void dequantize_int8_convrot_groups_kernel(
    const int8_t* __restrict__ q,
    const float* __restrict__ scales,
    OutputType* __restrict__ output,
    int K,
    int scale_size)
{
    constexpr int kGroupsPerBlock = BLOCK_THREADS / kConvRotGroup;
    extern __shared__ float smem[];

    const int group = static_cast<int>(blockIdx.x) * kGroupsPerBlock + threadIdx.x / kConvRotGroup;
    const int row = static_cast<int>(blockIdx.y);
    const int i = threadIdx.x % kConvRotGroup;
    const int sub = threadIdx.x / kConvRotGroup;
    const bool active = group < K / kConvRotGroup;
    const int64_t row_offset = static_cast<int64_t>(row) * K;
    const int col = group * kConvRotGroup + i;
    const float scale = scales[scale_size == 1 ? 0 : row];
    float* buf0 = smem + sub * (2 * kConvRotGroup);
    float* buf1 = buf0 + kConvRotGroup;

    buf0[i] = active ? static_cast<float>(q[row_offset + col]) * scale : 0.0f;
    __syncthreads();

    float* src = buf0;
    float* dst = buf1;
    #pragma unroll
    for (int stage = 0; stage < 4; ++stage) {
        const int s = (stage == 0) ? 1 : (stage == 1) ? 4 : (stage == 2) ? 16 : 64;
        const int d = (i / s) & 3;
        const int base = i - d * s;
        const float v = 0.5f * h4_row_dot(
            d, src[base], src[base + s], src[base + 2 * s], src[base + 3 * s]);
        dst[i] = v;
        __syncthreads();
        float* t = src; src = dst; dst = t;
    }

    if (active) {
        output[row_offset + col] = from_float<OutputType>(src[i]);
    }
}

template<int S>
__device__ __forceinline__ void convrot_fht_stage64(
    const float* __restrict__ src,
    float* __restrict__ dst,
    int lane)
{
    const int base = (lane % S) + (lane / S) * (4 * S);
    const float x0 = src[base];
    const float x1 = src[base + S];
    const float x2 = src[base + 2 * S];
    const float x3 = src[base + 3 * S];
    dst[base] = 0.5f * (x0 + x1 + x2 - x3);
    dst[base + S] = 0.5f * (x0 + x1 - x2 + x3);
    dst[base + 2 * S] = 0.5f * (x0 - x1 + x2 + x3);
    dst[base + 3 * S] = 0.5f * (-x0 + x1 + x2 + x3);
}

template<int S, typename OutputType>
__device__ __forceinline__ void convrot_fht_stage64_store(
    const float* __restrict__ src,
    OutputType* __restrict__ output,
    int lane)
{
    const int base = (lane % S) + (lane / S) * (4 * S);
    const float x0 = src[base];
    const float x1 = src[base + S];
    const float x2 = src[base + 2 * S];
    const float x3 = src[base + 3 * S];
    output[base] = from_float<OutputType>(0.5f * (x0 + x1 + x2 - x3));
    output[base + S] = from_float<OutputType>(0.5f * (x0 + x1 - x2 + x3));
    output[base + 2 * S] = from_float<OutputType>(0.5f * (x0 - x1 + x2 + x3));
    output[base + 3 * S] = from_float<OutputType>(0.5f * (-x0 + x1 + x2 + x3));
}

template<int S, typename OutputType>
__device__ __forceinline__ float convrot_fht_stage64_store_absmax(
    const float* __restrict__ src,
    OutputType* __restrict__ output,
    int lane)
{
    const int base = (lane % S) + (lane / S) * (4 * S);
    const float x0 = src[base];
    const float x1 = src[base + S];
    const float x2 = src[base + 2 * S];
    const float x3 = src[base + 3 * S];
    const float y0 = 0.5f * (x0 + x1 + x2 - x3);
    const float y1 = 0.5f * (x0 + x1 - x2 + x3);
    const float y2 = 0.5f * (x0 - x1 + x2 + x3);
    const float y3 = 0.5f * (-x0 + x1 + x2 + x3);
    output[base] = from_float<OutputType>(y0);
    output[base + S] = from_float<OutputType>(y1);
    output[base + 2 * S] = from_float<OutputType>(y2);
    output[base + 3 * S] = from_float<OutputType>(y3);
    return fmaxf(fmaxf(fabsf(y0), fabsf(y1)), fmaxf(fabsf(y2), fabsf(y3)));
}

template<int GROUPS_PER_BLOCK, typename OutputType>
__global__ void dequantize_int8_convrot_groups64_kernel(
    const int8_t* __restrict__ q,
    const float* __restrict__ scales,
    OutputType* __restrict__ output,
    int K,
    int scale_size)
{
    constexpr int kGroupThreads = 64;
    extern __shared__ float smem[];

    const int sub = threadIdx.x / kGroupThreads;
    const int lane = threadIdx.x % kGroupThreads;
    const int group = static_cast<int>(blockIdx.y) * GROUPS_PER_BLOCK + sub;
    const int64_t row = blockIdx.x;
    const bool active = group < K / kConvRotGroup;
    const int64_t row_offset = static_cast<int64_t>(row) * K;
    const int group_col = group * kConvRotGroup;
    const float scale = scales[scale_size == 1 ? 0 : row];

    float* buf0 = smem + sub * (2 * kConvRotGroup);
    float* buf1 = buf0 + kConvRotGroup;

    const int base = lane * 4;
    const int64_t q_offset = row_offset + group_col + base;
    const float x0 = active ? static_cast<float>(q[q_offset]) * scale : 0.0f;
    const float x1 = active ? static_cast<float>(q[q_offset + 1]) * scale : 0.0f;
    const float x2 = active ? static_cast<float>(q[q_offset + 2]) * scale : 0.0f;
    const float x3 = active ? static_cast<float>(q[q_offset + 3]) * scale : 0.0f;
    buf1[base] = 0.5f * (x0 + x1 + x2 - x3);
    buf1[base + 1] = 0.5f * (x0 + x1 - x2 + x3);
    buf1[base + 2] = 0.5f * (x0 - x1 + x2 + x3);
    buf1[base + 3] = 0.5f * (-x0 + x1 + x2 + x3);
    __syncthreads();

    convrot_fht_stage64<4>(buf1, buf0, lane);
    __syncthreads();
    convrot_fht_stage64<16>(buf0, buf1, lane);
    __syncthreads();

    if (active) {
        convrot_fht_stage64_store<64, OutputType>(buf1, output + row_offset + group_col, lane);
    }
}

template<int GROUPS_PER_BLOCK, typename InputType, typename OutputType>
__global__ void rotate_int8_convrot_groups64_amax_kernel(
    const InputType* __restrict__ x,
    OutputType* __restrict__ output,
    float* __restrict__ partial_absmax,
    int K)
{
    constexpr int kGroupThreads = 64;
    extern __shared__ float smem[];

    const int sub = threadIdx.x / kGroupThreads;
    const int lane = threadIdx.x % kGroupThreads;
    const int group = static_cast<int>(blockIdx.y) * GROUPS_PER_BLOCK + sub;
    const int64_t row = blockIdx.x;
    const int n_groups = K / kConvRotGroup;
    const bool active = group < n_groups;
    const int64_t row_offset = static_cast<int64_t>(row) * K;
    const int group_col = group * kConvRotGroup;

    float* buf0 = smem + sub * (2 * kConvRotGroup);
    float* buf1 = buf0 + kConvRotGroup;

    const int base = lane * 4;
    const int64_t x_offset = row_offset + group_col + base;
    const float x0 = active ? to_float(x[x_offset]) : 0.0f;
    const float x1 = active ? to_float(x[x_offset + 1]) : 0.0f;
    const float x2 = active ? to_float(x[x_offset + 2]) : 0.0f;
    const float x3 = active ? to_float(x[x_offset + 3]) : 0.0f;
    buf1[base] = 0.5f * (x0 + x1 + x2 - x3);
    buf1[base + 1] = 0.5f * (x0 + x1 - x2 + x3);
    buf1[base + 2] = 0.5f * (x0 - x1 + x2 + x3);
    buf1[base + 3] = 0.5f * (-x0 + x1 + x2 + x3);
    __syncthreads();

    convrot_fht_stage64<4>(buf1, buf0, lane);
    __syncthreads();
    convrot_fht_stage64<16>(buf0, buf1, lane);
    __syncthreads();

    float local_max = 0.0f;
    if (active) {
        local_max = convrot_fht_stage64_store_absmax<64, OutputType>(
            buf1, output + row_offset + group_col, lane);
    }
    buf0[lane] = local_max;
    __syncthreads();

    if (lane < 32) {
        float v = fmaxf(buf0[lane], buf0[lane + 32]);
        v = warp_reduce_max(v);
        if (lane == 0 && active) {
            partial_absmax[static_cast<int64_t>(row) * n_groups + group] = v;
        }
    }
}

template<int GROUPS_PER_BLOCK, typename InputType, typename OutputType>
__global__ void rotate_int8_convrot_groups64_kernel(
    const InputType* __restrict__ x,
    OutputType* __restrict__ output,
    int K)
{
    constexpr int kGroupThreads = 64;
    extern __shared__ float smem[];

    const int sub = threadIdx.x / kGroupThreads;
    const int lane = threadIdx.x % kGroupThreads;
    const int group = static_cast<int>(blockIdx.y) * GROUPS_PER_BLOCK + sub;
    const int64_t row = blockIdx.x;
    const bool active = group < K / kConvRotGroup;
    const int64_t row_offset = static_cast<int64_t>(row) * K;
    const int group_col = group * kConvRotGroup;

    float* buf0 = smem + sub * (2 * kConvRotGroup);
    float* buf1 = buf0 + kConvRotGroup;

    const int base = lane * 4;
    const int64_t x_offset = row_offset + group_col + base;
    const float x0 = active ? to_float(x[x_offset]) : 0.0f;
    const float x1 = active ? to_float(x[x_offset + 1]) : 0.0f;
    const float x2 = active ? to_float(x[x_offset + 2]) : 0.0f;
    const float x3 = active ? to_float(x[x_offset + 3]) : 0.0f;
    buf1[base] = 0.5f * (x0 + x1 + x2 - x3);
    buf1[base + 1] = 0.5f * (x0 + x1 - x2 + x3);
    buf1[base + 2] = 0.5f * (x0 - x1 + x2 + x3);
    buf1[base + 3] = 0.5f * (-x0 + x1 + x2 + x3);
    __syncthreads();

    convrot_fht_stage64<4>(buf1, buf0, lane);
    __syncthreads();
    convrot_fht_stage64<16>(buf0, buf1, lane);
    __syncthreads();

    if (active) {
        convrot_fht_stage64_store<64, OutputType>(buf1, output + row_offset + group_col, lane);
    }
}

template<typename InputType, int BLOCK_THREADS, bool STOCHASTIC>
__global__ void quantize_int8_rowwise_from_partials_kernel(
    const InputType* __restrict__ x,
    const float* __restrict__ partial_absmax,
    int8_t* __restrict__ q,
    float* __restrict__ scales,
    int K,
    uint64_t seed)
{
    constexpr int kWarps = BLOCK_THREADS / kThreadsPerWarp;
    __shared__ float warp_smem[kWarps];
    __shared__ float block_smem;

    const int row = static_cast<int>(blockIdx.x);
    const int tid = threadIdx.x;
    const int n_groups = K / kConvRotGroup;
    const int64_t row_offset = static_cast<int64_t>(row) * K;
    const float* row_partials = partial_absmax + static_cast<int64_t>(row) * n_groups;

    float abs_max = 0.0f;
    for (int g = tid; g < n_groups; g += BLOCK_THREADS) {
        abs_max = fmaxf(abs_max, row_partials[g]);
    }
    abs_max = block_reduce_max_t<kWarps>(abs_max, warp_smem, &block_smem);
    const float scale = fmaxf(
        finite_absmax_for_int8_scale<InputType>(abs_max) * (1.0f / 127.0f),
        1.0e-30f);
    if (tid == 0) {
        scales[row] = scale;
    }

    for (int col = tid; col < K; col += BLOCK_THREADS) {
        const int64_t idx = row_offset + col;
        const float scaled = quant_div_to_float<InputType>(x[idx], scale);
        float quantized;
        if constexpr (STOCHASTIC) {
            const InputType noise = stochastic_rng_value<InputType>(idx, seed);
            quantized = floorf(stochastic_sum_to_float<InputType>(scaled, noise));
        } else {
            quantized = nearbyintf(scaled);
        }
        quantized = fminf(127.0f, fmaxf(-128.0f, quantized));
        q[idx] = static_cast<int8_t>(quantized);
    }
}

// Optional activation applied on the way into the quantizer, so an MLP's
// `linear(act(proj(x)))` never writes act's output to HBM just to read it
// straight back. No reduction is involved, so it is nearly free here. The
// codes live in input_act_codes.h, shared with the nanobind layer.
template<int ACT>
__device__ __forceinline__ float apply_input_act(float v) {
    if constexpr (ACT == kActGeluTanh) {
        // Matches torch.nn.functional.gelu(x, approximate="tanh").
        constexpr float kBeta = 0.7978845608028654f;   // sqrt(2/pi)
        constexpr float kKappa = 0.044715f;
        const float inner = kBeta * (v + kKappa * v * v * v);
        return 0.5f * v * (1.0f + tanhf(inner));
    }
    return v;
}

// Reads one activated value: column `col` of the K-wide activated row starting
// at `in_row`. For SwiGLU the raw row is 2*K wide with the gate in the first
// half; every other activation reads the same K-wide row it writes.
template<int ACT, typename InputType>
__device__ __forceinline__ float load_input_act(
    const InputType* __restrict__ x, int64_t in_row, int col, int K)
{
    if constexpr (ACT == kActSwiGLU) {
        // Matches torch silu(gate) * up.
        const float gate = to_float(x[in_row + col]);
        const float up = to_float(x[in_row + K + col]);
        return (gate / (1.0f + expf(-gate))) * up;
    } else {
        return apply_input_act<ACT>(to_float(x[in_row + col]));
    }
}

template<typename InputType, int BLOCK_THREADS, bool STOCHASTIC, int ACT = kActNone>
__global__ void quantize_int8_rowwise_convrot64_kernel(
    const InputType* __restrict__ x,
    int8_t* __restrict__ q,
    float* __restrict__ scales,
    int K,
    uint64_t seed)
{
    constexpr int kGroupThreads = 64;
    constexpr int kGroupsInFlight = BLOCK_THREADS / kGroupThreads;
    constexpr int kWarps = BLOCK_THREADS / kThreadsPerWarp;

    extern __shared__ float smem[];
    float* row_buf = smem;
    float* tmp = smem + K;

    __shared__ float warp_smem[kWarps];
    __shared__ float block_smem;

    const int row = static_cast<int>(blockIdx.x);
    const int tid = threadIdx.x;
    const int sub = tid / kGroupThreads;
    const int lane = tid % kGroupThreads;
    const int64_t row_offset = static_cast<int64_t>(row) * K;
    // SwiGLU reads a [gate | up] raw row twice as wide as the K it writes.
    constexpr int kInWidth = (ACT == kActSwiGLU) ? 2 : 1;
    const int64_t in_row_offset = row_offset * kInWidth;
    const int n_groups = K / kConvRotGroup;

    float* buf0 = tmp + sub * (2 * kConvRotGroup);
    float* buf1 = buf0 + kConvRotGroup;
    float abs_max = 0.0f;

    const int iters = (n_groups + kGroupsInFlight - 1) / kGroupsInFlight;
    for (int it = 0; it < iters; ++it) {
        const int group = it * kGroupsInFlight + sub;
        const bool active = group < n_groups;
        const int base = lane * 4;
        const int group_col = group * kConvRotGroup;
        const int col = group_col + base;

        const float x0 = active ? load_input_act<ACT>(x, in_row_offset, col, K) : 0.0f;
        const float x1 = active ? load_input_act<ACT>(x, in_row_offset, col + 1, K) : 0.0f;
        const float x2 = active ? load_input_act<ACT>(x, in_row_offset, col + 2, K) : 0.0f;
        const float x3 = active ? load_input_act<ACT>(x, in_row_offset, col + 3, K) : 0.0f;
        buf1[base] = 0.5f * (x0 + x1 + x2 - x3);
        buf1[base + 1] = 0.5f * (x0 + x1 - x2 + x3);
        buf1[base + 2] = 0.5f * (x0 - x1 + x2 + x3);
        buf1[base + 3] = 0.5f * (-x0 + x1 + x2 + x3);
        __syncthreads();

        convrot_fht_stage64<4>(buf1, buf0, lane);
        __syncthreads();
        convrot_fht_stage64<16>(buf0, buf1, lane);
        __syncthreads();

        if (active) {
            abs_max = fmaxf(
                abs_max,
                convrot_fht_stage64_store_absmax<64, float>(buf1, row_buf + group_col, lane));
        }
        __syncthreads();
    }

    abs_max = block_reduce_max_t<kWarps>(abs_max, warp_smem, &block_smem);
    const float scale = fmaxf(
        finite_absmax_for_int8_scale<InputType>(abs_max) * (1.0f / 127.0f),
        1.0e-30f);
    if (tid == 0) {
        scales[row] = scale;
    }

    for (int col = tid; col < K; col += BLOCK_THREADS) {
        const int64_t idx = row_offset + col;
        const float scaled = quant_div_float_to_float<InputType>(row_buf[col], scale);
        float quantized;
        if constexpr (STOCHASTIC) {
            const InputType noise = stochastic_rng_value<InputType>(idx, seed);
            quantized = floorf(stochastic_sum_to_float<InputType>(scaled, noise));
        } else {
            quantized = nearbyintf(scaled);
        }
        quantized = fminf(127.0f, fmaxf(-128.0f, quantized));
        q[idx] = static_cast<int8_t>(quantized);
    }
}

} // namespace

} // namespace comfy
