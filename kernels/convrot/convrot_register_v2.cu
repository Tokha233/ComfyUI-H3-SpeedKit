// SPDX-License-Identifier: Apache-2.0
// Layout derived from SGLang PR #38040, commit 3299804f81b686d3752287e19d9c9d2987d1afc1.
// Copyright 2026 SGLang Team. All Rights Reserved.
// Modified 2026-09-11: retain Kitchen stage association, per-stage 0.5,
// BF16 quantization barriers, row scale and integer rounding. ACT=none only.
// V2: vectorize only global memory IO (uint4 BF16 loads / uint2 INT8 stores).
#include "convrot64_exact.cuh"
#include <cstddef>

#define H3_EXPORT extern "C" __attribute__((visibility("default")))

namespace {
constexpr unsigned kMask = 0xffffffffu;

// Each 8-lane cluster owns a 256-value group. Each lane owns 32 values.
// Index bits 0,1,2,4,6 are register bits; bits 3,5,7 are lane bits.
__device__ __forceinline__ int run_offset(int q, int j) {
  return ((q & 1) << 3) | ((j & 1) << 4) | (((q >> 1) & 1) << 5) |
         ((j >> 1) << 6) | (((q >> 2) & 1) << 7);
}

__device__ __forceinline__ void stage_register(float (&v)[32]) {
#pragma unroll
  for (int r = 0; r < 32; r += 4) {
    const float x0 = v[r], x1 = v[r + 1], x2 = v[r + 2], x3 = v[r + 3];
    // Verbatim Kitchen formulas, including operand order and signed zero.
    v[r]     = 0.5f * (x0 + x1 + x2 - x3);
    v[r + 1] = 0.5f * (x0 + x1 - x2 + x3);
    v[r + 2] = 0.5f * (x0 - x1 + x2 + x3);
    v[r + 3] = 0.5f * (-x0 + x1 + x2 + x3);
  }
}

template<int RegisterBit, int LaneXor>
__device__ __forceinline__ void stage_split(float (&v)[32], int lane) {
  const bool high = (lane & LaneXor) != 0;
#pragma unroll
  for (int r = 0; r < 32; ++r) {
    if ((r & RegisterBit) == 0) {
      const float own0 = v[r], own1 = v[r | RegisterBit];
      const float peer0 = __shfl_xor_sync(kMask, own0, LaneXor);
      const float peer1 = __shfl_xor_sync(kMask, own1, LaneXor);
      // Always recover canonical x0,x1,x2,x3. Reordering the additions to
      // start at this lane's values would change FP32 rounding/signed zero.
      const float x0 = high ? peer0 : own0;
      const float x1 = high ? peer1 : own1;
      const float x2 = high ? own0 : peer0;
      const float x3 = high ? own1 : peer1;
      if (high) {
        v[r] = 0.5f * (x0 - x1 + x2 + x3);
        v[r | RegisterBit] = 0.5f * (-x0 + x1 + x2 + x3);
      } else {
        v[r] = 0.5f * (x0 + x1 + x2 - x3);
        v[r | RegisterBit] = 0.5f * (x0 + x1 - x2 + x3);
      }
    }
  }
}

__device__ __forceinline__ float all_lane_max(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1)
    value = fmaxf(value, __shfl_xor_sync(kMask, value, offset));
  return value;
}

template<int K, bool Diagnostic>
__global__ void __launch_bounds__(((K + 1023) / 1024) * 32)
register_rotate_quant(const nv_bfloat16* __restrict__ x,
                     int8_t* __restrict__ q, float* __restrict__ scales,
                     float* __restrict__ rotated) {
  static_assert(K == 5376 || K == 7168);
  constexpr int kWarps = (K + 1023) / 1024;
  __shared__ float warp_max[kWarps];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int group = warp * 4 + lane / 8;
  const int cluster_lane = lane & 7;
  const bool active = group < K / 256;
  const int64_t row_offset = static_cast<int64_t>(blockIdx.x) * K;
  float v[32];
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const int col = group * 256 + run_offset(cluster_lane, j);
    // K and run_offset are multiples of eight BF16 values. The host ABI
    // requires 16-byte X alignment, so each lane owns one aligned uint4.
    uint4 raw = make_uint4(0, 0, 0, 0);
    if (active) raw = *reinterpret_cast<const uint4*>(x + row_offset + col);
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      const uint32_t word = e < 2 ? raw.x : e < 4 ? raw.y : e < 6 ? raw.z : raw.w;
      const unsigned short bits = static_cast<unsigned short>(word >> ((e & 1) * 16));
      // Bit reinterpretation + the original BF16-to-FP32 helper exactly
      // matches load_input_act<kActNone>, including negative zero.
      v[j * 8 + e] = comfy::to_float(__ushort_as_bfloat16(bits));
    }
  }
  stage_register(v);
  stage_split<4, 1>(v, lane);
  stage_split<8, 2>(v, lane);
  stage_split<16, 4>(v, lane);

  float local_max = 0.0f;
#pragma unroll
  for (int r = 0; r < 32; ++r)
    local_max = fmaxf(local_max, fabsf(v[r]));
  local_max = all_lane_max(local_max);
  if (lane == 0) warp_max[warp] = local_max;
  __syncthreads();
  // fmaxf over nonnegative fabsf values preserves the finite absmax exactly.
  // Every warp reads all row partials, so no second block barrier is needed.
  const float abs_max = all_lane_max(lane < kWarps ? warp_max[lane] : 0.0f);
  const float scale = fmaxf(
      comfy::finite_absmax_for_int8_scale<nv_bfloat16>(abs_max) * (1.0f / 127.0f),
      1.0e-30f);
  if (threadIdx.x == 0) scales[blockIdx.x] = scale;
  if (active) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int col = group * 256 + run_offset(cluster_lane, j);
      uint32_t low = 0, high = 0;
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const float value = v[j * 8 + e];
        if constexpr (Diagnostic) rotated[row_offset + col + e] = value;
        const float scaled = comfy::quant_div_float_to_float<nv_bfloat16>(value, scale);
        float quantized = nearbyintf(scaled);
        quantized = fminf(127.0f, fmaxf(-128.0f, quantized));
        const uint32_t byte = static_cast<unsigned char>(static_cast<int8_t>(quantized));
        if (e < 4) low |= byte << (8 * e);
        else high |= byte << (8 * (e - 4));
      }
      // CUDA targets are little-endian. Each packed byte is the same int8
      // conversion as r1; one aligned 8-byte vector store replaces eight.
      *reinterpret_cast<uint2*>(q + row_offset + col) = make_uint2(low, high);
    }
  }
}

struct Resources {
  int64_t abi, bytes, k, threads, warps, num_regs, static_shared_bytes;
  int64_t local_bytes, max_threads_per_block, active_blocks_per_sm;
  int64_t sm_count, max_threads_per_sm, binary_version, ptx_version;
};

template<int K>
int dispatch(const void* x, void* q, void* scale, void* rotated,
             int64_t rows, int mode, uintptr_t stream_ptr, Resources* resources) {
  constexpr int kThreads = ((K + 1023) / 1024) * 32;
  auto kernel = register_rotate_quant<K, false>;
  if (resources) {
    cudaFuncAttributes attr{};
    int active = 0, device = 0, sms = 0, max_threads = 0;
    cudaError_t error = cudaFuncGetAttributes(&attr, kernel);
    if (error != cudaSuccess) return error;
    error = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active, kernel, kThreads, 0);
    if (error != cudaSuccess) return error;
    error = cudaGetDevice(&device);
    if (error != cudaSuccess) return error;
    error = cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device);
    if (error != cudaSuccess) return error;
    error = cudaDeviceGetAttribute(&max_threads, cudaDevAttrMaxThreadsPerMultiProcessor, device);
    if (error != cudaSuccess) return error;
    *resources = {1, sizeof(Resources), K, kThreads, kThreads / 32, attr.numRegs,
                  static_cast<int64_t>(attr.sharedSizeBytes), static_cast<int64_t>(attr.localSizeBytes),
                  attr.maxThreadsPerBlock, active, sms, max_threads, attr.binaryVersion, attr.ptxVersion};
    return cudaSuccess;
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  if (mode == 2) {
    // Untouched original Kitchen FP32 rotation oracle, ACT=none only.
    constexpr int kGroups = 4;
    comfy::rotate_int8_convrot_groups64_kernel<kGroups, nv_bfloat16, float>
        <<<dim3(rows, (K / 256 + kGroups - 1) / kGroups), 64 * kGroups,
           kGroups * 2 * 256 * sizeof(float), stream>>>(
            static_cast<const nv_bfloat16*>(x), static_cast<float*>(rotated), K);
  } else if (mode == 1) {
    register_rotate_quant<K, true><<<rows, kThreads, 0, stream>>>(
        static_cast<const nv_bfloat16*>(x), static_cast<int8_t*>(q),
        static_cast<float*>(scale), static_cast<float*>(rotated));
  } else {
    register_rotate_quant<K, false><<<rows, kThreads, 0, stream>>>(static_cast<const nv_bfloat16*>(x),
        static_cast<int8_t*>(q), static_cast<float*>(scale), nullptr);
  }
  return cudaGetLastError();
}
} // namespace

H3_EXPORT int h3_convrot_register_abi() { return 1; }
H3_EXPORT size_t h3_convrot_register_resources_size() { return sizeof(Resources); }
H3_EXPORT const char* h3_convrot_register_error(int code) {
  return cudaGetErrorString(static_cast<cudaError_t>(code));
}
H3_EXPORT int h3_convrot_register_query(int k, Resources* out, size_t bytes) {
  if (!out || bytes != sizeof(Resources)) return cudaErrorInvalidValue;
  if (k == 5376) return dispatch<5376>(nullptr, nullptr, nullptr, nullptr, 0, 0, 0, out);
  if (k == 7168) return dispatch<7168>(nullptr, nullptr, nullptr, nullptr, 0, 0, 0, out);
  return cudaErrorInvalidValue;
}
H3_EXPORT int h3_convrot_register_run(const void* x, void* q, void* scale,
    void* rotated, int64_t rows, int k, int mode, uintptr_t stream_ptr) {
  if (!x || rows < 1 || rows > INT32_MAX || mode < 0 || mode > 2 ||
      (mode != 2 && (!q || !scale)) || (mode != 0 && !rotated)) return cudaErrorInvalidValue;
  if ((reinterpret_cast<uintptr_t>(x) & 15u) ||
      (mode != 2 && (reinterpret_cast<uintptr_t>(q) & 7u))) return cudaErrorInvalidValue;
  if (k == 5376) return dispatch<5376>(x, q, scale, rotated, rows, mode, stream_ptr, nullptr);
  if (k == 7168) return dispatch<7168>(x, q, scale, rotated, rows, mode, stream_ptr, nullptr);
  return cudaErrorInvalidValue;
}
