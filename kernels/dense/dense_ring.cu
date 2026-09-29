// Only the observed SM120 D128 INT8 -> BF16 Dense specialization.
// Arithmetic and KV iteration remain in the pinned, Apache-2.0 upstream header.
#include <cassert>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cstddef>
#include <type_traits>

#ifndef H3_VARIANT_ID
#define H3_VARIANT_ID 0
#endif
#include "qk_int_sv_i8_cuda.cuh"

struct H3DenseArgs {
  void *q, *k, *v, *o, *q_scale, *k_scale, *v_scale;
  uint32_t batch, qo_len, kv_len, qo_heads, kv_heads;
  uint32_t q_stride_b, q_stride_h, q_stride_s;
  uint32_t k_stride_b, k_stride_h, k_stride_s;
  uint32_t v_stride_b, v_stride_h, v_stride_d;
  uint32_t o_stride_b, o_stride_h, o_stride_s;
  float sm_scale;
};

extern "C" size_t h3_dense_args_size() { return sizeof(H3DenseArgs); }
extern "C" int h3_dense_query_group() { return H3_VARIANT_ID; }
extern "C" const char *h3_dense_error_string(int code) {
  return cudaGetErrorString(static_cast<cudaError_t>(code));
}

extern "C" int h3_dense_launch_window(const H3DenseArgs *a, uintptr_t stream_ptr, uint32_t start) {
  if (!a || !a->q || !a->k || !a->v || !a->o || !a->q_scale ||
      !a->k_scale || !a->v_scale || !a->batch || !a->qo_len ||
      !a->kv_len || !a->qo_heads || !a->kv_heads ||
      a->qo_heads % a->kv_heads != 0) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  constexpr uint32_t CTA_Q = 128, CTA_K = 128, WARP_Q = 16, WARP_K = 128;
  constexpr int smem_max = 4 * 128 * 128 + 1024;
  if(a->k_stride_s!=128 || a->k_stride_h!=a->kv_len*128 ||
     a->k_stride_b!=a->kv_heads*a->k_stride_h ||
     a->v_stride_h!=128*a->v_stride_d || a->v_stride_b!=a->kv_heads*a->v_stride_h)
    return (int)cudaErrorInvalidValue;
  alignas(64) CUtensorMap mapK,mapV;
  uint64_t kd[3]={128,a->kv_len,uint64_t(a->batch)*a->kv_heads};
  uint64_t ks[2]={a->k_stride_s,a->k_stride_h};
  uint64_t vd[3]={a->v_stride_d,128,uint64_t(a->batch)*a->kv_heads};
  uint64_t vs[2]={a->v_stride_d,a->v_stride_h};
  uint32_t box[3]={128,128,1},es[3]={1,1,1};
  auto encode=[&](CUtensorMap *m,void *p,uint64_t *d,uint64_t *st){
   return cuTensorMapEncodeTiled(m,CU_TENSOR_MAP_DATA_TYPE_UINT8,3,p,d,st,box,es,
    CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_128B,
    CU_TENSOR_MAP_L2_PROMOTION_L2_128B,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  };
  if(encode(&mapK,a->k,kd,ks)!=CUDA_SUCCESS || encode(&mapV,a->v,vd,vs)!=CUDA_SUCCESS)
    return (int)cudaErrorInvalidValue;
  auto kernel = qk_int_sv_i8_attn_kernel<
      CTA_Q, CTA_K, WARP_Q, WARP_K, 128, DataType::kInt8,
      QuantGranularity::kPerThread, QuantGranularity::kPerThread, float, false,
      nv_bfloat16, ComputeUnit::kCudaCore, MaskMode::kNone, false, true,
      false, false, true>;
  cudaError_t error = cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_max);
  if (error != cudaSuccess) return static_cast<int>(error);
  // Keep physical grid.x/y intact: the quantized Q scale ABI uses both.
  const uint32_t total_blocks=(a->qo_len+127)/128;
  if(start>=total_blocks)return (int)cudaErrorInvalidValue;
  if(start){
    // Only legal BHSD/BSHD output; byte-zero skipped rows before real compute.
    const size_t rows=size_t(start)*128;
    for(unsigned b=0;b<a->batch;b++){
      auto base=(char*)a->o+size_t(b)*a->o_stride_b*2;
      if(a->o_stride_s==a->qo_heads*128 && a->o_stride_h==128)
        error=cudaMemsetAsync(base,0,rows*a->qo_heads*128*2,(cudaStream_t)stream_ptr);
      else if(a->o_stride_s==128 && a->o_stride_h==a->qo_len*128)
        error=cudaMemset2DAsync(base,size_t(a->o_stride_h)*2,0,rows*128*2,a->qo_heads,(cudaStream_t)stream_ptr);
      else return (int)cudaErrorInvalidValue;
      if(error)return int(error);
    }
  }
  dim3 grid(total_blocks-start, a->qo_heads, a->batch);
  dim3 block(32,12);
  kernel<<<grid, block, smem_max, reinterpret_cast<cudaStream_t>(stream_ptr)>>>(
      static_cast<int8_t *>(a->q), static_cast<int8_t *>(a->k),
      static_cast<int8_t *>(a->v), static_cast<nv_bfloat16 *>(a->o), nullptr,
      static_cast<float *>(a->q_scale), static_cast<float *>(a->k_scale),
      static_cast<float *>(a->v_scale), nullptr, nullptr,
      0, 0, 0, 0, -1, a->qo_len, a->kv_len, a->qo_heads / a->kv_heads,
      a->q_stride_b, a->q_stride_s, a->q_stride_h,
      a->k_stride_b, a->k_stride_s, a->k_stride_h,
      a->v_stride_b, a->v_stride_h, a->v_stride_d,
      a->o_stride_b, a->o_stride_s, a->o_stride_h, a->sm_scale, mapK, mapV, start);
  return static_cast<int>(cudaGetLastError());
}

extern "C" int h3_dense_resources(int *out) {
  auto kernel = qk_int_sv_i8_attn_kernel<128,128,16,128,128,DataType::kInt8,
    QuantGranularity::kPerThread,QuantGranularity::kPerThread,float,false,
    nv_bfloat16,ComputeUnit::kCudaCore,MaskMode::kNone,false,true,false,false,true>;
  cudaFuncAttributes a; cudaError_t e=cudaFuncGetAttributes(&a,kernel);
  if(e) return int(e);
  out[0]=a.numRegs;out[1]=a.localSizeBytes;out[2]=a.sharedSizeBytes;
  out[3]=a.maxThreadsPerBlock;out[4]=a.binaryVersion;return 0;
}

extern "C" int h3_dense_launch(const H3DenseArgs *a,uintptr_t stream){return h3_dense_launch_window(a,stream,0);}
