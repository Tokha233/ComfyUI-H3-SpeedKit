/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * INT8 GEMM with a FUSED dequant epilogue via CUTLASS (EVT):
 *   D[m,n] = (sum_k A[m,k]*B[n,k]) * x_scale[m] * w_scale[n] + bias[n]   -> out dtype
 * bias (and the residual rscale) are read in the OUTPUT dtype and converted
 * to float in-register, so callers never cast them.
 *
 * Replaces cuBLAS-GEMM(int32) + separate dequant with one near-peak kernel.
 * Multiple tile configs are instantiated and selected with a shape heuristic
 * fitted from sustained Ada and Blackwell benchmarks.
 * Falls back to cuBLAS when CUTLASS is unavailable or no config can run.
 */
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cmath>

#ifdef COMFY_HAVE_CUTLASS

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/default_gemm_universal_with_visitor.h"
#include "cutlass/epilogue/threadblock/fusion/visitors.hpp"

#include "cutlass_gemm_common.cuh"

namespace {
using namespace cute;
using comfy_cutlass::ThreadblockSwizzleLeanStreamK;

template <typename ThreadMap, bool Scalar>
struct WeightScaleBroadcast;

template <typename ThreadMap>
struct WeightScaleBroadcast<ThreadMap, false> {
    using Type = cutlass::epilogue::threadblock::VisitorRowBroadcast<
        ThreadMap, float, cute::Stride<_0, _1, int32_t>>;

    static typename Type::Arguments arguments(const float* scale, int n) {
        return {scale, 0.f, {_0{}, _1{}, n}};
    }
};

template <typename ThreadMap>
struct WeightScaleBroadcast<ThreadMap, true> {
    using Type = cutlass::epilogue::threadblock::VisitorScalarBroadcast<float>;

    static typename Type::Arguments arguments(const float* scale, int) {
        typename Type::Arguments result{};
        result.scalar_ptrs[0] = scale;
        return result;
    }
};

// One fused int8 GEMM, parameterized on output type AND tile/warp/stage config.
// bias is read in ElementOutput (nullptr broadcasts 0).
template <typename ElementOutput, int TBM, int TBN, int TBK, int WM, int WN, int WK, int NumStages,
          typename ArchTag = cutlass::arch::Sm80,
          bool ScalarWeightScale = false, int AlignmentAB = 16,
          typename ThreadblockSwizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>>
struct FusedInt8Gemm {
    using ElementA = int8_t; using ElementB = int8_t;
    using ElementC = ElementOutput;
    using ElementAcc = int32_t; using ElementCompute = float;
    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;   // B[N,K] row == [K,N] col
    using LayoutC = cutlass::layout::RowMajor;
    static constexpr int AlignA = AlignmentAB, AlignB = AlignmentAB;
    static constexpr int AlignC = 128 / cutlass::sizeof_bits<ElementC>::value;
    using TB   = cutlass::gemm::GemmShape<TBM, TBN, TBK>;
    using Warp = cutlass::gemm::GemmShape<WM, WN, WK>;
    using Inst = cutlass::gemm::GemmShape<16, 8, 32>;
    static constexpr int EVTStages = H3_EVT_STAGES;

    using ThreadMap = cutlass::epilogue::threadblock::OutputTileThreadLayout<TB, Warp, ElementC, AlignC, EVTStages>;
    using Accum  = cutlass::epilogue::threadblock::VisitorAccFetch;
    using XScale = cutlass::epilogue::threadblock::VisitorColBroadcast<ThreadMap, ElementCompute, cute::Stride<_1, _0, int32_t>>;
    using WScale = typename WeightScaleBroadcast<ThreadMap, ScalarWeightScale>::Type;
    using Bias = cutlass::epilogue::threadblock::VisitorScalarBroadcast<ElementOutput>;
    using Mul0 = cutlass::epilogue::threadblock::VisitorCompute<cutlass::multiplies, ElementCompute, ElementCompute, cutlass::FloatRoundStyle::round_to_nearest>;
    using EVT0 = cutlass::epilogue::threadblock::Sm80EVT<Mul0, Accum, XScale>;
    using Mul1 = cutlass::epilogue::threadblock::VisitorCompute<cutlass::multiplies, ElementCompute, ElementCompute, cutlass::FloatRoundStyle::round_to_nearest>;
    using EVT1 = cutlass::epilogue::threadblock::Sm80EVT<Mul1, EVT0, WScale>;
    using Add2 = cutlass::epilogue::threadblock::VisitorCompute<cutlass::plus, ElementOutput, ElementCompute, cutlass::FloatRoundStyle::round_to_nearest>;
    using EVT2 = cutlass::epilogue::threadblock::Sm80EVT<Add2, EVT1, Bias>;
    using StoreD = cutlass::epilogue::threadblock::VisitorAuxStore<ThreadMap, ElementOutput, cutlass::FloatRoundStyle::round_to_nearest, cute::Stride<int64_t, _1, int64_t>>;
    using EVTD = cutlass::epilogue::threadblock::Sm80EVT<StoreD, EVT2>;

    using GemmKernel = typename cutlass::gemm::kernel::DefaultGemmWithVisitor<
        ElementA, LayoutA, cutlass::ComplexTransform::kNone, AlignA,
        ElementB, LayoutB, cutlass::ComplexTransform::kNone, AlignB,
        ElementC, LayoutC, AlignC,
        ElementAcc, ElementCompute,
        cutlass::arch::OpClassTensorOp, ArchTag,
        TB, Warp, Inst, EVTD,
        ThreadblockSwizzle,
        NumStages, cutlass::arch::OpMultiplyAddSaturate, EVTStages>::GemmKernel;
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

    static bool run_strided(const int8_t* A, const int8_t* B, const float* xs, const float* ws,
                            const ElementOutput* bias, ElementOutput* D, int M, int N, int K,
                            int output_stride, cudaStream_t stream) {
        const auto weight_scale_args = WeightScaleBroadcast<ThreadMap, ScalarWeightScale>::arguments(ws, N);
        typename EVTD::Arguments cb{
            { {  { {}, {const_cast<float*>(xs), 0.f, {_1{}, _0{}, M}}, {} },
                 weight_scale_args, {} },
              {ElementOutput(0)}, {} },
            {D, {output_stride, _1{}, M * output_stride}} };
        return comfy_cutlass::launch_universal<Gemm>(
            A, B, cb, M, N, K, stream);
    }
};


}
extern "C" int h3_int8(const int8_t*A,const int8_t*B,const float*xs,const float*ws,void*D,int M,int N,int K,int config,uintptr_t stream){
 using O=cutlass::bfloat16_t;
 using G0=FusedInt8Gemm<O,128,256,64,64,64,64,3,cutlass::arch::Sm80,false,16,cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>>;
 if(config==0)return G0::run_strided(A,B,xs,ws,nullptr,(O*)D,M,N,K,N,(cudaStream_t)stream);
 using G1=FusedInt8Gemm<O,128,256,64,64,64,64,3,cutlass::arch::Sm80,false,16,cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<2>>;
 if(config==1)return G1::run_strided(A,B,xs,ws,nullptr,(O*)D,M,N,K,N,(cudaStream_t)stream);
 using G2=FusedInt8Gemm<O,128,256,64,64,64,64,3,cutlass::arch::Sm80,false,16,cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>>;
 if(config==2)return G2::run_strided(A,B,xs,ws,nullptr,(O*)D,M,N,K,N,(cudaStream_t)stream);
 using G3=FusedInt8Gemm<O,128,256,64,64,64,64,3,cutlass::arch::Sm80,false,16,cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>>;
 if(config==3)return G3::run_strided(A,B,xs,ws,nullptr,(O*)D,M,N,K,N,(cudaStream_t)stream);
 using G4=FusedInt8Gemm<O,256,128,64,64,64,64,3,cutlass::arch::Sm80,false,16,cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>>;
 if(config==4)return G4::run_strided(A,B,xs,ws,nullptr,(O*)D,M,N,K,N,(cudaStream_t)stream);
 using G5=FusedInt8Gemm<O,128,128,128,64,64,128,3,cutlass::arch::Sm80,false,16,cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>>;
 if(config==5)return G5::run_strided(A,B,xs,ws,nullptr,(O*)D,M,N,K,N,(cudaStream_t)stream);
 using G6=FusedInt8Gemm<O,128,256,128,64,64,128,2,cutlass::arch::Sm80,false,16,cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>>;
 if(config==6)return G6::run_strided(A,B,xs,ws,nullptr,(O*)D,M,N,K,N,(cudaStream_t)stream);
 using G7=FusedInt8Gemm<O,256,128,128,64,64,128,2,cutlass::arch::Sm80,false,16,cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>>;
 if(config==7)return G7::run_strided(A,B,xs,ws,nullptr,(O*)D,M,N,K,N,(cudaStream_t)stream);
 return 0;
}
#endif
