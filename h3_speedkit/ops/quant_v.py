"""Experimental BF16 Kitchen 0.2.33 V quantizer; no server registration.

The CUDA wheel remains the correctness oracle. This module is a candidate,
not a claim of GPU qualification. See ../analysis/quant-v-design.md.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import triton
import triton.language as tl

KITCHEN_COMMIT = "e9ea99cf2f0af1d0c49c04690d4153a91c2b8668"
QUANT_V_SOURCE_SHA256 = "9f82e38c5f3045c15268952a9ef421281e977ea33f25aba0bf90ac16c75c1497"


@dataclass(frozen=True)
class QuantVConfig:
    partial_n: int = 128
    partial_d: int = 64
    partial_warps: int = 4
    quant_n: int = 64
    quant_d: int = 64
    quant_warps: int = 4


CONFIGS = {
    "d64": QuantVConfig(),
    "d32": QuantVConfig(256, 32, 4, 128, 32, 4),
    "d128": QuantVConfig(128, 128, 8, 64, 128, 8),
}


@triton.jit
def _partial_absmax(V, Partial, N: tl.constexpr, H: tl.constexpr,
                    D: tl.constexpr, SB: tl.constexpr, SH: tl.constexpr,
                    SN: tl.constexpr, PARTS: tl.constexpr,
                    BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr):
    ns = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    ds = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)
    bh = tl.program_id(2)
    base = (bh // H).to(tl.int64) * SB + (bh % H).to(tl.int64) * SH
    ptr = V + base + ns[:, None].to(tl.int64) * SN + ds[None, :]
    values = tl.load(ptr, (ns[:, None] < N) & (ds[None, :] < D), 0).to(tl.float32)
    maxima = tl.max(tl.abs(values), axis=0)
    dst = (bh * PARTS + tl.program_id(0)) * D + ds
    tl.store(Partial + dst, maxima, ds < D)


@triton.jit
def _finish_scales(Partial, Scale, Inverse, H: tl.constexpr,
                   D: tl.constexpr, PARTS: tl.constexpr,
                   BLOCK_PARTS: tl.constexpr, BLOCK_D: tl.constexpr):
    ps = tl.arange(0, BLOCK_PARTS)
    ds = tl.program_id(0) * BLOCK_D + tl.arange(0, BLOCK_D)
    bh = tl.program_id(1)
    ptr = Partial + (bh * PARTS + ps[:, None]) * D + ds[None, :]
    values = tl.load(ptr, (ps[:, None] < PARTS) & (ds[None, :] < D), 0)
    maxima = tl.max(values, axis=0)
    # Exact FP32 constants from float(1.f/127.f) and 1e-12f. Explicit PTX
    # prevents a compiler from changing multiplication into division or FMA.
    scales = tl.inline_asm_elementwise(
        "{ .reg .f32 product; mul.rn.ftz.f32 product, $1, 0f3C010204; "
        "max.f32 $0, product, 0f2B8CBCCC; }",
        constraints="=f,f", args=[maxima], dtype=tl.float32,
        is_pure=True, pack=1)
    # Upstream builds with --use_fast_math. This is deliberately the PTX
    # approximate reciprocal; the installed CUDA wheel must byte-verify it.
    inverse = tl.inline_asm_elementwise(
        "rcp.approx.ftz.f32 $0, $1;", constraints="=f,f", args=[scales],
        dtype=tl.float32, is_pure=True, pack=1)
    tl.store(Scale + bh * D + ds, scales, ds < D)
    tl.store(Inverse + bh * D + ds, inverse, ds < D)


@triton.jit
def _quantize_transpose(V, Out, Inverse, N: tl.constexpr, PADDED_N: tl.constexpr,
                        H: tl.constexpr, D: tl.constexpr,
                        SB: tl.constexpr, SH: tl.constexpr, SN: tl.constexpr,
                        BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr):
    dst = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    ds = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)
    bh = tl.program_id(2)
    # Destination -> source is [0,1,8,9,2,3,10,11,4,5,12,13,6,7,14,15].
    # Iterating destination indices gives contiguous packed stores. Every
    # source row still loads a contiguous D tile, including sliced QKV views.
    src = (dst & ~15) | (dst & 1) | ((dst & 2) << 2) | ((dst & 4) >> 1) | ((dst & 8) >> 1)
    base = (bh // H).to(tl.int64) * SB + (bh % H).to(tl.int64) * SH
    ptr = V + base + src[:, None].to(tl.int64) * SN + ds[None, :]
    values = tl.load(ptr, (src[:, None] < N) & (ds[None, :] < D), 0).to(tl.float32)
    inv = tl.load(Inverse + bh * D + ds, ds < D, 0)
    codes = tl.inline_asm_elementwise(
        "{ .reg .f32 product; mul.rn.ftz.f32 product, $1, $2; "
        "cvt.rni.sat.s8.f32 $0, product; }",
        constraints="=r,f,f", args=[values, inv[None, :]], dtype=tl.int32,
        is_pure=True, pack=1).to(tl.int8)
    out_ptr = Out + (bh * D + ds[:, None]).to(tl.int64) * PADDED_N + dst[None, :]
    tl.store(out_ptr, tl.trans(codes), (ds[:, None] < D) & (dst[None, :] < PADDED_N))


def validate_input(v: torch.Tensor, *, check_finite: bool = True) -> None:
    """Host contract; finite scan synchronizes and belongs outside timed launches."""
    if v.ndim != 4 or any(n <= 0 for n in v.shape) or v.shape[-1] != 128:
        raise ValueError("Candidate requires positive [B,H,N,128] extents")
    if v.dtype != torch.bfloat16 or not v.is_cuda:
        raise TypeError("Candidate requires a CUDA BF16 tensor")
    if v.stride(-1) != 1 or any(s < 0 for s in v.stride()):
        raise ValueError("Last dimension must be contiguous; strides must be nonnegative")
    if v.data_ptr() % 16 or any(n > 1 and (s * v.element_size()) % 16
                              for n, s in zip(v.shape[:3], v.stride()[:3])):
        raise ValueError("Input must satisfy Kitchen's 16-byte pointer/stride alignment")
    if check_finite and not bool(torch.isfinite(v).all().item()):
        raise ValueError("NaN and Inf are outside the exact finite-input contract")


class QuantVPlan:
    """Preallocated fixed-input experiment for allocation-free event timings.

    Construct once per immutable V. Constructor checks finiteness. Mutating V
    afterwards invalidates that check; call validate_input again before reuse.
    __call__ is asynchronous and launches only the three candidate kernels.
    """
    def __init__(self, v: torch.Tensor, *, cta_k: int = 128,
                 config: QuantVConfig = QuantVConfig(), check_finite: bool = True):
        validate_input(v, check_finite=check_finite)
        if cta_k not in (64, 128):
            raise ValueError("Kitchen packed V requires cta_k 64 or 128")
        for value in (config.partial_n, config.partial_d, config.quant_n, config.quant_d):
            if value < 16 or value & (value - 1):
                raise ValueError("Tile dimensions must be powers of two >= 16")
        if config.partial_warps not in (4, 8) or config.quant_warps not in (4, 8):
            raise ValueError("This experiment uses four or eight warps")
        self.v, self.config, self.cta_k = v, config, cta_k
        self.b, self.h, self.n, self.d = v.shape
        self.padded_n = triton.cdiv(self.n, cta_k) * cta_k
        self.parts = triton.cdiv(self.n, config.partial_n)
        self.out = torch.empty((self.b * self.h * self.d, self.padded_n),
                               device=v.device, dtype=torch.int8)
        self.scale = torch.empty(self.b * self.h * self.d, device=v.device, dtype=torch.float32)
        self.inverse = torch.empty_like(self.scale)
        self.partial = torch.empty((self.b * self.h, self.parts, self.d),
                                   device=v.device, dtype=torch.float32)

    def __call__(self) -> tuple[torch.Tensor, torch.Tensor]:
        c = self.config
        sb, sh, sn, _ = self.v.stride()
        scale_d = 16 if self.parts <= 512 else 8
        with torch.cuda.device(self.v.device):
            _partial_absmax[(self.parts, triton.cdiv(self.d, c.partial_d), self.b * self.h)](
                self.v, self.partial, self.n, self.h, self.d, sb, sh, sn,
                self.parts, c.partial_n, c.partial_d, num_warps=c.partial_warps,
                enable_fp_fusion=False)
            _finish_scales[(triton.cdiv(self.d, scale_d), self.b * self.h)](
                self.partial, self.scale, self.inverse, self.h, self.d,
                self.parts, triton.next_power_of_2(self.parts), scale_d, num_warps=4,
                enable_fp_fusion=False)
            _quantize_transpose[(triton.cdiv(self.padded_n, c.quant_n),
                                 triton.cdiv(self.d, c.quant_d), self.b * self.h)](
                self.v, self.out, self.inverse, self.n, self.padded_n, self.h,
                self.d, sb, sh, sn, c.quant_n, c.quant_d,
                num_warps=c.quant_warps, enable_fp_fusion=False)
        return self.out, self.scale


def quant_v_int8_exact(v: torch.Tensor, *, cta_k: int = 128,
                       config: QuantVConfig = QuantVConfig()) -> tuple[torch.Tensor, torch.Tensor]:
    """Checked one-shot entry point; use QuantVPlan for kernel-only benchmarking."""
    return QuantVPlan(v, cta_k=cta_k, config=config)()
