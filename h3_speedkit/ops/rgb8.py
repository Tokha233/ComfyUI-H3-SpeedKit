"""Experimental final-pixel -> contiguous FHWC RGB8 kernel; not installed.

Input must be the finalized pixels (after TRT stitching/normalization). The
explicit frame_dtype is the dtype the old CPU exporter would have received,
which can differ from the decoder dtype. This module changes no VAE arithmetic.
GPU numerical qualification and actual runtime source verification are pending.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _rgb8(X, Y, N: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
          S0: tl.constexpr, S1: tl.constexpr, S2: tl.constexpr, S3: tl.constexpr,
          CONTIGUOUS: tl.constexpr, FRAME_HALF: tl.constexpr,
          BLOCK: tl.constexpr):
    i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    if CONTIGUOUS:
        source = i
    else:
        c = i % 3
        w = i // 3 % W
        h = i // (3 * W) % H
        f = i // (3 * W * H)
        source = f * S0 + h * S1 + w * S2 + c * S3
    value = tl.load(X + source, i < N, other=0).to(tl.float32)
    # Match the cast at the GPU -> CPU frame boundary before CPU multiplication.
    if FRAME_HALF:
        value = value.to(tl.float16).to(tl.float32)
    scaled = value * 255.0
    if FRAME_HALF:
        # torch half multiplication stores half before clamp/byte. Omitting
        # this rounding changes pixels even when the decoded input is half.
        scaled = scaled.to(tl.float16).to(tl.float32)
    clipped = tl.minimum(tl.maximum(scaled, 0.0), 255.0)
    # Nonfinite inputs are diagnostic-only until GPU qualification. Finalized
    # production frames must be finite. Explicit NaN handling is deterministic.
    clipped = tl.where(scaled == scaled, clipped, 0.0)
    tl.store(Y + i, clipped.to(tl.uint8), i < N)


def quantize_finalized(frames, *, frame_dtype, block=1024):
    """Return a fresh CUDA uint8 FHWC Tensor; no host read or stream change.

    Layout/shape are logical: callers may pass a strided FHWC view of a BCTHW
    decoder output. Only finalized finite pixels have an intended contract.
    The caller retains the original image graph and chooses a separate RGB8
    terminal export route; uint8 is never substituted for Comfy IMAGE globally.
    """
    if not isinstance(frames, torch.Tensor) or frames.device.type != 'cuda':
        raise ValueError('expected CUDA final pixels')
    if frames.requires_grad or frames.dtype not in (torch.float16, torch.float32):
        raise ValueError('only no-grad FP16/FP32 inputs supported')
    if frame_dtype not in (torch.float16, torch.float32):
        raise ValueError('explicit original CPU frame dtype required')
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError('expected FHWC with three channels')
    f, h, w, _ = frames.shape
    if not (0 < f <= 400 and 0 < h <= 2048 and 0 < w <= 2048):
        raise ValueError('frame bounds exceeded')
    if any(s < 0 for s in frames.stride()) or block not in (256, 512, 1024, 2048):
        raise ValueError('unsupported stride or block')
    with torch.cuda.device(frames.device):
        out = torch.empty(frames.shape, dtype=torch.uint8, device=frames.device)
        _rgb8[(triton.cdiv(frames.numel(), block),)](
            frames, out, frames.numel(), h, w, *frames.stride(),
            frames.is_contiguous(), frame_dtype == torch.float16, block,
            num_warps=4, enable_fp_fusion=False)
        # Protect input storage if it was produced on another allocation stream.
        # Read-after-write ordering remains the caller's explicit obligation.
        frames.record_stream(torch.cuda.current_stream(frames.device))
    return out
