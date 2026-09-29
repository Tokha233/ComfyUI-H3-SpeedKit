"""ctypes-only consumer wrapper; quantization stays in the installed Kitchen."""
from __future__ import annotations

import ctypes as ct
from pathlib import Path

import torch


class DenseArgs(ct.Structure):
    _fields_ = [(name, ct.c_void_p) for name in
                ("q", "k", "v", "o", "q_scale", "k_scale", "v_scale")] + [
        (name, ct.c_uint32) for name in (
            "batch", "qo_len", "kv_len", "qo_heads", "kv_heads",
            "q_stride_b", "q_stride_h", "q_stride_s",
            "k_stride_b", "k_stride_h", "k_stride_s",
            "v_stride_b", "v_stride_h", "v_stride_d",
            "o_stride_b", "o_stride_h", "o_stride_s",
        )] + [("sm_scale", ct.c_float)]


def allocate_output(packed, layout="bhsd"):
    b, h, s, d = packed.q.shape
    if layout == "bhsd":
        return torch.empty((b, h, s, d), dtype=torch.bfloat16, device=packed.q.device)
    if layout == "bshd":
        # Logical shape stays BHSD, but transpose+reshape to [B,S,H*D] is a view.
        return torch.empty((b, s, h, d), dtype=torch.bfloat16,
                           device=packed.q.device).transpose(1, 2)
    raise ValueError(f"Unsupported output layout: {layout}")


class DenseLibrary:
    def __init__(self, path):
        self.path = Path(path).resolve()
        self.lib = ct.CDLL(str(self.path), mode=ct.RTLD_LOCAL)
        self.lib.h3_dense_args_size.argtypes = []
        self.lib.h3_dense_args_size.restype = ct.c_size_t
        self.lib.h3_dense_query_group.argtypes = []
        self.lib.h3_dense_query_group.restype = ct.c_int
        self.lib.h3_dense_launch.argtypes = [ct.POINTER(DenseArgs), ct.c_size_t]
        self.lib.h3_dense_launch.restype = ct.c_int
        self.lib.h3_dense_error_string.argtypes = [ct.c_int]
        self.lib.h3_dense_error_string.restype = ct.c_char_p
        if self.lib.h3_dense_args_size() != ct.sizeof(DenseArgs):
            raise RuntimeError("C/ctypes ABI struct size mismatch")
        self.group = self.lib.h3_dense_query_group()

    def bind(self, packed, output):
        q, k, v = packed.q, packed.k, packed.v
        tensors = [q, k, v, packed.q_scale, packed.k_scale, packed.v_scale, output]
        if any(not t.is_cuda or t.device != q.device for t in tensors):
            raise ValueError("Packed input/output devices must agree")
        if packed.cta_k != 128 or q.shape[-1] != 128 or packed.original_head_dim != 128:
            raise ValueError("This build only supports the observed D128/CTA_K128 path")
        if packed.input_dtype != torch.bfloat16 or output.dtype != torch.bfloat16:
            raise ValueError("This strict replay only supports the captured BF16 path")
        if packed.attn_mask is not None or output.shape != q.shape:
            raise ValueError("Only unmasked BHSD logical outputs are supported")
        if q.dtype != torch.int8 or k.dtype != torch.int8 or v.dtype != torch.int8:
            raise ValueError("Q/K/V must retain the original INT8 carrier")
        if any(t.dtype != torch.float32 for t in tensors[3:6]):
            raise ValueError("Quantization scales must be FP32")
        if any(not t.is_contiguous() for t in tensors[:-1]) or output.stride(-1) != 1:
            raise ValueError("Packed inputs must use the exact contiguous Kitchen ABI")
        b, h, s, d = q.shape
        hk, sk = k.shape[1:3]
        padded_sk = ((sk + 127) // 128) * 128
        if k.shape[0] != b or k.shape[-1] != d or h % hk:
            raise ValueError("Invalid Q/K heads or dimensions")
        if tuple(v.shape) != (b * hk * d, padded_sk):
            raise ValueError("V must use Kitchen's [B*Hkv*D, padded_Sk] layout")
        if tuple(packed.q_scale.shape) != (b, h, ((s + 127) // 128) * 32):
            raise ValueError("Invalid D128 per-thread Q scale layout")
        if tuple(packed.k_scale.shape) != (b, hk, ((sk + 127) // 128) * 4):
            raise ValueError("Invalid per-thread K scale layout")
        if packed.v_scale.numel() != b * hk * d:
            raise ValueError("Invalid V scale layout")
        pointers = [q.data_ptr(), k.data_ptr(), v.data_ptr(), output.data_ptr(),
                    packed.q_scale.data_ptr(), packed.k_scale.data_ptr(),
                    packed.v_scale.data_ptr()]
        ints = [b, s, sk, h, hk, *q.stride()[:3], *k.stride()[:3],
                hk * d * padded_sk, d * padded_sk, padded_sk, *output.stride()[:3]]
        if any(x < 0 or x > 0x7fffffff for x in ints):
            raise ValueError("Shape/stride exceeds the upstream int32 ABI")
        expected_bhsd = (h * s * d, s * d, d, 1)
        expected_bshd = (s * h * d, d, h * d, 1)
        if tuple(output.stride()) not in (expected_bhsd, expected_bshd):
            raise ValueError("Output must be BHSD or BSHD storage without overlap")
        params = DenseArgs(*pointers, *ints, packed.attention_scale)
        return BoundDense(self, packed, output, params)


class BoundDense:
    def __init__(self, library, packed, output, params):
        self.library, self.packed, self.output, self.params = library, packed, output, params
        self._params_ptr = ct.byref(self.params)

    def __call__(self):
        # Honor the actual current stream, including after non-default-stream use.
        with torch.cuda.device(self.output.device):
            stream = torch.cuda.current_stream(self.output.device).cuda_stream
            error = self.library.lib.h3_dense_launch(self._params_ptr, stream)
        if error:
            message = self.library.lib.h3_dense_error_string(error).decode()
            raise RuntimeError(f"Dense CUDA launch failed ({error}): {message}")
        return self.output
