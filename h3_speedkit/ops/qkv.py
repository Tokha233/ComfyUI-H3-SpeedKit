"""R85 QKV + RMS/RoPE + Q/K quantization, with current-input K anchors."""
import ctypes as ct
import torch
import triton
from comfy_kitchen.sage_attention import PrequantizedInt8Attention
from .quant_v import _finish_scales, _quantize_transpose


class QKV:
    def __init__(self, directory):
        self.libraries = [ct.CDLL(str(directory / (name + ".so")))
                          for name in ("qkv", "sample", "anchor")]
        self.full = self.libraries[0].h3_qkv_quant_kv
        self.full.argtypes = [ct.c_void_p] * 8 + [ct.c_float] + [ct.c_void_p] * 7 + [ct.c_int, ct.c_size_t]
        self.sample = self.libraries[1].h3_sample_konly
        self.sample.argtypes = [ct.c_void_p] * 8 + [ct.c_float] + [ct.c_void_p] * 3 + [ct.c_int, ct.c_size_t]
        self.gather = self.libraries[1].h3_sample_gather
        self.gather.argtypes = [ct.c_void_p] * 6 + [ct.c_int, ct.c_size_t]
        self.detect = self.libraries[2].h3_sample_anchor
        self.detect.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_size_t]
        for fn in (self.full, self.sample, self.gather, self.detect):
            fn.restype = ct.c_int

    def __call__(self, q, w, xs, ws, rope, qw, kw, eps):
        m = q.shape[0]
        if q.shape != (m, 5376) or w.shape != (21504, 5376):
            raise ValueError("QKV requires the H3 56 x 128, hidden=5376 geometry")
        if rope.shape != (1, m, 1, 48, 2, 2):
            raise ValueError("QKV requires the pinned partial split-half RoPE layout")
        expected = ((q, torch.int8, (m, 5376)), (w, torch.int8, (21504, 5376)),
                    (xs, torch.float32, (m, 1)), (ws, torch.float32, (21504, 1)),
                    (rope, torch.bfloat16, (1, m, 1, 48, 2, 2)),
                    (qw, torch.bfloat16, (128,)), (kw, torch.bfloat16, (128,)))
        if any(t.dtype != dtype or tuple(t.shape) != shape for t, dtype, shape in expected):
            raise ValueError("QKV operand dtype/shape does not match the CUDA ABI: " +
                             str([(str(t.dtype), tuple(t.shape), str(dtype), shape)
                                  for t, dtype, shape in expected]))
        device = q.device
        stream = torch.cuda.current_stream(device)
        st = stream.cuda_stream
        parts = triton.cdiv(m, 128)
        qi = torch.empty((1, 56, m, 128), device=device, dtype=torch.int8)
        ki = torch.empty_like(qi)
        qs = torch.empty((1, 56, parts * 32), device=device, dtype=torch.float32)
        ks = torch.empty((1, 56, parts * 4), device=device, dtype=torch.float32)
        vp = torch.empty((56, parts, 128), device=device, dtype=torch.float32)
        anchors = torch.empty((1, 56), device=device, dtype=torch.int32)
        sq = torch.empty((9, 5376), device=device, dtype=torch.int8)
        sx = torch.empty((9, 1), device=device, dtype=torch.float32)
        sr = torch.empty((1, 9, 1, 48, 2, 2), device=device, dtype=torch.bfloat16)
        sd = torch.empty((9, 21504), device=device, dtype=torch.bfloat16)
        out = torch.empty((m, 21504), device=device, dtype=torch.bfloat16)
        owners = (q, w, xs, ws, rope, qw, kw, qi, ki, qs, ks, vp, anchors, sq, sx, sr, sd, out)
        if any(not t.is_cuda or t.device != device or not t.is_contiguous() for t in owners):
            raise ValueError("QKV operands must be contiguous and on one CUDA device")
        for tensor in owners:
            tensor.record_stream(stream)

        def check(code):
            if code != 1:
                raise RuntimeError("R85 QKV launch failed; request aborted without retry")

        check(self.gather(*(t.data_ptr() for t in (q, xs, rope, sq, sx, sr)), m, st))
        check(self.sample(*(t.data_ptr() for t in (sq, w, sx, ws, sd, sr, qw, kw)),
                          eps, None, None, None, 9, st))
        sk = sd.view(3, 56, 9, 128)[1]
        check(self.detect(sk.data_ptr(), anchors.data_ptr(), st))
        check(self.full(*(t.data_ptr() for t in (q, w, xs, ws, out, rope, qw, kw)),
                        eps, *(t.data_ptr() for t in (qi, qs, vp, ki, ks, sk, anchors)), m, st))
        v = out.view(3, 56, m, 128)[2].unsqueeze(0)
        vi = torch.empty((56 * 128, parts * 128), device=device, dtype=torch.int8)
        vs = torch.empty((56 * 128,), device=device, dtype=torch.float32)
        inv = torch.empty_like(vs)
        width = 16 if parts <= 512 else 8
        _finish_scales[(triton.cdiv(128, width), 56)](
            vp, vs, inv, 56, 128, parts, triton.next_power_of_2(parts), width,
            num_warps=4, enable_fp_fusion=False)
        _quantize_transpose[(triton.cdiv(parts * 128, 64), 1, 56)](
            v, vi, inv, m, parts * 128, 56, 128, *v.stride()[:3], 64, 128,
            num_warps=8, enable_fp_fusion=False)
        return PrequantizedInt8Attention(
            q=qi, k=ki, v=vi, q_scale=qs, k_scale=ks, v_scale=vs,
            original_head_dim=128, input_dtype=torch.bfloat16,
            attention_scale=128 ** -0.5, cta_k=128, attn_mask=None)
