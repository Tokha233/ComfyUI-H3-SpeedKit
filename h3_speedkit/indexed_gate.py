# SPDX-License-Identifier: GPL-3.0-or-later
"""Opt-in ComfyUI consumer for Kitchen PR #219, independent of R85 binaries.

The attention/block sequence follows the pinned ComfyUI H3 implementation.
Only out_proj/FC2 GEMM + modulation gate + residual are replaced.
"""
from collections import Counter
from contextvars import ContextVar
import hashlib
import inspect
import logging
from pathlib import Path
import threading

LOG = logging.getLogger(__name__)
MODEL_SHA = "26ae72f5834cc9038a38600cd12c72d28da785cdaacd578d5b042b58ad48b09b"
KEY = "h3_speedkit_indexed_gate"


class Unsupported(ValueError):
    """Unsupported consumer configuration; use the original block."""


def segment_key(segments, tokens, gate_rows):
    """Validate the compact row map without reading CUDA tensor values."""
    key, cursor = [], 0
    for start, stop, row in segments:
        if (type(start) is not int or type(stop) is not int or type(row) is not int
                or start != cursor or stop < start or not 0 <= row < gate_rows):
            raise Unsupported("Requires contiguous segments with integer modulation rows")
        key.append((start, stop, row))
        cursor = stop
    if cursor != tokens:
        raise Unsupported("Modulation segments do not cover the packed input")
    return tuple(key)


def validate_linear(module, n, k):
    from comfy_kitchen.tensor.base import QuantizedTensor
    weight = module.weight
    if not isinstance(weight, QuantizedTensor) or weight._layout_cls != "TensorWiseINT8Layout":
        raise Unsupported("Requires INT8 ConvRot weights; merge the LoRA before quantizing")
    params = weight._params
    if (getattr(params, "transposed", False) or not getattr(params, "convrot", False)
            or getattr(params, "convrot_groupsize", 256) != 256
            or tuple(weight.shape) != (n, k) or module.bias is not None):
        raise Unsupported("Requires unbiased ConvRot256 H3 projections")
    if (module.weight_function or module.bias_function
            or getattr(module, "pre_quant_scale", None) is not None
            or getattr(module, "_full_precision_mm", False)
            or getattr(module, "comfy_force_cast_weights", False)):
        raise Unsupported("Active weight callbacks or precision overrides")


class IndexedGatePatch:
    def __init__(self, model, *, verify_new_shapes=True):
        self.model = model
        self.verify_new_shapes = verify_new_shapes
        self.verified = set()
        self.rejected = set()
        self.context = ContextVar(KEY, default=None)
        self.lock = threading.RLock()
        self.last_report = {}
        self.totals = Counter()
        self.warned = set()

    def preflight(self, args):
        import torch
        import comfy.model_management as mm
        from comfy.ldm.minimax import model as h3
        if torch.is_grad_enabled() or mm.in_training or torch.compiler.is_compiling():
            raise Unsupported("This ComfyUI consumer is qualified for eager inference only")
        x = args[0][0]
        if not x.is_cuda or torch.cuda.get_device_capability(x.device) != (12, 0):
            raise Unsupported("The consumer is qualified on SM120 only")
        if torch.cuda.is_current_stream_capturing():
            raise Unsupported("Graph capture is not qualified for the model wrapper")
        if any(getattr(torch.nn.modules.module, name, {}) for name in
               ("_global_forward_hooks", "_global_forward_pre_hooks", "_global_backward_hooks")):
            raise Unsupported("Global module hooks are active")
        if type(self.model) is not h3.MiniMaxH3Model or len(self.model.blocks) != 50:
            raise Unsupported("Requires the pinned 50-block H3 model")
        for block in self.model.blocks:
            if type(block) is not h3.DiTBlock or type(block.attn) is not h3.Attention or type(block.mlp) is not h3.MLP:
                raise Unsupported("Custom H3 block implementation")
            for module in block.modules():
                if ("forward" in module.__dict__ or getattr(module, "_compiled_call_impl", None) is not None
                        or any(getattr(module, name, {}) for name in
                               ("_forward_hooks", "_forward_pre_hooks", "_backward_hooks"))):
                    raise Unsupported("Module hooks or custom/compiled forward are active")
            validate_linear(block.attn.out_proj, 5376, 7168)
            validate_linear(block.mlp.fc2, 5376, 14336)
            if (block.attn.heads != 56 or block.attn.head_dim != 128
                    or block.attn.q_norm.eps != block.attn.k_norm.eps):
                raise Unsupported("Attention geometry differs from the qualified H3 model")
        return {"counts": Counter(), "rows": None, "row_key": None, "verified_keys": set()}

    def wrapper(self, executor, *args, **kwargs):
        with self.lock:
            try:
                frame = self.preflight(args)
            except Unsupported as error:
                frame = {"fallback": str(error), "counts": Counter()}
            token = self.context.set(frame)
            try:
                result = executor(*args, **kwargs)
                if "fallback" not in frame:
                    if frame["counts"]["blocks"] != 50:
                        raise RuntimeError("Incomplete indexed-gate block coverage")
                    self.verified.update(frame["verified_keys"])
                elif frame["fallback"] not in self.warned:
                    LOG.warning("H3 indexed-gate fallback: %s", frame["fallback"])
                    self.warned.add(frame["fallback"])
                self.last_report = {
                    "counts": dict(frame["counts"]),
                    "verified_layouts": len(self.verified),
                    "verification_blocks": frame.get("verification_blocks", 0),
                }
                if "fallback" in frame:
                    self.last_report["fallback"] = frame["fallback"]
                self.totals.update(frame["counts"])
                self.totals.update(forwards=1, fallback_forwards=int("fallback" in frame),
                                   verification_blocks=frame.get("verification_blocks", 0))
                return result
            finally:
                self.context.reset(token)

    @staticmethod
    def projection(linear, hidden, gate, rows, residual, input_act="none"):
        import torch
        import comfy.ops
        import comfy_kitchen as ck
        from comfy_kitchen.backends import cuda
        from comfy_kitchen.tensor.base import QuantizedTensor
        from comfy_kitchen.tensor.int8 import TensorWiseINT8Layout
        # Same offload ownership and want_requant policy as linear_input_act.
        with comfy.ops.CastBiasWeightContext(linear, hidden, offloadable=True,
                compute_dtype=hidden.dtype, want_requant=True) as (weight, bias):
            if not isinstance(weight, QuantizedTensor) or weight._layout_cls != "TensorWiseINT8Layout" or bias is not None:
                raise Unsupported("Weight cast no longer supplies INT8 ConvRot weights")
            w, ws = TensorWiseINT8Layout.get_plain_tensors(weight)
            q, qs = cuda.quantize_int8_rowwise_convrot64(hidden, 256, input_act=input_act)
            ws = ws.to(device=hidden.device, dtype=torch.float32).reshape(-1)
            if ws.numel() == 1:
                ws = ws.expand(w.shape[0])
            return ck.int8_gemm_indexed_gate(q, w.contiguous(), qs,
                ws.contiguous(),
                gate.to(dtype=torch.bfloat16).contiguous(), rows, residual)

    @staticmethod
    def attention_input(attn, x, rope, options):
        import comfy.model_management as mm
        import comfy.quant_ops
        from comfy.ldm.minimax import model as h3
        s = len(x)
        q, k, v = attn.qkv_proj(x).split(attn.heads * attn.head_dim, dim=-1)
        v = v.view(s, attn.heads, attn.head_dim)
        if rope is not None:
            q = q.view(1, s, attn.heads, attn.head_dim)
            k = k.view(1, s, attn.heads, attn.head_dim)
            qw = mm.cast_to(attn.q_norm.weight, device=x.device)
            kw = mm.cast_to(attn.k_norm.weight, device=x.device)
            comfy.quant_ops.ck.rms_rope_split_half_(q, k, rope, qw, kw,
                epsilon=attn.q_norm.eps, rot_dim=rope.shape[-3] * 2)
            q, k = q[0], k[0]
        else:
            q = attn.q_norm(q.view(s, attn.heads, attn.head_dim))
            k = attn.k_norm(k.view(s, attn.heads, attn.head_dim))
        v = v.clone()
        q, k, v = (h3.AttentionTensorContainer(t.transpose(0, 1).unsqueeze(0)) for t in (q, k, v))
        return h3.optimized_attention(q, k, v, attn.heads, mask=None,
            skip_reshape=True, transformer_options=options).squeeze(0)

    def block(self, block, inputs, rows):
        from comfy.ldm.minimax import model as h3
        x, segments = inputs["img"], inputs["mod_segments"]
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = block.adaln_proj(inputs["t_emb"])
        h = h3._mod_scale_shift(block.norm1(x), shift_a, scale_a, segments)
        h = self.attention_input(block.attn, h, inputs["rope_freqs"], inputs["transformer_options"])
        x = self.projection(block.attn.out_proj, h, gate_a, rows, x)
        h = h3._mod_scale_shift(block.norm2(x), shift_m, scale_m, segments)
        return self.projection(block.mlp.fc2, block.mlp.fc1(h), gate_m, rows, x, "swiglu")

    def replacement(self, index):
        def run(inputs, extra):
            import torch
            frame = self.context.get()
            original = extra["original_block"]
            if frame is None or "fallback" in frame:
                return original(inputs)
            x = inputs["img"]
            try:
                if x.dtype != torch.bfloat16 or not x.is_contiguous():
                    raise Unsupported("Packed hidden states must be contiguous BF16")
                key = segment_key(inputs["mod_segments"], len(x), inputs["t_emb"].shape[0] * 3)
                if key in self.rejected:
                    raise Unsupported("Layout previously failed exact verification")
                if frame["row_key"] != key:
                    rows = torch.empty(len(x), dtype=torch.int32, device=x.device)
                    for start, stop, row in key:
                        rows[start:stop].fill_(row)
                    frame["rows"], frame["row_key"] = rows, key
                # Both fused projections return fresh tensors; inputs['img'] remains intact.
                out = self.block(self.model.blocks[index], inputs, frame["rows"])
            except Unsupported as error:
                frame["fallback"] = str(error)
                return original(inputs)
            if self.verify_new_shapes and key not in self.verified:
                reference = original(inputs)["img"]
                frame["verification_blocks"] = frame.get("verification_blocks", 0) + 1
                if not torch.equal(out, reference):
                    self.rejected.add(key)
                    frame["fallback"] = f"Exact verification failed at block {index}"
                    return {"img": reference}
                frame["verified_keys"].add(key)
            frame["counts"].update(blocks=1, outproj=1, fc2=1)
            return {"img": out}
        return run


def patch_model(model, *, verify_new_shapes=True):
    """Clone MODEL and install only PR #219; never patch shared module methods."""
    import comfy_kitchen as ck
    import comfy.patcher_extension as pe
    from comfy.ldm.minimax import model as h3
    if not hasattr(ck, "int8_gemm_indexed_gate"):
        raise Unsupported("Build Kitchen PR #219 first; see docs/INDEXED-GATE.md")
    source = Path(inspect.getsourcefile(h3.MiniMaxH3Model))
    if hashlib.sha256(source.read_bytes()).hexdigest() != MODEL_SHA:
        raise Unsupported("Use the ComfyUI revision pinned in configs/compatibility.json")
    clone = model.clone()
    options = clone.model_options.setdefault("transformer_options", {})
    existing = options.get("patches_replace", {}).get("dit", {})
    if any(("double_block", i) in existing for i in range(50)):
        raise Unsupported("Start from an unpatched MODEL; do not stack this node with Optimize DiT")
    patch = IndexedGatePatch(model.model.diffusion_model, verify_new_shapes=verify_new_shapes)
    for index in range(50):
        clone.set_model_patch_replace(patch.replacement(index), "dit", "double_block", index)
    pe.add_wrapper_with_key(pe.WrappersMP.DIFFUSION_MODEL, KEY, patch.wrapper,
                            clone.model_options["transformer_options"])
    clone._h3_indexed_gate = patch
    return clone
