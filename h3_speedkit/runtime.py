# SPDX-License-Identifier: GPL-3.0-or-later
# The H3 block sequence follows ComfyUI's MiniMax H3 implementation.
"""Model-scoped Kitchen 0.2.36 R85 backend. No global function replacement."""
from collections import Counter
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
import ctypes as ct
import hashlib
import importlib.metadata
import inspect
import json
import logging
import threading
import types

import torch
from comfy_kitchen.backends import cuda
from comfy_kitchen.tensor.int8 import TensorWiseINT8Layout
from comfy_kitchen.tensor.base import QuantizedTensor
import comfy.ops
import comfy.model_management as mm
import comfy.patcher_extension as pe
from .ops.norm_quant import NormQuant, TORCH_COMMIT
from .ops.convrot import ConvRotRegister
from .ops.dense import DenseLibrary, DenseArgs, allocate_output
from .ops.qkv import QKV

LOG = logging.getLogger(__name__)
MODEL_SHA = "26ae72f5834cc9038a38600cd12c72d28da785cdaacd578d5b042b58ad48b09b"
QUALIFIED_ROWS = frozenset((90461, 87142, 82209, 82022, 80060, 67744, 79954,
                          80057, 84085, 84251, 84259, 90300, 90495, 90570, 99409))
KEY = "h3_speedkit_r85"


class Unsupported(ValueError):
    """A contract failed before any in-place model output was changed."""


class Backend:
    def __init__(self, directory=None):
        self.directory = Path(directory) if directory else Path(__file__).parent / "_native"
        if importlib.metadata.version("comfy-kitchen") != "0.2.36":
            raise Unsupported("Install comfy-kitchen==0.2.36 for this release")
        if torch.version.git_version != TORCH_COMMIT:
            raise Unsupported("This norm reduction is qualified for PyTorch 2.12.0 cu130; see configs/compatibility.json")
        if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
            raise Unsupported("This build requires an SM120 GPU")
        build = json.loads((self.directory / "build.json").read_text())
        for name in ("norm", "convrot", "qkv", "sample", "anchor", "dense", "fc1", "gate"):
            p = self.directory / (name + ".so")
            if hashlib.sha256(p.read_bytes()).hexdigest() != build["kernels"][name]["sha256"]:
                raise Unsupported("Kernel differs from its build manifest: " + name)
        self.norm = NormQuant(self.directory / "norm.so")
        self.convrot = ConvRotRegister(self.directory / "convrot.so")
        self.qkv = QKV(self.directory)
        self.dense = DenseLibrary(self.directory / "dense.so")
        self.window = self.dense.lib.h3_dense_launch_window
        self.window.argtypes = [ct.POINTER(DenseArgs), ct.c_size_t, ct.c_uint32]
        self.window.restype = ct.c_int
        self.libs = [ct.CDLL(str(self.directory / (name + ".so"))) for name in ("fc1", "gate")]
        self.fc1 = self.libs[0].h3_int8
        self.fc1.argtypes = [ct.c_void_p] * 5 + [ct.c_int] * 4 + [ct.c_size_t]
        self.gate = self.libs[1].h3_int8_gate
        self.gate.argtypes = [ct.c_void_p] * 8 + [ct.c_int] * 4 + [ct.c_size_t]
        self.fc1.restype = self.gate.restype = ct.c_int

    @staticmethod
    def check(code):
        if code != 1:
            raise RuntimeError("SpeedKit GEMM failed; the request is aborted, never retried after mutation")

    @staticmethod
    def touch(*operands):
        stream = torch.cuda.current_stream(operands[0].device)
        for tensor in operands:
            tensor.record_stream(stream)
        return stream.cuda_stream

    @staticmethod
    def validate_linear(module, n, k):
        weight = module.weight
        if not isinstance(weight, QuantizedTensor) or weight._layout_cls != "TensorWiseINT8Layout":
            raise Unsupported("Requires INT8 ConvRot weights; load the INT8 checkpoint first")
        params = weight._params
        if (getattr(params, "transposed", False) or not getattr(params, "convrot", False)
                or getattr(params, "convrot_groupsize", 256) != 256):
            raise Unsupported("Requires untransposed ConvRot256 weights")
        if tuple(weight.shape) != (n, k) or module.bias is not None:
            raise Unsupported("Unsupported linear geometry or bias")
        if (module.weight_function or module.bias_function or getattr(module, "pre_quant_scale", None) is not None
                or getattr(module, "_full_precision_mm", False) or getattr(module, "comfy_force_cast_weights", False)):
            raise Unsupported("Active weight callbacks, smoothing or full-precision override")
        if any(getattr(module, key, {}) for key in ("_forward_hooks", "_forward_pre_hooks", "_backward_hooks")):
            raise Unsupported("Custom module hooks")

    @contextmanager
    def weights(self, module, x):
        # Preserve Comfy's weight prefetch/offload ownership and LoRA cast behavior.
        with comfy.ops.CastBiasWeightContext(module, x, offloadable=True) as (weight, bias):
            if not isinstance(weight, QuantizedTensor) or weight._layout_cls != "TensorWiseINT8Layout" or bias is not None:
                raise RuntimeError("Weight policy changed after preflight")
            w, scale = TensorWiseINT8Layout.get_plain_tensors(weight)
            w, scale = w.contiguous(), scale.to(device=x.device, dtype=torch.float32).contiguous()
            if w.dtype != torch.int8 or scale.numel() != w.shape[0] or w.device != x.device:
                raise RuntimeError("Unexpected cast weight layout")
            yield w, scale

    def normalize(self, module, x, shift, scale, rows):
        weight = mm.cast_to(module.weight, device=x.device).detach()
        q = torch.empty(x.shape, device=x.device, dtype=torch.int8)
        qs = torch.empty((len(x), 1), device=x.device, dtype=torch.float32)
        self.norm.bind(x, weight, shift, scale, rows, q, qs,
                       eps=module.eps, check_values=False)()
        return q, qs

    def gate_projection(self, linear, q, qs, x, gate, rows, skip):
        with self.weights(linear, x) as (w, ws):
            g = gate.to(dtype=torch.bfloat16).contiguous()
            operands = (q[skip:], w, qs[skip:], ws, x[skip:], g, rows[skip:], x[skip:])
            st = self.touch(*operands)
            self.check(self.gate(*(t.data_ptr() for t in operands), len(x) - skip,
                                 5376, w.shape[1], 13, st))
            torch.autograd.graph.increment_version(x)

    def block(self, block, x, t_emb, segments, rope, *, rows, skip, counts):
        # All four projections and norm contracts were checked before this path.
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = block.adaln_proj(t_emb)
        qi, qs = self.normalize(block.norm1, x, shift_msa, scale_msa, rows)
        with self.weights(block.attn.qkv_proj, x) as (w, ws):
            qw = mm.cast_to(block.attn.q_norm.weight, device=x.device).contiguous()
            kw = mm.cast_to(block.attn.k_norm.weight, device=x.device).contiguous()
            packed = self.qkv(qi, w, qs, ws, rope, qw, kw, float(block.attn.q_norm.eps))
        del qi, qs
        output = allocate_output(packed, "bshd")
        bound = self.dense.bind(packed, output)
        st = self.touch(packed.q, packed.k, packed.v, packed.q_scale,
                        packed.k_scale, packed.v_scale, output)
        rc = self.window(bound._params_ptr, st, skip // 128)
        if rc:
            raise RuntimeError(self.dense.lib.h3_dense_error_string(rc).decode())
        del packed, bound
        attention = output.transpose(1, 2).reshape(len(x), 7168)
        aq = torch.empty(attention.shape, device=x.device, dtype=torch.int8)
        asc = torch.empty((len(x), 1), device=x.device, dtype=torch.float32)
        self.convrot.bind(attention, aq, asc, check_finite=False)()
        self.gate_projection(block.attn.out_proj, aq, asc, x, gate_msa, rows, skip)
        del output, attention, aq, asc
        qi, qs = self.normalize(block.norm2, x, shift_mlp, scale_mlp, rows)
        hidden = torch.empty((len(x), 28672), device=x.device, dtype=torch.bfloat16)
        if skip:
            hidden[:skip].zero_()
        with self.weights(block.mlp.fc1, x) as (w, ws):
            operands = (qi[skip:], w, qs[skip:], ws, hidden[skip:])
            self.check(self.fc1(*(t.data_ptr() for t in operands), len(x) - skip,
                                28672, 5376, 3, self.touch(*operands)))
        del qi, qs
        # Keep upstream SwigLU arithmetic and rounding barriers.
        fq, fs = cuda.quantize_int8_rowwise_convrot64(hidden, 256, input_act="swiglu")
        self.gate_projection(block.mlp.fc2, fq, fs, x, gate_mlp, rows, skip)
        counts.update(qkv=1, attention=1, norm_quant=2, fc1=1, gate=2)
        if skip:
            counts.update(last_block_window=1)
        return x


class ModelPatch:
    def __init__(self, model, backend, verify_new_shapes=True):
        self.model = model
        self.backend = backend
        self.context = ContextVar("speedkit_request", default=None)
        self.lock = threading.RLock()
        self.last_report = {}
        self.warned = set()
        self.verify_new_shapes = verify_new_shapes
        self.verified_shapes = Counter()
        self.rejected_shapes = set()

    def preflight(self, args, kwargs):
        from comfy.ldm.minimax import model as h3
        source = Path(inspect.getsourcefile(h3.MiniMaxH3Model))
        if hashlib.sha256(source.read_bytes()).hexdigest() != MODEL_SHA:
            raise Unsupported("ComfyUI H3 source differs from the validated revision")
        if type(self.model) is not h3.MiniMaxH3Model or len(self.model.blocks) != 50:
            raise Unsupported("Requires the 50-block MiniMax H3 model")
        if torch.is_grad_enabled() or mm.in_training or torch.compiler.is_compiling():
            raise Unsupported("Eager inference only")
        x = args[0][0]
        if not x.is_cuda or torch.cuda.get_device_capability(x.device) != (12, 0):
            raise Unsupported("SM120 GPU required")
        if torch.cuda.is_current_stream_capturing():
            raise Unsupported("CUDA graph capture is not qualified")
        payload = kwargs.get("minimax_payload") or (args[4] if len(args) > 4 else None) or {}
        layout = payload.get("layout")
        if layout is None:
            raise Unsupported("Missing prepared H3 layout")
        if not 1025 <= layout.seq_len <= 110000:
            raise Unsupported("Packed token count is outside the tested kernel allocation envelope")
        if layout.seq_len not in QUALIFIED_ROWS and not self.verify_new_shapes:
            raise Unsupported("New layout verification is disabled")
        if (kwargs.get("denoise_mask", args[5] if len(args) > 5 else None) is not None
                or kwargs.get("audio_denoise_mask", args[6] if len(args) > 6 else None) is not None):
            raise Unsupported("Masked generation uses the original model")
        targets = [(a, b) for a, b, kind in layout.segments if kind in ("video", "audio")]
        if not targets:
            raise Unsupported("Missing target video/audio spans")
        skip = min(a for a, _ in targets) // 128 * 128
        shape_key = (layout.seq_len, skip)
        if shape_key in self.rejected_shapes:
            raise Unsupported("This layout failed the exact reference comparison")
        if len(targets) != 2 or targets[-1][1] != layout.seq_len or targets[0][1] != targets[1][0]:
            raise Unsupported("Final-layer window requires contiguous final audio/video spans")
        if type(self.model.final_layer) is not h3.FinalLayer or "forward" in self.model.final_layer.__dict__:
            raise Unsupported("Custom final-layer implementation")
        if any(getattr(self.model.final_layer, name, {}) for name in ("_forward_hooks", "_forward_pre_hooks")):
            raise Unsupported("Final-layer hooks could observe omitted rows")
        if any(getattr(torch.nn.modules.module, name, {}) for name in ("_global_forward_hooks", "_global_forward_pre_hooks")):
            raise Unsupported("Global module hooks are active")
        for block in self.model.blocks:
            if type(block) is not h3.DiTBlock or type(block.attn) is not h3.Attention or type(block.mlp) is not h3.MLP:
                raise Unsupported("Custom block implementation")
            for obj in (block, block.attn, block.mlp, block.norm1, block.norm2, block.adaln_proj):
                if "forward" in obj.__dict__ or getattr(obj, "_compiled_call_impl", None) is not None:
                    raise Unsupported("Custom or compiled module forward")
                if any(getattr(obj, name, {}) for name in ("_forward_hooks", "_forward_pre_hooks")):
                    raise Unsupported("Custom block hooks")
            for obj, n, k in ((block.attn.qkv_proj, 21504, 5376), (block.attn.out_proj, 5376, 7168),
                              (block.mlp.fc1, 28672, 5376), (block.mlp.fc2, 5376, 14336)):
                self.backend.validate_linear(obj, n, k)
            for norm in (block.norm1, block.norm2):
                if norm.weight.dtype != torch.bfloat16 or tuple(norm.weight.shape) != (5376,):
                    raise Unsupported("Norm weights must be BF16 width 5376")
            if block.attn.heads != 56 or block.attn.head_dim != 128 or block.attn.q_norm.eps != block.attn.k_norm.eps:
                raise Unsupported("Unsupported attention geometry or normalization epsilon")
        options = kwargs.get("transformer_options", args[3] if len(args) > 3 else {})
        if options.get("optimized_attention_override") is not None and not options.get("_h3sla_dense", False):
            raise Unsupported("Competing attention override")
        return {"skip": skip, "rows": None, "row_key": None, "counts": Counter(), "tokens": layout.seq_len,
                "shape_key": shape_key,
                "verify": layout.seq_len not in QUALIFIED_ROWS and self.verified_shapes[shape_key] < 8}

    def wrapper(self, executor, *args, **kwargs):
        with self.lock:
            try:
                frame = self.preflight(args, kwargs)
            except Unsupported as error:
                reason = str(error)
                if reason not in self.warned:
                    LOG.warning("H3 SpeedKit fallback: %s", reason)
                    self.warned.add(reason)
                frame = {"fallback": reason}
            token = self.context.set(frame)
            try:
                result = executor(*args, **kwargs)
                if "fallback" not in frame and frame["counts"]["attention"] != 50:
                    raise RuntimeError("Incomplete SpeedKit block coverage")
                if frame.get("verify") and "verification_failed" not in frame:
                    self.verified_shapes[frame["shape_key"]] += 1
                    LOG.info("H3 SpeedKit: verified all 50 blocks, layout=%s, forward=%s/8",
                             frame["shape_key"], self.verified_shapes[frame["shape_key"]])
                self.last_report = {k: dict(v) if isinstance(v, Counter) else v for k, v in frame.items()
                                    if k not in ("rows", "row_key")}
                return result
            finally:
                self.context.reset(token)

    def replacement(self, index):
        def run(inputs, extra):
            frame = self.context.get()
            if frame is None or "fallback" in frame or "verification_failed" in frame:
                return extra["original_block"](inputs)
            x, segments = inputs["img"], inputs["mod_segments"]
            if x.dtype != torch.bfloat16 or not x.is_contiguous() or len(x) != frame["tokens"]:
                raise RuntimeError("Hidden activation changed after preflight")
            options = inputs["transformer_options"]
            if options.get("optimized_attention_override") is not None and not options.get("_h3sla_dense", False):
                raise RuntimeError("SpeedKit supports dense attention; sparse overrides must be removed")
            key = []
            cursor = 0
            for start, stop, row in segments:
                if type(row) is not int or start != cursor or stop < start or not 0 <= row < inputs["t_emb"].shape[0] * 3:
                    raise RuntimeError("Only contiguous integer modulation segments are qualified")
                key.append((start, stop, row));cursor = stop
            if cursor != len(x):
                raise RuntimeError("Incomplete modulation row map")
            key = tuple(key)
            if frame["row_key"] != key:
                rows = torch.empty((len(x),), device=x.device, dtype=torch.int32)
                for start, stop, row in key:
                    rows[start:stop].fill_(row)
                frame["rows"], frame["row_key"] = rows, key
            work = x.clone() if frame["verify"] else x
            # Original _mod_gate updates x in place: save the candidate input first.
            reference = extra["original_block"](inputs)["img"] if frame["verify"] else None
            output = self.backend.block(self.model.blocks[index], work, inputs["t_emb"], segments,
                inputs["rope_freqs"], rows=frame["rows"], skip=frame["skip"] if index == 49 else 0,
                counts=frame["counts"])
            if reference is not None:
                start = frame["skip"] if index == 49 else 0
                if not torch.equal(output[start:], reference[start:]):
                    self.rejected_shapes.add(frame["shape_key"])
                    frame["verification_failed"] = index
                    # Candidate only mutated its private clone. Original result is intact.
                    frame["fallback"] = "Exact layout verification failed at block " + str(index)
                    LOG.warning("H3 SpeedKit: %s; using reference", frame["fallback"])
                    return {"img": reference}
            return {"img": output}
        return run


def patch_model(model, directory=None, *, release_embeddings=True, verify_new_shapes=True):
    """Return a Comfy clone with explicit block replacements and a request wrapper."""
    clone = model.clone()
    options = clone.model_options.setdefault("transformer_options", {})
    existing = options.get("patches_replace", {}).get("dit", {})
    if any(("double_block", i) in existing for i in range(50)):
        raise Unsupported("A DiT block replacement is already installed; start from the unpatched MODEL output")
    from comfy.ldm.minimax import model as h3
    source = Path(inspect.getsourcefile(h3.MiniMaxH3Model))
    if hashlib.sha256(source.read_bytes()).hexdigest() != MODEL_SHA:
        raise Unsupported("Use the ComfyUI H3 revision pinned in configs/compatibility.json")
    if options.get("optimized_attention_override") is not None and not options.get("_h3sla_dense", False):
        raise Unsupported("Remove the competing attention patch before adding SpeedKit")
    from comfy.ldm.modules.attention import attention_comfy_kitchen_int8
    # Explicit model-local INT8 precision contract, including reference/fallback blocks.
    clone.set_model_optimized_attention(attention_comfy_kitchen_int8)
    options["_h3sla_dense"] = True
    patch = ModelPatch(model.model.diffusion_model, Backend(directory), verify_new_shapes)
    if release_embeddings:
        from .vendor.model_forward import _forward
        # Comfy owns install/restore with its normal model-patch lifecycle.
        clone.add_object_patch("diffusion_model._forward", types.MethodType(_forward, patch.model))
    for index in range(50):
        clone.set_model_patch_replace(patch.replacement(index), "dit", "double_block", index)
    pe.add_wrapper_with_key(pe.WrappersMP.DIFFUSION_MODEL, KEY, patch.wrapper,
                            clone.model_options["transformer_options"])
    clone._h3_speedkit = patch
    return clone
