# SPDX-License-Identifier: GPL-3.0-or-later
# Exact helpers from ComfyUI b2e31e89412a01a67be599571cc57ff74b242a82.
import torch
import comfy.ops
import comfy.model_management
import comfy.rmsnorm
from comfy.ops import INPUT_ACT_EAGER, QuantizedTensor, TensorWiseINT8Layout, cast_bias_weight, uncast_bias_weight
from comfy import quant_ops
def _eager_input_act(x, input_act, act_weight=None, act_eps=0.0):
    if input_act is None:
        return x
    if input_act == "rms_norm":
        return comfy.rmsnorm.rms_norm(x, act_weight, act_eps)
    return INPUT_ACT_EAGER[input_act](x)


def _fp16_linear_wanted(x):
    """kitchen's fp16-accumulate GEMM replaces a plain linear when the user opted into
        fp16 accumulation and the activation is fp16 on CUDA; weights come through cast_bias_weight."""
    return (getattr(torch.backends.cuda.matmul, "allow_fp16_accumulation", False)
            and x.dtype == torch.float16 and x.is_cuda)


def linear_input_act(linear, x, input_act, act_weight=None, act_eps=0.0,
                     residual=None, residual_scale=None):
    """``linear(act(x))``, with ``act`` folded into an INT8 activation quantizer.

    An INT8 linear quantizes its input anyway, so an elementwise activation can
    ride along inside that kernel instead of writing a full-size intermediate to
    HBM and reading it straight back. Worth it for an MLP's down-projection,
    where the intermediate is several times the hidden size, and for a pre-norm
    block's ``linear(rms_norm(x))`` ("rms_norm", which reads the norm's weight
    and eps from act_weight/act_eps).

    With ``residual``/``residual_scale`` the result is the pre-norm block's
    addcmul, ``residual + residual_scale * linear(act(x))``, fused into the
    INT8 GEMM epilogue where supported.

    """
    def _residual_out(out):
        if residual is None:
            return out
        return torch.addcmul(residual, out, residual_scale)

    weight = linear.weight
    full_precision_mm = getattr(linear, "_full_precision_mm", False)
    if (comfy.model_management.in_training
            or not isinstance(weight, QuantizedTensor)
            or weight._layout_cls != "TensorWiseINT8Layout"
            or getattr(weight._params, "transposed", False)
            or full_precision_mm):
        if (not comfy.model_management.in_training
                and not isinstance(weight, QuantizedTensor)
                and not full_precision_mm
                and _fp16_linear_wanted(x)):
            weight, bias, offload_stream = cast_bias_weight(linear, x, offloadable=True)
            try:
                return quant_ops.ck.fp16_linear(
                    _eager_input_act(x, input_act, act_weight, act_eps),
                    weight, bias, residual=residual, residual_scale=residual_scale)
            finally:
                uncast_bias_weight(linear, weight, bias, offload_stream)
        return _residual_out(linear(_eager_input_act(x, input_act, act_weight, act_eps)))

    # want_requant keeps a vbar-streamed layer on the INT8 path when a LoRA is
    # patched in on the fly; without it the cast hands back a dequantized weight.
    weight, bias, offload_stream = cast_bias_weight(
        linear, x, offloadable=True, compute_dtype=x.dtype, want_requant=True)
    try:
        if not isinstance(weight, QuantizedTensor):
            # A LoRA weight_function, or activations whose dtype differs from the
            # weight's, make the cast hand back a dequantized tensor.
            return _residual_out(torch.nn.functional.linear(
                _eager_input_act(x, input_act, act_weight, act_eps), weight, bias))
        qdata, scale = TensorWiseINT8Layout.get_plain_tensors(weight)
        return quant_ops.ck.int8_linear(
            x, qdata, scale, bias, x.dtype,
            convrot=getattr(weight._params, "convrot", False),
            convrot_groupsize=getattr(weight._params, "convrot_groupsize", 256),
            input_act=input_act,
            input_act_weight=act_weight,
            input_act_eps=act_eps,
            residual=residual,
            residual_scale=residual_scale,
        )
    finally:
        uncast_bias_weight(linear, weight, bias, offload_stream)


def install():
    comfy.ops.linear_input_act = linear_input_act
    comfy.ops._fp16_linear_wanted = _fp16_linear_wanted
