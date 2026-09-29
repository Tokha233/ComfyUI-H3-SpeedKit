"""Preallocated residual/norm2/mod -> ConvRot ABI; X is updated in place.

The bound object owns all original Torch tensors. Rows are immutable after
validation; callers order producers before bind/launch and before crossing
streams. Returned Q and FP32 scales are prequantized Linear input, never BF16.
"""
from __future__ import annotations

import ctypes as ct
import math
from pathlib import Path

import torch

K = 5376
TORCH_COMMIT = '7661cd9c6b841b62b7f411aa52ec51f05457263b'


class Resources(ct.Structure):
    _fields_ = [(name, ct.c_int64) for name in (
        'abi', 'bytes', 'k', 'threads', 'warps', 'num_regs',
        'static_shared_bytes', 'local_bytes', 'active_blocks_per_sm',
        'sm_count', 'max_threads_per_sm', 'binary_version', 'ptx_version')]


def span(tensor):
    """Conservative live address interval, including strided row padding."""
    end = 1 + sum((size - 1) * stride for size, stride in zip(tensor.shape, tensor.stride()))
    return tensor.data_ptr(), tensor.data_ptr() + end * tensor.element_size()


def check_disjoint(inputs, outputs):
    for name, tensor in outputs:
        left, right = span(tensor)
        for other_name, other in inputs:
            a, b = span(other)
            if left < b and a < right:
                raise ValueError(f'Output {name} overlaps input {other_name}')
    for i, (name, tensor) in enumerate(outputs):
        left, right = span(tensor)
        for other_name, other in outputs[:i]:
            a, b = span(other)
            if left < b and a < right:
                raise ValueError(f'Outputs {name} and {other_name} overlap')


def finite_bounded(tensor):
    if tensor.ndim < 2:
        if not bool(torch.isfinite(tensor).all().item()):
            raise ValueError('Nonfinite input is outside the contract')
    else:
        for start in range(0, tensor.shape[0], 128):
            if not bool(torch.isfinite(tensor[start:start+128]).all().item()):
                raise ValueError('Nonfinite input is outside the contract')


def validate(x, branch, weight, gate, shift, scale, rows, q, q_scale, *, eps,
             debug_norm=None, debug_rotated=None, debug_rstd=None,
             check_values=True, require_cuda=True):
    """Pure metadata checks can be exercised on CPU with require_cuda=False."""
    inputs = [('branch', branch), ('weight', weight), ('gate', gate), ('shift', shift), ('scale', scale), ('rows', rows)]
    outputs = [('x', x), ('q', q), ('q_scale', q_scale)]
    debug = (debug_norm, debug_rotated, debug_rstd)
    if any(t is not None for t in debug):
        if any(t is None for t in debug):
            raise ValueError('Debug requires norm, rotated, and rstd outputs together')
        outputs.extend(zip(('debug_norm', 'debug_rotated', 'debug_rstd'), debug))
    if any(type(t) is not torch.Tensor for _, t in inputs+outputs):
        raise ValueError('All operands must be original plain Torch tensors')
    if (require_cuda and not x.is_cuda) or any(t.device != x.device for _, t in inputs+outputs):
        raise ValueError('All operands must use one CUDA device')
    if any(t.requires_grad for _, t in inputs+outputs):
        raise ValueError('Inference only; detach only after validating model autograd policy')
    if (x.dtype != torch.bfloat16 or x.ndim != 2 or x.shape[1] != K
            or not 1 <= x.shape[0] <= 0x7fffffff or not x.is_contiguous()
            or x.stride(1) != 1 or (x.shape[0] > 1 and x.stride(0) != K)
            or x.data_ptr() % 8):
        raise ValueError('X requires aligned contiguous BF16 [M,5376]')
    if (branch.shape != x.shape or branch.dtype != torch.bfloat16
            or not branch.is_contiguous() or branch.stride(1) != 1
            or (branch.shape[0] > 1 and branch.stride(0) != K) or branch.data_ptr() % 8):
        raise ValueError('Branch requires aligned contiguous BF16 [M,5376]')
    m = x.shape[0]
    def dense(name, t, shape, dtype, alignment):
        if t.shape != shape or t.dtype != dtype or not t.is_contiguous() or t.data_ptr() % alignment:
            raise ValueError(f'{name} requires contiguous {dtype} {tuple(shape)} and {alignment}B alignment')
    dense('weight', weight, (K,), torch.bfloat16, 8)
    if (shift.ndim != 2 or shift.shape[1] != K or shift.shape[0] < 1
            or shift.shape[0] > 0x7fffffff or scale.shape != shift.shape or shift.dtype != scale.dtype
            or shift.dtype not in (torch.bfloat16, torch.float32)):
        raise ValueError('Tables require matching nonempty FP32/BF16 [T,5376]')
    if gate.shape != shift.shape or gate.dtype not in (torch.bfloat16, torch.float32):
        raise ValueError('Gate requires matching table shape and independent FP32/BF16 dtype')
    for t in (gate, shift, scale):
        if t.stride(1) != 1 or t.stride(0) < K or t.data_ptr() % t.element_size():
            raise ValueError('Tables require dense hidden axes and nonoverlapping rows')
    dense('rows', rows, (m,), torch.int32, 4)
    dense('q', q, (m, K), torch.int8, 8)
    dense('q_scale', q_scale, (m, 1), torch.float32, 4)
    if debug_norm is not None:
        dense('debug_norm', debug_norm, (m, K), torch.bfloat16, 2)
        dense('debug_rotated', debug_rotated, (m, K), torch.float32, 4)
        dense('debug_rstd', debug_rstd, (m,), torch.float32, 4)
    if not math.isfinite(float(eps)) or not torch.finfo(torch.float32).tiny <= float(eps) <= torch.finfo(torch.float32).max:
        raise ValueError('Epsilon must be finite, positive, normal FP32')
    check_disjoint(inputs, outputs)
    if check_values:
        if int(rows.min().item()) < 0 or int(rows.max().item()) >= shift.shape[0]:
            raise ValueError('Rows outside modulation table')
        for tensor in (x, branch, weight, gate, shift, scale):
            finite_bounded(tensor)
    return tuple(inputs), tuple(outputs)


class ResidualQuant:
    def __init__(self, library):
        self.path = Path(library).resolve()
        self.lib = ct.CDLL(str(self.path), mode=ct.RTLD_LOCAL)
        specs = {
            'abi': ([], ct.c_int), 'resources_size': ([], ct.c_size_t),
            'error': ([ct.c_int], ct.c_char_p),
            'query': ([ct.c_int, ct.c_int, ct.c_int, ct.POINTER(Resources), ct.c_size_t], ct.c_int),
            'run': ([ct.c_void_p]*12 + [ct.c_int64]*4 + [ct.c_float, ct.c_int, ct.c_int, ct.c_int, ct.c_size_t], ct.c_int)}
        for suffix, (arguments, result) in specs.items():
            fn = getattr(self.lib, 'h3_residual_quant_'+suffix)
            fn.argtypes, fn.restype = arguments, result
        if self.lib.h3_residual_quant_abi() != 1 or self.lib.h3_residual_quant_resources_size() != ct.sizeof(Resources):
            raise RuntimeError('ResidualQuant C/ctypes ABI mismatch')

    def check(self, code):
        if code:
            raise RuntimeError(f'ResidualQuant CUDA error {code}: '+self.lib.h3_residual_quant_error(code).decode())

    def query(self, gate_dtype=0, table_dtype=0, debug=0):
        data = Resources()
        self.check(self.lib.h3_residual_quant_query(gate_dtype, table_dtype, debug, ct.byref(data), ct.sizeof(data)))
        return {name: int(getattr(data, name)) for name, _ in data._fields_}

    def bind(self, x, branch, weight, gate, shift, scale, rows, q, q_scale, *, eps,
             debug_norm=None, debug_rotated=None, debug_rstd=None, check_values=True):
        inputs, outputs = validate(x, branch, weight, gate, shift, scale, rows, q, q_scale, eps=eps,
            debug_norm=debug_norm, debug_rotated=debug_rotated, debug_rstd=debug_rstd,
            check_values=check_values)
        with torch.cuda.device(x.device):
            if torch.cuda.get_device_capability(x.device) != (12, 0):
                raise RuntimeError('Only the intended SM120 device is supported')
            if torch.compiler.is_compiling() or torch.cuda.is_current_stream_capturing():
                raise RuntimeError('ResidualQuant qualification requires eager non-captured execution')
        pointers = [t.data_ptr() for t in (x, branch, weight, gate, shift, scale, rows, q, q_scale)]
        pointers += [t.data_ptr() if t is not None else 0 for t in (debug_norm, debug_rotated, debug_rstd)]
        args = pointers + [x.shape[0], gate.stride(0), shift.stride(0), scale.stride(0),
                           float(eps), int(gate.dtype == torch.bfloat16), int(shift.dtype == torch.bfloat16), int(debug_norm is not None)]
        return Bound(self, x.device, inputs, outputs, args, (x, q, q_scale))


class Bound:
    def __init__(self, library, device, inputs, outputs, args, result):
        self.library, self.device, self.inputs, self.outputs = library, device, inputs, outputs
        self.args, self.result = args, result
        self.owners = tuple(t for _, t in inputs+outputs)

    def __call__(self):
        with torch.cuda.device(self.device):
            stream = torch.cuda.current_stream(self.device)
            for owner in self.owners:
                owner.record_stream(stream)
            code = self.library.lib.h3_residual_quant_run(*self.args, stream.cuda_stream)
            # Raw stores may be queued before a reported CUDA error: no retry.
            torch.autograd.graph.increment_version(tuple(t for _, t in self.outputs))
            self.library.check(code)
        return self.result
