"""Preallocated, current-stream binding for the single ACT=none ConvRot candidate."""
from __future__ import annotations
import ctypes as ct
from pathlib import Path
import torch


class Resources(ct.Structure):
    _fields_ = [(n, ct.c_int64) for n in (
        'abi', 'bytes', 'k', 'threads', 'warps', 'num_regs', 'static_shared_bytes',
        'local_bytes', 'max_threads_per_block', 'active_blocks_per_sm',
        'sm_count', 'max_threads_per_sm', 'binary_version', 'ptx_version')]


def finite_bounded(x, rows=128):
    if rows < 1:
        raise ValueError('finite-check row chunk must be positive')
    for start in range(0, x.shape[0], rows):
        if not bool(torch.isfinite(x[start:start+rows]).all().item()):
            raise ValueError('NaN/Inf lies outside this finite-input replay contract')


def _validate(t, shape, dtype, device, name):
    if (not isinstance(t, torch.Tensor) or t.shape != shape or t.dtype != dtype
            or not t.is_cuda or t.device != device or not t.is_contiguous()):
        raise ValueError(f'{name} requires contiguous {dtype} {tuple(shape)} on {device}')
    if t.data_ptr() % t.element_size():
        raise ValueError(f'{name} is not element-aligned')


def _no_alias(tensors):
    spans = sorted((t.data_ptr(), t.data_ptr()+t.numel()*t.element_size(), n)
                   for n, t in tensors if t is not None)
    for left, right in zip(spans, spans[1:]):
        if left[1] > right[0]:
            raise ValueError(f'Overlapping buffers: {left[2]} and {right[2]}')


class ConvRotRegister:
    def __init__(self, library):
        self.path = Path(library).resolve()
        self.lib = ct.CDLL(str(self.path), mode=ct.RTLD_LOCAL)
        signatures = {
            'abi': ([], ct.c_int), 'resources_size': ([], ct.c_size_t),
            'error': ([ct.c_int], ct.c_char_p),
            'query': ([ct.c_int, ct.POINTER(Resources), ct.c_size_t], ct.c_int),
            'run': ([ct.c_void_p]*4 + [ct.c_int64, ct.c_int, ct.c_int, ct.c_size_t], ct.c_int)}
        for name, (args, result) in signatures.items():
            fn = getattr(self.lib, 'h3_convrot_register_'+name)
            fn.argtypes, fn.restype = args, result
        if self.lib.h3_convrot_register_abi() != 1:
            raise RuntimeError('Unsupported ConvRot register ABI')
        if self.lib.h3_convrot_register_resources_size() != ct.sizeof(Resources):
            raise RuntimeError('C/ctypes resource layout mismatch')

    def check(self, code):
        if code:
            raise RuntimeError(f'ConvRot CUDA error {code}: '+self.lib.h3_convrot_register_error(code).decode())

    def bind(self, x, q=None, scale=None, *, rotated=None, mode=0,
             act_code=0, check_finite=True):
        if (not isinstance(x, torch.Tensor) or not x.is_cuda or x.dtype != torch.bfloat16
                or x.ndim != 2 or not x.is_contiguous()):
            raise ValueError('X requires contiguous CUDA BF16 [M,K]')
        m, k = x.shape
        if m < 1 or m > 0x7fffffff or k not in (5376, 7168) or act_code != 0:
            raise ValueError('Only M=1..INT32_MAX, K=5376/7168, ACT=none is supported')
        if mode not in (0, 1, 2):
            raise ValueError('Mode must be production=0, diagnostic=1 or oracle=2')
        _validate(x, (m, k), torch.bfloat16, x.device, 'x')
        if mode != 2:
            _validate(q, (m, k), torch.int8, x.device, 'q')
            _validate(scale, (m, 1), torch.float32, x.device, 'scale')
        elif q is not None or scale is not None:
            raise ValueError('Oracle mode accepts only X and rotated output')
        if mode != 0:
            _validate(rotated, (m, k), torch.float32, x.device, 'rotated')
        elif rotated is not None:
            raise ValueError('Production mode does not accept rotated output')
        _no_alias([('x', x), ('q', q), ('scale', scale), ('rotated', rotated)])
        with torch.cuda.device(x.device):
            if torch.cuda.get_device_capability(x.device) != (12, 0):
                raise RuntimeError('This binary/experiment is restricted to SM120')
            if check_finite:
                finite_bounded(x)
            resource = Resources()
            self.check(self.lib.h3_convrot_register_query(k, ct.byref(resource), ct.sizeof(resource)))
        values = {n: int(getattr(resource, n)) for n, _ in resource._fields_}
        values['active_warps_per_sm'] = values['active_blocks_per_sm']*values['warps']
        values['theoretical_occupancy'] = (values['active_blocks_per_sm']*values['threads']
                                           / values['max_threads_per_sm'])
        return Bound(self, x, q, scale, rotated, mode, values)


class Bound:
    """Owns tensor refs and records allocator stream use; caller orders producers."""
    def __init__(self, library, x, q, scale, rotated, mode, resources):
        self.library, self.x, self.q, self.scale = library, x, q, scale
        self.rotated, self.mode, self.resources = rotated, mode, resources
        self.owners = tuple(t for t in (x, q, scale, rotated) if t is not None)
        self.pointers = tuple(ct.c_void_p(t.data_ptr()) if t is not None else None
                              for t in (x, q, scale, rotated))

    def __call__(self):
        with torch.cuda.device(self.x.device):
            stream = torch.cuda.current_stream(self.x.device)
            for owner in self.owners:
                owner.record_stream(stream)
            self.library.check(self.library.lib.h3_convrot_register_run(
                *self.pointers, *self.x.shape, self.mode, stream.cuda_stream))
        return self.rotated if self.mode == 2 else (self.q, self.scale)
