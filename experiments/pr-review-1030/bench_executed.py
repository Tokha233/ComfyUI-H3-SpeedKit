"""Pinned official H3 VAE ABBA experiment; not an end-to-end benchmark."""
import argparse
import gc
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import statistics
import sys
import time

import torch
from vllm_omni.diffusion.models.minimax_h3.ops.vae import dispatch
from vllm_omni.diffusion.models.minimax_h3.vae import MiniMaxH3VideoVAE


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(16 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


p = argparse.ArgumentParser()
p.add_argument('--root', type=Path, required=True)
p.add_argument('--source', type=Path, required=True)
p.add_argument('--mode', choices=['rgb8', 'fp32'], default='rgb8')
args = p.parse_args()
root = args.root
source = args.source.resolve()
manifest = json.loads((root / 'candidate-manifest.json').read_text())
output_path = root / ('decode-' + args.mode + '.json')
latent_path = Path('/tmp/h3-omni-1002/plain-video-latent.pt')
vae_path = Path('/tmp/h3-omni-1002/official/Ref2VA/video_vae')
torch.set_num_threads(4)
torch.manual_seed(0)
device = torch.device('cuda:0')
assert torch.cuda.get_device_capability() == (12, 0)
assert dispatch.resolve_h3_vae_operators(device) is not None
latent = torch.load(latent_path, weights_only=True, map_location='cpu').to(device, torch.float32)
report = {
    'scope': 'official video VAE decode only; excludes encoders, DiT, serving and MP4 export',
    'main_sha': '92595ee491d63aa00d3faa57ab944ddb9f0db4dc',
    'pr_sha': '1b55c8320f09857231a5fadcefe1298d17aaf7d5',
    'official_revision': '42ed227ee7df40d41602854ae760620d6eb651fe',
    'source': str(source), 'gpu': torch.cuda.get_device_name(),
    'capability': list(torch.cuda.get_device_capability()),
    'versions': {n: metadata.version(n) for n in ['torch', 'triton', 'vllm', 'transformers', 'diffusers', 'tokenizers', 'flashinfer-python']},
    'mode': args.mode, 'legacy_temporal': os.environ.get('VLLM_OMNI_VAE_LEGACY_TEMPORAL', ''),
    'latent_sha256': digest(latent_path), 'input_shape': list(latent.shape),
    'vae_files': {str(f.relative_to(vae_path)): digest(f) for f in sorted(vae_path.rglob('*')) if f.is_file() and '__pycache__' not in str(f)},
    'records': [],
}


def audit():
    loaded = {}
    for module in list(sys.modules.values()):
        f = getattr(module, '__file__', None)
        if not f:
            continue
        f = Path(f)
        if not f.is_absolute():
            # e.g. torch's generated _ops.py module has no backing file.
            continue
        f = f.resolve()
        try:
            name = str(f.relative_to(source))
        except ValueError:
            continue
        if f.suffix == '.py':
            loaded[name] = digest(f)
    mismatches = [n for n, h in loaded.items() if manifest.get(n) != h]
    report['imported_source_files'] = loaded
    report['source_mismatches'] = mismatches
    output_path.write_text(json.dumps(report, indent=2) + '\n')
    if mismatches:
        raise RuntimeError('Source mismatch: ' + repr(mismatches))


original = dispatch.H3_VAE_OPERATOR_TABLE
reference = None
with torch.inference_mode():
    for arm_index, arm in enumerate(('baseline', 'candidate', 'candidate', 'baseline')):
        dispatch.H3_VAE_OPERATOR_TABLE = () if arm == 'baseline' else original
        decoder = MiniMaxH3VideoVAE(str(vae_path), device=device, decode_only=True, trust_remote_code=True)
        installed = bool(getattr(decoder.model.decoder, '_omni_h3_vae_optimizations_installed', False))
        assert installed == (arm == 'candidate'), (arm, installed)
        for repeat in range(-1, 2):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            with torch.autocast('cuda', dtype=torch.float16):
                output = decoder.decode_latent(latent)
            torch.cuda.synchronize()
            ms = (time.perf_counter() - start) * 1000
            peak = torch.cuda.max_memory_allocated()
            cpu_output = output.cpu()
            del output
            if reference is None:
                reference = cpu_output.clone()
            expected = torch.uint8 if args.mode == 'rgb8' else torch.float32
            assert cpu_output.dtype == expected, (cpu_output.dtype, expected)
            exact = torch.equal(cpu_output, reference)
            row = {'arm': arm, 'arm_index': arm_index, 'repeat': repeat, 'warmup': repeat == -1,
                   'ms': ms, 'peak_memory_allocated_bytes': peak, 'exact': exact,
                   'shape': list(cpu_output.shape), 'dtype': str(cpu_output.dtype),
                   'max_abs': 0.0 if exact else (cpu_output.float() - reference.float()).abs().max().item(),
                   'sha256': hashlib.sha256(cpu_output.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()}
            report['records'].append(row)
            audit()
            print(json.dumps(row), flush=True)
            del cpu_output
            if not exact:
                raise RuntimeError('Output changed')
        del decoder
        gc.collect()
        torch.cuda.empty_cache()
dispatch.H3_VAE_OPERATOR_TABLE = original
report['medians_ms'] = {arm: statistics.median(r['ms'] for r in report['records'] if r['arm'] == arm and not r['warmup']) for arm in ['baseline', 'candidate']}
report['median_peak_bytes'] = {arm: statistics.median(r['peak_memory_allocated_bytes'] for r in report['records'] if r['arm'] == arm and not r['warmup']) for arm in ['baseline', 'candidate']}
report['latency_reduction_percent'] = 100 * (1 - report['medians_ms']['candidate'] / report['medians_ms']['baseline'])
report['complete'] = True
audit()
print(json.dumps({k: report[k] for k in ['medians_ms', 'median_peak_bytes', 'latency_reduction_percent', 'complete']}), flush=True)
