"""Compare reference and registered vLLM-Omni H3 VAE operators on a local latent."""
import argparse
import gc
import hashlib
import json
import statistics
import time
from pathlib import Path

import torch
from vllm_omni.diffusion.models.minimax_h3.ops.vae import dispatch
from vllm_omni.diffusion.models.minimax_h3.vae import MiniMaxH3VideoVAE
from vllm_omni.platforms import current_omni_platform

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--vae-dir', type=Path, required=True)
parser.add_argument('--latent', type=Path, required=True, help='torch.save of a plain [B,24,T,H,W] Tensor')
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--repeats', type=int, default=2)
args = parser.parse_args()
device = current_omni_platform.get_torch_device()
if dispatch.resolve_h3_vae_operators(device) is None:
    raise RuntimeError('Run the candidate checkout with a registered H3 VAE operator set')
latent = torch.load(args.latent, map_location='cpu', weights_only=True)
if not isinstance(latent, torch.Tensor) or latent.ndim != 5 or latent.shape[1] != 24:
    raise ValueError('Expected a plain [B,24,T,H,W] latent Tensor')
latent = latent.to(device=device, dtype=torch.float32)
torch.set_num_threads(4)
original = dispatch.H3_VAE_OPERATOR_TABLE
report = {'torch': torch.__version__, 'input_shape': list(latent.shape), 'records': [],
          'scope': 'video VAE decode only; excludes DiT, encoders and export'}
reference = None
with torch.inference_mode():
    for arm in ('baseline', 'candidate', 'candidate', 'baseline'):
        dispatch.H3_VAE_OPERATOR_TABLE = () if arm == 'baseline' else original
        decoder = MiniMaxH3VideoVAE(str(args.vae_dir), device=device, decode_only=True, trust_remote_code=True)
        installed = bool(getattr(decoder.model.decoder, '_omni_h3_vae_optimizations_installed', False))
        if installed != (arm == 'candidate'):
            raise RuntimeError(f'Unexpected optimization installation: {arm}, {installed}')
        for repeat in range(-1, args.repeats):
            current_omni_platform.synchronize()
            start = time.perf_counter()
            with torch.autocast(device.type, dtype=torch.float16):
                output = decoder.decode_latent(latent)
            current_omni_platform.synchronize()
            ms = (time.perf_counter() - start) * 1000
            output = output.cpu()
            if reference is None:
                reference = output.clone()
            exact = torch.equal(output, reference)
            row = {'arm': arm, 'repeat': repeat, 'ms': ms, 'exact': exact, 'dtype': str(output.dtype),
                   'shape': list(output.shape), 'max_abs': (output.float() - reference.float()).abs().max().item(),
                   'sha256': hashlib.sha256(output.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()}
            report['records'].append(row)
            args.output.write_text(json.dumps(report, indent=2) + '\n')
            print(json.dumps(row), flush=True)
            if not exact:
                raise RuntimeError('Decoded output changed')
        del decoder, output
        gc.collect()
        torch.accelerator.empty_cache()
dispatch.H3_VAE_OPERATOR_TABLE = original
medians = {arm: statistics.median(r['ms'] for r in report['records'] if r['arm'] == arm and r['repeat'] >= 0)
           for arm in ('baseline', 'candidate')}
print(json.dumps({'medians_ms': medians, 'latency_reduction_percent': 100 * (1 - medians['candidate'] / medians['baseline'])}))
