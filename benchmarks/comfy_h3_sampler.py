import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--arm', required=True, help='Label recorded in the report')
p.add_argument('--comfy-dir', type=Path, required=True)
p.add_argument('--kitchen-dir', type=Path, required=True)
p.add_argument('--weights', type=Path, required=True)
p.add_argument('--condition', type=Path, required=True, help='Trusted local torch fixture containing positive and latent')
p.add_argument('--output-dir', type=Path, required=True)
p.add_argument('--pair', type=int, required=True)
p.add_argument('--kitchen', default='installed')
a = p.parse_args()
r = a.output_dir
r.mkdir(parents=True, exist_ok=True)
sys.path[:0] = [str(a.comfy_dir.resolve()), str(a.kitchen_dir.resolve())]
import comfy.options

comfy.options.enable_args_parsing()
sys.argv = ['h3-sampler', '--disable-cuda-malloc']
import main  # noqa: F401, I001 -- initializes ComfyUI before model imports

import torch
import nodes
import comfy.sd
import comfy.ops
import comfy.model_management
import comfy_kitchen as ck
from comfy.ldm.modules.attention import attention_comfy_kitchen_int8
from comfy_extras.nodes_minimax_h3 import MiniMaxH3SigmaShift

torch.set_num_threads(4)
output_arm = a.arm
report = {'arm': output_arm, 'pair': a.pair, 'kitchen': ck.__file__, 'torch': torch.__version__,
              'gpu': torch.cuda.get_device_name(), 'comfy_dir': str(a.comfy_dir.resolve()), 'records': [],
              'scope': '8-step sampler including condition preparation; excludes encoding, VAE, export; user-supplied local Ref2VA fixture'}
def sha(t):
    return hashlib.sha256(t.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
def save():
    (r/f'sampler-{output_arm}-{a.kitchen}-{a.pair}.json').write_text(json.dumps(report, indent=2))
with torch.inference_mode():
    model = comfy.sd.load_diffusion_model(str(a.weights))
    model = MiniMaxH3SigmaShift.execute(model, 12., 4.)[0]
    model.set_model_optimized_attention(attention_comfy_kitchen_int8)
    saved = torch.load(a.condition, map_location='cpu', weights_only=False)
    positive, latent = saved['positive'], saved['latent']
    negative = nodes.ConditioningZeroOut().zero_out(positive)[0]
    report['refs'] = [{k: list(v.shape) if isinstance(v, torch.Tensor) else v for k,v in ref.items()}
                      for ref in positive[0][1].get('minimax_refs', [])]
    for repeat in range(-1, 2):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        sample = nodes.common_ksampler(model, 42, 8, 1., 'euler', 'beta', positive, negative, latent, denoise=1.)[0]
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        row = {'repeat': repeat, 'warmup': repeat < 0, 'seconds': elapsed,
                   'sha256': [sha(t) for t in sample['samples'].unbind()],
                   'peak_allocated_bytes': torch.cuda.max_memory_allocated()}
        report['records'].append(row)
        save()
        print(json.dumps(row), flush=True)
        del sample
    report['status'] = 'success'
    save()
