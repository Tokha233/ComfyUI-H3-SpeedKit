from pathlib import Path
import argparse
import hashlib
import json
import sys
import time

p = argparse.ArgumentParser()
p.add_argument('--arm', required=True)
p.add_argument('--pair', type=int, required=True)
p.add_argument('--kitchen', default='base')
p.add_argument('--last-probe', action='store_true')
a = p.parse_args()
r = Path('/tmp/h3-comfy-submit-1002')
dr = Path('/tmp/h3-kitchen-prs-0930')
sys.path[:0] = [str(r/a.arm), str(dr/a.kitchen), str(dr/'current-deps'), str(dr/'build-deps'), '/last/deps']
import comfy.options
comfy.options.enable_args_parsing()
sys.argv = ['h3-sampler', '--disable-cuda-malloc']
import main
import torch
import nodes
import comfy.sd
import comfy.ops
import comfy.model_management
import comfy_kitchen as ck
from comfy.ldm.modules.attention import attention_comfy_kitchen_int8
from comfy_extras.nodes_minimax_h3 import MiniMaxH3SigmaShift

torch.set_num_threads(4)
if a.last_probe:
    output_arm='last-probe'
else:
    output_arm=a.arm
report = dict(arm=output_arm, pair=a.pair, kitchen=ck.__file__, torch=torch.__version__,
              gpu=torch.cuda.get_device_name(), base='2d6b73283af2447bdd065ece4090b8c6b1784544', records=[],
              scope='8-step sampler including condition preparation; excludes encoding, VAE, export; public Ref2VA fixture')
def sha(t):
    return hashlib.sha256(t.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
def save():
    (r/f'sampler-{output_arm}-{a.kitchen}-{a.pair}.json').write_text(json.dumps(report, indent=2))
with torch.inference_mode():
    model = comfy.sd.load_diffusion_model('/study/speedkit036/public-model/larry_v4_int8_r2.safetensors')
    model = MiniMaxH3SigmaShift.execute(model, 12., 4.)[0]
    model.set_model_optimized_attention(attention_comfy_kitchen_int8)
    if a.last_probe:
        import types
        from comfy.ldm.minimax.model import _mod_scale_shift, _mod_gate
        def last_forward(self, x, t_emb, mod_segments, rope_freqs, transformer_options={}, attention=None):
            attention=self.attn if attention is None else attention
            shift_msa,scale_msa,gate_msa,shift_mlp,scale_mlp,gate_mlp=self.adaln_proj(t_emb)
            h=_mod_scale_shift(self.norm1(x),shift_msa,scale_msa,mod_segments)
            x=_mod_gate(x,gate_msa,attention(h,rope_freqs=rope_freqs,transformer_options=transformer_options),mod_segments)
            layout=transformer_options['minimax_h3_layout']
            start=min(s for s,e,kind in layout.segments if kind in ('audio','video'))
            segments=[(max(s,start)-start,e-start,row if isinstance(row,int) else row[max(start-s,0):]) for s,e,row in mod_segments if e>start]
            h=_mod_scale_shift(self.norm2(x[start:]),shift_mlp,scale_mlp,segments)
            _mod_gate(x[start:],gate_mlp,self.mlp(h),segments)
            return x
        block=model.model.diffusion_model.blocks[-1]
        block.forward=types.MethodType(last_forward,block)
        report['warning']='Experiment only: dead prefix values differ and full hook/replacement contracts are not preserved; not production implementation.'

    saved = torch.load('/study/speedkit036/public-run-r5/condition.pt', map_location='cpu', weights_only=False)
    positive, latent = saved['positive'], saved['latent']
    negative = nodes.ConditioningZeroOut().zero_out(positive)[0]
    report['refs'] = [{k: list(v.shape) if isinstance(v, torch.Tensor) else v for k,v in ref.items()}
                      for ref in positive[0][1]['minimax_refs']]
    for repeat in range(-1, 2):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        sample = nodes.common_ksampler(model, 42, 8, 1., 'euler', 'beta', positive, negative, latent, denoise=1.)[0]
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        row = dict(repeat=repeat, warmup=repeat < 0, seconds=elapsed,
                   sha256=[sha(t) for t in sample['samples'].unbind()],
                   peak_allocated_bytes=torch.cuda.max_memory_allocated())
        report['records'].append(row)
        save()
        print(json.dumps(row), flush=True)
        del sample
    report['status'] = 'success'
    save()
