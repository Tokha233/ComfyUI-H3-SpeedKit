from pathlib import Path
import sys,json,time,hashlib,argparse
p=argparse.ArgumentParser();p.add_argument('--arm',required=True);p.add_argument('--pair',type=int,required=True);a=p.parse_args()
r=Path('/tmp/h3-kitchen-followup-1001');dr=Path('/tmp/h3-kitchen-prs-0930');sys.path[:0]=[str(r/a.arm),str(dr/'comfy-current'),str(dr/'current-deps'),str(dr/'build-deps'),'/last/deps']
import comfy.options
comfy.options.enable_args_parsing();sys.argv=['vquant-sampler','--disable-cuda-malloc']
import main,torch,nodes,comfy.sd,comfy.ops,comfy.model_management
import comfy_kitchen as ck
from comfy.ldm.modules.attention import attention_comfy_kitchen_int8
from comfy_extras.nodes_minimax_h3 import MiniMaxH3SigmaShift
report=dict(arm=a.arm,pair=a.pair,records=[],kitchen=ck.__file__,note='Main Python and attention/V kernels; other linked objects common combined build. Only V scheduler differs between arms. Sampler excludes VAE/input encoding/export.')
def sha(t):return hashlib.sha256(t.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
def save():(r/f'sampler-{a.arm}-{a.pair}.json').write_text(json.dumps(report,indent=2))
torch.set_num_threads(4)
with torch.inference_mode():
 model=comfy.sd.load_diffusion_model('/study/speedkit036/public-model/larry_v4_int8_r2.safetensors');model=MiniMaxH3SigmaShift.execute(model,12.,4.)[0];model.set_model_optimized_attention(attention_comfy_kitchen_int8)
 saved=torch.load('/study/speedkit036/public-run-r5/condition.pt',map_location='cpu',weights_only=False);positive,latent=saved['positive'],saved['latent'];negative=nodes.ConditioningZeroOut().zero_out(positive)[0]
 for repeat in range(-1,2):
  torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
  sample=nodes.common_ksampler(model,42,8,1.,'euler','beta',positive,negative,latent,denoise=1.)[0]
  torch.cuda.synchronize();elapsed=time.perf_counter()-start
  row=dict(repeat=repeat,warmup=repeat<0,seconds=elapsed,sha256=[sha(t) for t in sample['samples'].unbind()],peak_allocated_bytes=torch.cuda.max_memory_allocated());report['records'].append(row);save();print(json.dumps(row),flush=True)
  del sample
 report['status']='success';save()
