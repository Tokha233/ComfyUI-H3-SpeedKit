import ast,sys,pathlib,types,json,time,statistics,importlib.util
import torch,triton,triton.language as tl
root=pathlib.Path('/tmp/h3-omni-1002')
from vllm_omni.diffusion.models.minimax_h3.ops.vae import scaled_residual, qk_norm_rope
mods={'scaled_residual':scaled_residual,'qk_norm_rope':qk_norm_rope}
report={'scope':'vLLM-Omni actual runtime exact VAE operators on SM120; not full decoder','torch':torch.__version__,'triton':triton.__version__,'gpu':torch.cuda.get_device_name(),'rows':[]}
def timing(fn):
 for _ in range(4):fn()
 torch.cuda.synchronize();result=[]
 for _ in range(12):
  start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);start.record()
  for i in range(5):fn()
  end.record();end.synchronize();result.append(start.elapsed_time(end)/5)
 return statistics.median(result)
def qref(x,cos,sin):
 n=torch.nn.functional.rms_norm(x.float(),(64,),None,1e-5).to(x.dtype);r=n[...,:48];a,b=r.chunk(2,dim=-1)
 return torch.cat((r*cos+torch.cat((-b,a),dim=-1)*sin,n[...,48:]),dim=-1)
with torch.inference_mode():
 for rows in (195,1797,8192,32768):
  torch.manual_seed(17);r=torch.randn(rows,2048,device='cuda');b=torch.randn(rows,2048,device='cuda',dtype=torch.float16);s=torch.randn(2048,device='cuda')
  base=lambda:r+b*s;opt=lambda:mods['scaled_residual'].try_scaled_residual_exact(r,b,s)
  expected,actual=base(),opt();report['rows'].append({'op':'scaled_residual','rows':rows,'exact':torch.equal(expected,actual),'max_abs':float((expected-actual).abs().max()),'base_ms':timing(base),'opt_ms':timing(opt)})
  del r,b,s,expected,actual
  qkv=torch.randn(1,rows,32,192,device='cuda',dtype=torch.float16);q,k,v=qkv.chunk(3,dim=-1);cos=torch.randn(1,rows,1,48,device='cuda',dtype=torch.float16);sin=torch.randn_like(cos)
  base=lambda:(qref(q,cos,sin),qref(k,cos,sin));opt=lambda:mods['qk_norm_rope'].try_qk_norm_rope_exact(q,k,(cos,sin),1e-5)
  expected,actual=base(),opt();report['rows'].append({'op':'qk_norm_rope','rows':rows,'exact':all(torch.equal(a,b) for a,b in zip(expected,actual)),'max_abs':max(float((a-b).abs().max()) for a,b in zip(expected,actual)),'different_fraction':max(float((a!=b).float().mean()) for a,b in zip(expected,actual)),'base_ms':timing(base),'opt_ms':timing(opt)})
  del qkv,q,k,v,cos,sin,expected,actual
  (root/'vae-ops-native.json').write_text(json.dumps(report,indent=2));print(json.dumps(report['rows'][-2:]),flush=True)
