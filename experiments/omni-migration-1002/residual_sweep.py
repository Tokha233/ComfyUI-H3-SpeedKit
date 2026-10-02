import json,statistics,torch,triton,triton.language as tl
from pathlib import Path
from vllm_omni.diffusion.models.minimax_h3.ops.vae.scaled_residual import _scaled_residual_exact_kernel
@triton.jit
def flat_kernel(O,R,B,S,N:tl.constexpr,BLOCK:tl.constexpr):
 i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
 r=tl.load(R+i,i<N,0).to(tl.float32);b=tl.load(B+i,i<N,0).to(tl.float32);s=tl.load(S+i%2048,i<N,0).to(tl.float32)
 p=tl.inline_asm_elementwise('mul.rn.f32 $0, $1, $2;',constraints='=f,f,f',args=[b,s],dtype=tl.float32,is_pure=True,pack=1)
 tl.store(O+i,r+p,i<N)
def timing(fn):
 for _ in range(10):fn()
 torch.cuda.synchronize();times=[]
 for _ in range(12):
  a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);a.record()
  for _ in range(20):fn()
  b.record();b.synchronize();times.append(a.elapsed_time(b)/20)
 return statistics.median(times)
report={'torch':torch.__version__,'gpu':torch.cuda.get_device_name(),'rows':[]}
with torch.inference_mode():
 for rows in [1,32,195,1797,7188,8192,32768]:
  torch.manual_seed(13);r=torch.randn(rows,2048,device='cuda');b=torch.randn_like(r,dtype=torch.float16);s=torch.randn(2048,device='cuda');o=torch.empty_like(r);expected=r+b*s
  variants={'ref':lambda: r+b*s}
  for warps in [4,8,16]:variants[f'row{warps}']=lambda w=warps:_scaled_residual_exact_kernel[(rows,)](o,r,b,s,2048,2048,2048,2048,num_warps=w)
  for block,warps in [(1024,4),(2048,4),(4096,4),(4096,8),(8192,4),(8192,8)]:variants[f'flat{block}w{warps}']=lambda n=block,w=warps:flat_kernel[(triton.cdiv(r.numel(),n),)](o,r,b,s,r.numel(),n,num_warps=w)
  for name,fn in variants.items():
   fn();actual=expected if name=='ref' else o
   result={'rows':rows,'name':name,'ms':timing(fn),'exact':torch.equal(actual,expected)};report['rows'].append(result);print(json.dumps(result),flush=True)
 Path('/tmp/h3-omni-1002/residual-sweep.json').write_text(json.dumps(report,indent=2))
