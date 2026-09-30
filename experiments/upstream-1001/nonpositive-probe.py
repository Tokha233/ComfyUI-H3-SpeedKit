from pathlib import Path
import argparse,sys,json,hashlib,statistics
p=argparse.ArgumentParser();p.add_argument('--arm',required=True);p.add_argument('--pair',type=int,default=0);a=p.parse_args()
r=Path('/tmp/h3-kitchen-prs-0930');sys.path[:0]=[str(r/a.arm),str(r/'build-deps'),'/last/deps']
import torch
from torch.nn.attention import SDPBackend,sdpa_kernel
import comfy_kitchen as ck
from comfy_kitchen.backends import cuda
result={'arm':a.arm,'pair':a.pair,'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,'source':ck.__file__,'extension':cuda._C.__file__,'positive':[],'nonpositive':[],'benchmarks':[]}
def digest(t):return hashlib.sha256(t.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
def inputs(d,n,dtype,lq=129,h=4,hk=2):
 torch.manual_seed(41)
 q=torch.randn(1,lq,h,d,device='cuda',dtype=dtype).transpose(1,2)
 k=torch.randn(1,n,hk,d,device='cuda',dtype=dtype).transpose(1,2)
 v=torch.randn(1,n,hk,d,device='cuda',dtype=dtype).transpose(1,2)
 return q,k,v
for d in [64,128,256]:
 for dtype in [torch.float16,torch.bfloat16]:
  for n in [1,64,65,512,513,1024,1025,8193]:
   q,k,v=inputs(d,n,dtype)
   out=ck.int8_attention(q,k,v)
   result['positive'].append({'head_dim':d,'kv_length':n,'dtype':str(dtype),'sha256':digest(out)})
for scale in [0.,-(128**-.5)]:
 torch.manual_seed(31)
 q=torch.randn(1,129,4,128,device='cuda',dtype=torch.bfloat16).transpose(1,2)
 k=torch.randn(1,8193,4,128,device='cuda',dtype=torch.bfloat16).transpose(1,2)
 v=torch.randn(1,8193,4,128,device='cuda',dtype=torch.bfloat16).transpose(1,2)
 out=ck.int8_attention(q,k,v,scale=scale)
 auto=torch.nn.functional.scaled_dot_product_attention(q,k,v,scale=scale)
 with sdpa_kernel(SDPBackend.MATH):ref=torch.nn.functional.scaled_dot_product_attention(q.float(),k.float(),v.float(),scale=scale)
 error=((out.float()-ref).square().mean()/ref.square().mean()).sqrt().item()
 result['nonpositive'].append({'scale':scale,'nrmse':error,'finite':bool(out.isfinite().all()),'nonzero':int(out.count_nonzero()),'auto_sdpa_finite':bool(auto.isfinite().all()),'reference_finite':bool(ref.isfinite().all())})
for lq,n,h,hk in [(129,8193,4,4),(4096,4096,16,16),(14850,14850,56,56),(32700,32700,42,42)]:
 q,k,v=inputs(128,n,torch.bfloat16,lq,h,hk);packed=ck.prequantize_int8_attention(q,k,v)
 for _ in range(5):out=ck.int8_attention_from_prequantized(packed)
 graph=torch.cuda.CUDAGraph()
 with torch.cuda.graph(graph):out=ck.int8_attention_from_prequantized(packed)
 times=[]
 for _ in range(8):
  start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
  start.record()
  for _ in range(10):graph.replay()
  end.record();end.synchronize();times.append(start.elapsed_time(end)/10)
 result['benchmarks'].append({'shape':[1,h,hk,lq,n,128],'samples_ms':times,'median_ms':statistics.median(times),'sha256':digest(out)})
 del graph,q,k,v,packed,out
path=r/f'nonpositive-probe-{a.arm}-{a.pair}.json';path.write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='positive'}),flush=True)
