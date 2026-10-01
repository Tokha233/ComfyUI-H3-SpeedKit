from pathlib import Path
import sys,json,torch,importlib.util,statistics,random,hashlib
r=Path('/tmp/h3-kitchen-followup-1001');dr=Path('/tmp/h3-kitchen-prs-0930');sys.path[:0]=[str(r/'base'),str(dr/'build-deps'),'/last/deps']
import comfy_kitchen as ck
from comfy_kitchen.backends import cuda
mods={}
for arm in ['base','candidate']:
 spec=importlib.util.spec_from_file_location(arm+'._C',r/(arm+'.so'));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);mods[arm]=m
report=dict(protocol='Same input; 1 warm graph each; 8 calls per CUDA graph; 8 interleaved timed replays per arm, median CUDA events. Exact INT8 bytes and FP32 scale bits.',gpu=torch.cuda.get_device_name(),torch=torch.__version__,rows=[],attention=[])
torch.manual_seed(57)
def save():(r/'benchmark.json').write_text(json.dumps(report,indent=2))
def stats(vals):return dict(samples_ms=vals,median_ms=statistics.median(vals))
for dtype in [torch.float16,torch.bfloat16]:
 for n in [8192,12288,14850,32700,87142,90461]:
  for layout in ['BHND','BNHD','QKV']:
   b,h,d=1,56,128
   if layout=='BHND':v=torch.randn(b,h,n,d,device='cuda',dtype=dtype)
   elif layout=='BNHD':v=torch.randn(b,n,h,d,device='cuda',dtype=dtype).transpose(1,2)
   else:v=torch.randn(b,n,3,h,d,device='cuda',dtype=dtype)[:,:,2].transpose(1,2)
   padded=(n+127)//128*128;buf={arm:(torch.empty((b,h,d,padded),device='cuda',dtype=torch.int8),torch.empty((b,h,d),device='cuda')) for arm in mods}
   def call(arm):mods[arm]._quant_v_int8(*(cuda._wrap_for_dlpack(t) for t in (v,*buf[arm])),padded,1 if dtype==torch.float16 else 2,torch.cuda.current_stream().cuda_stream)
   for arm in mods:call(arm)
   torch.cuda.synchronize();assert all(torch.equal(x.view(torch.uint8),y.view(torch.uint8)) for x,y in zip(buf['base'],buf['candidate']))
   graphs={}
   for arm in mods:
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
     for _ in range(3):call(arm)
    torch.cuda.current_stream().wait_stream(stream)
    g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g,stream=stream):
     for _ in range(8):call(arm)
    g.replay();graphs[arm]=g
   vals={arm:[] for arm in mods}
   for repeat in range(8):
    for arm in (['base','candidate'] if repeat%2==0 else ['candidate','base']):
     st=torch.cuda.Event(enable_timing=True);en=torch.cuda.Event(enable_timing=True);st.record();graphs[arm].replay();en.record();en.synchronize();vals[arm].append(st.elapsed_time(en)/8)
   row=dict(shape=[b,h,n,d],dtype=str(dtype),layout=layout,strides=list(v.stride()),exact=True,timing={arm:stats(vs) for arm,vs in vals.items()});row['time_reduction_percent']=100*(1-statistics.median(vals['candidate'])/statistics.median(vals['base']));report['rows'].append(row);save();print(json.dumps(row),flush=True)
   del graphs,buf,v
# Complete public API, including Q/K quantization, V quantization and attention.
for n in [12288,14850,32700]:
 for layout in ['BHND','QKV']:
  b,h,d=1,56,128
  if layout=='BHND':q,k,v=[torch.randn(b,h,n,d,device='cuda',dtype=torch.bfloat16) for _ in range(3)]
  else:
   qkv=torch.randn(b,n,3,h,d,device='cuda',dtype=torch.bfloat16);q,k,v=[qkv[:,:,i].transpose(1,2) for i in range(3)]
  outs={};vals={arm:[] for arm in mods}
  for arm in mods:cuda._C=mods[arm];outs[arm]=ck.int8_attention(q,k,v)
  torch.cuda.synchronize();assert torch.equal(outs['base'],outs['candidate'])
  for repeat in range(8):
   for arm in (['base','candidate'] if repeat%2==0 else ['candidate','base']):
    cuda._C=mods[arm];st=torch.cuda.Event(enable_timing=True);en=torch.cuda.Event(enable_timing=True);st.record();out=ck.int8_attention(q,k,v);en.record();en.synchronize();vals[arm].append(st.elapsed_time(en))
  row=dict(shape=[b,h,n,d],dtype='torch.bfloat16',layout=layout,exact=True,timing={arm:stats(vs) for arm,vs in vals.items()});row['time_reduction_percent']=100*(1-statistics.median(vals['candidate'])/statistics.median(vals['base']));report['attention'].append(row);save();print(json.dumps(row),flush=True)
report['status']='success';save()
