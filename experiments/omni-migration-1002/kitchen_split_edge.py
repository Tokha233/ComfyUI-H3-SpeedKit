import os,json,time,statistics,torch
from pathlib import Path
import comfy_kitchen
from sglang.multimodal_gen.runtime.layers.quantization import kitchen_int8 as mod
from sglang.multimodal_gen.runtime.layers.quantization.configs.kitchen_int8_config import KitchenInt8Config
method=mod.KitchenInt8LinearMethod(KitchenInt8Config(),group_size=256,is_checkpoint_serialized=True)
def timing(fn):
 for _ in range(3):fn()
 torch.cuda.synchronize();ts=[]
 for _ in range(10):
  a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);a.record();fn();b.record();b.synchronize();ts.append(a.elapsed_time(b))
 return {'median':statistics.median(ts),'min':min(ts),'max':max(ts)}
report={'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,'rows':[]}
with torch.inference_mode():
 for m in [12288,16384,32768]:
  for k,n in [(4096,8192),(8192,24576),(4096,24832),(14336,8192),(28672,21504)]:
   torch.manual_seed(12);layer=torch.nn.Module();layer.weight=torch.nn.Parameter(torch.randint(-127,128,(n,k),device='cuda',dtype=torch.int8),requires_grad=False);layer.weight_scale=torch.nn.Parameter(torch.rand(n,1,device='cuda')*.002,requires_grad=False)
   x=torch.randn(m,k,device='cuda',dtype=torch.bfloat16);bias=torch.randn(n,device='cuda',dtype=torch.bfloat16)
   mod._MAX_ROWS_PER_CALL=8192;base=method.apply(layer,x,bias)
   mod._MAX_ROWS_PER_CALL=0;opt=method.apply(layer,x,bias)
   results=[]
   for split in [8192,0,0,8192]:
    mod._MAX_ROWS_PER_CALL=split;results.append({'split':split,**timing(lambda:method.apply(layer,x,bias))})
   row={'m':m,'k':k,'n':n,'exact':torch.equal(base,opt),'max_abs':float((base.float()-opt.float()).abs().max()),'timings':results};report['rows'].append(row);print(json.dumps(row),flush=True)
   Path('/tmp/h3-omni-1002/kitchen-split-sweep.json').write_text(json.dumps(report,indent=2));del x,layer,base,opt,bias;torch.cuda.empty_cache()
