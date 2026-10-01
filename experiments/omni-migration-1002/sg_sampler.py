import os,sys,pathlib,time,json,hashlib,argparse
root=pathlib.Path('/tmp/h3-omni-1002')
p=argparse.ArgumentParser();p.add_argument('--split',type=int,default=8192);p.add_argument('--repeats',type=int,default=3);p.add_argument('--kitchen-attn',action='store_true');p.add_argument('--resident',type=int,default=0);p.add_argument('--lifetime',default='forward');p.add_argument('--prefetch',type=float,default=0.0);a,_=p.parse_known_args()
if __name__=='__main__':
 os.environ['SGLANG_KITCHEN_INT8_MAX_ROWS']=str(a.split)
 os.environ['H3_KITCHEN_ATTN']=str(int(a.kitchen_attn))
os.environ['HF_HUB_OFFLINE']='1'
sys.path[:0]=['/tmp/h3-comfy-submit-1002/base','/tmp/h3-kitchen-prs-0930/base',str(root/'sglang/python')]
import comfy.options
comfy.options.enable_args_parsing();sys.argv=['bench','--disable-cuda-malloc']
import main,torch,nodes,comfy.sd
from comfy_extras.nodes_minimax_h3 import MiniMaxH3SigmaShift
from sglang.multimodal_gen.apps.ComfyUI_SGLDiffusion.core.generator import SGLDiffusionGenerator

if os.environ.get('H3_KITCHEN_ATTN')=='1':
 from sglang.multimodal_gen.runtime.layers.attention.backends.sdpa import SDPAImpl
 import comfy_kitchen as ck
 original_varlen=SDPAImpl.forward_varlen
 attention_calls=0
 def kitchen_varlen(self,query,key,value,*,cu_seqlens,max_seqlen,cu_seqlens_host=None):
  global attention_calls
  if cu_seqlens_host is None or self.causal or self.dropout != 0.0 or query.dtype not in (torch.bfloat16,torch.float16) or query.shape[-1]!=128:
   return original_varlen(self,query,key,value,cu_seqlens=cu_seqlens,max_seqlen=max_seqlen,cu_seqlens_host=cu_seqlens_host)
  attention_calls+=1
  if attention_calls in (1,40,320): print(f"KITCHEN_USED pid={os.getpid()} calls={attention_calls} bounds={cu_seqlens_host}",flush=True)
  output=torch.empty_like(query)
  for start,stop in zip(cu_seqlens_host[:-1],cu_seqlens_host[1:]):
   if start==stop:continue
   out=ck.int8_attention(query[start:stop].transpose(0,1).unsqueeze(0),key[start:stop].transpose(0,1).unsqueeze(0),value[start:stop].transpose(0,1).unsqueeze(0),scale=self.softmax_scale)
   output[start:stop].copy_(out.squeeze(0).transpose(0,1))
  return output
 SDPAImpl.forward_varlen=kitchen_varlen

def sha(t):return hashlib.sha256(t.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
if __name__=='__main__':
 torch.set_num_threads(4)
 report={'scope':'SGLang Diffusion ComfyUI integrated H3 Larry8 DiT sampler; excludes VAE/export','split':a.split,'torch':torch.__version__,'gpu':torch.cuda.get_device_name(),'records':[],'kitchen_attention':a.kitchen_attn,'resident':a.resident,'lifetime':a.lifetime,'prefetch':a.prefetch}
 with torch.inference_mode():
  generator=SGLDiffusionGenerator.shared()
  model=generator.load_model('/study/speedkit036/public-model/larry_v4_int8_r2.safetensors',sgld_options={'model_type':'minimax_h3','num_gpus':1,'tp_size':1,'sp_degree':1,'dit_layerwise_offload':True,'dit_layerwise_resident_layers':a.resident,'dit_layerwise_residency_lifetime':a.lifetime,'dit_offload_prefetch_size':a.prefetch,'enable_cache_dit':False,'attention_backend':'torch_sdpa','master_port':29931})
  model=MiniMaxH3SigmaShift.execute(model,12.,4.)[0]
  saved=torch.load('/study/speedkit036/public-run-r5/condition.pt',map_location='cpu',weights_only=False)
  positive,latent=saved['positive'],saved['latent'];negative=nodes.ConditioningZeroOut().zero_out(positive)[0]
  for repeat in range(-1,a.repeats):
   torch.cuda.synchronize();start=time.perf_counter()
   sample=nodes.common_ksampler(model,42,8,1.,'euler','beta',positive,negative,latent,denoise=1.)[0]
   torch.cuda.synchronize();ms=(time.perf_counter()-start)*1000
   row={'repeat':repeat,'warmup':repeat<0,'ms':ms,'sha256':[sha(t) for t in sample['samples'].unbind()]}
   report['records'].append(row);(root/f'sg-sampler-{a.split}-ck{int(a.kitchen_attn)}-r{a.resident}-p{a.prefetch}-{a.lifetime}.json').write_text(json.dumps(report,indent=2));print(json.dumps(row),flush=True)
   if repeat==0:torch.save(sample,root/f'sg-sample-{a.split}-ck{int(a.kitchen_attn)}-r{a.resident}-p{a.prefetch}-{a.lifetime}.pt')
  generator.close_generator()
 print('DONE',flush=True)
