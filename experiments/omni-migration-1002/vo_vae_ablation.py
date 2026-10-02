import os,gc,time,json,hashlib,statistics
from pathlib import Path
from types import MethodType
os.environ['HF_HUB_OFFLINE']='1'
import torch
from vllm_omni.diffusion.models.minimax_h3.ops.vae import dispatch
from vllm_omni.diffusion.models.minimax_h3.vae import MiniMaxH3VideoVAE
root=Path('/tmp/h3-omni-1002');torch.set_num_threads(4)
latent=torch.load(root/'plain-video-latent.pt',weights_only=True).to('cuda',torch.float32)
original=dispatch.H3_VAE_OPERATOR_TABLE
arms=['baseline','weights','weights_ff','weights_qk','weights_residual','all']
report={'scope':'same-head VAE-only ablation; one warmup and two timed decodes per arm per order; forward then reverse','head':'ce3039e4a48b6026302b6b9dec107dec8ac69fd9','torch':torch.__version__,'gpu':torch.cuda.get_device_name(),'input_shape':list(latent.shape),'records':[]}
reference=None
with torch.inference_mode():
 for arm in arms+arms[::-1]:
  dispatch.H3_VAE_OPERATOR_TABLE=() if arm=='baseline' else original
  decoder=MiniMaxH3VideoVAE(str(root/'official/Ref2VA/video_vae'),device=torch.device('cuda:0'),decode_only=True,trust_remote_code=True)
  if arm!='baseline':
   assert decoder.model.decoder._omni_h3_vae_optimizations_installed
   for block in decoder.model.decoder.transformer_blocks:
    if arm not in ('all','weights_ff'):block.ff.forward=MethodType(type(block.ff).forward,block.ff)
    if arm not in ('all','weights_qk'):block.attn.forward=MethodType(type(block.attn).forward,block.attn)
    if arm not in ('all','weights_residual'):block.forward=MethodType(type(block).forward,block)
   del block
  for repeat in range(-1,2):
   torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
   with torch.autocast('cuda',dtype=torch.float16):output=decoder.decode_latent(latent)
   torch.cuda.synchronize();ms=(time.perf_counter()-start)*1000;peak=torch.cuda.max_memory_allocated();output=output.cpu()
   if reference is None:reference=output.clone()
   row={'arm':arm,'repeat':repeat,'ms':ms,'peak_bytes':peak,'exact':torch.equal(reference,output),'sha256':hashlib.sha256(output.contiguous().numpy().tobytes()).hexdigest()}
   print(json.dumps(row),flush=True);report['records'].append(row);(root/'vo-vae-ablation.json').write_text(json.dumps(report,indent=2))
   assert row['exact']
  del decoder,output;gc.collect();torch.cuda.empty_cache()
