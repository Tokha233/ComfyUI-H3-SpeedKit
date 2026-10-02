import os,gc,json
from pathlib import Path
os.environ['HF_HUB_OFFLINE']='1'
import torch
from vllm_omni.diffusion.models.minimax_h3.ops.vae import dispatch
from vllm_omni.diffusion.models.minimax_h3.vae import MiniMaxH3VideoVAE
root=Path('/tmp/h3-omni-1002');torch.set_num_threads(4)
latent=torch.load(root/'plain-video-latent.pt',weights_only=True).to('cuda',torch.float32)
original=dispatch.H3_VAE_OPERATOR_TABLE;report={}
with torch.inference_mode():
 for arm in ['baseline','all']:
  dispatch.H3_VAE_OPERATOR_TABLE=() if arm=='baseline' else original
  decoder=MiniMaxH3VideoVAE(str(root/'official/Ref2VA/video_vae'),device=torch.device('cuda:0'),decode_only=True,trust_remote_code=True)
  with torch.autocast('cuda',dtype=torch.float16):output=decoder.decode_latent(latent)
  del output;torch.cuda.synchronize()
  with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
   with torch.autocast('cuda',dtype=torch.float16):output=decoder.decode_latent(latent)
   torch.cuda.synchronize()
  events=sorted(prof.key_averages(),key=lambda e:e.self_device_time_total,reverse=True)
  report[arm]=[{'name':e.key,'count':e.count,'self_cuda_us':e.self_device_time_total,'self_cpu_us':e.self_cpu_time_total} for e in events[:35]]
  (root/'vo-vae-profile.json').write_text(json.dumps(report,indent=2));print(arm,report[arm][:5],flush=True)
  del decoder,output,prof;gc.collect();torch.cuda.empty_cache()
