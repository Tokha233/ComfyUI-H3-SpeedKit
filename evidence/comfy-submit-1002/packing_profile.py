from pathlib import Path
import argparse, json, sys, time
p=argparse.ArgumentParser(); p.add_argument('arm'); a=p.parse_args()
r=Path('/tmp/h3-comfy-submit-1002'); dr=Path('/tmp/h3-kitchen-prs-0930')
sys.path[:0]=[str(r/a.arm),str(dr/'base'),str(dr/'current-deps'),str(dr/'build-deps'),'/last/deps']
import torch
from comfy.ldm.minimax.model import MiniMaxH3Model, PackedLayout
from torch import nn
from torch.profiler import profile,ProfilerActivity
torch.set_num_threads(4)
# Real H3 row sizes. Small output hidden avoids timing unrelated GEMM throughput.
model=MiniMaxH3Model(hidden_size=8,num_layers=0,token_refiner_num_layers=0,num_attention_heads=1,attention_head_dim=8,ffn_hidden_size=8,latents_dim=24,audio_latents_dim=32,text_dim=5,timestep_input_dim=8,time_embed_hidden_size=8,time_embed_dim=8,dtype=torch.bfloat16,device='cuda',operations=nn)
video=torch.randn(1,24,37,32,48,device='cuda',dtype=torch.bfloat16)
audio=torch.randn(1,32,2,250,device='cuda',dtype=torch.bfloat16)
context=torch.randn(1,132,8,device='cuda',dtype=torch.bfloat16)
payload={'seed':42,'visual_cond_noise_aug':0.999,'audio_cond_noise_aug':0.8,'cond_video_latents':[torch.randn(1,24,1,32,48)],'cond_audio_latents':[torch.randn(1,32,2,30)],'refs':[{'kind':'image','latent_h':32,'latent_w':48},{'kind':'audio','ref_audio_t':30}]}
layout=PackedLayout(132,37,32,48,250,refs=payload['refs'])
with torch.inference_mode():
 if a.arm=='cond':payload.update(model.preprocess_reference_latents(payload,'cuda'))
 def run():return model._embed_and_pack(video,audio,context,layout,payload,{})
 for i in range(5):run()
 torch.cuda.synchronize()
 with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA]) as prof:run();torch.cuda.synchronize()
 events={e.key:{'calls':e.count,'cpu_us':e.cpu_time_total,'device_us':e.device_time_total} for e in prof.key_averages() if any(x in e.key for x in ['nonzero','index_put','bitwise','cat','randn','copy_','Synchronize'])}
 times=[]
 for i in range(50):
  torch.cuda.synchronize();t=time.perf_counter();run();torch.cuda.synchronize();times.append((time.perf_counter()-t)*1000)
 result={'arm':a.arm,'events':events,'helper_ms':times,'note':'Real helper on synthetic mixed visual/audio references, hidden=8; not full model speed.'}
 (r/f'profile-{a.arm}.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
