import os,sys,pathlib,time,json,gc,hashlib,statistics
os.environ['HF_HUB_OFFLINE']='1'
import torch
from vllm_omni.diffusion.models.minimax_h3.vae import MiniMaxH3VideoVAE
from vllm_omni.diffusion.models.minimax_h3.ops.vae import dispatch
root=pathlib.Path('/tmp/h3-omni-1002')
original=dispatch.H3_VAE_OPERATOR_TABLE
sm120=dispatch.H3VAEOperatorSet(supports=lambda device: device.type=='cuda' and torch.cuda.get_device_capability(device)==(12,0),qk_norm_rope=dispatch.try_qk_norm_rope_exact,scaled_residual=dispatch.try_scaled_residual_exact)
torch.set_num_threads(4)
# Input is a real sampled H3 latent; the equality gate compares the same adapter input.
sys.path.insert(0,'/tmp/h3-comfy-submit-1002/base')
sample=torch.load(root/'sg-sample-0-ck1.pt',map_location='cpu',weights_only=False)
latent=sample['samples'].unbind()[0].to('cuda',torch.float32)
report={'scope':'official full H3 video VAE through vLLM-Omni adapter, same sampled latent; no DiT or encoders','torch':torch.__version__,'gpu':torch.cuda.get_device_name(),'capability':torch.cuda.get_device_capability(),'input_shape':list(latent.shape),'records':[]}
def sha(t):return hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
reference=None
with torch.inference_mode():
 for arm in ('baseline','sm120','sm120','baseline'):
  dispatch.H3_VAE_OPERATOR_TABLE=original+((sm120,) if arm=='sm120' else ())
  decoder=MiniMaxH3VideoVAE(str(root/'official/Ref2VA/video_vae'),device=torch.device('cuda:0'),decode_only=True,trust_remote_code=True)
  installed=bool(getattr(decoder.model.decoder,'_omni_h3_vae_optimizations_installed',False))
  print('INSTALLED',arm,installed,flush=True)
  if arm=='sm120' and not installed:raise RuntimeError('official model rejected optimized installer; no performance claim')
  for repeat in range(-1,2):
   torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
   with torch.autocast("cuda",dtype=torch.float16):
    output=decoder.decode_latent(latent)
   torch.cuda.synchronize();ms=1000*(time.perf_counter()-start);output=output.cpu()
   if reference is None:reference=output.clone()
   delta=output.float()-reference.float();row={'arm':arm,'repeat':repeat,'warmup':repeat<0,'ms':ms,'installed':installed,'shape':list(output.shape),'dtype':str(output.dtype),'sha256':sha(output),'exact':torch.equal(output,reference),'max_abs':delta.abs().max().item(),'mae':delta.abs().mean().item(),'peak_bytes':torch.cuda.max_memory_allocated()}
   report['records'].append(row);(root/'vo-vae-full.json').write_text(json.dumps(report,indent=2));print(json.dumps(row),flush=True)
  del decoder,output;gc.collect();torch.cuda.empty_cache()
