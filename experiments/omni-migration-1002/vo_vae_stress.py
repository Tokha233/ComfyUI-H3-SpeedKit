import os
os.environ['HF_HUB_OFFLINE']='1'
os.environ['VLLM_OMNI_VAE_LEGACY_TEMPORAL']='1'
import gc, hashlib, json, pathlib, sys, time
import torch
from vllm_omni.diffusion.models.minimax_h3.vae import MiniMaxH3VideoVAE
from vllm_omni.diffusion.models.minimax_h3.ops.vae import dispatch
root=pathlib.Path('/tmp/h3-omni-1002')
sys.path.insert(0,'/tmp/h3-comfy-submit-1002/base')
sample=torch.load(root/'sg-sample-0-ck1.pt',map_location='cpu',weights_only=False)
full=sample['samples'].unbind()[0].float()
torch.manual_seed(713)
inputs={'sampled_crop':full[:,:,:17,:16,:24].contiguous(), 'zero':torch.zeros_like(full[:,:,:13,:16,:16]), 'noise':torch.randn_like(full[:,:,:21,:16,:24])*0.5}
original=dispatch.H3_VAE_OPERATOR_TABLE
reference={}
report={'scope':'vLLM native official full VAE float output; extra latent shapes and inputs; not end-to-end generation','torch':torch.__version__,'gpu':torch.cuda.get_device_name(),'records':[]}
torch.set_num_threads(4)
with torch.inference_mode():
 for arm in ('baseline','sm120'):
  dispatch.H3_VAE_OPERATOR_TABLE=() if arm=='baseline' else original
  decoder=MiniMaxH3VideoVAE(str(root/'official/Ref2VA/video_vae'),device=torch.device('cuda:0'),decode_only=True,trust_remote_code=True)
  installed=bool(getattr(decoder.model.decoder,'_omni_h3_vae_optimizations_installed',False))
  assert installed==(arm=='sm120')
  for name,cpu_latent in inputs.items():
   latent=cpu_latent.cuda()
   for repeat in (-1,0,1):
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); start=time.perf_counter()
    with torch.autocast('cuda',dtype=torch.float16):out=decoder.decode_latent(latent)
    torch.cuda.synchronize(); ms=1000*(time.perf_counter()-start);out=out.cpu()
    assert torch.isfinite(out).all()
    if name not in reference:reference[name]=out.clone()
    delta=out-reference[name]
    row={'arm':arm,'case':name,'input_shape':list(latent.shape),'output_shape':list(out.shape),'dtype':str(out.dtype),'repeat':repeat,'ms':ms,'exact':torch.equal(out,reference[name]),'max_abs':delta.abs().max().item(),'sha256':hashlib.sha256(out.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest(),'peak_bytes':torch.cuda.max_memory_allocated()}
    report['records'].append(row);(root/'vo-vae-stress.json').write_text(json.dumps(report,indent=2));print(json.dumps(row),flush=True)
    assert row['exact'],row
   del out,latent,delta
  del decoder;gc.collect();torch.cuda.empty_cache()
