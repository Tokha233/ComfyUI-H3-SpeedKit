from pathlib import Path
import sys,time,json,hashlib,gc
root=Path('/tmp/h3-omni-1002');dr=Path('/tmp/h3-kitchen-prs-0930')
sys.path[:0]=[str(dr/'combined'),'/tmp/h3-comfy-submit-1002/base',str(dr/'current-deps'),str(dr/'build-deps'),'/last/deps']
import comfy.options
comfy.options.enable_args_parsing();sys.argv=['quality','--disable-cuda-malloc']
import main,torch,comfy.sd,comfy.utils
from comfy_extras.nodes_audio import vae_decode_audio
from skimage.metrics import structural_similarity
import numpy as np
sources={'comfy':'comfy-control.pt','sg_sdpa':'sg-sample-0-ck0.pt','sg_int8':'sg-sample-0-ck1.pt','sg_resident24':'sg-sample-0-ck1-r24-p0.0.pt'}
report={'scope':'one public 768x512 124frame Larry8 Ref2VA fixture; same shared INT8 video VAE and FP32 audio VAE; no BF16 teacher run','records':[]}
def metrics(a,b):
 a=a.double().reshape(-1);b=b.double().reshape(-1);d=a-b
 return {'exact':torch.equal(a,b),'mae':d.abs().mean().item(),'rmse':d.square().mean().sqrt().item(),'max_abs':d.abs().max().item(),'cosine':torch.nn.functional.cosine_similarity(a,b,dim=0).item()}
with torch.inference_mode():
 torch.set_num_threads(4)
 samples={k:torch.load(root/v,map_location='cpu',weights_only=False) for k,v in sources.items()}
 sd,metadata=comfy.utils.load_torch_file('/decoder/models/minimax_h3_video_vae_int8_convrot.safetensors',return_metadata=True);video=comfy.sd.VAE(sd,metadata=metadata,dtype=torch.float16);del sd
 audio=comfy.sd.VAE(comfy.utils.load_torch_file('/models/vae/minimax_h3_audio_vae_fp32.safetensors'),dtype=torch.float32)
 outputs={}
 for name,sample in samples.items():
  parts=list(sample['samples'].unbind());start=time.perf_counter();pixels=video.decode(parts[0]).cpu()
  if pixels.ndim==5:pixels=pixels.squeeze(0)
  rgb=(pixels*255).clamp(0,255).byte();del pixels
  pcm=vae_decode_audio(audio,sample)['waveform'].cpu();outputs[name]=(rgb,pcm)
  if name=='comfy':refparts=parts
  row={'name':name,'decode_seconds':time.perf_counter()-start,'latent_video':metrics(parts[0],refparts[0]),'latent_audio':metrics(parts[1],refparts[1]),'rgb':metrics(rgb,outputs['comfy'][0]),'pcm':metrics(pcm,outputs['comfy'][1])}
  ref=outputs['comfy'][0];scores=[structural_similarity(x.numpy(),y.numpy(),data_range=255,channel_axis=-1) for x,y in zip(ref,rgb)];row['video_ssim_mean']=float(np.mean(scores));row['video_ssim_min']=float(np.min(scores));row['video_psnr']=float(-10*np.log10(np.mean((ref.numpy().astype(float)-rgb.numpy().astype(float))**2)/(255**2))) if not torch.equal(ref,rgb) else 'inf'
  spec=lambda x:torch.stft(x.float().reshape(-1,x.shape[-1]),n_fft=1024,hop_length=256,window=torch.hann_window(1024),return_complex=True).abs()
  row['audio_magnitude_spectral_cosine']=metrics(spec(pcm),spec(outputs['comfy'][1]))['cosine']
  report['records'].append(row);(root/'quality-compare.json').write_text(json.dumps(report,indent=2));print(json.dumps(row),flush=True)
  torch.save({'rgb':rgb,'pcm':pcm},root/f'quality-preview-{name}.pt')
