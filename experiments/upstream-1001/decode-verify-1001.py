from pathlib import Path
import sys,json,hashlib,time
r=Path('/tmp/h3-kitchen-prs-0930');sys.path[:0]=[str(r/'base'),str(r/'comfy-current'),str(r/'current-deps'),str(r/'build-deps'),'/last/deps']
import comfy.options
comfy.options.enable_args_parsing();sys.argv=['decode-verify','--disable-cuda-malloc']
import main,torch,comfy.sd,comfy.utils
from comfy_extras.nodes_audio import vae_decode_audio
from comfy.nested_tensor import NestedTensor
report={'video_vae':'minimax_h3_video_vae_fp16.safetensors','audio_vae':'minimax_h3_audio_vae_fp32.safetensors','records':[]}
def sha(t):return hashlib.sha256(t.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
torch.set_num_threads(4)
with torch.inference_mode():
    video=comfy.sd.VAE(comfy.utils.load_torch_file('/models/vae/minimax_h3_video_vae_fp16.safetensors'),dtype=torch.float16)
    audio=comfy.sd.VAE(comfy.utils.load_torch_file('/models/vae/minimax_h3_audio_vae_fp32.safetensors'),dtype=torch.float32)
    for arm in ['base','combined']:
        parts=torch.load(r/f'combined-latents-{arm}.pt',map_location='cpu',weights_only=True)
        start=time.perf_counter();pixels=video.decode(parts[0]).cpu();torch.cuda.synchronize()
        rgb=(pixels*255).clamp(0,255).byte();seconds=time.perf_counter()-start
        sample={'samples':NestedTensor(parts)}
        start=time.perf_counter();pcm=vae_decode_audio(audio,sample);torch.cuda.synchronize()
        row={'arm':arm,'video_seconds':seconds,'audio_seconds':time.perf_counter()-start,'shapes':[list(t.shape) for t in parts]+[list(rgb.shape),list(pcm['waveform'].shape)],'sha256':[sha(t) for t in parts]+[sha(rgb),sha(pcm['waveform'])]}
        report['records'].append(row);(r/'combined-decode-1001.json').write_text(json.dumps(report,indent=2));print(json.dumps(row),flush=True)
        del pixels,rgb,pcm,parts,sample
    report['status']='passed' if report['records'][0]['sha256']==report['records'][1]['sha256'] else 'mismatch'
    (r/'combined-decode-1001.json').write_text(json.dumps(report,indent=2))
