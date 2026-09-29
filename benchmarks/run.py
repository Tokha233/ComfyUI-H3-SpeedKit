#!/usr/bin/env python3
"""Run a real public Ref2VA input through stock and SpeedKit, with output hashes.

This is an opt-in GPU benchmark, never run by CI. Models/media stay local.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--comfyui', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True, help='INT8 ConvRot H3; merge a LoRA separately if desired')
    p.add_argument('--text-encoder', type=Path, required=True)
    p.add_argument('--video-vae', type=Path, required=True)
    p.add_argument('--audio-vae', type=Path, required=True)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--prompt', default='<Picture 1> is a colorful toy on a table. A clear voice says: Hello, welcome to our workshop. Static camera.')
    p.add_argument('--width', type=int, default=768)
    p.add_argument('--height', type=int, default=512)
    p.add_argument('--frames', type=int, default=124)
    p.add_argument('--steps', type=int, default=8)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--no-new-shape-verification', action='store_true', help='Exercise the original-model fallback for an unqualified layout')
    p.add_argument('--repeats', type=int, default=2)
    p.add_argument('--output', type=Path, default=Path('results/public'))
    a = p.parse_args()
    if a.output.exists():
        p.error('--output must be a new directory')
    a.output.mkdir(parents=True)
    sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(a.comfyui.resolve())]
    import comfy.options
    comfy.options.enable_args_parsing()
    sys.argv = ['speedkit-benchmark', '--disable-cuda-malloc']
    import main as comfy_main  # initialize the normal Comfy runtime
    import torch
    import numpy as np
    from PIL import Image
    import comfy.sd
    import comfy.utils
    import comfy.model_management as mm
    import nodes
    from comfy_extras.nodes_minimax_h3 import MiniMaxH3ReferenceToVideo, MiniMaxH3SigmaShift
    from comfy_extras.nodes_audio import vae_decode_audio
    from comfy.ldm.modules.attention import attention_comfy_kitchen_int8
    from h3_speedkit.runtime import patch_model
    from h3_speedkit.video import load_video_vae, decode_samples, RGBFrames
    from h3_speedkit.export import export_mp4
    torch.set_num_threads(4)
    report = {'schema': 1, 'status': 'running', 'records': [], 'precision': 'INT8 DiT + INT8 VAE / GPU FP32 audio',
              'timing': 'frozen condition to completed local MP4; hash and condition preparation excluded',
              'input': {'width': a.width, 'height': a.height, 'requested_frames': a.frames,
                        'steps': a.steps, 'seed': a.seed}, 'environment': {}}
    def save():
        (a.output / 'result.json').write_text(json.dumps(report, indent=2))
    def sha(t):
        return hashlib.sha256(t.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
    save()
    try:
        with torch.inference_mode():
            model = comfy.sd.load_diffusion_model(str(a.model), model_options={})
            model = MiniMaxH3SigmaShift.execute(model, 12.0, 4.0)[0]
            stock = model.clone()
            stock.set_model_optimized_attention(attention_comfy_kitchen_int8)
            video = load_video_vae(a.video_vae)
            audio = comfy.sd.VAE(comfy.utils.load_torch_file(str(a.audio_vae)), dtype=torch.float32)
            clip = comfy.sd.load_clip(ckpt_paths=[str(a.text_encoder)], embedding_directory=[], clip_type=comfy.sd.CLIPType.MINIMAX)
            image = torch.from_numpy(np.array(Image.open(a.reference).convert('RGB')).copy()).float().unsqueeze(0) / 255
            cond = MiniMaxH3ReferenceToVideo.execute(clip, video, audio, a.prompt, a.width, a.height,
                    a.frames, 'match', ref_images={'ref_image_0': image})
            positive, latent = cond[0], cond[1]
            negative = nodes.ConditioningZeroOut().zero_out(positive)[0]
            del clip, image
            mm.soft_empty_cache()
            fast = patch_model(model, verify_new_shapes=not a.no_new_shape_verification)
            for repeat in range(-1, a.repeats):
                order = [('stock', stock), ('speedkit', fast)] if repeat % 2 else [('speedkit', fast), ('stock', stock)]
                for arm, current in order:
                    torch.cuda.synchronize()
                    started = time.perf_counter()
                    sample = nodes.common_ksampler(current, a.seed, a.steps, 1.0, 'euler', 'beta',
                                                  positive, negative, latent, denoise=1.0)[0]
                    torch.cuda.synchronize()
                    dit = time.perf_counter() - started
                    t = time.perf_counter()
                    if arm == 'stock':
                        parts = sample['samples'].unbind()
                        pixels = video.decode(parts[0]).cpu()
                        if pixels.ndim == 5 and pixels.shape[0] == 1:
                            pixels = pixels[0]
                        frames = RGBFrames((pixels * 255).clamp(0, 255).byte(), 24.0)
                    else:
                        frames = decode_samples(sample, video)
                    pcm = vae_decode_audio(audio, sample)
                    torch.cuda.synchronize()
                    decode = time.perf_counter() - t
                    name = f'{arm}-{repeat}'
                    export_mp4(frames, a.output / (name + '.mp4'), pcm)
                    elapsed = time.perf_counter() - started
                    hashes = [sha(t) for t in sample['samples'].unbind()] + [sha(frames.pixels), sha(pcm['waveform'])]
                    row = {'arm': arm, 'repeat': repeat, 'warmup': repeat == -1, 'seconds': elapsed,
                           'dit_seconds': dit, 'decode_seconds': decode, 'sha256': hashes}
                    if arm == 'speedkit':
                        row['backend'] = fast._h3_speedkit.last_report
                    report['records'].append(row)
                    save()
                    print(json.dumps(row), flush=True)
                    del sample, frames, pcm
            references = {r['repeat']: r['sha256'] for r in report['records'] if r['arm'] == 'stock'}
            for r in report['records']:
                r['four_sha_equal_stock'] = r['sha256'] == references[r['repeat']]
            report['status'] = 'success'
            save()
    except BaseException as error:
        report.update(status='error', error_type=type(error).__name__)
        save()
        raise


if __name__ == '__main__':
    main()
