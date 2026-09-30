#!/usr/bin/env python3
"""Isolate Kitchen PR #219 in a real H3 workflow; both arms use stock VAE/export.

Requires an explicitly built PR #219 Kitchen, the pinned ComfyUI, and local
public models. Frozen condition files must be trusted (they contain pickle).
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import statistics
import sys
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--comfyui', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--video-vae', type=Path, required=True)
    p.add_argument('--audio-vae', type=Path, required=True)
    p.add_argument('--condition', type=Path, help='Trusted frozen positive/latent file; replaces text/reference preparation')
    p.add_argument('--text-encoder', type=Path)
    p.add_argument('--reference', type=Path)
    p.add_argument('--prompt', default='<Picture 1> is a colorful toy on a table. A clear voice says: Hello, welcome to our workshop. Static camera.')
    p.add_argument('--width', type=int, default=768)
    p.add_argument('--height', type=int, default=512)
    p.add_argument('--frames', type=int, default=124)
    p.add_argument('--steps', type=int, default=8)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--repeats', type=int, default=4)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.repeats < 2 or a.steps < 1:
        p.error('Use at least two formal repeats and one step')
    if not a.condition and not (a.reference and a.text_encoder):
        p.error('Supply --condition or both --reference and --text-encoder')
    if a.output.exists():
        p.error('--output must be a new directory')
    a.output.mkdir(parents=True)
    sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(a.comfyui.resolve())]
    import comfy.options
    comfy.options.enable_args_parsing()
    sys.argv = ['indexed-gate-benchmark', '--disable-cuda-malloc']
    import main as comfy_main  # noqa: F401 -- initialize normal Comfy runtime
    import torch
    import comfy.sd
    import comfy.utils
    import comfy.model_management as mm
    import nodes
    from comfy_extras.nodes_minimax_h3 import MiniMaxH3ReferenceToVideo, MiniMaxH3SigmaShift
    from comfy_extras.nodes_audio import vae_decode_audio
    from comfy.ldm.modules.attention import attention_comfy_kitchen_int8
    from h3_speedkit.indexed_gate import patch_model
    from h3_speedkit.video import load_video_vae, RGBFrames
    from h3_speedkit.export import export_mp4
    torch.set_num_threads(4)
    report = {'schema': 1, 'status': 'running', 'records': [],
              'optimization': 'Kitchen PR #219 only: outproj + FC2 indexed gate epilogues',
              'timing': 'frozen condition to completed local MP4; hashes/conditioning/cold load excluded',
              'input': {'width': a.width, 'height': a.height, 'requested_frames': a.frames,
                        'steps': a.steps, 'seed': a.seed, 'frozen_condition': bool(a.condition)},
              'environment': {'torch': torch.__version__, 'torch_git': torch.version.git_version,
                              'gpu': torch.cuda.get_device_name(),
                              'kitchen_version': importlib.metadata.version('comfy-kitchen')}}

    def save():
        (a.output / 'result.json').write_text(json.dumps(report, indent=2))

    def sha(t):
        return hashlib.sha256(t.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()

    save()
    try:
        with torch.inference_mode():
            model = comfy.sd.load_diffusion_model(str(a.model))
            model = MiniMaxH3SigmaShift.execute(model, 12.0, 4.0)[0]
            model.set_model_optimized_attention(attention_comfy_kitchen_int8)
            stock, fast = model.clone(), patch_model(model)
            video = load_video_vae(a.video_vae)
            audio = comfy.sd.VAE(comfy.utils.load_torch_file(str(a.audio_vae)), dtype=torch.float32)
            if a.condition:
                saved = torch.load(a.condition, map_location='cpu', weights_only=False)
                positive, latent = saved['positive'], saved['latent']
            else:
                import numpy as np
                from PIL import Image
                clip = comfy.sd.load_clip(ckpt_paths=[str(a.text_encoder)], embedding_directory=[], clip_type=comfy.sd.CLIPType.MINIMAX)
                image = torch.from_numpy(np.array(Image.open(a.reference).convert('RGB')).copy()).float().unsqueeze(0) / 255
                cond = MiniMaxH3ReferenceToVideo.execute(clip, video, audio, a.prompt, a.width, a.height,
                        a.frames, 'match', ref_images={'ref_image_0': image})
                positive, latent = cond[0], cond[1]
                del clip, image
            negative = nodes.ConditioningZeroOut().zero_out(positive)[0]
            mm.soft_empty_cache()
            for repeat in range(-1, a.repeats):
                arms = [('stock', stock), ('indexed_gate', fast)]
                if repeat >= 0 and repeat % 2 == 0:
                    arms.reverse()
                for arm, current in arms:
                    before = fast._h3_indexed_gate.totals.copy()
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    started = time.perf_counter()
                    sample = nodes.common_ksampler(current, a.seed, a.steps, 1.0, 'euler', 'beta',
                                                  positive, negative, latent, denoise=1.0)[0]
                    torch.cuda.synchronize()
                    dit = time.perf_counter() - started
                    t = time.perf_counter()
                    # Identical stock decode, CPU RGB conversion and export on both sides.
                    parts = sample['samples'].unbind()
                    pixels = video.decode(parts[0]).cpu()
                    if pixels.ndim == 5 and pixels.shape[0] == 1:
                        pixels = pixels[0]
                    frames = RGBFrames((pixels * 255).clamp(0, 255).byte(), 24.0)
                    del pixels
                    torch.cuda.synchronize()
                    video_seconds = time.perf_counter() - t
                    t = time.perf_counter()
                    pcm = vae_decode_audio(audio, sample)
                    torch.cuda.synchronize()
                    audio_seconds = time.perf_counter() - t
                    t = time.perf_counter()
                    export_mp4(frames, a.output / f'{arm}-{repeat}.mp4', pcm)
                    export_seconds = time.perf_counter() - t
                    elapsed = time.perf_counter() - started
                    row = {'arm': arm, 'repeat': repeat, 'warmup': repeat < 0,
                           'seconds': elapsed, 'dit_seconds': dit, 'video_seconds': video_seconds,
                           'audio_seconds': audio_seconds, 'export_seconds': export_seconds,
                           'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                           'sha256': [sha(t) for t in parts] + [sha(frames.pixels), sha(pcm['waveform'])]}
                    if arm == 'indexed_gate':
                        row['backend'] = dict(fast._h3_indexed_gate.last_report)
                        row['forward_counts'] = dict(fast._h3_indexed_gate.totals - before)
                    report['records'].append(row)
                    save()
                    print(json.dumps(row), flush=True)
                    if arm == 'indexed_gate' and 'fallback' in row['backend']:
                        raise RuntimeError('Indexed gate fell back: ' + row['backend']['fallback'])
                    if arm == 'indexed_gate' and row['backend'].get('counts') != {'blocks': 50, 'outproj': 50, 'fc2': 50}:
                        raise RuntimeError('Missing real indexed-gate execution coverage')
                    if arm == 'indexed_gate':
                        counts = row['forward_counts']
                        if (counts.get('forwards') != a.steps or counts.get('blocks') != 50 * a.steps
                                or counts.get('outproj') != 50 * a.steps or counts.get('fc2') != 50 * a.steps
                                or counts.get('fallback_forwards', 0)):
                            raise RuntimeError('Incomplete optimized sampling coverage')
                        if repeat >= 0 and counts.get('verification_blocks', 0):
                            raise RuntimeError('First-use verification leaked into formal timing')
                    del sample, parts, frames, pcm
            references = {r['repeat']: r['sha256'] for r in report['records'] if r['arm'] == 'stock'}
            for r in report['records']:
                r['four_sha_equal_stock'] = r['sha256'] == references[r['repeat']]
            if not all(r['four_sha_equal_stock'] for r in report['records']):
                raise RuntimeError('Output signature mismatch')
            means = {arm: {key: statistics.mean(r[key] for r in report['records']
                if r['arm'] == arm and not r['warmup']) for key in ('dit_seconds', 'seconds')}
                for arm in ('stock', 'indexed_gate')}
            report['summary'] = {'means': means, 'time_reduction_percent': {
                key: 100 * (1 - means['indexed_gate'][key] / means['stock'][key])
                for key in ('dit_seconds', 'seconds')}}
            report['status'] = 'success'
            save()
    except BaseException as error:
        report.update(status='error', error_type=type(error).__name__, error=str(error))
        save()
        raise


if __name__ == '__main__':
    main()
