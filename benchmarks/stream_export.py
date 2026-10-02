#!/usr/bin/env python3
"""Compare serial/streamed video output from an existing local H3 latent.

No model downloads or DiT generation. Audio is decoded before timed intervals.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--comfyui", type=Path, required=True)
    p.add_argument("--video-vae", type=Path, required=True)
    p.add_argument("--audio-vae", type=Path, required=True)
    p.add_argument("--latent", type=Path, required=True,
                   help="torch.save({'parts': [video_tensor, audio_tensor]}, path)")
    p.add_argument("--fps", type=float, default=24)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--order", default="serial,stream,stream,serial")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.repeats < 1 or any(x not in ("serial", "stream") for x in a.order.split(",")):
        p.error("Use at least one repeat and serial/stream arms")
    a.output.mkdir(parents=True, exist_ok=False)
    sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(a.comfyui.resolve())]
    import comfy.options
    comfy.options.enable_args_parsing()
    sys.argv = ["stream-benchmark", "--disable-cuda-malloc", "--reserve-vram", "1.5"]
    import main as comfy_main
    import av
    import torch
    import comfy.sd
    import comfy.utils
    import comfy.model_management as mm
    from comfy.nested_tensor import NestedTensor
    from comfy_extras.nodes_audio import vae_decode_audio
    from h3_speedkit.video import load_video_vae, decode_rgb, RGBFrames
    from h3_speedkit.export import export_mp4
    from h3_speedkit.stream_export import StreamExport

    torch.set_num_threads(a.threads)
    for name in ("allow_fp16_accumulation", "allow_fp16_reduced_precision_reduction",
                 "allow_bf16_reduced_precision_reduction", "allow_tf32"):
        setattr(torch.backends.cuda.matmul, name, False)
    mm.set_cudnn_benchmark()
    report = {"status": "running", "rows": [], "torch": torch.__version__, "av": av.__version__,
              "gpu": torch.cuda.get_device_name(), "cpu_threads": a.threads,
              "scope": "Video decode to completed MP4; excludes audio decode, model load, DiT and hashes"}

    def save():
        (a.output / "result.json").write_text(json.dumps(report, indent=2))

    with torch.inference_mode():
        parts = torch.load(a.latent, map_location="cpu", weights_only=True)["parts"]
        video = load_video_vae(a.video_vae)
        audio = comfy.sd.VAE(comfy.utils.load_torch_file(str(a.audio_vae)), dtype=torch.float32)
        pcm = vae_decode_audio(audio, {"samples": NestedTensor(parts)})
        pcm = {**pcm, "waveform": pcm["waveform"].detach().float().cpu()}
        mm.unload_model_and_clones(audio.patcher, unload_additional_models=False)
        del audio
        mm.load_models_gpu([video.patcher], memory_required=6 * 1024**3, force_full_load=True)
        latent = parts[0].to(device=video.device, dtype=torch.float16)
        _, c, t, h, w = video.first_stage_model.decode_output_shape(latent.shape)
        signatures = None
        save()
        for group, arm in enumerate(a.order.split(",")):
            for rep in range(-1, a.repeats):
                path = a.output / f"{group}-{arm}-{rep}.mp4"
                torch.cuda.synchronize()
                start = time.perf_counter()
                if arm == "stream":
                    with StreamExport((t, h, w, c), a.fps, path, pcm) as encoder:
                        rgb = decode_rgb(video.first_stage_model, latent, on_frames=encoder.write)
                        torch.cuda.synchronize()
                        decoded = time.perf_counter()
                else:
                    rgb = decode_rgb(video.first_stage_model, latent)
                    torch.cuda.synchronize()
                    decoded = time.perf_counter()
                    export_mp4(RGBFrames(rgb, a.fps), path, pcm)
                end = time.perf_counter()
                hashes = [hashlib.sha256(rgb.numpy().tobytes()).hexdigest(),
                          hashlib.sha256(pcm["waveform"].numpy().tobytes()).hexdigest()]
                with path.open("rb") as f:
                    hashes.append(hashlib.sha256(f.read()).hexdigest())
                if signatures is None:
                    signatures = hashes
                if hashes != signatures:
                    raise RuntimeError("RGB, PCM or MP4 changed")
                row = {"group": group, "arm": arm, "warmup": rep < 0,
                       "decode_seconds": decoded - start, "total_seconds": end - start,
                       "sha256_rgb_pcm_mp4": hashes}
                report["rows"].append(row)
                save()
                print(json.dumps(row), flush=True)
                del rgb
        report["status"] = "success"
        save()


if __name__ == "__main__":
    main()
