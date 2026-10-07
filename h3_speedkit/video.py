# SPDX-License-Identifier: GPL-3.0-or-later
# Temporal decode sequence derived from ComfyUI's MiniMax H3 VAE.
"""INT8 decode with fused RGB8 finalization and bounded asynchronous D2H."""
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import threading
import torch
import comfy.model_management as mm
from .ops.finalize_rgb import quantize_raw
from .vendor.h3_vae import MiniMaxH3VideoVAE


@dataclass(frozen=True)
class RGBFrames:
    pixels: torch.Tensor
    fps: float


class RGBSink:
    def __init__(self, shape, device, on_frames=None):
        if shape[0] != 1 or shape[1] != 3:
            raise ValueError("RGB output requires a single H3 video")
        self.shape = shape
        self.output = torch.empty((shape[2], shape[3], shape[4], 3), dtype=torch.uint8)
        self.device = device
        self.stream = torch.cuda.Stream(device=device)
        self.slots = [None, None]
        self.index = self.position = 0
        self.on_frames = on_frames
        self.worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="h3-rgb-drain") if on_frames else None
        self.pending = [None, None]

    def drain(self, index):
        if self.worker is not None:
            if self.pending[index] is not None:
                self.pending[index].result()
                self.pending[index] = None
            return
        slot = self.slots[index]
        if slot is not None and slot.get("event") is not None:
            slot["event"].synchronize()
            count = slot["stop"] - slot["start"]
            self.output[slot["start"]:slot["stop"]].copy_(slot["pinned"][:count])
            slot["event"] = None

    def write(self, rgb):
        index = self.index % 2
        self.drain(index)
        count = len(rgb)
        slot = self.slots[index]
        if slot is None or len(slot["pinned"]) < count:
            slot = {"pinned": torch.empty_like(rgb, device="cpu", pin_memory=True)}
            self.slots[index] = slot
        self.stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(self.stream):
            slot["pinned"][:count].copy_(rgb, non_blocking=True)
            rgb.record_stream(self.stream)
            event = torch.cuda.Event()
            event.record(self.stream)
        slot.update(start=self.position, stop=self.position + count, event=event)
        if self.worker is not None:
            self.pending[index] = self.worker.submit(
                self.deliver, event, slot["pinned"], self.position, self.position + count)
        self.position += count
        self.index += 1

    def deliver(self, event, pinned, start, stop):
        import numpy as np
        event.synchronize()
        # The callback owns an immutable output view, never a reusable pinned slot.
        np.copyto(self.output[start:stop].numpy(), pinned[:stop-start].numpy())
        self.on_frames(start, self.output[start:stop])

    def close(self):
        try:
            self.stream.synchronize()
        finally:
            if self.worker is not None:
                self.worker.shutdown(wait=True)

    def finish(self):
        for index in sorted(range(2), key=lambda i: self.slots[i]["start"] if self.slots[i] else -1):
            self.drain(index)
        if self.position != len(self.output):
            raise RuntimeError("Incomplete RGB frame sequence")
        return self.output


def decode_rgb(model, latent, *, on_frames=None):
    """Return CPU FHWC uint8. Blending finishes before quantization."""
    if type(model) is not MiniMaxH3VideoVAE:
        raise ValueError("Use the H3 SpeedKit Video VAE Loader")
    sink = RGBSink(model.decode_output_shape(latent.shape), latent.device, on_frames)
    def write(part):
        count = min(part.shape[2], len(sink.output) - sink.position)
        if count > 0:
            frames = part[0, :, :count].permute(1, 2, 3, 0)
            sink.write(quantize_raw(frames, model.pixel_std, model.pixel_mean))

    try:
        z = latent * model.latents_std.view(1, -1, 1, 1, 1).to(latent)
        z = z + model.latents_mean.view(1, -1, 1, 1, 1).to(latent)
        if z.shape[2] == 1:
            write(model._adaptive_decode(z)[:, :, -1:])
        else:
            pad, chunks = model._decode_temporal_chunks(z.shape[2])
            if pad:
                z = torch.cat((z, z[:, :, -1:].repeat(1, 1, pad, 1, 1)), dim=2)
            previous = None
            chunk_frames = model.tokens_chunk_size * model.vae_ratio_t
            for index in range(chunks):
                mm.throw_exception_if_processing_interrupted()
                start = index * model.tokens_chunk_size
                clip = model._adaptive_decode(z[:, :, start:start + model.tokens_chunk_size + model.token_overlap])
                for half in range(int(model.token_drop > 0) + 1):
                    part = clip[:, :, half * chunk_frames:min((half + 1) * chunk_frames, clip.shape[2])]
                    part = part[:, :, model.frame_pre_padding:]
                    if half == 0:
                        if previous is not None:
                            part = model.blend(previous, part, model.frame_overlap, dim=-3)
                            previous = None
                        write(part)
                    else:
                        previous = part.contiguous()
                if index == chunks - 1 and previous is not None:
                    write(previous)
                    previous = None
                del clip, part
        return sink.finish()
    finally:
        sink.close()


def load_video_vae(filename):
    """Load the pinned decoder per-instance; no global Comfy class mutation."""
    import comfy.sd
    import comfy.utils
    import comfy.ops
    state, metadata = comfy.utils.load_torch_file(str(filename), return_metadata=True)
    wrapper = comfy.sd.VAE(state, metadata=metadata, dtype=torch.float16)
    if type(wrapper.first_stage_model).__name__ != "MiniMaxH3VideoVAE":
        raise ValueError("Selected file is not a MiniMax H3 video VAE")
    quant = comfy.utils.detect_layer_quantization(state, "")
    operations = comfy.ops.mixed_precision_ops(quant, torch.float16) if quant is not None else comfy.ops.disable_weight_init
    decoder = MiniMaxH3VideoVAE(operations=operations)
    missing, unexpected = decoder.load_state_dict(state, strict=False, assign=True)
    if missing or unexpected:
        raise ValueError(f"VAE state mismatch: missing={missing}, unexpected={unexpected}")
    decoder.eval()
    decoder.to(dtype=torch.float16)
    wrapper.first_stage_model = decoder
    wrapper.patcher = type(wrapper.patcher)(decoder, load_device=wrapper.device,
                                            offload_device=wrapper.patcher.offload_device)
    wrapper._speedkit_lock = threading.RLock()
    return wrapper


def decode_samples(samples, vae, fps=24.0, *, on_frames=None):
    parts = samples["samples"]
    latent = parts if isinstance(parts, torch.Tensor) and parts.ndim == 5 else parts.unbind()[0]
    if latent.ndim != 5 or latent.shape[1] != 24:
        raise ValueError("Expected H3 video latent [1,24,T,H,W]")
    with vae._speedkit_lock, torch.inference_mode():
        mm.load_models_gpu([vae.patcher], memory_required=6 * 1024 ** 3, force_full_load=True)
        pixels = decode_rgb(vae.first_stage_model, latent.to(device=vae.device, dtype=torch.float16),
                            on_frames=on_frames)
    return RGBFrames(pixels, float(fps))


def decode_to_mp4(samples, vae, path, audio=None, *, fps=24.0, crf=23):
    """Overlap CPU encoding with decode; return only a completed local MP4."""
    from .stream_export import StreamExport
    parts = samples["samples"]
    latent = parts if isinstance(parts, torch.Tensor) and parts.ndim == 5 else parts.unbind()[0]
    if latent.ndim != 5 or latent.shape[1] != 24:
        raise ValueError("Expected H3 video latent [1,24,T,H,W]")
    if audio is not None:
        audio = {**audio, "waveform": audio["waveform"].detach().float().cpu()}
    b, c, t, h, w = vae.first_stage_model.decode_output_shape(latent.shape)
    with StreamExport((t, h, w, c), fps, path, audio, crf=crf) as exporter:
        decode_samples(samples, vae, fps, on_frames=exporter.write)
    return exporter.result
