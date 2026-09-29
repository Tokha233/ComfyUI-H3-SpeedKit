# SPDX-License-Identifier: Apache-2.0
"""CPU H.264/AAC exporter matching the measured R85 color/codec contract."""
from pathlib import Path
from fractions import Fraction
import math
import os
import uuid


def export_mp4(frames, path, audio=None, *, crf=23, threads=4):
    import av
    import numpy as np
    import torch
    from av.video.reformatter import ColorPrimaries, ColorRange, ColorTrc
    rgb = frames.pixels
    if rgb.device.type != "cpu" or rgb.dtype != torch.uint8 or rgb.ndim != 4 or rgb.shape[-1] != 3:
        raise ValueError("Expected CPU FHWC RGB8 frames")
    if not 1 <= frames.fps <= 120 or not 0 <= crf <= 51 or not 1 <= threads <= 64:
        raise ValueError("Invalid frame rate, CRF or thread limit")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + "." + uuid.uuid4().hex + ".partial.mp4")
    fps = Fraction(round(frames.fps * 1000), 1000)

    def color(item):
        item.color_primaries = ColorPrimaries.BT709
        item.color_trc = ColorTrc.IEC61966_2_1
        item.colorspace = 1
        item.color_range = ColorRange.MPEG

    try:
        with av.open(str(temporary), mode="w", format="mp4", options={"movflags": "use_metadata_tags+faststart"}) as output:
            video = output.add_stream("h264", rate=fps)
            video.width, video.height = rgb.shape[2], rgb.shape[1]
            video.pix_fmt = "yuv420p"
            video.codec_context.thread_count = threads
            video.options = {"crf": str(crf)}
            color(video.codec_context)
            track = None
            if audio is not None:
                pcm = audio["waveform"].detach().float().cpu()
                if pcm.ndim != 3 or pcm.shape[0] != 1 or pcm.shape[1] not in (1, 2, 6):
                    raise ValueError("Audio must be [1,channels,samples], channels=1,2,6")
                rate = int(audio["sample_rate"])
                if rate not in (32000, 44100, 48000):
                    raise ValueError("Unsupported audio sample rate")
                layout = {1: "mono", 2: "stereo", 6: "5.1"}[pcm.shape[1]]
                track = output.add_stream("aac", rate=rate, layout=layout)
            for pixels in rgb:
                frame = av.VideoFrame.from_ndarray(pixels.contiguous().numpy(), format="rgb24")
                frame = frame.reformat(format="yuv420p", dst_colorspace=1)
                color(frame)
                output.mux(video.encode(frame))
            output.mux(video.encode(None))
            if track is not None:
                wave = np.ascontiguousarray(pcm[0, :, :math.ceil(rate / fps * len(rgb))].numpy())
                frame = av.AudioFrame.from_ndarray(wave, format="fltp", layout=layout)
                frame.sample_rate, frame.pts = rate, 0
                output.mux(track.encode(frame))
                output.mux(track.encode(None))
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return str(path)
