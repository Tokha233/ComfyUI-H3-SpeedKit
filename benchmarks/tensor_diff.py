# SPDX-License-Identifier: Apache-2.0
"""Compare saved AV latents or raw PCM from trusted local experiments.

python benchmarks/tensor_diff.py reference.pt candidate.pt --output diff.json

Inputs: {"parts": [video_tensor, audio_tensor]} or
{"waveform": tensor, "sample_rate": int}. Does not read MP4/AAC or align signals.
"""

import argparse
import json
from pathlib import Path

import torch


def tensor_metrics(reference, candidate):
    if reference.shape != candidate.shape:
        raise ValueError(f"shape mismatch: {reference.shape} / {candidate.shape}")
    ref = reference.flatten().float()
    cur = candidate.flatten().float()
    if not torch.isfinite(ref).all() or not torch.isfinite(cur).all():
        raise ValueError("non-finite tensor")
    delta = cur - ref
    rms = ref.square().mean().sqrt()
    rmse = delta.square().mean().sqrt()
    denominator = torch.linalg.vector_norm(ref) * torch.linalg.vector_norm(cur)
    return {
        "shape": list(reference.shape),
        "exact": torch.equal(reference, candidate),
        "max_absolute_difference": delta.abs().max().item(),
        "mean_absolute_difference": delta.abs().mean().item(),
        "rmse": rmse.item(),
        "reference_rms": rms.item(),
        "normalized_rmse": (rmse / rms).item() if rms > 0 else None,
        "cosine": (torch.dot(ref, cur) / denominator).item() if denominator > 0 else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    ref = torch.load(args.reference, map_location="cpu", weights_only=True)
    cur = torch.load(args.candidate, map_location="cpu", weights_only=True)
    if "parts" in ref and "parts" in cur:
        if len(ref["parts"]) != 2 or len(cur["parts"]) != 2:
            raise ValueError("expected video/audio pair")
        report = {
            name: tensor_metrics(a, b)
            for name, a, b in zip(["video_latent", "audio_latent"], ref["parts"], cur["parts"])
        }
    elif "waveform" in ref and "waveform" in cur:
        if ref["sample_rate"] != cur["sample_rate"]:
            raise ValueError("sample rates differ")
        report = {"sample_rate": ref["sample_rate"], "pcm": tensor_metrics(ref["waveform"], cur["waveform"])}
    else:
        raise ValueError("expected parts or waveform dictionaries")
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
