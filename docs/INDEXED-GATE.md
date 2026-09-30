# Experimental Kitchen #219 consumer

This optional node demonstrates a real consumer for
[Kitchen PR #219](https://github.com/Comfy-Org/comfy-kitchen/pull/219).
It fuses H3's attention output projection and MLP FC2 with their indexed
modulation gates and residual additions. It does not require SpeedKit's
separate R85 native libraries. The normal **Optimize DiT** node remains the
complete acceleration path; use one model optimization node at a time.

**This is an unmerged upstream experiment, not a released Kitchen feature.**
An ordinary `pip install comfy-kitchen==0.2.36` does not contain the API.

## Build and connect

Use a separate environment with the [qualified ComfyUI/Torch versions](../configs/compatibility.json),
Linux, an SM120 GPU and CUDA 13 nvcc. Build the exact tested PR head:

```bash
git clone https://github.com/Comfy-Org/comfy-kitchen.git kitchen-pr219
cd kitchen-pr219
git fetch origin pull/219/head
git checkout dc7c739fe5fc7502b50206e81fbe26c6fe3930a1
git submodule update --init --recursive
COMFY_CUDA_ARCHS=120 COMFY_KITCHEN_BUILD_NO_HIP=1 python -m pip install .
python -c 'import comfy_kitchen as ck; print(ck.int8_gemm_indexed_gate)'
```

Install this repository in `ComfyUI/custom_nodes` and restart ComfyUI.
Connect **Load INT8 ConvRot H3 → H3 Sigma Shift → H3 SpeedKit · Indexed Gate
(Kitchen PR 219) → KSampler**. Continue to use your usual VAE/audio/output
nodes. Keep the sampler, steps, LoRA and attention setting fixed for an A/B.
The node preserves the incoming attention setting.

Start from an unpatched MODEL. Do not stack this node with **Optimize DiT**
or another double-block replacement. Use a merged and then quantized Larry
checkpoint; active weight callbacks use the original model. Missing PR API
or a different H3 source revision produces a clear installation error.

## Numerical and fallback contract

- Original QKV, RMS/RoPE, attention, FC1, norm/modulation, sampler and VAE.
- Original ConvRot256 activation quantization, including fused SwiGLU for FC2.
- Preserve the BF16 GEMM result rounding before FP32 gate/residual FMA and
  final BF16 rounding. No change to INT8 model precision or sampling steps.
- Gate table `[G,5376]` plus per-token INT32 row map; no expanded gate matrix.
- Respect ComfyUI's cast/un-cast and offload ownership. Both fused projections
  return new tensors, so validation fallback retains the original block input.
- Model-local clones and a request wrapper; no shared forward-method replacement.
- The first diffusion forward for each new segment layout compares every
  block with the original. This is deliberately slower and excluded from
  steady-state timing. A mismatch returns the original result and rejects
  that layout. Passing these checks is evidence for tested inputs, not a
  mathematical guarantee for all checkpoints, seeds or future library versions.
- Training, graph capture, modified module forwards/hooks, unsupported weights,
  precision overrides or non-integer/gapped modulation segments use the original
  path with a logged reason. CUDA execution failures abort rather than silently
  retrying a potentially unhealthy device.

The node currently qualifies only the pinned H3 implementation on SM120.
The underlying Kitchen API has broader fallbacks; that does not imply the
entire ComfyUI consumer has been qualified on other architectures.

## Reproduce a complete A/B

Run from this repository with locally obtained public models and a reference:

```bash
python benchmarks/indexed_gate.py \
  --comfyui /path/to/ComfyUI \
  --model /path/to/merged-larry-v4-int8.safetensors \
  --text-encoder /path/to/qwen3vl_32b_minimax_h3_int8_convrot.safetensors \
  --video-vae /path/to/minimax_h3_video_vae_int8_convrot.safetensors \
  --audio-vae /path/to/minimax_h3_audio_vae_fp32.safetensors \
  --reference /path/to/reference.png \
  --width 768 --height 512 --frames 124 --steps 8 --seed 42 \
  --repeats 4 --output results/indexed-gate-new
```

A trusted frozen conditioning file with `positive` and `latent` entries can
replace `--text-encoder` and `--reference` via `--condition`. Do not load
untrusted pickle files. The saved input dimensions must match the arguments.

Both arms use the same PR Kitchen build: the stock arm never calls the new
API, whose addition does not change `int8_linear`. Both use the same video
decoder, CPU RGB conversion, FP32 GPU audio decoder and synchronous MP4 export.
Only the two DiT projection epilogues differ. Timing starts with frozen
conditioning and ends when the local MP4 is complete; it excludes conditioning,
cold model reads and output hashing. This is single-request latency, not a
sustained serving throughput benchmark.

The benchmark alternates A/B and B/A, saves warmups separately, records every
formal run, checks actual coverage (50 outproj + 50 FC2 calls per forward),
and compares SHA256 of both latents, RGB8 frames and PCM audio. It fails if
the patch falls back, is not executed, or any output signature differs.

## Validation

`python tools/verify_indexed_gate.py` recomputes the published timings and checks
all output signatures and the consumer source hashes.

### RTX 5090 D v2 result, 2026-09-30

Public Larry v4 merged INT8 checkpoint, 768×512, 124 requested frames, packed
sequence 14,850 tokens, eight Euler/beta steps, CFG 1, seed 42, shift 12/4.
Torch 2.12.0+cu130, Kitchen PR head `dc7c739`, pinned ComfyUI `2504e68`.
Two independently loaded processes, each one warmup pair plus four alternating
formal pairs: **16 formal requests total**, all included.

| Measurement | Stock | PR #219 consumer | Less time |
|---|---:|---:|---:|
| Process 1, DiT | 15.902738 s | 15.644471 s | 1.62% |
| Process 1, complete local-file cycle | 19.206741 s | 19.020499 s | 0.97% |
| Process 2, DiT | 15.900018 s | 15.621659 s | 1.75% |
| Process 2, complete local-file cycle | 19.159501 s | 18.960541 s | 1.04% |
| Combined, DiT | 15.901378 s | 15.633065 s | **1.69%** |
| Combined, complete local-file cycle | 19.183121 s | 18.990520 s | **1.00%** |

Both video/audio latents, raw RGB8 and PCM SHA256 match the stock output in
every formal request and warmup. The second process verifies 400 outproj and
400 FC2 API calls per eight-step request, zero fallback and zero first-use
verification during formal timing. Two modulation layouts appear in this input;
their first-use verification checks 100 blocks in total. That warmup took
19.574 s of DiT / 22.805 s for the local-file cycle and is saved separately.

The earlier integration diagnostic also compared all 400 blocks in one
eight-step sample, with identical final outputs. Normal node use checks the
first forward of each new layout, rather than repeating this expensive check.

Individual timings overlap, and an audio-decode outlier remains in the raw
samples. No samples were discarded. This is a small measured average benefit
on one public input, not a universal 1% promise. The synthetic kernel-chain
11.49–18.70% result is a different measurement. Full H3 already uses segment-wise
gate broadcasting instead of a gathered full gate tensor, and only part of
each block changes. These effects and the rest of the pipeline limit the
whole-request gain.

Torch's reported peak allocated memory increased from 1,836,176,896 to
1,996,000,768 bytes (about 152 MiB) in these runs. This counter excludes external
aimdo allocations, so it is **not total GPU memory**. Keep the original node
if this memory tradeoff is unsuitable. [Every timing, signature and source hash](../evidence/indexed-gate-consumer.json).

This is evidence that the new API can accelerate a real consumer while
preserving tested output. It is not an additional 1% on top of full R85:
Optimize DiT already has its own fused gate implementation.

CPU tests cover segment validation, model-clone isolation, wrapper registration
across Comfy's options replacement, competing patches, fallback and exception
cleanup. The opt-in GPU benchmark checks the actual model and all four outputs.
Kitchen's PR additionally contains independent integer-oracle kernel tests,
streams, CUDA Graph replay, compile and fallback cases.
