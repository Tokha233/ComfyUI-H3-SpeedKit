## Motivation

Loading an H3 single-file INT8 ConvRot checkpoint through the ComfyUI integrated path currently fails with `Parameter ...comfy_quant not found in custom model state dict`. The loader reads the H3 metadata but discards its per-layer quantization markers, so it builds ordinary floating-point Linear parameters.

## Modifications

Resolve the serialized INT8 config using the existing H3 loader utilities, pass it into model construction, and filter metadata-only keys. Preserve CPU checkpoint placement for layerwise offload. Reject conflicting runtime quantization, FSDP, and unsupported serialized formats explicitly. The existing ComfyUI QKV layout remains unchanged.

## Accuracy Tests

- Regression writes a real small safetensors checkpoint and runs the shared loader. INT8 weights and FP32 scales must remain exactly equal, parameters frozen, and ComfyUI QKV ordering preserved.
- The test fails on the original loader with the missing `proj.comfy_quant` error, and passes with this change.
- H3 ComfyUI unit suite: **28 passed, 1 skipped, 3 xfailed**.
- Actual 1x RTX 5090 D v2, PyTorch 2.13.0+cu130: the full Larry v4 baked INT8 checkpoint loads and completes 8-step Ref2VA sampling. Fixed seed 42, 768x512, 124 frames, Euler/beta, CFG 1, identical video/audio latent hashes across three timed repetitions.

## Speed Tests and Profiling

This is a loading correctness fix, not a kernel performance claim. Successful native SGLang sampling with SDPA measured 32.905 / 32.916 / 32.947 seconds after warmup; excludes condition encoders, VAE and export. The original integrated loader fails before sampling, so there is no valid speedup ratio. No experimental attention or row-splitting changes are included in this PR.

## Checklist

- [x] Changed-file pre-commit checks pass.
- [x] Added a regression test with actual safetensors loading.
- [x] Included accuracy and runtime validation above.
- [x] Followed existing loader/config conventions.

Codex assisted with implementation and testing. The reported GPU runs were actually executed; no human review is implied.

<!-- pr-states:start -->
---
### CI States

Latest PR Test (Base): <!-- slot:pr-test:start -->:x: [Run #36918637044](https://github.com/sgl-project/sglang/actions/runs/36918637044)<!-- slot:pr-test:end -->
Latest PR Test (Extra): <!-- slot:pr-test-extra:start -->:x: [Run #36918636803](https://github.com/sgl-project/sglang/actions/runs/36918636803)<!-- slot:pr-test-extra:end -->
Latest PR Test (AMD ROCm 10): <!-- slot:pr-test-amd:start -->:x: [Run #36918637016](https://github.com/sgl-project/sglang/actions/runs/36918637016)<!-- slot:pr-test-amd:end -->
<!-- pr-states:end -->
