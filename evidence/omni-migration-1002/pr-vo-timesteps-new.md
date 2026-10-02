## Purpose

MiniMax H3 rebuilds the same video/audio target and condition positions every time it fills a denoise timestep tensor. On CUDA, four boolean indexing operations invoke `nonzero` and synchronize the stream. Scalar indexed writes add further host-to-device scalar copies.

Compute the four position tensors once per request branch, then fill uniform timesteps with `index_fill_`. Per-row edit timesteps retain their existing indexed assignment and length check. Request-mode and continuously batched execution share the same method. No timestep values, solver math, cached activations, quantization or model weights change.

## Test Plan

Base `68003cf`; head `6925191`. RTX 5090 D v2 24GB, driver 580.126.20, PyTorch 2.13.0+cu130, Triton 3.7.1, vLLM 0.30.0.

45 tests passed, including request/step-mode equality at 1/8/50 steps, packed-layout tests, latent masks, interleaved video/audio conditions, locked audio, per-row edits and incorrect target-length rejection. Added profiler assertion prevents reintroducing `aten::nonzero` in timestep fill. Relevant pre-commit hooks all pass, including mypy, markers, forbidden imports, platform API and SPDX checks.

```sh
python -m pytest -q \
  tests/diffusion/models/minimax_h3/test_minimax_h3_step_execution.py \
  tests/diffusion/models/minimax_h3/test_minimax_h3_parallel.py \
  tests/diffusion/models/minimax_h3/test_minimax_h3_latent_mask.py
# CI-like CPU selection:
python -m pytest -q --run-level=core_model -m 'core_model and cpu' \
  tests/diffusion/models/minimax_h3/test_minimax_h3_step_execution.py
```

GPU reproduction, with vLLM dependencies installed; no model download needed:

```sh
git show 68003cf:vllm_omni/diffusion/models/minimax_h3/denoise_loop.py > /tmp/h3_denoise_reference.py
python benchmarks/minimax_h3/benchmark_timestep_preparation.py \
  --baseline /tmp/h3_denoise_reference.py --output /tmp/h3_timestep_results.json
```

## Bottleneck and measured value

Checked-in reproduction ran on the same candidate, in ABBA order, with 20 warmup calls per arm and 30 measured groups of eight fills per arm (60 group means per variant). Profiling is separate from timing. Every filled FP32 tensor is exact.

| packed rows | reference median (min–max), ms | candidate median (min–max), ms | latency reduction |
|---|---:|---:|---:|
| 4672 | 0.166366 (0.161738–0.171890) | 0.022796 (0.022505–0.024798) | 86.30% |
| 15040 | 0.228388 (0.226749–0.233505) | 0.029313 (0.028984–0.030089) | 87.17% |
| 32320 | 0.229254 (0.227685–0.231965) | 0.029448 (0.029072–0.031877) | 87.15% |

The single-call profiler shows `aten::nonzero` 4 → 0 and `cudaStreamSynchronize` 8 → 0 for all three layouts. Both retain the same two device synchronizations from the measurement/profiler boundary. The gain saves about 0.20 ms per preparation call on the larger layouts (about 1.6 ms over eight steps), not seconds of denoising.


The additional request-local indices cost eight bytes per video/audio row (about 251 KiB at the largest measured layout). They are released with the branch; there is no cross-request cache. Timestep values and validation results are exact. This is a small preparation-stage optimization, not a claim of faster GEMMs, full DiT inference or serving QPS. The model-weight path was not benchmarked in this PR.

Self-review covered immutable layout ownership, both fill consumers, edit/locked-audio paths, all changed code, simplification and examples policy. Codex assisted implementation and testing; no independent human review is claimed.

Raw timings, profiler counts, tests and audit: https://github.com/Tokha233/ComfyUI-H3-SpeedKit/tree/8650e87/evidence/omni-migration-1002
