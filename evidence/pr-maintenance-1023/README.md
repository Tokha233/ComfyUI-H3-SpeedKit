# H3 PR validation — 2026-10-23

Current-source regression evidence and serialized microbenchmarks from an RTX 5090 D v2 (SM120). The test environment used Python 3.12, PyTorch 2.12.0+cu130 for Kitchen/ComfyUI, and PyTorch 2.13.0+cu130 for SGLang/Omni. This is not a serving-throughput benchmark. Production queues and occupied GPUs were not modified.

## Changes and validation

| Repository / PR | Current-source result | Scope |
|---|---|---|
| ComfyUI #16681 | 53 passed | Current quantized dispatch, indexed gate fallback, hooks, gradients, FP32 residual |
| ComfyUI #16712 | 9 passed / 2 platform skips on each CPU+CUDA and CPU+MPS run | Continuous reference/target packing, video and audio references |
| ComfyUI #16713 | 12 passed | CPU request-local preprocessing; actual CUDA→CPU→CUDA row transfer; exact video/audio rows |
| Kitchen #219 | 54 passed / 1 HIP skip | Native indexed INT8 epilogue and residual tests |
| Kitchen #223 | 37 passed | Native float-input quantization and indexed gate API |
| Kitchen #224 | 523 passed / 594 HIP/ROCm skips | Nonpositive attention scale correctness |
| Kitchen #227 | 435 passed / 594 skips / 2 failures | The two D128 nonpositive-scale failures reproduce on main; #224 fixes them. Candidate positive-scale D64 exact tests pass. |
| SGLang #42249 | 17 passed | Lazy per-backend imports after upstream registry/alias removal |
| SGLang #42257 | 114 passed | Eager disabled/merged/zero-scale LoRA; current scaling and chunked merge regressions |
| SGLang #42359 | 26 passed twice | Full failure-policy file, normal and expired inherited deadline |
| SGLang #42186 | 38 passed | Current row-limit policy and full native ConvRot backend tests |
| sglang-omni #2481 | 25 passed | Real SHM round trip, dtype/device, cancellation and backpressure |
| vllm-omni #8414 | 68 passed | VAE operators, temporal patching and residency |
| vllm-omni #8416 | 53 passed | Step execution, parallel preparation and latent masks |

`tested-heads.json` maps results to exact commits and imported-source counts. The final #16713 test-only change has a separately verified SHA-256. `test-summary.json` is derived from JUnit artifacts; platform skips and the #227 failures are retained, not hidden. Local CPU/MPS counts are additional and are not added into the GPU totals.

## Timing files

- `bench-219-serial.json`: native indexed epilogue, all outputs exact. Large matrices reduce operator latency by 11.73–18.60%; M1024 cases regress about 0.44–0.48%. No full-model inference gain is implied.
- `bench-227-serial.json`: separate-process ABBA D64 attention. All 12 dtype/shape hashes agree. B2/S1025 improves 10.75–11.17%; B4/S1797 4.28–4.42%; B8/S1797 2.72–3.36%. Small/aligned cases show no convincing improvement. The baseline extension contains main's exact attention code and #219's unrelated GEMM addition; that GEMM API is not called here. Its `comparison` section pools per-arm timed samples before taking medians.
- `bench-8416-serial.json`: forward_kwargs only, no DiT weights or HTTP serving. `nonzero` calls 4→0 and stream synchronization 9→2; exact prepared tensors. Use the pooled-sample `summary` medians.
- `bench-42186.json`: explicit Kitchen backend, current main0.2.40 plus #224's attention-only correction (GEMM unchanged); M/K/N32700/5376/16128, five warmups and twenty CUDA-event samples per ABBA arm. Median of arm medians 12.479072→10.535864 ms, exact. **This unsplit benefit is already in SGLang main via #44179.** The updated #42186 contributes tests/benchmark only.
- `reference-memory.json`: long CPU FP32 reference video [1,24,124,128,128] plus image/audio. Prepared GPU row storage 197,285,888→0 bytes; preparation CUDA peak393,740,288→0 bytes. CPU staging costs ~7.3786ms/step versus0.00496ms lookup in the older GPU-cached candidate. Exact rows; no new complete sampler timing for this residency revision.

## Upstream integration and limits

SGLang #44179 already provides the broader unsplit Kitchen default, and #44508 already batches failure-policy subprocesses. The related author PRs were narrowed to remaining regression/benchmark value. Their historical acceleration is not re-claimed as new work against current main.

Kitchen native sources were rebuilt against main `f8b7e97a` with pinned CUTLASS/FlashAttention sources on SM120. CUDA wheel approval and native HIP/ROCm validation are still maintainer tasks. Kitchen #223 depends on #219; ComfyUI's native consumer depends on the release containing those APIs. ComfyUI #16678 additionally waits for Kitchen #220's output_layout release.

SGLang #42249 has `run-ci`; Extra jobs require `run-ci-extra`. Author-local CUDA tests do not substitute for the repository's native AMD/NPU lanes. vLLM-Omni C1/C32 serving measurements and the L4/SM120 VAE lane question remain outstanding. sglang-omni #2481 was tested on Torch2.13, not the current full-stack Torch2.14.1 pin. Its changed-file pre-commit passes; the attempted full-repository run encounters already-upstream large Voxt assets and missing local cargo.

A genuine test isolation failure introduced while updating #42186 was fixed before push: reloading the canonical backend module changed class identity used by other tests. The final version loads the import-time environment policy in an isolated module, and both complete test files pass together. The initial JIT toolchain PATH issue was also corrected before the final GPU run.

No new complete H3 sampler, image/audio quality, or service-throughput improvement is claimed by this maintenance report. The public data here contains numeric test/benchmark evidence, not production requests or media.
