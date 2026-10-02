# Candidate pre-submission audit

## vLLM-Omni fixed timestep positions

- Base: 68003cf (latest fetched main). Scope: H3 branch initialization and timestep fill, plus focused regression and benchmark.
- Owners: request-local immutable packed layout. Searched production and tests: img/audio positions and update masks have no post-init assignments. Request and batched packing both call the same fill method.
- Cache only the four index tensors. Timesteps remain recomputed every call, including per-row edits and locked-audio semantics. No reuse of activations, no scheduler arithmetic changes.
- Scalar index_fill preserves FP32 cast and is used only for uniform values; arbitrary target tensors retain indexed assignment and length checks.
- Existing step/request equality suite covers 1, 8, 50 steps; new regression has interleaved video/audio anchors, edits, locked audio, out-of-range lengths and a profiler guard against aten::nonzero.
- GPU benchmark isolates preparation, no full DiT quality or serving throughput claim. Prior full-VAE PR is independent.
- New benchmark is under benchmarks/, not a model-specific example. No new dependency, torch.cuda call, allowlist, broad exception, global cache, input mutation, per-step clone or cross-request state.
- SGLang already uses precomputed timestep indices; this change follows the same general optimization principle, using vLLM's own layout and fill contract.
- Lint / mypy / markers / forbidden-import / SPDX hooks pass locally. Real GPU suite initially lacked pytest-mock; installed test dependency then reran: 45 passed. Initial environment failure excluded from pass claim.

## SGLang Kitchen SM120 row policy

- Base: b906cd3. Independent from loader PR #42121; public sampler requires its loader fix in both arms for the serialized Comfy weight input. Only row policy is proposed here.
- Standard wheel 0.2.36 tested. 30 matrix pairs with BF16 inputs, INT8 weights/scales, BF16 bias, ConvRot256 are bit-exact.
- Broad no-split policy rejected due wide FC1 11.7% regression. Very large K offers negligible/mixed gains, so keep old behavior above K8192. Keep old behavior above N24832, for FP16/FP32, non-SM120, or either explicit env override.
- Capability is cached by concrete activation device, avoiding a driver query per linear while keeping multi-device identity distinct. No CPU accelerator initialization on the short-circuit fallback.
- 14 policy regression tests pass. Completed sampler ABBA: all 12 warm/timed video/audio hashes exact; median 33562.724 to 33396.079 ms, but large baseline spread prevents a stable sampler acceleration claim. Checked-in matrix benchmark independently rerun: about 15.44% reduction at M32700/K5376/N16128.
- Full changed-file pre-commit passes. No Kernel code changes, arithmetic changes or weight transformation in this PR.

## Negative / duplicate outcomes

- Residual row/flat Triton scheduling sweep: no persuasive benefit across H3 shapes. All exact, but isolated 8192-row improvement does not justify a broad new dispatcher. No PR.
- SHM receive zero-fill and single-tensor copy removal: already in SGLang-Omni #2368; our #2481 concerns multi-tensor GPU-to-host packing and remains complementary. No duplicate PR.
- SGLang H3 VAE QK/RMS/RoPE fusion: #41906 already in review. No duplicate PR.
- SGLang FC1 raster tuning: existing upstream work; no claim of authorship for existing kernels.
- vLLM VAE fallback QKV recomputation: real source observation, but default official decoder uses the supported fast path. Changing the remote fallback contract without an impacted supported workload is not qualified for a performance PR.
- Denosing-loop clones retain callback snapshots and model-input ownership; no blind removal.
- ComfyUI #16677 confirmed merged 2026-10-01T12:04:56Z. Existing open count of 17 excludes this merged PR.

## Publication / final checks

- vLLM-Omni #8416, 6925191, formal OPEN. Hosted pre-commit, Python3.11/3.12 wheels and DCO pass; docs pending. Self-review posted once.
- SGLang #42186, d41531d, formal OPEN. CI GPU gate fails solely due missing run-ci, verified job steps/log; maintainer authorization requested once.
- CI-like CPU command using --run-level=core_model executed: 21 passed (subset of the 45-test suite).
- Final six committed source files on GPU host hash-equal local commits. Capability override detection changed to honor both env vars; GPU source matched before final portable benchmark.
