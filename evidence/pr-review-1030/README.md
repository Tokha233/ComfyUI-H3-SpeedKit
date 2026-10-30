# H3 VAE SM120 recheck — 2026-10-30

Author-side validation for [vLLM-Omni #8414](https://github.com/vllm-project/vllm-omni/pull/8414), responding to the request for full-decode parity on the vLLM 0.32 stack. This measures one official video-VAE workload, not full generation or serving throughput.

## Pinned inputs and execution

- Main: `92595ee491d63aa00d3faa57ab944ddb9f0db4dc`, merged without conflicts with PR head `1b55c8320f09857231a5fadcefe1298d17aaf7d5`.
- GPU: one NVIDIA RTX 5090 D v2, 24 GB, SM120; driver 580.126.20.
- Runtime: PyTorch 2.13.0+cu130, Triton 3.7.1, vLLM 0.32.0, Transformers 5.16.1, Diffusers 0.40.0, tokenizers 0.23.2, flashinfer-python 0.7.0.post1.
- Official model/remote-code revision: `MiniMaxAI/MiniMax-H3` at `42ed227ee7df40d41602854ae760620d6eb651fe`, `Ref2VA/video_vae`.
- Official weight SHA256: `5f0c2e161d895a9fee7645ca32d4a7e3a22b90cacfcbeba62ec999cdbbefe0d3`. It matches the pinned Hub LFS metadata in [official-vae-source-metadata.json](official-vae-source-metadata.json).
- One fixed sampled latent `[1,24,37,32,48]`; output `[1,3,124,512,768]`. Input and VAE file hashes are recorded in both result JSON files. The latent, prompts, media and model weights are not redistributed.

Both arms use the same merged checkout. Baseline sets `H3_VAE_OPERATOR_TABLE=()` before constructing a fresh decoder; candidate restores the registered table. The harness checks installation state for each arm. This is an operator-set off/on comparison, not two separately installed repository versions. The enabled set includes decoder Linear FP16 precast.

Each output mode runs baseline/candidate/candidate/baseline, with one warmup and two measured decodes per block. There are **four timed samples per arm**, and 12 outputs including warmups per mode. These are repeats of one input, not 12 distinct workloads. Timings synchronize the GPU around native `decode_latent`; CPU output copy, hashing, file I/O and model construction are outside the timed interval.

Both modes use FP16 autocast. The separate FP32-output experiment sets `VLLM_OMNI_VAE_LEGACY_TEMPORAL=1`; it is not full-FP32 computation or an instrumentation of the default RGB8 path before conversion.

## Results

| Metric | Operator set off | Operator set on | Comparison |
|---|---:|---:|---|
| Default RGB8 decode, median | 6179.305 ms | 4577.766 ms | 25.92% lower latency |
| RGB8 peak PyTorch allocation | 10,660,823,552 bytes | 5,824,971,264 bytes | Allocator peak, not whole-service memory |
| Legacy-temporal FP32-output decode, median | 6186.045 ms | 4588.588 ms | 25.82% lower latency |
| FP32-output peak PyTorch allocation | 11,099,127,296 bytes | 6,263,275,008 bytes | Includes the larger FP32 output |

All 12 RGB8 outputs have SHA256 `689397e3869f4a5ad30518d04fc17fef30d427056947f72fb5a3fa9977664a9d`.

All 12 FP32 outputs have SHA256 `904fd0ac28afabff75aabb050de8ae2ccd1d032c4e67cd1126d251f3a742eb80`, with `max_abs=0`.

The three existing VAE test files passed: **69 passed, 0 skipped**. [pytest.log](pytest.log) preserves the output. [source-audit.json](source-audit.json) records 4,596 matching source/config files; each decode report also contains 315 matching imported project source hashes. The results correspond to the pinned merge above; a later comment-only clarification does not change these tested commit identities.

Raw records: [RGB8](decode-rgb8.json), [FP32](decode-fp32.json). [artifact-sha256.json](artifact-sha256.json) identifies the unchanged source artifacts. Logs retain development-version, torch.jit, remote-config and optional FlashInfer cubin/toolchain warnings; they are not silently discarded. An initial recorder failure on PyTorch's virtual `_ops.py` filename was fixed before both complete ABBA runs; that incomplete attempt is excluded.

## Reproduce and verify

[The executed harness and commands](../../experiments/pr-review-1030/README.md) document the environment, checkout, fixed fixture paths and test invocation. The recorded latent is not public: exact published output hashes and timings require that input; using another same-shape latent is a new validation case.

The numeric and artifact checks need only Python's standard library, with no GPU or model download:

```bash
python experiments/pr-review-1030/verify_evidence.py
```

The scope excludes DiT, audio, encoders, DLO, multi-GPU VAE patch parallelism, HTTP serving and MP4 encoding. Recipe E2E validation and upstream SM120 CI admission remain open. This result is not additional acceleration over a separately optimized INT8 VAE deployment. Codex assisted with the experiment and evidence preparation; the GPU runs were actually executed.
