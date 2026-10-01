# October 1 PDMD and Kitchen evidence

`results.json` contains the sanitized per-segment measurements, aggregate means and separately recorded sampler/decode/export durations. Both INT8 PDMD weight paths cover all 15 business segments. BF16 PDMD and 8-NFE extrapolation cover two representative segments. Candidate/reference audio metrics use decoded AAC; raw PCM differences are separate.

- `latent-pcm-diff.json`: direct tensor comparisons, including pre-AAC FP32 PCM.
- `remote-gate-bench.json`: six scheduling candidates, ten gate shapes; no stable gain.
- `remote-d64-bench.json`: direct attention kernel candidates.
- `remote-d64-extension-check.json`: prequantized API checks.
- `remote-d64-attention-only-decode.json`: attention-only integration comparison.
- `d64-isolated-decode.json`: one extension per process, ABBA complete decoder experiment; use this for standalone-extension integration evidence.
- `g0-d64-isolated-micro.json`: independent-process public API benchmark.
- `g0-d64-fullpair-decode.json`, `g0-d64-op-profile.json`: diagnostic multi-extension swapping results. Unchanged GEMM timings were affected; do not interpret these as installed-library performance. Independent-process testing removed the effect.
- `pr227-status.json`: PR snapshot and review fix.
- `validation.json`: completeness, aggregation, link and artifact checks.

No private prompts, condition tensors, task identifiers or business media are included. Detailed scope and interpretation: [report](../../docs/PDMD-KITCHEN-TESTS-20261001.zh-CN.md).
