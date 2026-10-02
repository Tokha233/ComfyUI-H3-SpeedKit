Fuse MiniMax H3 MLP FC2 with its indexed modulation gate through Kitchen's public float-input wrapper. Keep fallback gates compact by segment, reuse fresh linear output where safe, and preserve hooks, gradients, weight casting/requantization, FP32 residuals, and uncast cleanup.

Compatibility: use the indexed-gate API only when it is available. With the pinned Kitchen 0.2.36 wheel, retain the existing INT8 linear plus segmented-gate fallback. Kitchen Comfy-Org/comfy-kitchen#223 (stacked on #219) still needs to be released and adopted to enable the fusion for ordinary installs; the current pinned dependency no longer raises an AttributeError or requires an unpublished wheel for tests.

Validation on current head `85f21d224f39800cdf566dc9d89221d5305d3d29`:

- Actual pip-installed Kitchen 0.2.36: **30 focused tests passed**, including API-present/API-absent paths, compact FP32/BF16 fallbacks, global/module hooks, gradients, residual dtype, and existing H3/mixed-precision regressions.
- **14/14 GitHub checks passed**, including Linux, macOS and Windows unit suites.
- RTX 5090 D v2, PyTorch 2.13.0+cu130, same Kitchen #223 build on both sides, Larry v4 baked INT8, Euler/beta 8 steps, seed 42, CFG 1, shift 12/4, 768x512, 124 frames.
- One warmup plus two formal sampler requests per arm: median **15.738291 -> 15.580160 seconds (1.00% lower latency)**. All video/audio latent SHA256 values match. This small-sample measurement covers the sampler only, not end-to-end generation or serving throughput.
- Earlier four-pair measurements remain available in the linked reports; this follow-up does not replace them or claim a larger gain.

[Current code, test logs and measurements](https://github.com/Tokha233/ComfyUI-H3-SpeedKit/blob/9c9093d/docs/OMNI-MIGRATION-20261002.zh-CN.md). [Earlier GPU protocol](https://github.com/Tokha233/ComfyUI-H3-SpeedKit/blob/9c9093d/docs/UPSTREAM-RECOVERY-TESTS-20261001.zh-CN.md).

AI assistance: Codex assisted with code, tests, documentation and GPU validation. Measurements were executed on the stated hardware; this is not a claim of human review. DCO signed off.
