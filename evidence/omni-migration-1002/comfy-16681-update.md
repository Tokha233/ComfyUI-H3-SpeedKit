Fixed the released-Kitchen CI failures in 85f21d2. The consumer now delegates to the indexed-gate op only when that public API exists; with pinned Kitchen 0.2.36 it retains the INT8 linear + segmented gate fallback. Tests exercise both API-present and API-absent behavior without requiring an unpublished wheel.

Validation: actual installed Kitchen 0.2.36, 30 focused tests passed. On RTX 5090 D v2 / Torch 2.13.0+cu130 with the same Kitchen #223 build on both sides, full Larry8 sampler medians were 15.738291 → 15.580160 s (1.00%); all video/audio latent hashes match. Every current-head GitHub check passes (14/14, including all three OS unit suites).

The published Kitchen release is still required to enable the fusion for ordinary installs, but absence of that API no longer breaks the pinned dependency. [Code, test logs and measurements](https://github.com/Tokha233/ComfyUI-H3-SpeedKit/blob/d6d6da7/docs/OMNI-MIGRATION-20261002.zh-CN.md).
