## Purpose

Enable the existing exact MiniMax-H3 video VAE operator set on SM120, validated on RTX 5090 D v2. The current allowlist leaves this device on the reference path. Add one operator-set entry; preserve the installer and all existing dtype, layout, execution-mode and remote-model guards. Extend the existing operator tests to representative larger shapes and unsupported SM121.

This reuses upstream QK norm/RoPE, scaled residual, SiluAndMul, and FP16 decoder-block weight pre-casting. It does not introduce a new quantization format or change the requested decode autocast.

## Validation

1x RTX 5090 D v2 24 GB, SM120, driver 580.126.20, PyTorch 2.13.0+cu130, Triton 3.7.1, vLLM 0.30.0. Baseline `7b15d22f48a4af2ba83f9a209944d0496348fc91`. Official weights/remote code: `MiniMaxAI/MiniMax-H3` revision `42ed227ee7df40d41602854ae760620d6eb651fe`, `Ref2VA/video_vae`.

Actual native `MiniMaxH3VideoVAE.decode_latent`, sampled latent `[1,24,37,32,48]`, FP16 autocast matching the pipeline, output `[1,3,124,512,768]`. ABBA, one warmup and two measured decodes per block:

- Median video decode **6202.364 → 4631.699 ms (25.32% lower latency)**.
- PyTorch allocation peak **10,660,823,552 → 5,824,971,264 bytes**.
- All RGB8 outputs bit-exact. A separate legacy-temporal float-output run is also bit-exact before RGB conversion (`max_abs=0`), 6213.711 → 4625.177 ms.
- Cropped sampled, zero and random latent inputs (43/56/69 frames, 256x256 and 256x384) also have bit-exact full FP32 outputs.
- Both optimized operators pass direct bit-exact checks through the real vLLM runtime. Small 195-row residual calls regress ~0.0103 → 0.0182 ms in isolation; the full decode improves. No hidden per-op universal-speed claim.
- The parameterized public benchmark was separately executed: **6199.159 → 4631.679 ms**, exact outputs again.

These measurements cover video VAE decode only, not audio, DiT, full generation or serving throughput. They are not additional acceleration over a separately optimized INT8 VAE deployment.

[Raw measurements](https://github.com/Tokha233/ComfyUI-H3-SpeedKit/tree/d6d6da7/evidence/omni-migration-1002), [executed benchmark sources and exact reproduction commands](https://github.com/Tokha233/ComfyUI-H3-SpeedKit/tree/d6d6da7/experiments/omni-migration-1002).

**Tests: 69 passed.** With this checkout on PYTHONPATH and its pinned dependencies, an available supported CUDA GPU; unit tests need no model downloads:

```bash
python -m pytest tests/diffusion/models/minimax_h3/test_minimax_h3_vae_ops.py -q
python -m pytest tests/diffusion/models/minimax_h3/test_minimax_h3_vae_ops.py tests/diffusion/models/minimax_h3/test_minimax_h3_vae_temporal_patches.py tests/diffusion/models/minimax_h3/test_minimax_h3_vae_split_residency.py -q --run-level core_model -m 'core_model and diffusion'
```

Changed-file pre-commit passes including locally enforced mypy, marks, SPDX and forbidden-import / torch.cuda gates. No CI allowlist exceptions added. Full self-review completed against the existing model-owned operator boundary; the only measured target added is SM120.

Codex assisted with implementation and testing. No human review is implied; all reported GPU runs were actually executed.
