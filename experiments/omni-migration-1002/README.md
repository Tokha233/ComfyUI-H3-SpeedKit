# Omni migration experiments — 2026-10-02

These scripts and JSON/logs are the actual isolated RTX 5090 D v2 experiment evidence. No weights or media are distributed. The scripts except `bench_h3_vae.py` retain laboratory paths; adapt them to prepared isolated checkouts before use. No script is a production installer.

Baseline revisions: SGLang `a71c7d19e7154f5cf2a698418486706f7ce314f6`, SGLang-Omni `1523543056bf334bc2f4d69db23aad0a002cedf2`, vLLM-Omni `7b15d22f48a4af2ba83f9a209944d0496348fc91`, ComfyUI `2d6b73283af2447bdd065ece4090b8c6b1784544`. Torch 2.13.0+cu130, Triton 3.7.1, driver 580.126.20; vLLM runtime 0.30.0. Official VAE weights and remote code: `MiniMaxAI/MiniMax-H3` commit `42ed227ee7df40d41602854ae760620d6eb651fe`, `Ref2VA/video_vae`.

## Portable full video VAE comparison

Use the candidate vLLM-Omni checkout, its pinned vLLM dependencies and one idle SM120 GPU. Prepare a **plain Tensor** file containing `[B,24,T,H,W]` model latents and the official local VAE directory. The experiment used a sampled latent `[1,24,37,32,48]` that decodes to 124×512×768 RGB. The original scripts `vo_vae_full.py` and `vo_vae_float.py` generated the archived records; this parameterized harness reruns the same decode/equality protocol:

```bash
PYTHONPATH=/path/to/vllm-omni HF_HUB_OFFLINE=1 python bench_h3_vae.py --vae-dir /path/to/Ref2VA/video_vae --latent /path/to/latent.pt --output vae-rgb.json
PYTHONPATH=/path/to/vllm-omni HF_HUB_OFFLINE=1 VLLM_OMNI_VAE_LEGACY_TEMPORAL=1 python bench_h3_vae.py --vae-dir /path/to/Ref2VA/video_vae --latent /path/to/latent.pt --output vae-float.json
```

ABBA, one warmup plus two timed decodes per block, synchronized before/after decode. FP16 autocast matches the real pipeline. The baseline disables only the VAE operator table; candidate uses the registered SM120 implementation. The VAE temporal path remains the same between arms. Output equality is required. No full serving throughput or audio-decoder claim.

## SGLang-Omni transport

`omni_transaction.py` compares the pinned original `stage_io.py` with the candidate production module: multiple tensors → pack → SHM write/read/ack → restore original device. Save the baseline module to the script's `stage_io_baseline.py` path and prepare the candidate import. It uses ABBA, 3 warmups and 15 timed transfers per block. It includes small/large CPU and CUDA payloads with mixed scalar alignment. These are **transport transactions without model execution**, not speech or H3 serving QPS.

## SGLang H3 sampler

`sg_sampler.py` uses native SGLang DiT through its ComfyUI integrated adapter, Larry v4 baked INT8, seed42, 8 Euler/beta steps, CFG1, shift12/4, one 768×512 / 124-frame public fixture. It excludes encoders, VAE and export. `quality_compare.py` decodes the ComfyUI and SGLang latents with the same video/audio VAEs. Cross-framework outputs currently differ substantially; faster SGLang variants are not qualified as an output-preserving replacement. This is not a comparison to the BF16 50-step teacher or the five 30-second business cases.
