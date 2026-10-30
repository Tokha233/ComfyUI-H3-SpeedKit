# Reproduce the vLLM 0.32 H3 VAE recheck

[Results and scope](../../evidence/pr-review-1030/README.md). [bench_executed.py](bench_executed.py) is the exact executed harness, including its two fixed input paths. It is kept unchanged for provenance.

## Prerequisites

Use an SM120 CUDA GPU and an isolated Python 3.12 environment matching PyTorch 2.13.0+cu130, Triton 3.7.1, vLLM 0.32.0, Transformers 5.16.1 and the pinned Omni checkout's requirements. The experiment used Diffusers 0.40.0 and tokenizers 0.23.2. The VAE experiment uses official weights/remote code; obtain them under their upstream terms. Unit/operator tests do not need those weights.

Prepare the tested source tree:

```bash
git clone https://github.com/vllm-project/vllm-omni.git omni-h3-review
cd omni-h3-review
git checkout --detach 92595ee491d63aa00d3faa57ab944ddb9f0db4dc
git fetch https://github.com/Tokha233/vllm-omni.git 1b55c8320f09857231a5fadcefe1298d17aaf7d5
git merge --no-commit --no-ff 1b55c8320f09857231a5fadcefe1298d17aaf7d5
export OMNI_SOURCE="$PWD"
export PYTHONPATH="$OMNI_SOURCE"
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
```

With the matching environment active, run from the Omni root:

```bash
python -m pytest -q tests/diffusion/models/minimax_h3/test_minimax_h3_vae_ops.py

python -m pytest -q \
  tests/diffusion/models/minimax_h3/test_minimax_h3_vae_ops.py \
  tests/diffusion/models/minimax_h3/test_minimax_h3_vae_temporal_patches.py \
  tests/diffusion/models/minimax_h3/test_minimax_h3_vae_split_residency.py \
  --run-level core_model -m 'core_model and diffusion'
```

## Complete video VAE decode

The unchanged executed harness expects:

- `/tmp/h3-omni-1002/official/Ref2VA/video_vae`: the official `Ref2VA/video_vae` directory from revision `42ed227ee7df40d41602854ae760620d6eb651fe`, including its `source` subdirectory and remote-code modules.
- `/tmp/h3-omni-1002/plain-video-latent.pt`: `torch.save` of a plain tensor `[1,24,37,32,48]`. The recorded sampled latent is not distributed. Supply your own latent for a new parity test; its absolute output hashes and performance are not the published case.

Use an isolated container or create those paths as links to your chosen fixtures without replacing existing files. The harness explicitly trusts the pinned checkpoint's Python code. It loads no DiT or audio model.

Set `SPEEDKIT_SOURCE` to this evidence repository's absolute checkout path. Each mode writes into a fresh result directory, leaving the published evidence unchanged:

```bash
export SPEEDKIT_SOURCE=/path/to/ComfyUI-H3-SpeedKit
export RUN_ROOT="$(mktemp -d /tmp/h3-vae-recheck.XXXXXX)"
cp "$SPEEDKIT_SOURCE/evidence/pr-review-1030/candidate-manifest.json" "$RUN_ROOT/"

CUDA_VISIBLE_DEVICES=0 VLLM_OMNI_VAE_LEGACY_TEMPORAL=0 \
  python "$SPEEDKIT_SOURCE/experiments/pr-review-1030/bench_executed.py" \
  --root "$RUN_ROOT" --source "$OMNI_SOURCE" --mode rgb8

CUDA_VISIBLE_DEVICES=0 VLLM_OMNI_VAE_LEGACY_TEMPORAL=1 \
  python "$SPEEDKIT_SOURCE/experiments/pr-review-1030/bench_executed.py" \
  --root "$RUN_ROOT" --source "$OMNI_SOURCE" --mode fp32
```

Compare each mode's complete `records`: all per-output SHA256 values must agree within that mode. `torch.equal` alone is insufficient to distinguish signed-zero bit patterns; the published evidence verifier also enforces identical hashes. Exclude `warmup=true` records when computing per-arm medians. Preserve failures and do not relax tolerances to claim exactness.
