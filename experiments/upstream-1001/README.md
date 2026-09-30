# Upstream integration qualification — 2026-10-01

These are isolated experiment runners, not ComfyUI custom nodes or production installers. The main SpeedKit installation remains unchanged. Results and limitations are in [the report](../../docs/UPSTREAM-RECOVERY-TESTS-20261001.zh-CN.md).

The integration snapshot is [Tokha233/comfy-kitchen, commit 8710121](https://github.com/Tokha233/comfy-kitchen/tree/8710121). Baseline: Kitchen 19ea55b, ComfyUI 8cfe5e1, Torch 2.12.0+cu130, CUDA 13.0.88, aimdo 0.5.5. The tested GPU is RTX 5090 D v2.

The scripts retain the exact laboratory paths to identify the measured environment. Before reuse, adapt `/tmp/h3-kitchen-prs-0930`, `/study/speedkit036/public-model`, `/study/speedkit036/public-run-r5/condition.pt`, `/models/vae`, and dependency paths to your isolated checkout. Obtain the model and condition fixture separately; they are not bundled in this directory.

Required experiment root:

- `base/`: compiled Kitchen baseline.
- `combined/`: compiled integration snapshot. The build script emits `_C.abi3.so` in `build-combined-clean/`; copy it into `combined/comfy_kitchen/backends/cuda/` before testing.
- `comfy-current/`: ComfyUI baseline with its dependencies.
- `model-release-1001.py`: `comfy/ldm/minimax/model.py` from ComfyUI PR #16677 at `20db23b`.
- `cutlass/` and `flash-attention/`: the baseline's pinned build dependencies, with the corresponding `third_party` links.

Run each stage on an otherwise idle GPU:

```bash
python combined/build_combined_clean.py
cp build-combined-clean/_C.abi3.so combined/comfy_kitchen/backends/cuda/
python run-combined-1001.py
python decode-verify-1001.py
python qkv-large-1001.py
```

`run-combined-1001.py` starts separate baseline/candidate processes in AB/BA order. Each process warms up once and measures twice, for eight formal samples per arm. Initial loading and warmup remain recorded but are excluded from the formal mean. `decode-verify-1001.py` compares video/audio latent, RGB8 and PCM hashes using the same FP16 video/FP32 audio VAE. Its cold/warm decode timings are not a speed comparison.

The runner fixes Kitchen INT8 attention and uses private binding access for the experimental prequantized projection. Do not copy that integration into generic ComfyUI model code: a formal consumer must preserve backend dispatch, patches/hooks, training, casting and offload behavior. The optimization is dense, with no token removal, timestep cache or changed LoRA.


## Nonpositive attention scale regression (PR #224)

`nonpositive-probe.py` checks 48 positive-scale cases, the original zero/negative-scale reproducer, and CUDA Graph timing. `nonpositive-sampler.py` verifies the full H3 sampler using isolated base, fixed, and combined environments. `run-nonpositive.py` and `run-combined-nonpositive.py` record exit codes and serialize validation after builds. These scripts retain the experiment directory convention and require its prepared dependencies/models; they are evidence runners, not a standalone installer. See [fix report](../../docs/ATTENTION-NONPOSITIVE-SCALE-FIX-20261001.zh-CN.md) for exact revisions and interpretation.
