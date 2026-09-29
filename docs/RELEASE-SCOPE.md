# Release scope

v0.1 is a Linux SM120 **source release** for the pinned Kitchen 0.2.36 / Torch 2.12 / ComfyUI H3 stack. It includes all default R85 DiT kernels, old encoder / INT8 VAE decode integration, RGB8 transfer, MP4 export, a bounded service export queue, build scripts, workflows and benchmark evidence.

The optimized DiT preserves the selected INT8 baseline. INT8 VAE is a separate, very-small-error change from FP16. No claim of mathematical equality to BF16 dense inference is made. No GPU binary, model weight, Windows certification, Comfy Registry listing or arbitrary-GPU support is implied.

Historical R85 documents retain their original .35 scope. Fresh .36 results and public integration validation are reported separately. The precise tested state is in [release-status.json](../configs/release-status.json).
