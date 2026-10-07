# PR maintenance validation — 2026-10-08

ComfyUI #16681, commit `c7155580faf3adc26c03b81376b6dc6f282e2063`: the custom-forward output-gradient regression fails before the guard and passes after it; 31 targeted tests pass locally and on the CUDA host. The public Larry8 sampler uses upstream `af89add63f71a487fde45efc7fba744f3455ae73` MLP/DiTBlock.forward as the baseline, with explicitly built Kitchen #219/#223 dependencies. One warmup pair and four measured alternating pairs have identical video/audio latent hashes. Median sampler time 15.731014 -> 15.568599 s (1.032% reduction). This is not an incremental speedup over already-fused R85 and is not a released-wheel benchmark.

SGLang #42257, commit `5d32348ad30840a582ee0e19ae383cff0e47780d`: latest-main merge plus runtime-scale-zero eager dispatch, preserving base parameter rebind and output offsets. 25 tests pass on Torch 2.13.0+cu130 / RTX 5090 D v2, covering CPU and CUDA cases. No inference-throughput claim.

Source verification covered 971 ComfyUI and 4,145 SGLang Python files with zero hash mismatches. ComfyUI uses Torch 2.12.0+cu130. Test timestamps, expected failure, checks and hashes are retained here.
