Responding to the three performance-evidence questions with a same-head ablation. This PR only adds SM120 dispatch; the ablated kernels and weight-precast logic already exist upstream.

**1. Bottleneck.** The current reference VAE repeatedly casts decoder Linear weights and materializes intermediate QK norm/RoPE, gated activation and FP32 residual operations across tiles. A separate warmed PyTorch trace of this input records `aten::copy_` 43,111→6,823, `_fused_rms_norm` 12,096→6,048, `cat` 12,585→489, and `mul` 21,839→671. GEMM call count (`addmm`) remains 12,180 in both paths. This trace diagnoses removed work; profiler instrumentation introduced command-buffer stalls, so its timings are **not** used for speedup or utilization claims.

**2. User value / workload.** Official `MiniMaxAI/MiniMax-H3` VAE and remote code revision `42ed227ee7df40d41602854ae760620d6eb651fe`, local Ref2VA/video_vae; actual sampled latent `[1,24,37,32,48]`, RGB8 output `[1,3,124,512,768]`. One RTX 5090 D v2 24GB (SM120), driver580.126.20, torch2.13.0+cu130, Triton3.7.1, vLLM0.30.0. Decode only: no DiT, audio, encoders, MP4 or serving-QPS claim. It is not an incremental gain over a production INT8 VAE.

**3. A/B and ablation.** Both modes run on submitted `ce3039e4a48b6026302b6b9dec107dec8ac69fd9`; reference disables the dispatch table, matching pre-PR SM120 behavior. Original pre-PR comparison base was `7b15d22f48a4af2ba83f9a209944d0496348fc91`. Each ablation reloads official weights. Sequence is reference / weights / weights+FF / weights+QK / weights+residual / all, then reverse. Each group has one untimed warmup and two timed decodes; four samples per arm. No concurrent GPU jobs. Wrappers disabled for ablation are restored to the original class forwards. The reference arithmetic and autocast dtype remain unchanged.

| Arm | Median ms | Mean ms | Min–max ms | Peak allocated bytes |
|---|---:|---:|---:|---:|
| baseline | 6204.470 | 6205.571 | 6195.725–6217.621 | 10660823552 |
| weights | 5375.281 | 5374.993 | 5359.920–5389.491 | 5824971264 |
| weights_ff | 5206.627 | 5206.543 | 5191.183–5221.735 | 5824971264 |
| weights_qk | 4878.474 | 4878.802 | 4867.504–4890.755 | 5824971264 |
| weights_residual | 5303.970 | 5304.202 | 5288.718–5320.151 | 5824971264 |
| all | 4641.391 | 4640.229 | 4626.332–4651.800 | 5824971264 |


All 36 warmup/timed RGB8 outputs are bit-exact, SHA256 `689397e3869f4a5ad30518d04fc17fef30d427056947f72fb5a3fa9977664a9d`. Existing separate FP32-pre-conversion and three edge-input exactness evidence remains linked in the description. Peak values are PyTorch allocator bytes, not total device/service memory. Do not sum the marginal percentages: each one-kernel arm compares to the weights-only arm and interactions affect the combined path.

The first ablation harness retained the last loop variable pointing to a decoder block, adding ~133 MB to later allocator measurements. The final harness explicitly releases it, reruns every arm in both orders, and verifies stable per-arm memory. Only that corrected run is reported above; original PR full-decode ABBA did not contain that loop variable.

EVIDENCE_LINK_PENDING
