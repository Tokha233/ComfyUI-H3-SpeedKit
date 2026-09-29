# Reproduce the source release

Use `python benchmarks/run.py --help` for a real GPU stock/SpeedKit comparison. The reference image in `workflows/reference.png` is original and public. Example (model paths are local):

```bash
python benchmarks/run.py --comfyui /path/to/ComfyUI \
  --model /path/to/h3_larry_v4_int8.safetensors \
  --text-encoder /path/to/qwen3vl_32b_minimax_h3_int8_convrot.safetensors \
  --video-vae /path/to/minimax_h3_video_vae_int8_convrot.safetensors \
  --audio-vae /path/to/minimax_h3_audio_vae_fp32.safetensors \
  --reference workflows/reference.png --width 768 --height 512 --frames 124 \
  --repeats 2 --output results/my-comparison
```

The first pair is warmup. New layouts compare every block for eight model evaluations. Formal pairs reverse arm order, use the same prepared condition, and end when the MP4 is saved. Hashes are computed outside timing. `result.json` stores video/audio latent, RGB and PCM signatures; inspect fallback and verification fields. The private business baseline and public model merge are different model identities. Do not compare their raw times as if only one switch changed.

The benchmark intentionally selects Kitchen INT8 attention for both arms. Compare against a separate BF16 reference only as a quality experiment; that changes arithmetic. No source model file is modified.

## Historical evidence

- `native-fp16-vs-r85.json`: 8 formal full requests, two clips, two repeats/arm/clip, AB/BA per paired device and four warmups.
- `native-fp16-vae-quality.json`: same 15 latents, 60 formal decoder timings, 30 warmups in the historical protocol, RGB error and compressed media metrics.
- `stock035-vs-r85.json`: both arms use INT8 VAE; separates fixed-precision pipeline gain from switching video VAE.
- `lora-quality.json`: distinct historical groups; never pool groups with different inputs/protocols into a universal winner.

Full timing starts with frozen condition and ends with local MP4. Private business media, conditions and source GPU logs are not redistributed. Anonymous signatures and source hashes make the recorded aggregation auditable; they do not make the original private-input run independently reproducible.

## Required public release benchmark

Use public, licensed inputs and keep an immutable manifest of every file, prompt, seed, resolution, frame count, actual NFE, model/LoRA SHA and encoding parameters. Use a small smoke workflow plus representative 10/15-second segments and longer multi-shot assemblies. Five 30-second stories may be composed of multiple generated segments; do not relabel segment timing as an entire 30-second request.

Compare `.35 stock`, `.35 R85`, `.36 stock`, `.36 ported`, keeping all other factors fixed. Within each comparison, report paired-device ABBA/BAAB order, independent warmup, repeat counts, distributions and all failures. Extend repeats when variability exceeds the claimed gain. Do not label n=2 as a confidence interval.

For exact INT8 mode compare per-step video/audio latents, raw RGB and PCM; kernel-level mismatch must be investigated even if a compressed MP4 looks similar. VAE quantization mode uses identical input latent and PCM, then reports RGB PSNR/MAE/percentiles and separately MP4 SSIM/LPIPS. Human side-by-side review is useful but is not replaced by SSIM or audio cosine.

For serving throughput submit a sustained queue and measure completed outputs per wall time, warm/cold state, P50/P95, queue depth, CPU RAM, pinned memory, GPU memory and failure rate. GPU-finished events are not final completion. Report latency reduction as `1 - optimized / baseline`; serial capacity equivalent as `baseline / optimized - 1`; these are different from actual service throughput.

## Public asset manifest fields

`case_id`, `source_url`, `license`, `sha256`, `prompt`, `seed`, `width`, `height`, `frames`, `fps`, `sampler`, `scheduler`, `nfe`, `cfg`, `shift`, `model_sha256`, `lora_sha256`, `vae_sha256`, `audio_sha256`, `kernel_build_id`, `runtime_manifest`, `encoding_parameters`.

No sample manifest with fictitious hashes is shipped. Populate it from real publishable inputs when the generation runner is extracted.
