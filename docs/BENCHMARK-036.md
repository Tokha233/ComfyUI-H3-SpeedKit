# Kitchen 0.2.36 measurements

Two representative private Ref2VA segments on RTX 5090 D v2, two formal runs per arm after independent warmups. These are segments of longer business cases, not 30-second complete stories. Same Larry v4 pre-merged INT8 ConvRot checkpoint, Euler/beta 8 steps, CFG 1, shifts 12/4, frozen old-encoder conditioning, GPU FP32 audio, and identical CPU codec settings.

| Mean seconds | Clip A | Clip B | Pooled mean |
|---|---:|---:|---:|
| Kitchen .36, native FP16 VAE | 270.576 | 258.303 | **264.439** |
| Kitchen .36, INT8 VAE | 257.440 | 243.459 | **250.450** |
| SpeedKit .36, INT8 VAE / RGB8 | 239.949 | 227.303 | **233.626** |

- Complete measured cycle versus FP16 VAE: **−11.6523%**; inverse-latency capacity equivalent **+13.1892%**.
- Complete measured cycle versus INT8 VAE: **−6.7173%**.
- DiT: **233.986 → 218.795 s**, **−6.4922%**.
- FP16 VAE/output stage versus optimized INT8 output: **25.591 → 10.085 s**. This includes a quantization change and output engineering.

The timing includes component switching, decode, CPU handoff and completed MP4. It excludes text/reference preparation, disk-cold model reads, integrity hashes, network, queue admission and upload. Separate processes were run sequentially; this .36 comparison is not a randomized concurrent service trial. Differences smaller than its observed variation should not be treated as stable wins. Do not extrapolate directly to every duration, resolution or GPU.

## Correctness and precision

All four formal optimized runs match the stock INT8 path in video latent, audio latent, RGB8 and PCM SHA. Every request has eight model evaluations and 50 optimized attention blocks per evaluation. The FP16-VAE arm has the same latents and PCM but different RGB, as expected.

Historical .35 same-latent testing across 15 clips measured FP16→INT8 raw RGB PSNR 58.6703 dB and MAE 0.088106/255, encoded SSIM 0.9890704 and LPIPS 0.00758937. These are **historical 15-clip quality data**, not a newly run .36 15-clip test. The new .36 INT8 RGB signatures align with the historical baseline on the two measured clips. See [VAE quality](R85-NATIVE-VAE-QUALITY.md).

The main timings use the shared historical CPU exporter to isolate compute/output effects. The new in-process PyAV exporter was separately exercised through the Save Video node: 260 frames, decoded frames equal to the historical MP4, about 3.21 s for that individual export. Do not subtract that from the table to invent a new full-cycle speedup. Final source/public-input validation is recorded separately.

## Reproduce / inspect

[Raw anonymized records](../evidence/kitchen036-comparison.json) include each formal run, stage durations, all four SHA values and call counts. Run:

```bash
python tools/verify_kitchen036.py
python benchmarks/run.py --help
```

Private prompts, references and tensors cannot be reconstructed from these records. Use the benchmark CLI with your own public input for an independently shareable comparison. The private baked checkpoint identity and the public merge-tool output identity are separate; do not assume their byte SHA or generated trajectory is identical.

## Public input: small integration case

A newly drawn robot reference, the public base model plus Larry v4 merged with `tools/merge_lora.py`, 384×256, 124 requested frames (124 actual), seed 42 and eight Euler/beta steps. This is a small integration case, not a business-resolution speed claim. Its first eight model evaluations compared every optimized block to the original INT8 block and passed; all video/audio latent, RGB and PCM signatures then matched for warmup and both formal repetitions.

- Stock formal cycles: 4.3508 / 4.6225 s.
- SpeedKit formal cycles: 3.9451 / 3.9460 s.
- Same model-local INT8 attention selection and same in-process PyAV exporter in both arms.
- [Raw records](../evidence/public-input-small.json), [original public reference](../workflows/reference.png), [merge identity](../evidence/public-merge.json).

The public merge has its own SHA; these results do not claim equivalence to the private pre-merged checkpoint. The condition was initially generated with the public node, then replayed locally while resolving integration checks; preparation is excluded from timing.

## Public input: 768×512

The same public reference and merged model were run from fresh reference/Qwen conditioning, with the public pinned `comfy/ops.py` restored in an isolated checkout. Model, model-patcher and attention source hashes match the public pin. First-use 50-block comparisons passed for all eight evaluations, followed by two formal repetitions with all four signatures equal.

| Formal seconds | Run 1 | Run 2 | Mean |
|---|---:|---:|---:|
| Stock INT8 | 19.4650 | 19.1550 | **19.3100** |
| SpeedKit INT8 | 16.7660 | 16.7612 | **16.7636** |

This one public scenario measured **13.19% less complete-cycle time**. It uses the same INT8 VAE and the public in-process exporter in both arms. The warmup includes shape verification and is excluded. It is not pooled with business clips, and neither dataset establishes sustained service throughput. [Raw records](../evidence/public-input-medium.json).

## Final VAE migration regression

The packaged Kitchen 0.2.36 INT8 VAE + RGB8 path was rerun on all 15 historical saved latents. **15/15 raw RGB hashes exactly match R85**. This confirms output migration for those inputs; the single-pass durations are not an additional speed benchmark. [Records](../evidence/vae15-migration.json).

The final guarded node revision was rechecked on Clip A: two formal cycles **239.753 / 239.885 s**, both with all four R85 signatures equal. This validates the final adapter; the main comparison table retains its original paired dataset. [Record](../evidence/final-business-validation.json).
