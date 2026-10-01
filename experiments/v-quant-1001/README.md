# Kitchen #231 — original experiment evidence

These files preserve the actual 2026-10-01 laboratory scripts and outputs. They are not a portable installer. Adapt the absolute paths only after preparing equivalent isolated checkouts. No weights or condition media are included.

- Kitchen Python and base attention/V source: `12389a30463c62c93670b049d59bf3fa56c0316d`.
- Candidate V source: Kitchen #231 `6dd6f95a7d4b8fd9ea1d6b72a695b513c2ac045c`; both exact V source files are archived here.
- Common object build: integration snapshot `8710121`, from the sibling `upstream-1001/build_combined_clean.py`; base attention launcher is separately rebuilt from `12389a3`.
- ComfyUI: `8cfe5e1ecb97512dea8deaac15e1228d7e6feeb1`.
- GPU: RTX 5090 D v2; Torch 2.12.0+cu130; CUDA compiler 13.0.88; aimdo 0.5.5.
- Weights SHA256: `422dffed547dbe9d1e693ace73ee66c85cdf7fc62bb5514973851aa6f84547dc`.
- Trusted local condition fixture SHA256: `9e3a1dc9eef79b7753bcd8273711134b8b5c1afb2f10cb52190f58c311282cd9`.

`remote-extensions-build.json` records the complete compile/link argv, including reused objects. This was a controlled replacement of one V object, **not a clean rebuild of Kitchen main**. Do not interpret it as extra acceleration over R85's separate fused QKV/V preparation.

The actual experiment root was `/tmp/h3-kitchen-followup-1001`; it contained these scripts, sources, base/candidate Python overlays and extensions. `/tmp/h3-kitchen-prs-0930/build-combined-clean` supplied common objects; `/tmp/h3-pdmd-gate-1001/d64-neutral-build.json` supplied the base-attention link list. With those artifacts prepared, the exact commands were:

```bash
python /tmp/h3-kitchen-followup-1001/build_extensions.py
python /tmp/h3-kitchen-followup-1001/bench_extensions.py
python /tmp/h3-kitchen-followup-1001/sampler.py --arm base --pair 0
python /tmp/h3-kitchen-followup-1001/sampler.py --arm candidate --pair 0
python /tmp/h3-kitchen-followup-1001/sampler.py --arm candidate --pair 1
python /tmp/h3-kitchen-followup-1001/sampler.py --arm base --pair 1
python /tmp/h3-kitchen-followup-1001/profile_sampler.py --arm candidate --pair 9
```

`bench_extensions.py` measures both V-only and complete attention in interleaved AB/BA order on the same tensors. It swaps two separately loaded native extensions while preserving the same Python dispatcher. The sampler uses independent processes, one warmup and two timed requests per process. Profiling is separate. The recorded mean 15.688698 → 15.537895 seconds includes condition preparation in the sampler but excludes condition encoders, VAE, export and initial model loading.

`SHA256.json` identifies all archived files. The simpler `benchmarks/bench_v_quant.py` in Kitchen is a smoke benchmark, not this paired measurement. Public parameterized sampler reuse is available in `../../benchmarks/comfy_h3_sampler.py`; it follows the same warmup/timing settings but is not the original source of these archived records.
