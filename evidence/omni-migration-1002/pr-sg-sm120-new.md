## Motivation

The Kitchen INT8 row-split default was tuned to avoid expensive Stream-K execution on RTX 4090. On RTX 5090 D v2, splitting H3 QKV matrices into 8192-row calls adds launches and a full output copy and is slower than a single call.

## Modifications

Keep a single call by default for SM120, BF16 activations, K <= 8192 and 8192 <= N <= 24832. Keep the existing policy for wider outputs, larger reductions, other dtypes/devices, and whenever either row-split environment variable is explicitly set. Cache the capability by activation device. No weight or arithmetic changes.

The bounds are deliberate: a blanket no-split policy regressed M14850/K7168/N28672 by 11.7%, while K28672 cases were essentially tied. Both remain on the previous path.

## Accuracy Test

RTX 5090 D v2 (24GB, SM120), driver 580.126.20, PyTorch 2.13.0+cu130, Triton 3.7.1, **stock comfy-kitchen 0.2.36**. All 30 tested matrix pairs (15 H3 shapes and 15 boundary/reduction probes) were bit-exact; max_abs=0. BF16 activations and bias, INT8 weights, FP32 row scales, ConvRot256.

14 policy tests passed; all applicable pre-commit hooks passed.

```sh
python -m pytest -q python/sglang/multimodal_gen/test/unit/test_kitchen_int8_row_split.py
python -m sglang.multimodal_gen.benchmarks.bench_kitchen_int8_row_split \
  --rows 32700 --input-features 5376 --output-features 16128 --repeats 20
```

## Benchmark & Profiling

Base `b906cd3f45bf9d47e92f2d9d7e3d93ad69b06177`; candidate `d41531d`. Matrix sweep uses ABBA, three warmups and ten event-timed repetitions in each arm. Example medians of arm medians:

| M/K/N | split8192 | single call | latency reduction |
|---|---:|---:|---:|
| 8193/5376/16128 | 3.2457 ms | 2.7743 ms | 14.52% |
| 14850/5376/16128 | 5.7207 ms | 4.8743 ms | 14.79% |
| 32700/5376/16128 | 12.4758 ms | 10.5752 ms | 15.23% |
| 32700/5376/8192 | 6.5016 ms | 5.6269 ms | 13.45% |

No new kernel is introduced; it avoids four GEMMs plus result copies where a single GEMM is more efficient. This is not a universal policy for every Blackwell shape.

Complete SGLang Comfy-integrated Larry v4 INT8 eight-step sampler: 768×512, 124 frames, seed42, Euler/beta, CFG1, sigma shift12/4, torch SDPA, no resident layers. ABBA process order, one warmup and two timed samplers per process. Both arms include the loader correction from #42121 because current main cannot load that serialized checkpoint; this PR does not include the loader change. Both use stock Kitchen 0.2.36.

- Baseline four samples: 33960.216, 33967.192, 33140.687, 33165.233 ms (median **33562.724 ms**).
- Candidate four samples: 33365.072, 33370.351, 33421.806, 33426.852 ms (median **33396.079 ms**, 0.50% lower).
- **All video/audio latent SHA256 values agree across all 12 runs including warmups.**
- The reference process spread is larger than the aggregate difference, and its final trial is faster than the candidate. This does **not** establish stable sampler/serving acceleration. The measured benefit is strongest in the targeted matrix path; no serving-QPS or VAE gain is claimed.
- An existing ComfyUI destructor warning occurs after successful shutdown in all arms; processes exit zero. It is outside timing and is not claimed fixed.


Implementation and verification used Codex assistance. No independent human review is claimed.

## Checklist

- [x] Pre-commit formatting/checks.
- [x] Policy regression tests.
- [x] GPU correctness and benchmark, including negative controls.
- [x] Preserve explicit configuration and other hardware defaults.

Raw timings, profiler counts, tests and audit: https://github.com/Tokha233/ComfyUI-H3-SpeedKit/tree/8650e87/evidence/omni-migration-1002
