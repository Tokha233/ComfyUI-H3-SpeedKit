# PR maintenance validation — 2026-10-05

Four PR branches synchronized with upstream and conflict-free. The tested SGLang base is `bab04cd7913ff50d1bf1a677a2d10e5015dc4c3b`; SGLang-Omni base is `c00c327930f5f10c4870c111b933408d8ac371f6`. Candidate heads are in `heads.json`, and final diffs against those bases are included.

Validation: one idle RTX 5090 D v2, Python 3.12, PyTorch 2.13.0+cu130. Source contents were checked against the complete Git index plus final working-tree files, with symlinks preserved. No production services changed.

| PR | Scope | Result |
|---|---|---|
| SGLang #42250 | E2E/load performance guards, baseline and sequential request tests | 95 passed with CUDA policy; 95 passed with simulated HIP policy |
| SGLang #42256 | Residual dispatch and full Qwen CUDA file | 44 passed, real CUDA execution; HIP eligibility simulated |
| SGLang #42249 | Cold imports, registry/export/alias behavior, import isolation, ConvRot and transformer quantization | 117 passed + 24 subtests |
| SGLang-Omni #2481 | Payload device restoration + SHM relay lifecycle | 26 passed, including 11-dtype GPU transfers, backpressure, cancellation and upstream event filtering |

Changed-file pre-commit passed for all SGLang branches. SGLang-Omni full-repository pre-commit passed, including Rust formatting. Actual AMD/XPU/NPU kernel execution was not performed locally.

## Transfer benchmark

`omni_transaction.py` measures pack + actual SHM write/read/ack + device restoration, with CUDA synchronization at timed boundaries. ABBA, 3 warmups and 15 timed samples per block (30 samples per arm per size). Four CPU threads. Baseline and candidate share the new upstream transport implementation; only `pack_tensors` is swapped. All packed bytes, restored tensors and devices matched exactly.

| GPU payload | Baseline | Candidate | Latency reduction |
|---|---:|---:|---:|
| ~1.35 MB | 1.13818 ms | 1.13743 ms | 0.07% |
| ~40 MiB | 53.70915 ms | 45.87470 ms | 14.59% |
| ~160 MiB | 229.53074 ms | 194.06681 ms | 15.45% |

CPU 40/160 MiB changes were -0.65%/-0.04% (no gain). This is host-local transfer latency, not H3 full inference or serving QPS. Raw samples are in `omni-transaction.json`. Host contention can alter absolute times; compare paired arms in this run.

## Environment repairs before successful validation

The first source comparison accidentally omitted locally sparse-checkout files and copied symlink targets as regular files. Validation was blocked, the isolated test trees were recreated preserving symlinks, and all four complete source digests then matched before tests ran. No source mismatch was ignored.

The first Omni test invocation could not import its SGLang dependency because the new isolated checkout was missing from PYTHONPATH. Adding the verified upstream SGLang source path fixed collection; the successful full test result is `omni-relay-fixed.log`. Optional Mooncake/NIXL/MUSA packages are absent; this run specifically exercises SHM and CUDA tensor restoration, not those transports.

The existing legacy warmup warning about process-group cleanup appeared at the end of the Qwen process. All 44 assertions passed and the process exited. No tolerance or required test was weakened.

Hosted CI still requires upstream maintainer authorization and review; these local results do not claim all hosted checks are green.
