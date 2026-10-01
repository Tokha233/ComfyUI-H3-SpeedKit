## Motivation

When a multi-tensor payload containing CUDA tensors is routed through CPU shared memory, packing currently creates separate host copies and then concatenates them. Large stage payloads pay an avoidable extra full CPU copy.

## Modifications

Compute aligned offsets first, allocate the final private byte buffer once, then copy each source directly into its final slice. Keep padding zeroed. Copies remain blocking and precede relay-credit waits, preserving snapshot semantics when a producer reuses its tensors under backpressure. CPU-only, single-tensor and direct CUDA IPC routes retain their existing paths.

## Related Issues

Complementary to #2368, which targets the already materialized single-tensor case. This change targets multiple tensors with CUDA sources packed for a CPU destination; it does not modify relay-credit or atomic metadata transport behavior.

## Accuracy Test

32 passed, 1 skipped in the actual CUDA environment: payload restoration + SHM relay tests, including 11 dtypes, strided/empty tensors, original device placement, producer reuse during backpressure, and cancellation followed by credit reuse. Full-repository pre-commit passes. Round-trip tensor contents are exactly equal.

## Benchmark & Profiling

RTX 5090 D v2 24 GB, PyTorch 2.13.0+cu130, one otherwise idle experiment GPU. Complete transfer transaction: pack + SHM write/read/ack + restore receiver device. ABBA, 3 warmups and 15 timed transfers per arm per block, CUDA synchronized boundaries. No model execution or serving QPS claim.

Pinned baseline: `1523543056bf334bc2f4d69db23aad0a002cedf2`. Complete-transaction medians across 30 timed samples per arm:

| Source / payload | Baseline ms | Candidate ms | Latency reduction |
| --- | ---: | ---: | ---: |
| GPU / ~1.35 MB | 1.048607 | 1.057946 | -0.89% |
| GPU / ~40 MiB | 49.997057 | 43.252745 | 13.49% |
| GPU / ~160 MiB | 189.113012 | 157.749784 | 16.58% |
| CPU / ~40 MiB | 31.758029 | 32.035689 | -0.87% |
| CPU / ~160 MiB | 119.708824 | 120.825679 | -0.93% |

Small and CPU payloads have no measured gain. These are host-local SHM transactions, not independent speech workers or end-to-end model throughput. A separate earlier run also improved the two large GPU payloads by ~11% / ~15.6%; compare arms within each run because absolute host latency varies.

[Raw records and test logs](https://github.com/Tokha233/ComfyUI-H3-SpeedKit/tree/d6d6da7/evidence/omni-migration-1002); [original benchmark and instructions](https://github.com/Tokha233/ComfyUI-H3-SpeedKit/tree/d6d6da7/experiments/omni-migration-1002).

## Checklist

- [x] Full-repository pre-commit passes.
- [x] Tests cover actual SHM lifecycle and tensor ownership.
- [x] Existing public interfaces and metadata remain unchanged.
- [x] Performance scope is limited to transfer transactions.

Codex assisted with implementation and testing. No human review is implied. A maintainer's `run-ci` label is needed for hosted GPU CI.
