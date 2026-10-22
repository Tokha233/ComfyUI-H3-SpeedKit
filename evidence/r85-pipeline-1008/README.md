# R85 pipeline queue check — 2026-10-08

See `docs/R85-STREAM-EXPORT.zh-CN.md` for timing scope and limitations. Six independent runs, each with one full warmup and a two-request measured GPU queue. The original CPU subprocess export and an in-process resident-codec control are separate baselines; neither is a native vLLM-Omni C32 server benchmark. Video/audio latents, RGB8, PCM and completed MP4 hashes match the historical R85 outputs. Private reference media and conditioning are not included. Output artifact paths in these copied records are reduced to basenames; numeric values and hashes are unchanged.

Throughput fields are inverse measured two-request queue duration, not sustained service QPS. GPU2's resident-codec comparison saves 2.907 s at the queue tail; its GPU completion window is essentially unchanged.
