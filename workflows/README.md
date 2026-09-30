# Workflows

`ref2va-api.json` is a ComfyUI **API-format** graph. Set model filenames and the reference image, then send it to `/prompt` using your local ComfyUI API. `ref2va-canvas.json` is the corresponding importable canvas graph; its visual layout is provided for convenience and needs ComfyUI frontend confirmation. The API graph passed ComfyUI validate_prompt and PromptExecutor through completed MP4; see evidence/workflow-validation.json. The benchmark also exercises the same public node APIs. The graph contains standard H3 conditioning, sigma shift, Euler/beta eight-step sampling, SpeedKit DiT, RGB8 video decode, FP32 audio decode and final MP4 save.

1. Obtain model weights from upstream under their licenses.
2. Merge Larry once with `tools/merge_lora.py`; the stock ModelSave node on the pinned revision fails for INT8 Parameter serialization, so it is not recommended here.
3. Start ComfyUI with `--disable-cuda-malloc` and the tested stack.
4. Use `H3 SpeedKit · Optimize DiT` after Sigma Shift. Remove competing sparse/cache/attention patches.
5. Use the SpeedKit Video VAE Loader for both reference encoding and video decode; audio keeps its normal FP32 VAE loader.

The benchmark CLI accepts your own image and models and reports timing and hashes. Do not expect a new input size to reproduce private-business timings; first-use exact layout verification also adds overhead.

`reference.png` is a newly drawn, original toy-robot fixture (Apache-2.0), used by the public-input benchmark. Copy it to your ComfyUI input directory.

For a focused upstream experiment, use the [Kitchen #219 Indexed Gate node](../docs/INDEXED-GATE.md)
instead of Optimize DiT. It requires the unmerged Kitchen source build. The
dedicated `benchmarks/indexed_gate.py` compares both arms with identical stock
decode/export so other SpeedKit optimizations do not enter its result.
