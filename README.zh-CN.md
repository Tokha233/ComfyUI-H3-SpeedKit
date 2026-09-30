<p align="center"><img src="docs/assets/hero.svg" alt="H3 SpeedKit：Kitchen 0.2.36，完整周期耗时减少 11.65%" width="100%"></p>

[English](README.md) · [安装接入](docs/INSTALL.md) · [逐项迁移](docs/MIGRATION-036.md) · [实测口径](docs/BENCHMARK-036.md) · [工作流](workflows/README.md)

# H3 SpeedKit

**面向 RTX 5090 D v2 的 MiniMax H3 / ComfyUI 加速节点，包含可自行编译的完整 CUDA 源码。** 保持 Larry v4、8 步、seed 与采样器，用一个 MODEL 节点接入 DiT 融合；视频输出另有 INT8 VAE、RGB8 搬运和 MP4 保存节点。

Kitchen **0.2.36** 上，实测完整周期平均 **264.44 → 233.63 秒，耗时减少 11.65%**。两臂都使用 INT8 VAE 时，仍减少 **6.72%**；DiT 阶段减少 **6.49%**。

这些数字来自两个代表性业务片段、每臂每片段两次正式运行，起点为准备好的 condition，终点为本地 MP4 完成。**不包含 Qwen/参考素材编码、冷读权重、排队、网络和上传**。11.65% 包含原生 FP16 → INT8 VAE 收益，不是每一项算子百分比相加；旧文档中的“全 BF16 满血版”实际使用 FP16 视频 VAE，不能改称 BF16 VAE 基线。

| 比较项 | 结果 |
|---|---:|
| 同 INT8 精度、同 8 步的 DiT | 耗时 **−6.49%** |
| 原生 FP16 VAE 基础路径 → 完整优化路径 | 完整周期 **−11.65%** |
| 按耗时倒数计算的串行容量 | **+13.19%**；不是持续服务吞吐实测 |
| 同 INT8 路径的视频/音频 latent、RGB、PCM | 本轮四次正式请求均 **SHA 一致** |
| 历史 15 段相同 latent，FP16 → INT8 VAE | 原始 RGB PSNR **58.67 dB**，成片 SSIM **0.98907**，LPIPS **0.00759** |

INT8 VAE 差异很小，但不是像素完全相同。DiT 融合保留的是**所选 INT8 基线**，不是宣称等价于 BF16 dense。音频继续 GPU FP32。

**公开输入复验：** 768×512、124 帧、公开 Larry 合并权重，**19.31 → 16.76 秒（−13.19%）**，四项输出 SHA 一致。首次逐层验证不计入正式时间，见[单独数据](evidence/public-input-medium.json)。

**上游贡献：** 已提交[三个独立 Kitchen PR](docs/KITCHEN-UPSTREAM-PRS-20260930.zh-CN.md)，
附源码、回归测试和原始实测。新增的 [Indexed Gate 实验节点](docs/INDEXED-GATE.md)
可以单独验证 #219 的完整 H3 收益；需源码编译尚未合并的 PR，与完整 Optimize DiT 节点二选一。
三个 PR 仅覆盖部分优化；[整套 6.49% DiT 方案的上游拆分与剩余工作](docs/FULL-UPSTREAM-PLAN.zh-CN.md)。

## 安装与接入

首版支持 **Linux / 5090 D v2 / Torch 2.12.0+cu130 / Triton 3.7.0 / Kitchen 0.2.36 / 固定 ComfyUI H3 源码版本**。[完整版本表](configs/compatibility.json)。建议单独建环境，插件不会自动替换 PyTorch。

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Tokha233/ComfyUI-H3-SpeedKit.git
cd ComfyUI-H3-SpeedKit
python -m pip install comfy-kitchen==0.2.36 av
git clone --branch v4.5.0 --depth 1 https://github.com/NVIDIA/cutlass.git /tmp/h3-cutlass
python tools/build_kernels.py --cutlass /tmp/h3-cutlass
python -m h3_speedkit --probe-cuda
```

以 `--disable-cuda-malloc` 启动 ComfyUI。按此顺序连线：

1. 加载 INT8 ConvRot H3 / 合并后的 Larry → H3 Sigma Shift → **H3 SpeedKit · Optimize DiT** → KSampler。
2. KSampler 使用 8 步、Euler、beta、CFG 1；video/audio shift 为 12/4。
3. **H3 SpeedKit · Video VAE Loader** 加载上游 INT8 VAE，参考图沿用旧 encoder。
4. 采样结果 → **Decode to RGB8**；音频 → 标准 FP32 Audio VAE Decode。
5. 两路接入 **H3 SpeedKit · Save Video**，文件写完才返回。

Larry 首次使用需按[详细说明](docs/INSTALL.md)运行 `tools/merge_lora.py`，不要在已合并权重上重复挂 LoRA。不要同时使用其他 sparse/cache/attention patch。新 token 长度会先逐层对照八次模型前向，因此首次更慢；验证失败使用原结果并关闭该长度的加速。历史长度直接使用已测路径。

需要后接调色、upscale 等 IMAGE 节点时，保留普通 VAE Decode；DiT 节点可以独立使用。源码编译需要 CUDA Toolkit 13.0；没有预编译 wheel，也没有捆绑模型权重。

## 开源内容

- SM120 INT8 dense 主循环：三槽 TMA、Q 寄存器复用、BSHD 输出与最后一层有效 query 范围。
- Norm / modulation / ConvRot 量化融合，QKV / RMS / RoPE / QK 量化融合，当前输入 K anchor 与 V 量化转置。
- FC1 raster 调度，以及 outproj/FC2 的 gate/residual epilogue，保留 BF16 舍入位置。
- 旧 encoder + 上游 INT8 decoder，最终 blend 后转 RGB8，双 pinned buffer 异步 D2H。
- CPU PyAV 编码和按任务数/字节背压的服务导出队列；普通 ComfyUI 不会被全局修改队列调度。
- 构建脚本、公开输入 benchmark、逐项迁移说明、30 项历史优化与负结果、脱敏原始数据和复算工具。

[逐项实现与数值要求](docs/MIGRATION-036.md) · [历史试验](docs/R85-EXPERIMENTS.md) · [历史 LoRA 质量](docs/R85-QUALITY.md)

## 复验

```bash
python -m unittest discover -s tests -v
python tools/verify_kitchen036.py
python tools/verify_metrics.py
python tools/check_repository.py
python benchmarks/run.py --help
```

离线检查不能代替 GPU 测试。benchmark 接收你自己的参考图、模型和提示词；新环境、新尺寸、新权重都应重新比较。Windows、其他 GPU、其他 Torch/ComfyUI 版本尚未认证。

## 归属与许可

感谢 ComfyUI、Comfy Kitchen、SageAttention、CUTLASS、PyTorch、SGLang；Larry v4 LoRA 和 INT8 VAE 均为上游成果。本项目公开硬件专项适配、融合、工程接入与验证，不将上游模型/算子宣称为原创。

组合插件按 [GPL-3.0-or-later](LICENSE) 发布；内核和其他文件的原有 Apache/BSD 许可保留，见 [NOTICE](NOTICE)、[来源索引](evidence/kernel-provenance.json)。H3 模型权重另受社区许可证的地域与商业条件约束。业务素材、私有提示词和合并权重不公开。

[Hugging Face 项目页](https://huggingface.co/StellarVoyager/MiniMax-H3-SpeedKit-RTX5090Dv2) · [Kitchen 上游 PR 机会分析](docs/KITCHEN-PR-OPPORTUNITIES.zh-CN.md)

**10 月 1 日上游组合实测：** [完整报告与复现证据](docs/UPSTREAM-RECOVERY-TESTS-20261001.zh-CN.md)。在更新的固定 Kitchen/ComfyUI 基线上，完整 8 步采样 **15.737→14.201 s（−9.76%）**，视频/音频 latent、RGB8 和 PCM 哈希一致。此实验集成与上述已发布插件的历史口径分别统计。
