# ComfyUI 原生优化提交与实测：2026-10-02

本轮新增两个正式 PR，并修复现有 FC2 融合 PR 的三类边界。所有提交以 ComfyUI `2d6b73283af2447bdd065ece4090b8c6b1784544` 为对照。这里的结果不替代历史 R85 的整套 6.4922% DiT 加速，也不能相加成部署吞吐收益。

| PR | 内容 | 提交 | 合并条件 |
|---|---|---|---|
| [#16712](https://github.com/Comfy-Org/ComfyUI/pull/16712) | 条件/目标行直接拼接，移除布尔索引散写 | `6a9276f32653d3907fdde40394ffedc99627ecc0` | 正式 Ready；不依赖新增 Kitchen API |
| [#16713](https://github.com/Comfy-Org/ComfyUI/pull/16713) | 固定视觉/音频参考行每次采样准备一次 | `560558678c49ea157fb6fc62d56a2ee606390226` | 正式 Ready；不依赖新增 Kitchen API |
| [#16681](https://github.com/Comfy-Org/ComfyUI/pull/16681) | indexed FC2 gate/residual 消费者；修复全局 hooks、FP32 residual、梯度资格 | `5fa0a87` | Ready for review；仍须 Kitchen #219/#223 正式发布并更新 pin，当前 0.2.36 不含 API |

## 完整 Ref2VA 采样结果

RTX 5090 D v2，Torch 2.12.0+cu130，Larry v4 预合并 INT8，Euler/beta，8 steps，CFG 1，seed 42，video/audio shift 12/4，768×512、124 帧。输入含一张参考图，其 latent 为 `[1,24,1,16,24]`。条件文件与预合并权重是本地固定测试制品，不是本仓库的可下载资产；来源及权重口径见 [安装说明](INSTALL.md)。公开脚本支持用自己的可信条件文件重跑。

每个独立进程先跑一次 warmup，再记录两次正式请求。基线/拼接/参考预处理按 `base → packing → cond → cond → packing → base` 顺序运行，中间插入 gate 的独立对照。以下包含采样条件准备，**不包含 Qwen/VAE 编码、VAE 解码、导出或服务排队**。计时前后同步 CUDA；哈希在计时外计算；性能计时没有挂 profiler/hooks。

| 改动 | 正式请求数 | sampler 平均秒 | 相对基线 | 峰值 Torch allocation |
|---|---:|---:|---:|---:|
| 当前主线 | 4 | 15.735140 | — | 1,664,144,896 B |
| #16712 直接拼接 | 4 | 15.738751 | +0.023% 耗时，无可分辨增益 | 相同 |
| #16713 参考预处理 | 4 | 15.739265 | +0.026% 耗时，无可分辨增益 | 1,664,218,624 B，增加 72 KiB |
| 末层 MLP 裁行探索，未提交 | 2 | 15.741296 | 无可分辨增益 | 相同 |

显存数据来自 PyTorch allocator，不包含 aimdo 外部权重存储，不是总卡显存。

FC2 融合另用**相同 Kitchen gate 构建**对照，每侧 1 warmup + 2 正式请求：`15.740015 → 15.581095 s`，耗时减少 **1.0097%**；Torch 峰值增加 153 KiB。这是小样本复测，不能宣称生产 QPS 已提升 1.01%。其源码依赖为 Kitchen #223 head `6bf0cb0babea71e511b9da4bcf6f6ee663108b31`（基于 #219）；实测环境使用已有编译对象，不能将其称为正式 0.2.36 wheel 集成测试。

本批 9 个进程、27 次采样（含 warmup）的两模态 latent SHA256 全部一致：

- 视频：`8df41570a04e49d6e71924eacfcf5967baddab5762299589e813f6212d12abf1`
- 音频：`72e7410dc0809f58071a923c9cd8b4c80a2abf3239cafe5910060b4dfb98aefb`

本批没有重复解码并宣称新的视频 SSIM/音频频谱指标；这里验证的是进入后续 VAE 的两模态 tensor 逐字节相同。

## 小补丁减少了什么

独立 CUDA helper 实验调用实际 `_embed_and_pack`，使用合成的视觉+音频参考、目标 video latent `[1,24,37,32,48]`，参考图 `[1,24,1,32,48]`、参考音频 `[1,32,2,30]`。为隔离数据组织开销，projection 输出 hidden=8；它不是完整 H3 block benchmark。每侧 5 warmup、单独一次 profiler、然后 50 次同步 wall-clock 测量。50 次中位值：

| 实现 | helper 中位 ms | `nonzero` | `index_put_` | `cudaStreamSynchronize` | 每次 RNG |
|---|---:|---:|---:|---:|---:|
| 主线 | 0.795363 | 4 | 4 | 8 | 2 |
| 直接拼接 | 0.621977 | 0 | 0 | 2 | 2 |
| 参考预处理 | 0.320928 | 4 | 4 | 6 | 0 |

直接拼接减少约 21.80% helper 时间；参考预处理减少约 59.65% helper 时间，**后者不含一次性的准备成本**。这些毫秒级变化被完整采样的约 15.74 秒主干淹没，不能将 helper 百分比当作部署加速。

直接拼接只依赖既有“参考行在前、目标行在后”的模态行顺序，projection 次数/形状不变。ControlNet 仍使用的 layout masks 保留。现有 H3 6 项回归通过；另有 96 组 helper CPU 对照及 32 组 layout/梯度对照逐位相同。

参考预处理把 patchify/pack、确定性噪声增强和搬运移到 `extra_conds`；数据由本次条件 payload 持有，无模型全局缓存。目标 latent 和 projection 每步照常计算。启用梯度时保留逐次 helper 路径，避免跨步复用 autograd 图。10 项 H3 回归通过，包括增强系数 1/0.999/0.5、seed 变化、strided 输入和条件作用域。参考越长，额外驻留 tensor 越大；本例的 72 KiB 不代表所有输入的上限。

## #16681 的修复

- 全局 forward/pre-hook 与本地 hook 都保留 ungated 模块输出；hook 返回零也不会错误抹掉 residual。
- indexed fusion 要求 BF16 residual；FP32 residual 进入支持该类型的 fallback，不强行降精度。
- 任一相关操作数需要梯度时，走 eager module 和非原地分段累加。
- 29 项实际环境回归通过，覆盖 H3、mixed precision、indexed residual；全仓 Ruff 通过。
- 尚未发布的 Kitchen API 仍是显式合并阻塞条件。没有虚构依赖版本或加动态兼容探测来规避。

## 没有硬提交的候选

1. **末层 MLP-only 裁行**：完成真实模型探索。当前样例最终 latent 相同，但未测出净收益；实验会改变已无最终消费者的前缀 tensor，尚不满足全局/block/MLP/FinalLayer hook 和替换接口的观测契约。因此只保留实验脚本，不作为生产 PR。
2. **VAE 浮点异步 D2H**：当前 buffer API 不拥有有界 stream/event 生命周期。单纯加 `non_blocking=True` 不能证明正确重叠；完整实现需 wrapper/内存管理层的 buffer、取消、OOM 和返回前同步验证。本轮未完成该候选 GPU 验证，不提交未经验收的默认行为。
3. **Norm/QKV/outproj 消费者**：相关 Kitchen #219/#221/#222/#223 尚待发布。更大融合需要单独保持 backend、hooks、offload 和舍入契约；本轮未把私有 runtime 复制进 ComfyUI。已有 BSHD #16678 和 FC2 #16681 不重复开 PR。
4. 已合入的 INT8 VAE、embedding 生命周期、无 V clone 等不重复申报；静态 tag 准备与已有 #15560 重叠，未重复提交。

## 复现与原始证据

- [全部 JSON、原始执行脚本与 manifest](../evidence/comfy-submit-1002/)
- [聚合结果](../evidence/comfy-submit-1002/summary.json)
- [参数化完整采样脚本](../benchmarks/comfy_h3_sampler.py)

在所选 ComfyUI checkout 的完整环境中运行，提前安装相应 Kitchen 和依赖；A/B 必须固定同一模型、条件、依赖/编译二进制、attention 后端与硬件：

```bash
python benchmarks/comfy_h3_sampler.py \
  --arm base --pair 0 \
  --comfy-dir /path/to/ComfyUI-base \
  --kitchen-dir /path/to/comfy-kitchen \
  --weights /path/to/larry-v4-merged-int8.safetensors \
  --condition /path/to/trusted-condition.pt \
  --output-dir /path/to/results
```

条件文件须是本地可信的 `torch.save` 制品，字典含 `positive`（正常 ComfyUI conditioning）与 `latent`（含 `samples` 的正常 latent 字典），并已导入其序列化对象所需模块。脚本使用 `weights_only=False`，不要给它不可信的 pickle 文件。先以同一输入检查 A/B latent hash，再评估时间。完整输入/模型未分发，因此其他输入不保证复现本页的绝对时长或哈希；无需模型权重的 helper 原型和上游回归可独立重跑。

原始环境脚本保留当时路径；参数化脚本适合在另一台机器运行，已在同一 GPU 环境额外完成 1 warmup + 2 正式请求，视频/音频 latent hash 保持一致（该额外验证未加入上表统计）。性能、正确性和 CI 是分别记录的证据，不将排队中的 CI 写成已经通过。

## 首轮 GitHub CI 核查

- #16712：14 项 check run 全部成功，mergeable_state 为 clean。
- #16713：13 项成功；Windows 2022 的既有 mixed-precision 测试在 `comfy/ops.py` 的 CPU `_forward` 原生运算处触发 `0xc000001d`（illegal instruction）。此 PR 未修改该代码，其他 Windows/macOS/Linux jobs 成功；日志尚未显示本次参考预处理测试断言失败。应先重跑失败 runner 验证，不能把这一条直接记为通过，也不能据此改坏无关 CPU kernel。
- #16681：Kitchen 0.2.36 缺新增 API 的 release/pin 前置条件仍在，正式合并前必须补齐；本地含 #223 的测试通过不替代发布依赖下的 CI。
