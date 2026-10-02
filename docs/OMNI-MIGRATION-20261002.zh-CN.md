# 2026-10-02：Omni 迁移实验与 PR 修复

## 当前结论

本轮已修复 ComfyUI #16681 的正式依赖兼容问题，提交 `85f21d224f39800cdf566dc9d89221d5305d3d29` 已推送，14 项 GitHub CI 全部通过，包含 Windows、macOS、Linux。Kitchen #231 的复现资料评审意见也已修复并推送。GitHub 发布通道已恢复，新增 [SGLang-Omni #2481](https://github.com/sgl-project/sglang-omni/pull/2481) 和 [vLLM-Omni #8414](https://github.com/vllm-project/vllm-omni/pull/8414)，均为正式 PR。

新框架中最明确的可迁移收益是 **vLLM-Omni H3 视频 VAE 的 SM120 支持：6.202 → 4.632 秒，解码耗时减少 25.32%，FP32 解码输出逐字节一致**。这是该框架内参考路径对比，不能算作当前线上 INT8 VAE 的额外收益。

SGLang-Omni 的多 Tensor 传输优化已完成真实 SHM 测试，40 MiB / 160 MiB GPU 来源载荷的完整传输耗时减少 13.49% / 16.58%。它没有运行语音模型，不代表模型端到端 QPS。

SGLang 原生 H3 的 Larry8 采样可以跑通，但优化后仍比当前 ComfyUI 慢，且生成结果存在明显差异，**尚不能替换当前生产方案**。

## 测试环境和口径

- 仅使用实验容器内 GPU0：RTX 5090 D v2 24 GB，SM120；未修改其他 GPU 服务。
- 驱动 580.126.20，PyTorch 2.13.0+cu130，Triton 3.7.1。
- vLLM-Omni `7b15d22f48a4af2ba83f9a209944d0496348fc91`，实际 vLLM 0.30.0；独立虚拟环境，避免旧 `/last/deps` 遮蔽依赖。
- SGLang `a71c7d19e7154f5cf2a698418486706f7ce314f6`；SGLang-Omni `1523543056bf334bc2f4d69db23aad0a002cedf2`。
- 公共采样用例：768×512，124 帧，Larry v4 已合并 INT8，seed42，Euler/beta 8 步，CFG1，shift12/4。
- 本轮不是此前 5 个约 30 秒业务 case，也没有重新运行 BF16 50 步 teacher。质量指标参考同环境 ComfyUI Larry8。
- 原始记录：[evidence/omni-migration-1002](../evidence/omni-migration-1002/)；实验脚本和复现说明：[experiments/omni-migration-1002](../experiments/omni-migration-1002/)。

## PR 失败及修复

| PR | 原因 | 操作与验证 | 状态 |
|---|---|---|---|
| [ComfyUI #16681](https://github.com/Comfy-Org/ComfyUI/pull/16681) | CI 固定 Kitchen 0.2.36 没有 `int8_linear_indexed_gate` | 接口存在才融合；不存在用已有 INT8 linear 与 segmented gate；真实 0.2.36 下 30 项测试通过 | `85f21d2` 已推送，14 项 CI 全绿 |
| [Kitchen #231](https://github.com/Comfy-Org/comfy-kitchen/pull/231) | 评审指出 smoke 命令不能复现 attention / sampler 数据 | 补原始测试脚本、完整编译链接参数、固定版本、输入 hash 和精确命令 | `98f7fb9` 已推送；3 项检查通过，Build Wheels 为 action_required，待维护者批准运行 |
| [SGLang #42121](https://github.com/sgl-project/sglang/pull/42121) | `Require run-ci label` 门禁失败，后续汇总作业连带失败 | 正文已补齐真实 loader 回归和 GPU 采样结果，保留机器人 CI 状态区 | 已留言申请维护者加 `run-ci` |
| [ComfyUI #16712](https://github.com/Comfy-Org/ComfyUI/pull/16712)、[#16713](https://github.com/Comfy-Org/ComfyUI/pull/16713) | 当前无失败项 | 各 14 项检查通过 | 等审核 |
| [SGLang-Omni #2481](https://github.com/sgl-project/sglang-omni/pull/2481) | 新贡献者工作流为 `action_required` | CUDA 环境 32 passed、1 skipped；全仓 pre-commit 通过 | 正式 PR，无合并冲突；已留言申请工作流授权和 `run-ci` |
| [vLLM-Omni #8414](https://github.com/vllm-project/vllm-omni/pull/8414) | 新提交，正在运行上游检查 | 69 项相关测试及 changed-file pre-commit 已通过 | 正式 PR，无合并冲突；DCO、Python 3.11/3.12 构建、pre-commit 通过，文档构建进行中 |

最后一次 API 核对：#16681 全部 workflow 成功、mergeable_state=clean；旧机器评审的 changes requested 记录仍在，需要评审方重新确认，不等于已批准合并。#231 当前没有新提交对应的新增评审意见；Build Wheels 尚未获得运行授权。

ComfyUI #16681 的新旧消费者使用相同 Kitchen #223 构建复测：每侧 1 次 warmup + 2 次正式请求，采样中位数 15.738291 → 15.580160 秒，约 1.00%。全部视频、音频 latent SHA256 一致。该次结果只证明当前修复保持正确性和原有小幅收益，不替换此前更大样本测试。

GitHub CLI 已以 Tokha233 完成认证。#42121 正文已更新并[请求 CI 授权](https://github.com/sgl-project/sglang/pull/42121#issuecomment-5944930250)；#16681 正文已改为当前兼容逻辑，并[补充验证回复](https://github.com/Comfy-Org/ComfyUI/pull/16681#issuecomment-5944932706)。#16681 最新 CodeRabbit 评论没有新增可执行问题，旧的正式 `CHANGES_REQUESTED` 记录仍需评审方确认。维护者的工作流授权与合并审核不会因本机登录而自动完成。

## vLLM-Omni：视频 VAE 精确算子迁移

### 改动

在已有 `H3_VAE_OPERATOR_TABLE` 增加 SM120。复用框架已有的 QK RMSNorm＋RoPE 精确融合、FP32 residual 精确融合、SiluAndMul 和 decoder block Linear 的 FP16 权重预转换。没有新增量化、改权重数值精度策略或改变时间分块。FP16 权重预转换匹配原 decode autocast 使用的计算 dtype。

正式视频 VAE 权重及 remote code：`MiniMaxAI/MiniMax-H3` 固定 `42ed227ee7df40d41602854ae760620d6eb651fe` 的 `Ref2VA/video_vae`。使用实际采样 latent `[1,24,37,32,48]`；默认输出 `[1,3,124,512,768]`。

### 完整解码

| 测试 | 参考 ms | SM120 ms | 耗时减少 | 输出检查 |
|---|---:|---:|---:|---|
| 默认 RGB8 解码，ABBA | 6202.364 | 4631.699 | 25.32% | 全部逐字节一致 |
| 保留 FP32 输出的解码 | 6213.711 | 4625.177 | 25.56% | 全部逐字节一致，max_abs=0 |
| 采样 latent 裁剪，56×256×384 | 443.818 | 322.649 | 27.30% | FP32 逐字节一致 |
| 零 latent，43×256×256 | 221.381 | 161.563 | 27.02% | FP32 逐字节一致 |
| 噪声 latent，69×256×384 | 591.425 | 430.835 | 27.15% | FP32 逐字节一致 |

主测试按 baseline/candidate/candidate/baseline 顺序，每组 1 次 warmup + 2 次正式解码；统计 4 个正式样本的中位数。补充 FP32 与三种输入为正确性检查，每侧 1 warmup + 2 次正式解码，不具有主测试一样的平衡顺序。

参数化复用脚本也已在相同 GPU 完成一次独立 ABBA 复测：6.199159 → 4.631679 秒，减少 25.29%，所有输出 hash 一致（`vo-portable.json`）。

主测试 PyTorch allocator 峰值约 10.661 → 5.825 GB。该数值不等于全卡或全服务总显存。收益主要包括减少每次 Linear 的权重转换/缓存和融合中间算子。

### 算子级结果

| 操作 / rows | 参考 ms | 优化 ms | 精确一致 |
|---|---:|---:|---|
| QK norm＋RoPE / 195 | 0.102432 | 0.031949 | 是 |
| QK norm＋RoPE / 8192 | 0.883917 | 0.100230 | 是 |
| QK norm＋RoPE / 32768 | 5.608381 | 0.511011 | 是 |
| scaled residual / 195 | 0.010275 | 0.018202 | 是；小形状变慢 |
| scaled residual / 8192 | 0.223792 | 0.119552 | 是 |
| scaled residual / 32768 | 1.060157 | 0.579174 | 是 |

没有隐藏小形状 residual 退化；最终资格以完整 decode 的收益为依据。原生测试共 69 passed，覆盖新增大形状、支持/不支持 capability、安装契约、temporal patch 和 split residency。完整 changed-file pre-commit 通过，包含 mypy、CI markers、SPDX 和禁止新增 `torch.cuda` 等本地检查。

分支 `perf/h3-vae-sm120`、提交 `ce3039e` 已推送至 Tokha233/vllm-omni，并创建正式 [PR #8414](https://github.com/vllm-project/vllm-omni/pull/8414)。补丁、PR 正文与最终 pre-commit 日志均已归档。提交前已确认可与新上游 `bbee488` 无冲突合并；性能测量仍对应上文固定基线，没有混用版本重新计算收益。

最初缺少 autocast 的 19 秒试跑、dtype 不匹配报错，以及测试环境缺 pytest/xdist 的准备失败均已排除，不计入速度统计。

## SGLang-Omni：多 Tensor host 打包

SGLang-Omni 是多阶段语音/全模态运行时；H3 的原生扩散实现位于 SGLang Diffusion，不能把二者混为一谈。

本改动先计算各 tensor 的对齐偏移，一次分配最终 CPU byte buffer，CUDA tensor 直接复制到最终 slice，省掉单独 D2H 临时 buffer 后的 CPU cat。阻塞复制在等待 relay credit 之前完成，保留 producer 随后复用输入内存时的快照语义；padding 清零。没有改成不安全的异步 D2H，也没有改变全 CUDA IPC 路径。

隔离复测采用真实 `write_payload → SHM write/read → ack → device restore`，ABBA，每块 3 次 warmup + 15 次正式事务，汇总每侧 30 个正式样本中位数：

| 输入来源与总量 | 参考 ms | 优化 ms | 耗时减少 |
|---|---:|---:|---:|
| GPU / 约 1.35 MB | 1.048607 | 1.057946 | -0.89% |
| GPU / 约 40 MiB | 49.997057 | 43.252745 | 13.49% |
| GPU / 约 160 MiB | 189.113012 | 157.749784 | 16.58% |
| CPU / 约 40 MiB | 31.758029 | 32.035689 | -0.87% |
| CPU / 约 160 MiB | 119.708824 | 120.825679 | -0.93% |

小载荷与 CPU 路径没有实质收益，不能宣传统一加速。此前独立一轮的大 GPU 载荷也有约 11% / 15.6% 收益；不同轮次主机负载影响绝对时间，所以只在每轮内部比较。

32 passed，1 skipped；包含 11 种 dtype、strided/empty tensor、device 保持、背压期间 producer 复用、取消后 credit 回收。所有 round-trip 值精确相等。分支 `perf/direct-multi-tensor-host-pack`、提交 `9897501` 已推送并创建正式 [PR #2481](https://github.com/sgl-project/sglang-omni/pull/2481)。与 #2368 的单 tensor 打包优化互补；提交前确认可与新上游 `fd8369e` 无冲突合并。

## SGLang：完整 8 步采样

先修复实际 H3 INT8 checkpoint 的 `.comfy_quant` 元数据加载失败：复用已有量化配置和 QKV layout，原 INT8 字节、FP32 scales 保持相同；提交 `f851045`，PR #42121。实际 safetensors 回归在原版报错、改后通过，相关 suite 28 passed、1 skipped、3 xfailed。

| 配置 | 采样 ms，中位数 | 比较说明 |
|---|---:|---|
| SDPA，split8192，0 resident | 32915.950 | SGLang 原设置 |
| SDPA，split0，0 resident | 32486.782 | 同框架输出逐字节一致，约 1.30% |
| Kitchen INT8 attention，0 resident | 26185.975 | attention 精度改变，不能称无损 |
| INT8 attention，24 resident / forward | 22811.053 | 与上行输出逐字节一致 |
| INT8 attention，24 resident / permanent | 21646.720 | 与 INT8 0 resident 输出逐字节一致 |
| 同环境 ComfyUI＋已有融合 | 15656.289 | 目前更快 |

这些时间包括 8 步采样及 condition preparation，排除 encoders、VAE、导出。residency 优化在 SGLang 内可保留输出，但整体迁移尚未保留 ComfyUI 结果。

### 相对同环境 ComfyUI Larry8 的质量

全部用相同 INT8 video VAE＋FP32 audio VAE 解码；对相同帧计算 SSIM，对相同长度 PCM / 幅度谱计算 cosine，float64 累积。指标衡量输出一致性，不代表独立主观质量评分。

| SGLang 配置 | 视频 SSIM | PSNR dB | PCM cosine | 幅度谱 cosine |
|---|---:|---:|---:|---:|
| SDPA | 0.666624 | 17.898 | 约 0.020327 | 约 0.12829 |
| Kitchen INT8 | 0.655392 | 16.614 | 0.027677 | 0.129514 |
| INT8＋resident24 | 同上 | 同上 | 同上 | 同上 |
| 去掉额外 audio slope 的探索 | 0.523042 | 13.737 | 0.036344 | 0.194736 |

最后一行仅为诊断实验：代码对比发现 SGLang adapter 比当前 ComfyUI 多一次 audio slope，因此做了独立移除测试。音频一致性稍升但视频一致性下降，整体仍不合格，**没有作为正式修复提交**。需进一步逐层对齐精度、条件处理与 timestep 语义。

## 后续门槛

1. 跟踪已提交的 #42121、SGLang-Omni #2481、vLLM-Omni #8414：维护者授权后完成上游 CI，对实际失败继续修复。发布通道恢复、正文更新和两个新 PR 创建均已完成。
2. 继续观察 #16681 新 head 的审核；Kitchen release 后才能使普通用户实际启用新融合接口，现有 pin 下先保证不报错。
3. 完整 H3 框架迁移必须先解决输出差异，再补 5 个 30 秒业务 case、BF16 teacher 和服务并发测量。当前不宣称端到端服务吞吐提升，也不替换生产基线。
