# 2026-10-01：Kitchen 后续贡献与最新 H3 加速方案

> 后续新增 GPU 实验：Gate 分带未得到稳定收益；D64 小查询块已提交 #227，完整 decoder 独立进程对照 −0.96%；PDMD 已进行 Ref2VA 音视频对照。见 [实测报告](PDMD-KITCHEN-TESTS-20261001.zh-CN.md)。以下保留下午检索时的研究快照。

核查时间：北京时间 2026-10-01 下午，资料检索截至约 15:10。目标场景：单卡 RTX 5090 D v2 / 24 GB、Ref2VA、Larry v4 8 步、INT8 DiT、已接受的 INT8 视频 decoder、保留旧 encoder。本文是源码审计、公开资料核查和 PR 推进记录，**本轮没有新增 GPU 性能实验**。

建议继续以现有经过验证的组合为部署基线。最有价值的后续工作是：**把最新 GEMM 分带调度接到 indexed gate；迁移历史 VAE D64 精确优化；补齐通用消费者边界。** 新模型首先评估 PDMD 4-NFE，其次是 LynnReal LightVAE 和 LongLive-Plug；Veda 值得作为稀疏研究支线。尚无新方案经过本项目五个业务 case 验证，能同时证明超过 Larry v4 8 步的速度与音视频质量。

## 1. Draft 可以直接转；本轮实际做了什么

PR 作者可以点击 **Ready for review**。这是告诉维护者代码可以开始正式审查，不等于通过审查、CI 已绿或能够立即合并。有前置 PR 的变更也可以提前送审，把合并顺序写清即可。

已将以下四个 Kitchen PR 转为非 Draft，并更新说明中的测试数量、已完成的验证与依赖：

| PR | 内容 | 本轮结果 | 合并前置条件 |
|---|---|---|---|
| [#220](https://github.com/Comfy-Org/comfy-kitchen/pull/220) | INT8 attention 直接输出连续 BSHD | **Ready for review** | #218 先合并 |
| [#221](https://github.com/Comfy-Org/comfy-kitchen/pull/221) | indexed RMSNorm/modulation/ConvRot | **Ready for review** | 审核窄 Torch/SM120 适用条件和公共 API |
| [#222](https://github.com/Comfy-Org/comfy-kitchen/pull/222) | H3 QKV 投影与 RMS/RoPE/INT8 准备融合 | **Ready for review** | 审核模型专用 API、回退与正式消费者边界 |
| [#223](https://github.com/Comfy-Org/comfy-kitchen/pull/223) | float-input indexed gate 组合 API | **Ready for review** | #219 先合并 |

Kitchen #217、#218、#219、#224 和 ComfyUI #16677 原本已经是非 Draft。此次送审后，整个贡献批次是 **11 个开放 PR：9 个非 Draft、2 个 Draft**；查询时没有已合并项。

保留 Draft 的是 [ComfyUI #16678](https://github.com/Comfy-Org/ComfyUI/pull/16678) 和 [#16681](https://github.com/Comfy-Org/ComfyUI/pull/16681)：它们依赖尚未正式发布的 Kitchen API，当前 requirements 仍固定 Kitchen 0.2.36。#16678 还存在关于依赖 pin 的正式修改要求。这是实际运行依赖尚未满足，不是 GitHub 禁止转换。待库发布，更新 pin、复测并转正式即可。

#221、#222 的算子 API 已完成独立测试，因此可以先审库；这不代表通用 ComfyUI 消费者也已经完成。实验消费者固定 INT8 后端的做法不能直接冒充通用接入。

既有 CLA/Socket 检查成功，外部贡献 workflows 曾为 `action_required`，需要维护者批准执行。Ready 状态不自动解决这一点。此次后段公共 GitHub API 出现限流，#222/#223 的转换和描述保存以登录网页回读核实；没有把旧 CI 快照写成新一轮 CI 已通过。

## 2. 先固定收益口径

| 测量 | 基线与范围 | 已有结果 | 不代表什么 |
|---|---|---|---|
| 发布插件 .36 | 两个代表业务片段，prepared condition → 本地 MP4；原生 FP16 VAE → INT8 VAE | 264.44 → 233.63 s，耗时 **−11.65%** | 不含 condition、网络和冷加载；不是纯 DiT 融合 |
| 发布插件同 INT8 VAE | 相同视频 decoder，完整上述周期 | **−6.72%** | 不能再叠加上一行 |
| 历史 r85 DiT | 当时固定版本与业务对照 | **−6.49%** | 不能与新组合的百分比相加 |
| 10/1 上游组合 | Kitchen `19ea55b` + ComfyUI `8cfe5e1`，768×512 / 124 帧 / 14,850 tokens，Larry8；仅完整 sampler | 15.737106 → 14.200641 s，**−9.7633%** | 不含 condition/VAE/export，不是在 r85 上又快 9.76% |

10/1 组合每臂 8 次正式采样，视频/音频 latent 相同；两臂使用相同 decoder 后 RGB8/FP32 PCM 也相同。倒数换算的处理能力为 +10.8197%，不是在线并发 QPS 实测。后续 #224 正确性修复复验没有改变此 H3 输出。

原生 VAE 的既有发布对照实际是 **FP16**，应沿用日志记录，不改写为全 BF16。历史 15 份相同 latent 的 INT8 VAE 对照：raw PSNR 58.67 dB、MP4 SSIM 0.98907、LPIPS 0.00759；差异很小，但不是逐像素相等。

来源：[.36 基准](BENCHMARK-036.md)、[10/1 组合实测](UPSTREAM-RECOVERY-TESTS-20261001.zh-CN.md)、[非正 scale 修复](ATTENTION-NONPOSITIVE-SCALE-FIX-20261001.zh-CN.md)。

## 3. Kitchen 和相关库到底更新了什么

### 3.1 Kitchen 主线已向前推进，包版本仍是 0.2.36

本次源码审计固定主线 [`12389a30463c62c93670b049d59bf3fa56c0316d`](https://github.com/Comfy-Org/comfy-kitchen/commit/12389a30463c62c93670b049d59bf3fa56c0316d)。相对上次基线 `19ea55b`，主要变化如下，时间均为北京时间：

| 更新 | 合并/提交时间 | 实际作用 | 对当前基线的判断 |
|---|---|---|---|
| [#216](https://github.com/Comfy-Org/comfy-kitchen/pull/216) | 10/1 02:34 | CUDA W4A8/W6A8 ConvRot 重量化融合 | 更贴近低比特权重加载/LoRA patch；不是新的 INT8 decoder |
| [#225](https://github.com/Comfy-Org/comfy-kitchen/pull/225) | 10/1 06:39 | #216 的 HIP 移植 | AMD 收益，不是 5090 D v2 新内核 |
| [#215](https://github.com/Comfy-Org/comfy-kitchen/pull/215) | 10/1 06:57 | INT8 GEMM N-banded Stream-K、L2/shape 选择与 wave guard | **值得验证，尤其长序列普通 GEMM 和自定义 gate** |
| `12389a3` | 10/1 09:17 | 进一步优化 W4A8 requant | 不自动改善现有纯 INT8 稳态主循环 |

重要区别：`pip install comfy-kitchen==0.2.36` 不意味着已经拿到这些最新 main 提交。需要明确源码 SHA 与编译产物；也不能只升级 wheel，就认为 SpeedKit 的独立 CUDA 库已经更新。

### 3.2 查到的发布版本

| 库/项目 | 本次版本 | 发布时间或源码信息 | 判断 |
|---|---|---|---|
| comfy-kitchen | 0.2.36 | PyPI 9/29；main 见上文 | main 有实质新调度，先做隔离源码 A/B |
| PyTorch | **2.14.1** | PyPI 10/1 01:43 | 新 patch release，不等于自定义 CUDA 自动提速 |
| Triton | 3.8.0 | PyPI 8/29 | 已知版本，不能当作今天新加速 |
| flashinfer-python | 0.7.0.post1 | PyPI 9/30 07:34 | 新发行包不证明其中 H3 路径超过当前 INT8 |
| comfy-aimdo | 0.5.5 | PyPI 9/15 | 未查到比当前使用版本更高的发布 |
| [H3-Optimizations](https://github.com/Zironic/H3-Optimizations) | **0.2.45** | head `862774944a33`，9/25 00:45 | 仍是已测 FastH3 VSA 那版，没有新 0.2.46 证据 |

版本来源：[Kitchen](https://pypi.org/project/comfy-kitchen/)、[Torch](https://pypi.org/project/torch/)、[Triton](https://pypi.org/project/triton/)、[FlashInfer](https://pypi.org/project/flashinfer-python/)、[aimdo](https://pypi.org/project/comfy-aimdo/)。

本项目之前 PyTorch 2.14/Triton 3.8 的完整 DiT 实验没有稳定速度优势，且输出发生变化。2.14.1 可单独复测，不能直接继承之前版本的质量结论，也不能把重编译本身记成收益。

## 4. 最值得继续优化、投稿的地方

| 优先级 | 方向 | 贡献应投哪里 | 现有证据 | 下一步必须补什么 |
|---|---|---|---|---|
| P0 | 最新分带调度接 indexed gate GEMM | Kitchen #219 后续小 PR，或按维护者意见并入 | gate 仍使用旧 swizzle；#215 已有长序列微基准收益 | 同 epilogue 的窄 A/B、真实 outproj/FC2、完整 sampler |
| P0 | 将已有算子可靠接入通用 ComfyUI | ComfyUI，依赖 Kitchen 发布 | 实验组合 −9.76%，API 大部分已提交 | attention patch/hook/offload/训练契约、pin、消费者测试 |
| P1 | VAE D64 cached-Q / 小 Q tile | Kitchen 独立窄 dispatch | 历史局部 attention 约 −11% | 最新主线迁移、不同 batch/长度、完整相同 latent decode |
| P1 | VAE D64 RMS/RoPE → Q 旋转量化 | Kitchen API + ComfyUI consumer 分开 | 历史 R234v2 已有精确验收 | 保留 K anchor、FP16 舍入、非目标形状回退 |
| P1 | 把同请求 MP4 编码提前到 VAE 解码期间 | SpeedKit/服务层，通用部分再投 ComfyUI | 上游有新实测；我们已有 RGB8/D2H/跨请求队列 | 先确认当前 exporter 缺的究竟是哪段重叠，测最后字节时间 |
| P2 | SM120 dense 主循环资源调度 | 先完善 #218，再小步追加 | 主线 cached-Q 仍只为 SM89 开启；我们已有 SM120 TMA | 真实 stalls、寄存器/共享内存瓶颈和严格输出对照 |
| P2 | #221 扩大已验证 Torch reduction 范围 | Kitchen #221 后续兼容补丁 | 当前 native 只允许测试过的 Torch revision | 新版 reduction/量化边界逐值验收，不能删 gate 硬开 |
| P2 | workspace/设备属性缓存的并发与图捕获 | Kitchen 正确性或工程小 PR | 静态代码有值得核查的点 | 先复现问题与净开销，再定实现；尚无性能收益数据 |

### 4.1 最明确的增量：indexed gate 接新 GEMM 调度

最新普通 INT8 GEMM 在 `comfy_kitchen/backends/cuda/ops/cutlass_gemm_int8.cu` 使用 `ThreadblockSwizzleLeanStreamKT<32>`；`cutlass_gemm_common.cuh` 按 L2 与 SM 数调整 band。我们的 `cutlass_gemm_indexed_gate.cu` 仍实例化旧 `ThreadblockSwizzleLeanStreamK`，其别名等于 `<0>`。

即：**普通 GEMM 已按 N 分带，自定义 gate GEMM 没有自动继承这一变化。** 将相邻 CTA 的权重工作集控制在 L2 内，可减少长 M 场景的重复权重读取；实际收益还取决于激活大小、N/K、tile 和 epilogue。

我们的既有独立测试针对 #215 早期 head `92e9aa6`：

| M×N×K | 基础 | #215 | 时间变化 |
|---|---:|---:|---:|
| 32700×21504×5376 | 13.661868 ms | 13.362028 ms | −2.19% |
| 87142×21504×5376 | 36.377665 ms | 35.437028 ms | −2.59% |
| 87142×28672×5376 | 47.965631 ms | 47.186445 ms | −1.62% |
| 14850×21504×5376 | 6.144162 ms | 6.152232 ms | 无收益，略慢 |

这不是最新 main 与我们的 gate 组合实测。应先比较最新 main 的普通 GEMM，再测试 gate 的 band=0/8/16/32/64。保持 INT32 accumulation、BF16 输出舍入、gate/addcmul、bias、residual 顺序一致；记录 small-M 回退与长期重复结果。

新主线还已有面向约 2k-row decoder tile 的 **wave guard**。因此“按 SM 数修正 VAE GEMM tile”已不完全是空白，不能重复发一份相同选择器。先检查真实 VAE shape 触发哪个配置，确认增量。

若实测有收益，PR 应突出“让 indexed gate 复用已有分带调度”，署名引用 kijai #215，不把分带算法本身当作原创。若主要改变已有 PR 文件，优先与维护者确认放在 #219 还是后续；不是为了增加 PR 数量重复提交。

### 4.2 VAE D64：小而明确的下一份算子贡献

视频 decoder 为 32 heads × D64，常见 attention shape `[4,32,1797,64]`；DiT 是 56 heads × D128、hidden 5376、partial RoPE 96。**#222 不能简单复用于视频 VAE。**

历史 R244 将 Q 保留寄存器、CTA_Q 128→64、WARP_Q 32→16、K tile 保持 64；寄存器约 196→128，驻留 CTA 约 2→4，Tensor active 53.42%→63.28%。该真实 shape 的 attention 局部约快 11%。

R234v2 则融合 RMS/RoPE 后的 Q 旋转量化，保持 K 的原 FP16 写回与 anchor 检测。第一版因 FMA/倒数边界变化被拒绝，只有 v2 通过原有精确标准。

两项组合相对“静态 VAE＋R207”使全视频 decode 约 **−0.632%**；不是整个请求 −11%，也未达到全部 VAE <5s。历史 15 份 latent、30 次预热与 60 次正式 decode 是可复用的验收资产，但仍需迁移到当前主线重跑。

提交建议分两步：先提交不改 API 的 D64 小 tile dispatch，再考虑需要 prequantized container 的准备融合。保留原量化 scale 索引、softmax/PV 累加顺序、batch/stride/mask/empty 支持；不能用“都 close”覆盖既有 exact 要求。

### 4.3 不要重复提交已存在的 VAE 融合

当前 ComfyUI `comfy/ldm/minimax/vae.py` 已通过 `linear_input_act` 融合 pre-norm 与量化，FC2 带 SwiGLU 和 residual epilogue，Q/K 用 Kitchen `rms_rope_split_half_`，量化 decoder 走 INT8 attention。

SGLang [#41906](https://github.com/sgl-project/sglang/pull/41906) 新增 FP32 RMSNorm、scaled residual→norm、D64/RoPE48 融合，作者报告 H200 单 temporal unit 656.3→589.5 ms（−10.2%），4×H200 完整请求 11.00→10.75s（−2.3%）。但其对照是 SGLang 的 eager 路径，且误差约 5e-5；1×H200 的 10s decode 数值是 unit×14 推算，不是独立整段测量。

我们的当前库已覆盖部分融合，不能再声称迁入它就会额外快 10%。真正要比较的是**剩余内存往返**和不同实现的舍入语义。其 residual→norm 若与现有 GEMM residual epilogue 争用同一边界，应测完整链，不能只叠加局部百分比。

[ComfyUI #16698](https://github.com/Comfy-Org/ComfyUI/pull/16698) 已修复 VAE offload 问题；#16657 已支持轻量 decoder。避免为已解决的 qk norm buffer 设备问题重复提 PR。

### 4.4 SM120 attention 可以继续，但不能承诺大幅无损提升

最新 Kitchen 普通 dense 核的 `cache_query` 仍只对 SM89 / D128 的限定组合开启。我们的 #218 已提供 SM120 TMA 路径，因此下一轮应该针对它实际 profile 出来的长延迟依赖、共享内存冲突、数据搬运、occupancy 和 epilogue 调整。

可探索：按短/长序列选择 CTA 和 pipeline stage；控制寄存器与共享内存占用；保持 K/V 访问顺序的预取；省掉消费者布局转换。改变 softmax 归约、split-K 合并顺序、删补偿项或稀疏跳块会引入数值变化，应与 exact 方向分开。

SM120 与 SM100/SM103 虽同属 Blackwell，指令及资源假设不同，不能将 B200 的 kernel 改一个架构宏就宣称得到 5090 专用实现。GPU busy 高也不等于 Tensor Core 达到理论上限；应以同一精度、相同时钟下的 roofline、CUPTI/Nsight 指标判断。本文没有新的 GPU 利用率测量。

### 4.5 主机调度与 workspace：先证明问题

`cutlass_gemm_common.cuh` 当前 workspace 按 device/stream 缓存，查表及扩容由全局 mutex 保护；扩容时在锁内 cudaFree/cudaMalloc。新增设备属性缓存使用静态数组。值得验证并发初始化、首次 CUDA Graph capture、workspace 变大时的行为和调用开销。

这是静态审计线索，不是已经复现的服务 bug，更不是已证实的吞吐瓶颈。不要仅因“有锁”就写无锁缓存，也不要在 GPU 主循环之外的微小开销上承诺大幅速度收益。

## 5. 最新少步模型：哪些真正值得测

### 5.1 PDMD 4-NFE / 2-NFE：优先级最高的新 LoRA 候选

原作者权重：[4-NFE](https://huggingface.co/pdmd2026/pdmd_4NFE_lora)、[2-NFE](https://huggingface.co/pdmd2026/pdmd_2NFE_lora)；[论文](https://arxiv.org/abs/2609.35768)。本次 HF 快照两者 9/30 更新，revision 分别 `770549d…`、`81337b0…`。

PDMD 将 DMD 更新投影到与 student–critic endpoint residual 正交的空间。它改变蒸馏训练，不额外要求推理时再跑一个 critic。权重约 1.38GB，rank/alpha=128，覆盖 50 个主块与 2 个 refiner；视频/音频 VAE 和 condition 使用基础模型。

先试 4-NFE，质量合格再试 2-NFE。原 recipe 是 `transformer/` 基础模型、strength=1、shift=12/3，尚未给出我们 Ref2VA 的业务验收。不能因为 tensor 形状匹配，就认为 reference image/video/audio 保真已经证明。

ComfyUI 转换需特别小心：

- [4-NFE 转换](https://huggingface.co/Iwannapose/minimax_h3_pdmd_4nfe_comfyui) 当前应使用 **`minimax_h3_pdmd_4nfe_comfyui_v6.safetensors`**；模型卡明确说明旧版存在 key 无效和 SwiGLU 半区错误。
- [2-NFE 转换](https://huggingface.co/Iwannapose/minimax_h3_pdmd_2nfe_comfyui) 也必须固定具体 revision。
- q/k/v 合并需 block-diagonal LoRA 布局与正确 alpha/rank；FC1 需处理 Diffusers `[value; gate]` 到 ComfyUI `[gate; up]` 的顺序。
- 转换者称 208 个 target key 与 Ref2VA INT8 权重形状一致，这是加载兼容性证据，仍不是生成质量测试。
- Kijai 实验仓还有 average-rank 38/57 的压缩版本。先测完整 rank，再单独测压缩，避免把“蒸馏质量”和“二次低秩压缩”混为一项。

四步相对八步减少模型调用，但 condition/VAE/export 不随之减半，不能直接宣传整请求 2×。任意加至 6/8 步也不保证更好，先忠实复现训练 schedule。

### 5.2 LongLive-Plug：有价值，但 H3 配方并非普通四步 Euler

[官方项目](https://github.com/NVlabs/LongLive/tree/main/LongLive-Plug) 9/28 宣布；[H3 few-step 模型](https://huggingface.co/Efficient-Large-Model/LongLive-Plug-MiniMax-H3-few-step) 9/29 上传/更新。源码审计 head `fb16a879f46e604df4a0ca48ce2bf45adfa90352`，HF revision `b3686f432d3f20d6d6462a91d4c09ba7f189c0a2`。

研究目标是一次蒸馏得到可迁移到同一骨干下游模型的功能 LoRA。官方展示 H3 ControlNet/SolarWM 等下游例子，有音视频四次 forward 配置，值得评估。

但有三个实际限制：

1. 通用 README 说 few-step=1.0 与 CFG=0.5 一起用；**H3 专属模型卡明确要求当前分开使用，不推荐叠加**。不能套用 Wan 合并示例。
2. HF 配套 `infer_student_4step.py` 明确检查 **Base T2VA**，Ref2VA 会被拒绝。项目下游迁移展示不等于这个脚本已支持我们的 Ref2VA。
3. 配套 sampler 使用 **fresh-noise**：每次预测 clean，再按下一 sigma 加新噪声；固定 4 次模型调用、5 个 sigma endpoint、video/audio shift=12/3，并维护两模态 RNG 顺序。普通 Euler 4-step 不是相同算法。

模型文件可做 ComfyUI 映射，但还需实现正确采样器与条件语义。先做官方 T2VA 阳性对照，再研究 Ref2VA，优先级低于接入成本较低的 PDMD4。H3 的训练文档在 GitHub 主说明中仍标注稍后发布，HF 已含推理快照；应区分发布渠道及完整程度。

### 5.3 未找到可以直接替换 Larry 的新版本

[Larry 原始仓](https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora) 当前仍有 v4 step600 / EMA，未找到作者发布 v5 的证据。[LightX2V Turbo](https://huggingface.co/lightx2v/Minimax-h3-Turbo) 仍是既有 Ref2VA 4-step v0.1 和 8-step v1.0 768p。HyperFlow 也未发现新训练版本可推翻本项目既有结果。

Larry 是当前速度、视频和音频指标之间的业务选择，并不是每一项 reference similarity 都最高。无 LoRA16 曾更接近参考但明显更慢；新 LoRA 应与 Larry 和 BF16 50-step 参考同时比较。

## 6. 最新 VAE：LynnReal LightVAE 是换模型

[原始轻量 decoder](https://huggingface.co/stdstu123/LynnReal-Onmi-light-vae) 已在 9 月中旬出现；新进展是 [ComfyUI #16657](https://github.com/Comfy-Org/ComfyUI/pull/16657) 于 9/30 北京时间合入支持，以及 [Kijai INT8 ConvRot 转换](https://huggingface.co/Kijai/MiniMax-H3-experimental/blob/main/minimax_h3_lynnreal_light_vae_int8_convrot.safetensors)。

核心是用 **26 个 decoder Transformer blocks 代替 36 个**，并加载对应训练权重。它不是简单删掉最后十层的无损优化。encoder、latent 空间和音频 VAE 可以继续使用原版，所以特别适合用现有保存 latent 单独评估。

[转换者报告](https://huggingface.co/corechan/MiniMax-H3-LightVAE) 的 RTX PRO 6000 / 1280×704 / 124 帧：轻量 TRT 5.2s、原版 TRT 7.1s、轻量 PyTorch 8.3s。**52.8dB 是同一轻量模型 TRT 对 PyTorch，不是轻量模型对原始 36 层 VAE。** Kijai 实验卡另报告约 1.3× decode，加速分母和硬件均应按原文核对。

建议使用同一批 15 份 latent 比较：原36层 FP16、当前36层 INT8、26层 FP16、26层 INT8。固定 tile/blend/RGB8/export 与音频 PCM，检查 SSIM、LPIPS、PSNR、帧间闪烁、细节/脸/文字和接缝，再计时。不能将 26/36 的层数比例直接当成总 decode 耗时比例。

vLLM-Omni [#6131](https://github.com/vllm-project/vllm-omni/pull/6131) 的新 FP8 VAE 是另一条路线：B300 上纯 FP8 decode 1.205→1.104s（−8.4%），SSIM≈0.98535。其 0.447s 是 FP8+compile+stacked 组合，不是单独 FP8；也不是 5090 D v2 或当前 INT8 decoder 的对照。本项目当前优先保留 INT8。

## 7. 稀疏与其他推理框架更新

### 7.1 Veda / Miowtion：有新的 SM120 源码，但 Ref2VA 未验收

[Veda H3 T2VA preview](https://huggingface.co/Veda-Sparse/Minimax-H3-T2VA-Veda-8NFE-600Step-Preview) 在 9/30 更新。它使用训练的小 predictor 选择约 10% attention，不是 LoRA；训练与评估配合 Larry 的 8-step。作者称可兼容其他 LoRA/微调模型，但仍应以具体任务验证。

作者 RTX PRO 6000 /14.4s 的 attention 6.79×、表格所谓 E2E 3.12×，其表前文字限定为 **per denoising step**。不能据此宣传从输入到 MP4 的整请求加速，更不能把其 dense 分母替换成我们的融合 INT8。

本轮发现文档与源码不同步：[部署说明](https://huggingface.co/Veda-Sparse/Minimax-H3-T2VA-Veda-8NFE-600Step-Preview/blob/9a1fd3a41b4a754a7886e64e82edbddf599fd1bd/AGENTS.md) 的 SM120 段仍提示视 FA4 build 手动加架构；实际 [源码 `fa4.py`](https://github.com/veda-sparse/Miowtion/blob/45866c809db9167892b54915dbf9ef2ccc2dd56a/miowtion/kernels/fa4.py) 已设置 `PATCHED_MAJOR_ARCHS=(8,12)`。

[vendored patch](https://github.com/veda-sparse/Miowtion/blob/45866c809db9167892b54915dbf9ef2ccc2dd56a/miowtion/kernels/fa4_sm8x/__init__.py) 明确支持 SM120，借其继承的 SM80 kernel 增加 block sparse 主循环，锁定 FA4 4.0.0b32 / `d15f1531…`，并校验被替换五个模块及两个 SM120 模块的哈希。必须在其他 `flash_attn.cute` import 前安装。**不是仅改 whitelist，也不是上游 FA4 任意最新版开箱即用。**

这使其有实际可测试的 5090 代码路径；本轮没有执行 CUDA。24GB 文档要求 block offload 与 MLP chunking，PCIe 可能吞掉 attention 加速。发布 predictor 是 T2VA，源码虽有 Ref2VA pipeline，也不代表 predictor 训练覆盖参考音视频。

实施顺序：先验证确实走 sparse kernel而非 dense，再跑自己的 dense 对照，随后相同 Ref2VA case 做质量评估；必要时探索 early steps 保持 dense 和较高 keep ratio。改变 keep ratio 本身也偏离训练点，应单独记账。这属于近似推理，不列入无损 PR。

### 7.2 SGLang 新 PR 中哪些可借鉴

| PR | 新内容/公开数据 | 对当前项目的价值 |
|---|---|---|
| [#41819](https://github.com/sgl-project/sglang/pull/41819) | VAE 出最终帧后立刻编码 MP4；H200 上 H3 5s 请求约省 0.34–0.36s；作者验证 MP4 字节相同 | 借鉴同请求 chunk callback、buffer 所有权和失败回退 |
| [#41906](https://github.com/sgl-project/sglang/pull/41906) | VAE norm/RoPE 融合 | 与 Kitchen 已有融合对照，避免重复 |
| [#37662](https://github.com/sgl-project/sglang/pull/37662) | FastH3 V2 正确训练 rungs、batched VAE、sort-free 路由、SM100 VSA | 批量/路由工程可借鉴；4×GB300 不能映射单5090；T2VA学生已测不适合直接Ref2VA |
| [#41965](https://github.com/sgl-project/sglang/pull/41965)、[#41985](https://github.com/sgl-project/sglang/pull/41985) | SubBlock int64 大索引、按head分块路由降低临时显存 | 单卡长序列的工程参考；稀疏本身仍改变模型输出 |

这些查询时仍未合并。#41985 的单 B200 case 即使分块，临时峰值仍约30.5GiB，不能直接装进24GB。按 head 分块相对同一稀疏算法可保持相同输出，不等于稀疏相对 dense 无损。

#41819 中音频先在 GPU 解码，再让 CPU/ffmpeg 与视频 GPU 解码重叠。我们历史已有最终 blend→RGB8、双 pinned D2H、跨请求 CPU export。新增价值取决于是否还存在**同请求编码尾巴**，不能把原有能力重新记成新收益。保持相同 x264/AAC 参数；改 preset 不是编码结果不变的优化。

音频 VAE 移 CPU 已测 4.49–9.44s，而 GPU FP32 约0.11–0.14s、PCM 不同，因此不重新优先考虑。可做的 CPU 重叠重点是编码、I/O 和素材准备。

### 7.3 vLLM-Omni 更新

- **[#8322](https://github.com/vllm-project/vllm-omni/pull/8322) 已于10/1 14:24合并**：LBH latent upscale 后高分辨率再采样。公开示例用8×H200、50 base steps+18 refine steps，视频/音频都会改变；无受控速度优势证明。是用户之前提出的方向的框架接入，但不是新训练 upscaler，也不能推翻本项目此前低分辨率两阶段的负面结果。
- [#8072](https://github.com/vllm-project/vllm-omni/pull/8072)：Ref KV 周期刷新、pinned host ring、跳过复用步参考投影；4×Ascend作者单次测量省7.65–15.80%。**跨 denoising step 的 Ref KV 是近似缓存**，输入 reference 不变不意味着深层 hidden/KV 不变；未有客观质量验收，不能当成精确“推测解码”。
- [#6894](https://github.com/vllm-project/vllm-omni/pull/6894)：支持 Comfy pruned INT8 ConvRot 权重。2×A800 对BF16报告−10.91%，这是新框架接入，不是相对我们当前INT8又快10.91%。
- [#8268](https://github.com/vllm-project/vllm-omni/pull/8268)：TaoMate-H3因果流式 talking-head，2/4×H200、历史KV、音频teacher、增量decode。2卡每卡139–143GB，属于不同任务与资源预算，不是24GB Ref2VA替换包。
- [#7711](https://github.com/vllm-project/vllm-omni/pull/7711)：SVDQuant/NVFP4融合有局部17–35%收益，但公开完整Ref2VA对照基本持平甚至略慢，尚无校准质量验收。当前无需转回此方向。

### 7.4 FlashInfer / LightX2V 的新标题不等于新可部署收益

FlashInfer [#5660](https://github.com/flashinfer-ai/flashinfer/pull/5660) 新增 SM120 NVFP4 `nodelta` attention：省掉 Q block mean 的补偿项，完整算子相对同库 NVFP4约1.02–1.10×；Gaussian oracle relative-L2约0.191→0.210，短块可更差。它主动放弃一项计算，不符合当前“INT8优先、效果不下降”，本轮只记录，不建议部署。

近期大量 SM120 PR 是 DeepSeek sparse MLA、MoE、NVFP4，不是 H3 D128 dense。架构关键词相同不能证明可直接迁移。

LightX2V [#1570](https://github.com/ModelTC/LightX2V/pull/1570) 仍开放，最新更新主要是 offload launcher/目录整理，说明明确未重跑整模型；不能当作新的吞吐结果。[VDN #1528](https://github.com/ModelTC/LightX2V/pull/1528) 也仍开放，最近更新9/26；未找到足以证明新Ref2VA质量胜过现有基线的发布。INT8 linear-branch 转换主要改变内存/量化，不是新的VDN训练。

## 8. 后续实验与投稿顺序

### 无损算子路线

1. 冻结最新 main `12389a3` 与同工具链 baseline，独立比较 #215，不混入低比特新权重或新 LoRA。
2. 对 indexed gate 的分带候选做 microbenchmark。真实 M覆盖14,850/32,700/87,142/90,461，outproj/FC2分别测，检查 BF16输出和scale的位模式。
3. 只保留重复AB/BA有稳定优势的配置，再跑完整 Larry8、相同condition/seed、视频和音频latent哈希；最终RGB/PCM相同。
4. 迁移D64 VAE候选，先跑全部15份相同latent；至少保留冷/热分离及多次交错测量。
5. 检查持续请求完成间隔、p50/p95、显存峰值、CPU编码队列背压和取消/失败；单kernel快不等于服务吞吐快。

### 新模型路线

建议先做便宜的单case正确加载与短片排错，再扩大到五个约30秒业务case；每个候选先固定作者配方，后做可解释的消融：

| 候选 | 第一次试验 | 第二次试验 | 主要质量风险 |
|---|---|---|---|
| PDMD4 | 完整rank、4NFE、12/3、strength1，确认全部target生效 | Ref2VA五case；通过后测2NFE或压缩rank | 参考一致性、运动细节、音色/高频/口型 |
| LongLive-Plug | 官方T2VA四forward＋fresh-noise阳性对照 | 正确Ref2VA条件与sampler移植 | 普通Euler误用、两LoRA误叠、RNG/音频调度 |
| LightVAE | 同latent26层FP16对36层FP16 | 26层INT8与当前INT8，接相同历史算子 | 细纹、脸/文字、闪烁、tile接缝 |
| Veda | 锁定FA4补丁，实际SM120 sparse dispatch＋dense对照 | Ref2VA＋不同keep/early-dense；固定同一LoRA | 音频/参考token的跨模态attention损失、offload成本 |

报告同时给出完整请求耗时、DiT/VAE/export分项、同case相对Larry变化。视频至少SSIM/LPIPS及关键帧/时间一致性；音频沿用历史同一采样率/对齐/频谱与embedding协议，不能更换指标后跨表排名。五case逐项和最差case都展示，不能只看均值。

BF16 50-step 视频是参考输出，不是唯一真值；SSIM变高只能说明更相似。若声称“生成质量提高”，还应有人物/参考遵循、文本指令、口型与听感审阅。对算子无损路线，继续使用latent/RGB/PCM exact标准。

## 9. 哪些应归功于上游，哪些属于我们的贡献

- Larry蒸馏权重、INT8 VAE、Kitchen已有input_act/residual/RMS-RoPE、kijai分带调度属于各自上游。
- 我们的贡献是限定SM120的适配与融合、indexed公共API、数值边界保持、消费者集成、工程重叠和真实验收。
- 新PR应有独立问题、最小补丁、适用范围、回退与实测，不把整个发布插件放入Kitchen。
- MP4队列、模型最后层非目标query裁剪、ComfyUI节点属于模型/应用层；不能为凑Kitchen PR数量塞进通用算子库。
- 后续优先推进已有11个PR的反馈，新的两个最清晰算子候选是**gate分带增量**与**VAE D64小tile**；拿到当前主线的实测后再提交，不用旧分母冒充新证据。

本次没有更改线上服务、替换权重或发布新性能版本。PR转正式及研究文档不改变已发布插件的运行行为。

## 10. 复用资料

- [30项优化目录](R85-OPTIMIZATION-CATALOG.md)
- [历史成功/失败实验](R85-EXPERIMENTS.md)
- [音视频质量比较](R85-QUALITY.md)
- [上游批次与单项证据](UPSTREAM-BATCH2.zh-CN.md)
- [10/1组合与RGB/PCM验收](UPSTREAM-RECOVERY-TESTS-20261001.zh-CN.md)
- [本轮公共来源与PR状态摘要](../evidence/research-1001/manifest.json)

原始公开API、模型卡、PR正文与源码快照已保留在研究目录；公开摘要仅包含来源、revision、摘要和哈希，不含浏览器页面快照、私有业务媒体或凭据。
