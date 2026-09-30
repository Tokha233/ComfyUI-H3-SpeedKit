# Kitchen 上游贡献机会：2026-09-30

> 本文保留首批 PR 提交前的审计。三个 PR 后续已提交，见
> [提交与实测状态](KITCHEN-UPSTREAM-PRS-20260930.zh-CN.md)。整套 DiT 的
> 6.49% 口径、未提交的融合和当前 ComfyUI 差分见 [完整上游计划](FULL-UPSTREAM-PLAN.zh-CN.md)。

本次核对 GitHub 主仓库源码、全部 44 个开放 PR、最近更新的 30 个关闭 PR，以及候选 PR 的 diff。结论：**有适合贡献的内容。建议第一个 PR 做 CUDA 寄存器 ConvRot256；SM120 dense attention 做第二个独立 PR。先对比 #215，再决定 GEMM 调度是否还有增量。** 本文是代码审查和实施建议，没有新增 GPU 性能测量，也没有向 Kitchen 提交 PR。

## 当前上游状态

- `pyproject.toml` 仍为 **0.2.36**。main 为 [`19ea55b`](https://github.com/Comfy-Org/comfy-kitchen/commit/19ea55b9ebdaf77942dab36223e1222009d3ce11)，北京时间 9/30 03:59:57；仅修改 HIP attention 的 `-inf` mask NaN 和相关测试，不是 NVIDIA H3 新加速。
- 我们已实测的版本提交是 `888b13e`，与当前 main 的 CUDA 文件没有这个提交带来的差异。
- [#215](https://github.com/Comfy-Org/comfy-kitchen/pull/215)，Kijai，**开放 Draft**，本次 head `92e9aa6`。N 方向分带的 Stream-K 调度，按 L2 容量选择 band；作者报告 5090 的 H3 FC1 权重读取从约 26 GB 降到 0.9 GB、完整工作流约快 2%。这些是作者报告，尚未在我们的 5090 D v2 / R85 环境复测。
- [#208](https://github.com/Comfy-Org/comfy-kitchen/pull/208) 已于 9/27 合入，含 Q 缓存、PV fragment 流水等。当前 CUDA 源码的 `cache_query` 仅在 `__CUDA_ARCH__ == 890`、Q128/D128/K128/无 mask 时启用；SM120 明确走 false。
- [#183](https://github.com/Comfy-Org/comfy-kitchen/pull/183) 是 **HIP** 寄存器 ConvRot，尚未合入。作者报告 gfx1200/1201 算子级提升约 17.9%/19.4%；不是 NVIDIA/SM120 数据。
- [#80](https://github.com/Comfy-Org/comfy-kitchen/pull/80) CUDA ConvRot 小 K launch/chunk 改进已关闭、未合并。当前主线已有 K 分档线程配置；不能再把“去掉固定 1024 线程”作为我们的新贡献。
- [#124](https://github.com/Comfy-Org/comfy-kitchen/pull/124) 涉及 fused SwiGLU 的存储类型舍入，仍开放。它提示融合数值语义必须逐阶段核对；本次不改变 SwiGLU，避免与其重叠。

## 建议拆分

| 顺序 | 候选 | 与上游的区别 | 审阅/维护成本 | 当前证据与缺口 |
|---|---|---|---|---|
| 1 | CUDA BF16 ConvRot256 寄存器路径 | 8-lane/256-value layout，以寄存器和 shuffle 取代逐阶段共享内存转置 | 较低；不改 Python API，只增加窄分派和回退 | 已在发布插件中运行；缺当前 Kitchen 独立单算子完整 sweep、非有限值契约测试 |
| 2 | SM120 D128 dense attention TMA pipeline | 当前 cached-Q 仅 SM89；我们的三槽 TMA、Q 常驻、BSHD 输出是另一实现组合 | 中高；与 #208、mask/大索引支持共享核心代码 | 整链/业务输出验证已有；缺移植到当前公共 attention API 后的单算子和跨 shape 对照 |
| 3 | INT8 GEMM indexed gate/residual epilogue | 上游已有列广播 `rscale` 的 residual epilogue；我们增加按 token/segment 选择 gate 及原 BF16/FMA 边界 | 中；需设计通用 API 和 reference fallback | 发布链 outproj/FC2 已使用；缺通用 bias、dtype、索引异常与跨模型测试 |
| 先测再定 | #215 的 N-banded Stream-K | 与我们的 FC1 raster8 方向重叠，但算法不相同；可能适用于目前 config13 的 gate GEMM | 较低到中；先用原 PR 的方法做差分验证 | 尚无 #215 与 R85 的 A/B，不能承诺还有 2% |

“较低”是基于改动范围的工程判断，不是维护者认可或合并概率保证。

## 第一个 PR 的具体边界

建议标题：`[CUDA] Add a register-based BF16 ConvRot256 path for SM120`。

来源：[`kernels/convrot/convrot_register_v2.cu`](../kernels/convrot/convrot_register_v2.cu)。它基于 SGLang PR #38040 的寄存器 layout，但保留 Kitchen 的四级运算关联、逐级 0.5、BF16 除法/量化屏障、absmax 处理和整数舍入；向量化 16-byte 输入和 8-byte INT8 输出。来源及 Apache 声明应完整保留。

当前候选是 **BF16、连续、group_size=256、ACT=none、确定性量化、K=5376/7168、SM120**；它不是任意 K/dtype/activation 通用实现。5376 使用 192 线程，7168 使用 224 线程。不要与历史 `convrot_schedule035` 的 768/640 线程实验混淆。

当前发布链实际在 attention outproj 的 **K=7168** 使用独立 register ConvRot；norm→modulation 路径使用自己的融合核，FC2 继续 Kitchen `input_act=swiglu`。不能按两个 K 都在所有 projection 调用来计算整模收益。

计划修改上游 `comfy_kitchen/backends/cuda/ops/int8_linear.cu` 及一个小型私有 header，接入现有 launcher；不把本项目 ctypes ABI、模型 SHA allowlist、ComfyUI patch 和九个动态库整体塞进 Kitchen。所有现有 dtype、activation、stochastic、非目标设备/shape 分派保持原入口。设备查询要缓存，避免每次 launch 增加主机开销；同时保留 stream 和 graph-capture 契约。

重要缺口：现有插件 wrapper 的显式有限值检查说明它的公开实验契约较窄。上游不能直接依赖每次 `isfinite().item()` 扫描，也不能声称该 wrapper 已覆盖 NaN/Inf。提交前需验证 signed zero、非有限值、dtype 极值造成的旋转溢出、scale floor；若分歧，要在 kernel 内保留主线语义或继续调整，不能用隐藏 CPU 同步换正确性。

历史 Kitchen .35 累计消融里，加寄存器 ConvRot 前后是 **29.404285→29.386356 s**，约 **0.061%**，仅代表那组 forward 的相邻差。累计 2.885% 不是这个单核的收益；本项目发布的 11.65%/13.19% 也不能作为此 PR 的单项收益。

### 提交前的可复现验证

1. 固定 main SHA、编译器、GPU、功耗/时钟、Torch/CUDA；baseline 与候选同工具链。
2. 公共随机输入：M 覆盖 1/16/128/1024/4096/32700/65536/约 90k，K=5376/7168；再覆盖非目标 K 和 dtype 的回退。大 M 会占内存，按实际预算逐 shape 分配。
3. exact 比较 **INT8 codes + FP32 scale 的位模式**；另比较诊断 rotation。不能仅用 dequant 后 allclose 掩盖量化尺度差异。
4. 测零、负零、tiny、量化 half-way 边界、极值、NaN/Inf；检查非默认 stream、当前设备、重复调用、empty tensor、合法/非法对齐、stochastic/activation 回退。
5. 交替 A/B，充分预热，多组 CUDA event 中位数/p95；Nsight 辅助记录 DRAM 字节、shared-memory traffic、register/spill 和 occupancy。随后跑公开 H3 输入、同 seed 八步，对四项 SHA 和 local MP4 周期。
6. 新 PR 只报此 patch 的测量结果；我们历史整链数据作为外部动机和链接。

## SM120 attention 的第二个 PR

我们已有 [`kernels/dense/`](../kernels/dense/) 源码；潜在新贡献是三槽 TMA 和 SM120 的资源布局，**不是简单把上游 `890` 改成 `1200`**。主线注释明确说 cached-Q 只在 Ada 测得收益，说明直接开启未必更快。

先迁移无 mask、D128 的限定 fast path，覆盖 Q/K 边界、batch/head stride、非整 tile、长索引和不同 Q/K 长度；保留全部 K/V 和 online softmax/PV 顺序。对缺条件情况回退主线，不能损坏 #207 mask 或 b54f851 大 offset 支持。BSHD 输出如需 API 扩展应单独说明 strides；最后一层 non-target Q 裁剪依赖模型语义，留在 ComfyUI/SpeedKit，不作为通用 attention 默认规则。

独立比较：主线、简单 cached-Q 分支、完整 TMA 候选，报告 kernel-only 与实际 call（包含 prequantization/output layout）两种口径，再报告 DiT 周期。只看 kernel 峰值无法证明工作流收益。

## #215 如何进入我们下一轮实验

当前 base `888b13e`，head `92e9aa6`，共修改 6 个文件。关键是 `cutlass_gemm_common.cuh` 的 `ThreadblockSwizzleLeanStreamKT<32>`，large-L2 默认 32 个 N tile，small-L2 使用 SM 数和 tile 几何估算带宽最优 band；主线 selector 也按 activation footprint 转 config13。

我们的 FC1 是 config3/raster8，gate 是 config13，但私有实例化还引用 `ThreadblockSwizzleLeanStreamK`。**只升级 Kitchen wheel 不会自动重编/改写 SpeedKit 私有 CUDA 库**。要分别测试：原版 .36 → 原 PR #215；R85 FC1 → banded 变体；R85 gate13 → banded 变体；组合后整链。每组保留同一 epilogue 和四 SHA 检查。

不要让 #215 的 W4A8/W6A8 解包改动混入纯 INT8 结果，也不要把 5090 与 5090 D v2 的作者数据视为同机 A/B。若收益可复现，优先向现有 PR 提供测试证据/小补丁，而不是创建重复的调度 PR。

## 哪些内容不适合投给 Kitchen

- INT8 VAE decoder、基础 fused input activation、基础 residual epilogue 已来自上游。应署名引用，不重新作为原创提交。
- MP4 export queue、双缓冲 D2H、RGB8 产品流、ComfyUI node 和模型最后层裁剪，主要属于 ComfyUI/插件集成。
- 复制的 ComfyUI GPL forward/vae 不能直接作为 Kitchen Apache 文件；候选只抽取已有独立 Apache/BSD 来源的 kernel 部分。PyTorch 原生 reduction 融合的版本依赖也使它不适合作为第一个 PR。

## 贡献要求

[`CONTRIBUTING.md`](https://github.com/Comfy-Org/comfy-kitchen/blob/main/CONTRIBUTING.md) 要求 Apache-2.0 与 DCO sign-off。当前 [`cla.yml`](https://github.com/Comfy-Org/comfy-kitchen/blob/main/.github/workflows/cla.yml) 还会检查 PR 作者的独立 [CLA](https://github.com/Comfy-Org/comfy-cla/blob/main/comfyui_icla.md)。二者都应核对；DCO 不能替代 CLA。本文没有代签、发评论或提交 PR。

## 已发布宣传页

[StellarVoyager/MiniMax-H3-SpeedKit-RTX5090Dv2](https://huggingface.co/StellarVoyager/MiniMax-H3-SpeedKit-RTX5090Dv2)，沿用此前 DeepSeek 部署说明页的方式。中英文、实测表格、安装/源码链接，明确不提供模型权重。完整测试和代码仍以 GitHub 为主。
