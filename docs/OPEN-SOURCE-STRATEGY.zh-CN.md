# 开源方案：先交付一个好用的产品，再扩大内核复用

调研时间：2026-09-30，北京时间。本文基于公开仓库、官方 ComfyUI 文档和已有 R85 实验记录；本轮没有新增 GPU benchmark。原始网页 URL、抓取时间和内容 SHA 见 [research-sources.json](../evidence/research-sources.json)。

## 1. 我的判断

建议开源 **一个主仓库，里面同时交付 ComfyUI 节点、独立内核模块、完整评测与部署示例**。主仓库不能只有宣传与论文式说明，再让用户跳到另一个仓库寻找核心实现。用户从搜索到成功生成，应只需要理解一个项目名、一个安装入口和一份兼容表。

本地工作名 `ComfyUI-H3-SpeedKit`，对外显示名 `H3 SpeedKit`。GitHub 名称搜索在本次返回 0 个同名结果，但未核实、占用 PyPI 或 Registry 名称，也不是商标检索。Registry 项目名建议 `h3-speedkit`，正式注册前再确认；官方建议 Registry ID 不含 `ComfyUI`。

当前最准确的定位是：**专为 5090 D v2 优化、保留既有 INT8 计算结果的 H3 推理工具包，带可选近无损视频 VAE 路径。** 不宜宣称通用 GPU 加速器、所有显卡适用，或比全部 H3 框架都快。我们的工程知识可以逐步泛化，但目前通过验收的是特定模型、权重格式、shape、采样和硬件组合。

最值得发布的不是“30 个开关”，而是用户插入一个节点就能获得适用的优化，并明确知道哪些优化真正命中。对外少量节点，对内保留完整算子和调度细节。

## 2. 同类项目给出的实际信号

| 项目 | 本次 star 快照 | 可借鉴的交付方式 | 对我们意味着什么 |
|---|---:|---|---|
| [ComfyUI-nunchaku](https://github.com/nunchux-ai/ComfyUI-nunchaku) | 2,929 | 工作流、安装文档、可选节点、平台/Torch 对应 wheel | 让非 CUDA 用户真正装得上；其内核库独立并有多个模型，拆仓有现实基础 |
| [Nunchaku core](https://github.com/nunchux-ai/nunchaku) | 3,958 | 核心算法、量化模型、二进制分发、研究材料 | 内核和用户产品都可吸引关注，但不能把两个仓库 star 简单相加当用户量 |
| [SageAttention](https://github.com/thu-ml/SageAttention) | 3,953 | 独立调用接口、明确方法、跨模型 benchmark | 通用性来自真实接口和数据，不来自仓库名字写着通用 |
| [Comfy-WaveSpeed](https://github.com/chengzeyi/Comfy-WaveSpeed) | 1,235 | 模型加载后接少数 patch 节点，编译与 cache 分开 | 原工作流改动小、用途清楚，比暴露几十个 kernel 参数更有利采用 |
| [ComfyUI-TeaCache](https://github.com/welltop-cn/ComfyUI-TeaCache) | 1,096 | Manager 安装、按模型给配置、示例图 | 近似计算可能有更大倍率；必须把质量代价与精确算子路线区分 |
| [H3-Optimizations](https://github.com/Zironic/H3-Optimizations) | 121 | 已有 H3 memory、sparse、FastH3 VSA、native backend 与回退 | 我们不是首个 H3 加速节点；需要和它做同口径对比与冲突检测 |
| [Comfy Kitchen](https://github.com/Comfy-Org/comfy-kitchen) | 228 | 多后端算子、pip/wheel、稳定 ABI 思路、能力分派 | 底层影响力不等于 star 数；优先贡献可通用化改动，减少长期 fork 负担 |

这些数字只是 9 月 30 日的观察，没有控制项目年龄、作者影响力、推广投入和模型热度，不能据此预测我们的 star。最可操作的经验是：**现成工作流 + 二进制安装 + 可核实收益 + 持续兼容维护**，四项一起提供。

H3-Optimizations 的 README 明确区分 memory 优化和有质量代价的 sparse；FastH3 节点要求 VSA 训练权重。其部分平台“编译了二进制”仍不等于做过真机测试。我们也应公开“编译支持 / 真机正确性 / 性能验证”三个状态。

## 3. 最新发现会改变发布优先级

Kitchen `main` 已到 **0.2.36**，commit `888b13e2c0e721f6576fe351a2ad79894b1c451f`。近期包括：

- [#208](https://github.com/Comfy-Org/comfy-kitchen/pull/208)，9 月 27 日合入：Q 寄存器缓存、PV 整数点积与转换/FP32 累加流水、部分形状的 Q tile 分派等。
- [0432be9](https://github.com/Comfy-Org/comfy-kitchen/commit/0432be9a3ac8193e4c516ad167560adabab3f146)，9 月 28 日：SM120 上一组 4096–4608 token、30–32 heads 的 tile 选择结合 SM 数判断。
- [b54f851](https://github.com/Comfy-Org/comfy-kitchen/commit/b54f85128ee4246cb42b9888d1e0ae1a0d829049)，9 月 28 日：batch/head offset 扩到 64 位，保留受限的 head 内索引与边界验证，扩大尺寸覆盖。

其中部分思想和我们的历史路径相近。不能因为“我们已做过 Q cache”就认定新版没价值，也不能凭 PR 名称认为新版一定更快。**先对齐 .36，再决定保留哪些补丁**，比立即把所有 .35 实验脚本产品化更划算。

正式发布至少需要四臂：`.35 stock`、`.35 + frozen R85`、`.36 stock`、`.36 + ported SpeedKit`。前两臂用于延续历史口径，后两臂用于回答安装当前产品的实际价值。精确模式与 INT8 VAE 模式分别测，分阶段记录延迟、命中数量、峰值显存和最终结果。

只对两臂都完成的相同任务计算收益。更快的上游已经吸收的部分应从自有增量中删除。不要把 .35 的 6.47% 直接贴在 .36 插件首页。

## 4. 具体开源哪些内容

| 层级 | 应交付内容 | 首发取舍 |
|---|---|---|
| 用户入口 | `MODEL → MODEL` 的 H3 DiT 优化节点；默认 dense，自动判断适用性；日志能解释未命中原因 | 核心产品，优先 |
| 核心算子 | SM120 D128 dense、Norm/mod/residual/ConvRot 融合、GEMM epilogue、QKV/RMS/RoPE/V quant、layout/载体 | 源码、构建脚本、单算子校验、许可证一起交付 |
| 内核运行时 | shape/dtype/stride/quant contract、current stream、所有权和 request state、原实现回退 | 与算子同等必要，不能只发 `.so` |
| 视频输出 | 上游 INT8 VAE 接入；blend 后 RGB8；pinned D2H；可选高效输出节点 | 第二个明确 opt-in 功能，单独说明量化差异 |
| 配方 | Larry v4 8-step、旧 encoder、GPU FP32 audio 的已验证配置；权重来源、revision、SHA | 发布配方与本地转换方法，不把作者 LoRA 说成自训练 |
| 评测 | 原始匿名时间、复算脚本、全链路与算子 benchmark、质量协议、负结果 | 当前已能交付历史证据；公开可重跑输入仍要补 |
| 服务示例 | 有界 CPU export、下一请求素材准备、完成事件/排空、模型缓存 | 作为可选示例，不强制用户换服务系统 |
| 项目维护 | wheel、源码包、版本矩阵、示例工作流、变更日志、报错模板 | 正式发布的一部分，不能当后续可选美化 |

逐项对应 30 个历史优化的边界、优先级和迁移要求见 [发布范围](RELEASE-SCOPE.md)。

**不放入默认产品**：所有历史失败 kernel、NVFP4、CPU audio、未验收跳步/cache/sparse、私有 condition/latent/业务素材、堡垒机及容器操作脚本、生产凭据、只有二进制没有可重建源码的组件。负结果可以公开摘要与合法可复现脚本，不应占据普通节点界面。

## 5. 节点应该是什么样

第一节点建议 `H3 SpeedKit · DiT Optimize`，接在模型加载、所需 LoRA 处理之后，采样器之前。用户只选择启用与否，必要时有“严格要求优化生效”选项。默认 dense，不改 scheduler、step、seed、CFG、condition 或 LoRA。已 baked 的 Larry 权重不能再挂一遍同一 LoRA。

第二节点为独立视频 decode/output 功能。应显示高精度 VAE 与 INT8 VAE 的区别，拒绝对错误 checkpoint 套用 INT8 路径。**标准 ComfyUI `IMAGE` 常需 float tensor；如果把 RGB8 转回 float 并再次搬运，R85 输出收益就可能消失。** 完整吞吐优化应提供明确的帧流/文件输出链，或兼容外部 VideoCombine 的适配，不能伪装成普通 `VAE → VAE` patch 后就宣称拿到原收益。

第三个节点是可选诊断：硬件、运行栈、后端、命中次数、回退原因和结果身份。它帮助社区提交可行动的 issue。诊断不应自动安装编译器、升级 Torch、下载模型或上传日志。本仓库先实现了环境诊断部分。

底层配置可以完整开放给开发者；普通用户路径应保持简短。每个开关只有在真实收益、适用条件、数值变化都有解释时才公开。

## 6. 首页怎样写最有说服力

建议用英文首页配中文入口，首屏依次展示：一句用途、当前状态、支持平台、两三个带精度和基线说明的实测数字、工作流图和安装入口。下方再展开内核机制、质量曲线、失败实验。

可以引用的历史结论是：

> On RTX 5090 D v2, our frozen R85 deployment reduced DiT time by 6.47% against Kitchen 0.2.35. With the upstream INT8 VAE and optimized output transfer, frozen-condition-to-MP4 latency fell by 11.59%. The packaged release is being validated against Kitchen 0.2.36.

不应写“无损 2×”“一节点吞吐 +13%”“BF16 VAE 无差别”“所有 GPU 通用”。当前的 13.11% 是单请求延迟倒数推导，且 11.59% 涉及 VAE 量化；这些混写会让外部复现者很快失去信任。

Larry v4 相对 50-step 满血参考的质量指标用来解释配方选择，不能当成本项目训练了更好的 LoRA。INT8 VAE 的 SSIM 0.98907 属于相同 latent 的 decoder 对比，不能挪去描述 Larry 与 50-step 的整体差异。

例如最初同五故事的 Q1 组：旧 Turbo4 配置 8 步 SSIM 0.694995、LPIPS 0.362692、音频频谱 cosine 0.808250；Larry v4 8 NFE 为 0.736385、0.280848、0.868469，支持当时的选择。但 Larry 的音频 nRMSE 并非全组最低。后续 Q2 中无 LoRA16、DMD12 又能取得更高参考相似度，代价是更多时间。因此公开配方应写“既定速度预算下的均衡选择”，不要写“全部指标全球最优”。各组的原始口径见 [LoRA 质量说明](R85-QUALITY.md)。

## 7. 想获得更多 star，优先做什么

**第一优先级是让别人跑成。** Linux wheel 先通过干净环境验证；随后补 Windows 5090 真机，不能只跨编译就宣布支持。首次完成时间应可预测，工作流不带无法访问的路径；下载只引用原作者的合法模型来源。

**第二优先级是公开可复现的素材与实验。** 选有明确授权的自制图像/音频；包括静态人像、小动作、快运动、细纹理、多个参考、长视频与音频对齐。保留少量边界失败例。单独给“相同 INT8 权重精确模式”和“INT8 VAE 近无损模式”工作流，用户能自己做 A/B。

**第三优先级是上游合作与可解释技术。** 最容易通用化的 dense/layout/epilogue 改动整理为小 PR，附 shape、反例和测试。不要把改过的 Kitchen 整仓藏在 vendor 中。主仓库继续承担产品安装、配置与整链路收益，底层改动被上游吸收后及时切回上游。

**第四优先级是有节奏地发布。** 技术首发包含 30–60 秒公开素材对比、安装录屏、完整 benchmark 和可下载 wheel；再做一篇具体到 TMA、寄存器、舍入语义、跨 block 所有权的文章。README 链接可复核数据，在 ComfyUI 社区、模型社区和 CUDA 社区按各自关注点介绍。这里是发布建议，本次没有替用户发帖或联系他人。

首月跟踪：冷安装成功率、获得首个视频的耗时、活跃复现者数量、支持的真实配置、问题响应时间、外部 PR、Registry 安装与 GitHub 流量（在可取得的范围内）。star 是结果指标，不设“保证达到某个数”的承诺。适度支持 5090/5090 D v2/后续 4090 会扩大受众，但先拿证据再贴标签。

## 8. 许可证会影响产品边界和传播范围

新写的工具包可采用 Apache-2.0；Kitchen 与 SageAttention 也是 Apache-2.0。ComfyUI 为 GPL-3.0，复制或改写其模型实现时需要保留来源并按衍生关系处理，不能简单覆盖为 Apache。CUTLASS 的 C++ 与 CuTe DSL 也不能视为同一许可。

本次读取的 MiniMax H3 Community License 明确列出排除地区与商业条件；Larry 模型卡写 Apache-2.0 并不消除底座模型限制。不要打包权重或把整个模型称为全球无条件可用。独立通用内核的开源可以扩大可复用范围，但其实际来源仍需逐文件核实。具体原文与处置见 [许可证审计](LICENSING.md)。

这不是停止工作的理由：当前仓库已创建，新写工具与匿名证据可以整理；对待提取的第三方衍生代码逐文件保留许可，避免把许可不明的参考实现直接搬进来。

## 9. 一个仓库，何时才拆成两个

初期逻辑分包即可：`h3_speedkit` 放 Python runtime/适配，`kernels` 放 C++/CUDA，用户通过仓库根入口接入 ComfyUI。wheel 与节点可有不同构建产物，不需要不同 Git 仓库。

只有出现以下实际需求再拆内核仓库：已有多个独立下游；多个模型共用稳定算子 API；内核和插件的发布节奏明显不同；维护者能够分别承担 CI/版本兼容。Nunchaku 的核心研究与多模型生态符合这类条件，我们目前尚未达到。

不要先建一个空主仓库再建一个“小插件仓库”来分散 issue、文档和 stars。也不建议先 fork 整个 ComfyUI——升级冲突和许可证范围都会扩大。

## 10. 本次已经做完与正式首发的差距

本次已创建本地 Git 仓库，整理公开来源、30 项优化发布范围、历史 HTML 与数据、复算脚本，写入诊断 CLI/节点、发布规格、issue 模板与 CI。仓库 readiness JSON 明确记录了实际状态。

正式加速首发仍需：完整 R85 源码依赖闭包提取、当前 Kitchen 比较、逐文件许可、构建 wheel、公开工作流、公开样本 A/B、ComfyUI 加载/旁路/冲突测试。详见 [路线图](ROADMAP.md)。这轮请求是开源范围研究与本地建仓；没有把现有 GPU 实验重新运行或声称插件已经达到历史收益。
