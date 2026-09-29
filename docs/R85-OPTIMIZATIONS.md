> 历史 R85 证据说明，来源于 2026-09-30 交接资料；不代表本仓库已交付 GPU 后端。旧资料中的内部路径索引、runner 和黄金素材不在本仓库。当前开源方案见 [开源策略](OPEN-SOURCE-STRATEGY.zh-CN.md)。

# 算子融合、内存与执行流水线

本文解释固定 R85 的实际实现及其来源。模块状态分为：**固定基线**、**服务工程已验证**、**解码候选**、**未采用**。固定基线不是本次核实过的所有线上实例状态。

来源 ID 可在 [source-identities.json](../evidence/source-identities.json) 查到原文件 SHA；内部交接另有对应源码绝对路径。

## 1. 从原始 Kitchen 到最终方案的层次

原 Kitchen 已提供 INT8 量化 GEMM、ConvRot、attention 及部分输入激活融合。原部署还已包含 DynamicVRAM、异步导出等功能。因此本项目贡献应写成“在上游之上适配、组合和验证”，逐项区分引入和新增优化。

| 模块 | 优化 | 当前状态 | 关键验收依据 |
|---|---|---|---|
| 权重 | 捕获运行期有效权重，预合并固定 Larry LoRA | 固定基线 | 权重 SHA；不重复挂载 LoRA |
| QKV 交接 | container 载体、提前释放所有权、去除冗余 V clone | 固定基线 | 真实 forward 回放与最终输出 |
| Norm | RMSNorm＋分段 scale/shift | 固定基线 | 保留 BF16 落点、stride 资格 |
| Residual | gate/residual＋norm2＋调制 | 固定基线 | 浮点操作顺序和历史四 SHA |
| Quant | Norm/调制→ConvRot INT8；Residual/Norm/调制→INT8 | 固定基线，限定布局 | 实際调用计数；量化 code、scale 一致 |
| ConvRot | K5376/K7168 的线程、寄存器与 shuffle 调度 | 固定基线 | 分宽度测试；不覆盖新版未知 act 语义 |
| Dense | 直接 BSHD 输出、精确 V 量化、3 槽 TMA、Q 寄存器缓存、早释放 K | 固定基线 | 真实长 QKV、边界、stream、完整生成 |
| Dense | o_scale==1 时消去恒等乘法 | 固定基线 | 原 KV 顺序、量化和累加保留 |
| Embedding | packed copy 后释放不再消费的原引用 | 固定基线 | 内存下降；单独速度收益不稳定 |
| 跨 block | FC2 residual→下一 block norm1/量化 | 固定基线，限定业务 | 392 次跨层＋8 次末层；所有权和代际检查 |
| GEMM | out-proj/FC2 的 gate/residual epilogue | 固定基线 | 15 片段、70 完整请求，历史四 SHA |
| GEMM 调度 | FC1 config3/swizzle8；QKV config5 | 固定基线 | 短/长 M 分离；正式完整周期复测 |
| QKV 后处理 | RMS/RoPE/Q quant/V partial-max 融合 | 固定基线 | 原中间舍入与 anchor 逻辑 |
| 有效工作量 | 当前输入 9 行 K-only anchor；末层不消费的 Q/GEMM 行裁剪 | 固定基线 R85 | 不是跨 step cache；全 15 片段四 SHA |
| 视频 VAE | .35 INT8 ConvRot ViT3D，旧 encoder 分离加载 | 固定基线，INT8 decode 用户确认已部署 | TRT/INT8 同 latent 质量与耗时；像素非完全相同 |
| VAE 输出 | 预分配 canvas、tile 引用释放、原 overlap | 固定基线 | tile 边界和原 RGB |
| 帧交接 | GPU RGB8、双 pinned buffer、异步 D2H | 固定基线/服务工程已验证 | RGB、event/stream、CPU 生命周期 |
| CPU | x264/AAC/MP4 与下一请求 GPU 并行 | 服务工程已验证 | 真实队列完成事件、排空和成功语义 |
| 权重存储 | 本地不可变模型缓存、Qwen 避免反复 mark_cold | 服务工程已验证，需部署层接入 | 读盘字节/次数、主机内存、完整请求 |
| VAE 新候选 | 静态加载＋R234v2 RoPE/Q＋R244 cached-Q | 解码候选 | 60 正式 decode；尚无整套累计服务计时 |
| CPU audio / NVFP4 / graph / 更大 tile | 详见负结果 | 未采用 | 不进入默认配置 |

## 2. LoRA 预合并：省去每次加载时的重复工作

进入本轮算子研发之前，部署已经使用 INT8 ConvRot DiT/Qwen、AdaLN 压缩底模、TRT FP16视频解码、原生 FP16参考编码和FP32音频解码，以及 native allocator（`--disable-cuda-malloc`）。这些是起点的一部分，不能重复作为R85新增贡献。所谓pruned主要涉及AdaLN时间嵌入/基底压缩，仍保留50个transformer blocks。

历史另测过 `max_split_size_mb:512`、显存余量、MLP分块和极端多图/参考视频容量。512是进程级分配器参数，不是图像尺寸，也不是DynamicVRAM开关；它未在所有历史版本默认启用。参考视频/极端长宽比的容量路由和分块可能改变输出，不纳入本15片段R85的无损结论。公开基准须写清allocator及内存余量，不能混用历史压力测试与正常业务的耗时。

原运行期 LoRA patch 会在基础权重上执行反量化、低秩增量叠加及重新量化。固定业务使用固定 LoRA 时，将**实际 post-cast 生效的权重**封存为 baked checkpoint，避免反复执行同一操作。

不能简单把低秩矩阵直接加到旋转后的 INT8 字节。转换流程须与原有效权重路径一致：恢复对应浮点/旋转空间 → 加正确缩放的 LoRA 更新 → 按原规则再旋转/量化；AdaLN 等浮点部分按其真实语义保留。原项目旧 Turbo 的特殊 alpha/16 规则不能套给 Larry。

固定 Larry 权重 SHA 为 `9eccf52e4fe6e764f4aac8cdb6045a58bc86d8458c783c35b7035edca71fec12`。当前入口已经 baked，不能再挂一次 Larry。这里保留的是选定方案的运行语义，不构成对 BF16 底模的无损量化证明。

## 3. Norm、调制、残差和 ConvRot 融合

典型 block 的 unfused 数据流是：

```text
x → RMSNorm → BF16 中间结果
  → 按 video/audio/text 区段 scale/shift → BF16
  → ConvRot → 行/组量化 code 与 scale → INT8 GEMM

attention/MLP branch → BF16 gate → residual 加法
  → 下一次 Norm/调制/量化
```

融合减少大尺寸 BF16 张量的写回和重读、kernel launch 以及 Python 载体处理，但必须保留旧路径的数值边界：

- RMS reduction 归约树、epsilon、scale/shift 广播和 token 分段不能改变。
- 如果旧路径先舍入到 BF16 再参与下游，融合中也需要同样落点，不能以“精度更高”为由取消。
- ConvRot 的分组、旋转顺序、row scale、padding、随机量化开关必须原样保留。
- strided 输入需要单独资格，不把非连续 tensor 当连续内存读写。

早期 `norm_quant/residual_quant` 只验收 packed rows 42520/54803/78196/113872；9/19 Case1 回放没有命中它们，不能把开关存在当成获得其收益。后续逐步扩展实际业务覆盖，R85 最终限定 15 个已验收布局。开源版应按真实布局/stride/语义分派，不能只有一个 token 数白名单。

K5376/K7168 的 register ConvRot 是另一路精确优化。9/19 迁移的线程配置分别为 768/640；收益是否成立须以该版本实际调用为准。K14336 SwiGLU/ConvRot 原本已有输入激活融合，后续多次尝试更大融合未形成稳定完整收益。

## 4. Kitchen .33→.35 的接口迁移

低层 `quantize_int8_rowwise_convrot64` 从 8 参数变为 10 参数：

```text
旧：x, q, scale, group, stochastic, act, seed, stream
新：x, q, scale, group, stochastic, act, seed, act_weight, act_eps, stream
```

新增参数承载 RMSNorm 等输入激活语义，不能直接用旧 wrapper 忽略它们。本项目迁移方法：

1. 固定新版 Python 源码、实际 code object 与 CUDA 二进制 SHA。
2. AST 核对唯一目标调用的参数顺序和 tensor owner。
3. 仅在旧资格满足时进入自定义核，例如特定 BF16 宽度、非随机量化、合法 act/group 等；其余把完整新参数回交 Kitchen。
4. 保留新版 postquant tail 的 residual/residual_scale、小 M 选择和反量化语义。
5. 作用域退出在 finally 中恢复入口；入队后异常不能重复执行同一个写操作。

真实第一步回放 9 组共 36 次含预热输出一致；stock 30.259366 s，迁入原优化 29.392929 s。它证明原融合可迁移，并非证明“升级 .35 本身带来 2.863% 新收益”。来源 S02、S31。

## 5. SM120 INT8 dense 主循环

### 数据流

不显式生成整个 N×N attention 矩阵。Q/K 量化后用整数 MMA 计算分块 QK，执行原在线 softmax，概率量化为 U8，与 S8 V 进行 PV，再按原 scale 累积。量化规则和 online softmax 次序共同决定输出。

当前 DiT 主形状 D128，Q128/K128 分块，实际示例 `[1,56,90461,128]`。其主要特征为：

- 3 槽 TMA ring 隐藏 K/V 搬运；完成消费后提前释放可复用 K 槽。
- Q 保留寄存器，降低反复 shared-memory 读取。
- 8 个 compute warp，另一个 4-warp group 搬运并让渡寄存器；动态额度 240/24。
- 动态 shared memory 66,560 B，按设备实际资源检验。
- 概率量化和打包留在寄存器；`o_scale==1` 时去掉对分母和输出的乘 1，其他分支保留。
- 直接生成 BSHD 布局，减少下游 transpose/copy；精确 V 量化保留原 scale 和 padding。

这里的 dense 表示不额外删掉 attention block。INT8 attention 相比高精度 attention 已有量化误差；后续主循环优化的“精确”只相对于固定 INT8 attention。

### 硬件定向而非通用 Blackwell 参数

本机为 SM120 / GB202 系列，不是数据中心 B200 / SM100。已查询到 170 SM、48 warp/SM、65,536 个 32-bit register/SM、CTA opt-in shared 101,376 B、L2 96 MiB；运行时报告的每 SM shared 为 102,400 B。资源准入使用设备查询结果。

使用 TMA、`mma.sync` 和符合约束的 `setmaxnreg`；不能原样移植依赖 `wgmma` 或 `tcgen05`/TMEM 的实现。`setmaxnreg` 的 warp-group 约束也是不能随意删掉三个 producer warp 的原因。

“低 occupancy”不是空闲算力的直接比例。真实长序列已有数万个 CTA 工作项，增加请求并发、拆 K 再合并或增加流水深度不一定更快，还可能改动浮点归约。后续尝试取消 producer group、Q64 多 CTA、概率查表、max3、双行流水和更多缓存提示，多数更慢，详见负结果。

来源 S15、S17、S27。SageAttention/Kitchen 为基础实现与思想来源；不能将这些全部命名为本项目原创 attention 算法。

## 6. 跨 block 残差消费与 GEMM epilogue

### 跨 block

MLP 的 residual 结果通常紧接下一 block 的 norm1/量化。对已验收的串行上下文，可以把消费链连接起来，省去某些中间写读。8 steps×49 个跨 block 连接形成 **392 次跨层消费**；8 次最后一层保留专门收尾。

状态必须绑定请求、step、block、generation、owner 与 pointer；禁止使用上一请求或上一 step 的 residual。候选载体是单消费者协议，不可重复消费。未知布局、同实例并发或所有权冲突应拒绝该快路径并安全回退。当前不是一般化的并发服务适配器。

### Out-proj / FC2 gate-residual

将 GEMM 的反量化输出处理与 gate/residual 合并，但保留以下真实顺序：

```text
INT32 accumulator
  → FP32 × row scale × column scale
  → 原 bias/加零语义
  → BF16 branch 舍入
  → BF16 gate
  → FP32 FMA 加 residual
  → BF16 输出
```

若省去中间 BF16 舍入、改变 gate 精度或 FMA 边界，结果即可能不同。无需再读取的 branch buffer 仅作为合法所有权元数据保留，不能读取未初始化数据。

9/26 FC2 实测选择 stage1/config13；真实长 capture 算子链 27.393008→26.082304 ms，减少 4.783%。但完整 out-proj＋FC2 组合固定工作量容量仅提高约 0.468%，不能宣传整机快 4.8%。全 15 片段、70 次完整请求（含预热）通过历史四 SHA。来源 S16。

## 7. R85：QKV、当前 anchor 与末层有效行

R85 吸收此前 FC1 raster config3/swizzle8、QKV config5 及 QKV 后处理融合；其总收益对照已经包含 gate epilogue 和其他历史优化。

QKV 融合连接原 INT8 projection、RMS/RoPE、Q 量化与 V partial-max，减少中间量化准备。K 的中心化/anchor 依赖必须保留，不能为少一次读写改变定义。

R85 的当前输入 **9 行 K-only anchor projection** 只计算 anchor 所需 K，不为这些行生成不会消费的 Q/V；配合 4-warp K fusion。它每次读取当前输入，**不是用前一步 K 替代当前 K**。

最后一个 transformer block 的部分 reference prefix Q、out-proj、FC1/FC2 行不会影响最终需要导出的输出，因此按已验证消费者关系裁剪。其他 query 所需 K/V 仍保留；原实现需要的 prefix residual/填充和对齐保持。它是死行消除，不是降低分辨率、少算一层、早停或近似稀疏注意力。

完整 15 片段配对组均值之和 3295.244819→3232.463433 s；四 SHA 一致。R126 另以 36 次正式请求复现约 1.897% 降时。此两项是同一收益的验证，不重复累计。来源 S04、S05、S15、S27。

## 8. INT8 视频 VAE

### Encoder 与 decoder 必须拆开理解

旧参考 encoder 为 FP16 CNN 路径；生成视频 decoder 为 36 层 ViT3D。新版 INT8 权重中 144 个量化 Linear 对应每层 QKV、attention output、FFN w1/w2；输入输出层、归一化等仍存在浮点运算。

仅更换 Kitchen wheel 不会自动启用新 decoder；需要配套模型类、权重和 helper。项目以局部 namespace 引入新版 VAE/helper，避免全局替换使 Qwen/DiT 的其他 shape 误进新路径；参考 encoder 继续原类和原 FP16 权重。

### 来自上游的核心能力

- RMSNorm 或 SwiGLU 通过 `linear_input_act` 并入 ConvRot/激活量化。
- 适用 GEMM 的 residual/residual_scale epilogue。
- Q/K norm 与 RoPE 融合，decoder INT8 完整 attention。
- tile 最大批量 4，显存不足时按规则缩小；原空间和时间 overlap 保留。
- 及时释放旧 tile 引用，直接写预分配 canvas，避免重复拼接输出。

上游来源：Kitchen #167、ComfyUI #16187、#16332；版本与权重记录见 [SOURCES.md](R85-SOURCES.md)。量化权重不是本项目重新训练的 decoder。

### 精确输出工程

不能对尚未完成 blend 的 tile 先转 uint8，否则叠加顺序与精度改变。项目保留原 tile 权重与 blend，完成可提交区间后再做 terminal RGB8，并维持原 canvas overlap 生命周期。

INT8 相比 TRT 的像素差异单独验收；在同一 INT8 decoder 上新增融合则要求历史 RGB8/PCM 相同。来源 S03、S06、S18、S31。

### 最新解码候选

**静态加载**：在 VAE 阶段将需要的 decoder 权重按原 dtype/scale 预先 cast/load，避免逐层 dynamic wrapper 的重复处理；仍需在 DiT 与 VAE 之间控制驻留和释放。它不意味着把完整 DiT＋VAE 永久放入 24 GB，也不能用静态加载的短脚本证明跨请求显存安全。

**R234v2**：CTA 内计算 32 token 的原 RMS/RoPE，Q 写 shared 后按原 32-row/8 组 scale 立即旋转量化；K 保持原 FP16 写回与 anchor 检测。首版因 FMA/倒数边界不一致被拒绝，v2 保留数值语义后通过。

**R244**：VAE D64、真实 shape `[4,32,1797,64]` 下，Q 放寄存器复用，CTA_Q 从 128 降为 64、WARP_Q 从 32 降为 16，K tile 仍 64；原 128-row 量化 scale 索引、softmax/PV 次序保留。原核 196 register、2 CTA/SM；候选 128 register、4 CTA/SM，Tensor active 约 53.42%→63.28%。局部 attention 快约 11%，全视频 decode 只快约 0.632%。

这些变化只在 VAE 作用域启用，退出后恢复，不能让 D64 VAE hook 覆盖 DiT D128。来源 S07、S08、S20。

## 9. RGB8、双锁页缓冲与异步 D2H

原大输出以 FP32 送 CPU，占 4 byte/通道。GPU 完成原 terminal 转换后以 uint8 传输，每帧数据变为四分之一。例如 case1 的 3,009,871,872 byte 降至 752,467,968 byte。

转换保留 `FP32 × std + mean → clamp → ×255 → truncate` 的实际顺序，禁用会改变边界结果的 FMA contraction。GPU/CPU 的 round、truncate 不能混换。

双 pinned slot 的所有权协议：

```text
compute stream 完成 chunk 的 blend / RGB8
    → record ready event
copy stream wait ready → async D2H → record copied event
CPU 等 copied 后消费 pinned buffer
    → CPU 消费完成才把 slot 交还下一 chunk
```

通过 `record_stream`、event 和强引用确保 GPU 源 tensor、锁页目标和 CPU consumer 生命周期正确。仅使用 `non_blocking=True` 不足以保证重叠或避免提前复用。buffer 数和总字节有上限，满载时背压，不无限缓存整段 FP32 视频。

9/13 原部署累计测试中，10 秒帧交接从 1.757→0.303 s，15 秒从 2.617→0.465 s；包含其他累计优化，不能全部算作 D2H 独立收益。28 个含预热成片 MP4 对黄金字节一致。来源 S01、S34。

## 10. CPU 成片与下一请求 GPU 重叠

一张卡串行执行生成；请求 N 的 RGB/PCM 交给 CPU 后，CPU 执行 x264/AAC/MP4，而 GPU 可进入请求 N+1。服务端原来已有异步 export，后续优化保留该能力；不能每轮再重复宣传“首次实现异步”。

要点包括任务独占的 mmap/缓冲、限量 CPU 编码队列、按总字节背压、失败重试和排空。GPU 完成不能立即把业务任务标为成功：MP4、上传（如启用）与结果提交完成后，finalizer 才改变任务状态。取消或异常须释放当前任务资源，不能释放 CPU 仍在读的帧。

吞吐要测实际成片完成事件。若以 `3600/单请求延迟` 表示串行容量，应与队列完成间隔分别命名。9/13 的队列实测已经包含 CPU/GPU 重叠，不能把 CPU 编码秒数再减一次。来源 S01、S19、S34。

## 11. 权重预取、CPU 缓存和 condition

DynamicVRAM 权重 H2D 预取原来就存在。一次历史 DiT profile 中 H2D 1.593947 s，其中 1.546261 s 与计算重叠（约 97.01%）；R85 另一场 profile 中未被 kernel 覆盖的 copy 仅 0.045101 s。两次采样不同，不能合并成同一测量。

服务层还可将下一请求的素材预取与当前GPU工作重叠，当前CPU编码/上传与下一请求生成重叠。预取仍要核对素材版本、任务所有权、失败和取消，不把不同业务请求的最终condition按相似prompt直接缓存复用。R85 benchmark冻结condition是用于控制变量，不是线上默认跨请求复用condition。

模型本地不可变缓存减少远端共享文件系统反复读；Qwen 的 `mark_cold=False` 防止刚用完的活跃页被驱逐。实测一次 591 次 / 27.12 GB 的权重读取后，连续 5 请求没有新增读取；代价是主机内存从约 56.49→78.58 GiB。它是 GPU 显存外的资源交换，部署需预留主机内存，不能给每个副本无限复制模型。

Qwen 编码计划拆成独立服务。此路线可以按条件长度批处理和复用模型驻留，但需要计入编码器占用的 CPU/GPU、排队、传输和失败处理；不属于 R85 固定 condition 计时中的已取得收益。

音频搬 CPU 则并不适合当前瓶颈：4 线程 CPU 4.49–9.44 s 对 GPU 0.11–0.14 s，PCM 也不同。即使完全隐藏 GPU audio，理论上只节省当前完整周期约 0.05%；实际并行收益尚未测试。来源 S14、S15、S19。

## 12. 正确性和移植合同

公开代码应保留以下必要契约，而非只交付一组更快的 `.so`：

1. 固定源代码、编译器、编译参数、架构、二进制与模型身份；实际导入路径写入运行结果。
2. 按 dtype、stride、shape、layout、quant group、mask、act、bias 和 residual 语义检查分派。
3. 对失败/未知 shape 使用原实现；不能吞掉异常继续使用半初始化输出。
4. 输入不可变校验、poison 输出覆盖、非默认 stream 后继消费、最后一次计时的真实输出检查。
5. 局部边界与 sanitizer、真实 step/layer capture、全部业务完整输出逐级验证；不能把短输入 sanitizer 当全模型证明。
6. 四 SHA 比较固定历史黄金，不只让本轮 A/B 彼此相同，防止双方一起改了。
7. 一实例的 stateful hook 保持串行；生产并发应通过独立实例/隔离上下文实现。

本项目已经通过的资格具有明确边界；它们不能自动覆盖新分辨率、多参考视频、横屏、不同长度、训练或其他 GPU 架构。
