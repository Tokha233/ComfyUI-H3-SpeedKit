> 历史 R85 证据说明，来源于 2026-09-30 交接资料；不代表本仓库已交付 GPU 后端。旧资料中的内部路径索引、runner 和黄金素材不在本仓库。当前开源方案见 [开源策略](OPEN-SOURCE-STRATEGY.zh-CN.md)。

# 每项优化的实现与验收

共30项，区分共同基础、固定R85、服务层、后续候选和未采用实验。所有单项数字保留原分母，不与总收益相加。代码名在内部 CODE_INDEX.md 中可定位。

## 01. Larry v4 固定 LoRA 预合并

**阶段：权重与输入｜状态：共同基础｜来源：项目适配与优化**

**原始路径：** 运行期 patch 在权重反量化后叠加低秩增量，再重新旋转和量化；每次加载都可能重复这条路径。

**具体改动：** 捕获实际 post-cast 生效的权重，按原旋转空间、alpha 和 scale 规则固化为 baked INT8 ConvRot checkpoint。两臂都直接加载它。

**为何更快/选择理由：** 固定 adapter 不再反复计算或保存临时更新张量；模型身份也更容易复现。

**数值与适用边界：** 不得把 LoRA 矩阵直接加到 INT8 字节；不得重复挂 Larry；旧 Turbo 的 alpha/16 规则不能套用。冻结 SHA 为 9eccf52e…71fec12。

**验收证据：** S04/S05；R85 manifest；当前同场比较两臂权重 SHA 相同。

**实测收益口径：** 当前基础版已 baked，这一项没有计入本次 A→B 的新增收益。

**代码入口：** bake_models_remote.py；r130_r85_baseline.py

## 02. 原生分配器与 DynamicVRAM 权重预取

**阶段：权重与输入｜状态：共同基础｜来源：上游已有能力，项目沿用**

**原始路径：** 大模型无法把所有组件永久同时驻留显存；按需加载时，串行 H2D 会阻塞计算。

**具体改动：** 保留 native allocator（--disable-cuda-malloc）与原 DynamicVRAM；使用 pinned 主机权重和预取流，让下一层搬运与当前层计算重叠。

**为何更快/选择理由：** 隐藏权重搬运，控制 24 GB 显存中的组件切换。

**数值与适用边界：** event、源 tensor 与目标 owner 必须活到消费结束；不能把 allocator 的 max_split_size_mb:512 当成 DynamicVRAM 开关。

**验收证据：** S15/S19：一次 profile 的 H2D 为 1.593947 s，其中 1.546261 s 与计算重叠。

**实测收益口径：** 约 97.01% 是该次 H2D 重叠比例；基础版已有此能力，不能再算一遍 R85 新增收益。

**代码入口：** ComfyUI model_management；R85 driver

## 03. QKV 载体、所有权转移与去 V clone

**阶段：DiT｜状态：R85 固定｜来源：项目适配与优化**

**原始路径：** Q/K/V 通过普通张量载体反复交接，下游重排或生命周期不明时产生额外 clone 和大张量引用。

**具体改动：** 使用 AttentionTensorContainer 显式交接量化载体；消费完成后提前释放原引用，去掉已证明不必保留的 V clone。

**为何更快/选择理由：** 减少显存带宽、复制和 Python 载体处理；降低同时存活的大张量数量。

**数值与适用边界：** 只删不再使用的副本；量化 code/scale、layout 和 stream 消费顺序保持。未知 owner/stride 回退。

**验收证据：** S02/S31：.35 迁移的 container/no-clone/modulation 累计组合 forward 30.259366→29.984625 s；正式完整输出另有四 SHA 验证。

**实测收益口径：** 组合减少 0.908%，包含调制，未单独拆出 container 或 clone 的完整请求收益。

**代码入口：** teacher_migrated；runtime_compat035；model_dispatch035

## 04. RMSNorm 与 video/audio/text 分段调制融合

**阶段：DiT｜状态：R85 固定｜来源：项目适配与优化**

**原始路径：** RMSNorm 写出 BF16，再按 token 区段分别 scale/shift，之后再次写回。

**具体改动：** 同一 kernel 完成归一化、对应区段调制和输出，减少中间张量；对合法 stride 单独分派。

**为何更快/选择理由：** 少一次或多次全量 BF16 写读，以及小 kernel launch。

**数值与适用边界：** RMS reduction、epsilon、广播、token 边界和两次 BF16 舍入照旧，不能因融合而用更高精度贯穿。

**验收证据：** S02/S31 的逐项迁移回放与 S04/S05 全业务历史四 SHA。

**实测收益口径：** 迁移消融累计到本层 29.459518 s，相对 stock 累计 −2.643%；不是这一项独享 2.643%。

**代码入口：** model_dispatch035；teacher_migrated；Norm CUDA kernels

## 05. Residual、gate、norm2 与调制融合

**阶段：DiT｜状态：R85 固定｜来源：项目适配与优化**

**原始路径：** attention/MLP 分支做 gate 和 residual，写出结果后另起 Norm/调制 kernel。

**具体改动：** 在合法消费链中合并 residual 与后续 norm2/调制；保留真实浮点落点。

**为何更快/选择理由：** 减少 residual 张量读写和 launch；缩短 block 内的串行依赖链。

**数值与适用边界：** gate 的 BF16 乘法与 residual 的 FP32 FMA/最终 BF16 必须按原顺序；strided 输入不得冒充 contiguous。

**验收证据：** S02/S31：残差 Norm/strided 累计 forward 29.404285 s；S04/S05 全业务验证。

**实测收益口径：** 相对 stock 的组合累计 −2.826%，未把相邻消融差异当成独立服务收益。

**代码入口：** residual/norm CUDA；model_dispatch035

## 06. Norm/Residual/调制直接产生 ConvRot INT8 输入

**阶段：DiT｜状态：R85 固定｜来源：项目适配与优化**

**原始路径：** 融合后的 BF16 仍需写入显存，再由旋转量化 kernel 读取。

**具体改动：** 限定业务 shape、stride 和 act 下，Norm/调制或 Residual/Norm/调制直接接 ConvRot、scale 与 INT8 code 生成。

**为何更快/选择理由：** 省掉 GEMM 输入前的一次大张量往返。

**数值与适用边界：** 保留 group、旋转顺序、scale、padding、随机量化开关与原 BF16 中间落点；不满足资格时转回上游。

**验收证据：** S02/S04/S05：早期 Case1 不命中这些开关，后续才逐步覆盖当前 15 片段布局。

**实测收益口径：** 单纯打开开关不代表加速。该项在最终整组对照中验证，没有独立完整请求百分比。

**代码入口：** cross_quant.py；quant_shadow；runtime_int8.py

## 07. K5376/K7168 ConvRot 寄存器与线程调度

**阶段：DiT｜状态：R85 固定｜来源：项目适配与优化**

**原始路径：** 通用旋转量化 kernel 对业务宽度使用统一的线程和中间存储策略。

**具体改动：** 按 5376/7168 宽度调整 register、shuffle 与线程组织；9/19 迁移配置分别为 768/640 线程。

**为何更快/选择理由：** 减少 shared-memory 往返和不合适的线程工作划分。

**数值与适用边界：** 每个宽度独立资格；不得覆盖新版未知 act_weight/act_eps 语义。K14336 SwiGLU 更大融合不在默认路径。

**验收证据：** S02/S31：逐宽度和真实第一步回放，最终完整输出验证。

**实测收益口径：** 迁移组合累计到寄存器 ConvRot 为 29.386356 s；不宣传单独 −2.885%。

**代码入口：** convrot_schedule035；ConvRot CUDA sources

## 08. Attention 直接输出 BSHD 与精确 V 量化

**阶段：DiT｜状态：R85 固定｜来源：项目适配与优化**

**原始路径：** attention 的输出布局不直接符合 projection 消费者；V 量化也有额外转换和缓冲。

**具体改动：** attention 直接写 BSHD，使用保留原 scale/padding 的 V 量化路径，减少 transpose/copy。

**为何更快/选择理由：** 以消费者布局产生结果，避免全量激活搬运。

**数值与适用边界：** V 的行/组 scale、量化饱和和 padding 与基线相同；仅改地址映射不能改有效元素顺序。

**验收证据：** S02/S31：加入 BSHD/dense8102 的累计 forward 29.854025 s，再加 V quant 为 29.624339 s。

**实测收益口径：** 对应相对 stock 累计 −1.340% / −2.099%；这些是有继承关系的组合。

**代码入口：** selected-dense；runtime_compat035；teacher_migrated

## 09. SM120 dense：3 槽 TMA、Q 寄存器缓存、提前释放 K

**阶段：DiT｜状态：R85 固定｜来源：基于 SageAttention/Kitchen 的硬件适配**

**原始路径：** 基础 dense 分块 QK→online softmax→PV，内核调度存在 shared-memory 读取及搬运等待。

**具体改动：** D128、Q128/K128；3 槽 TMA ring；8 compute warp 加 4 producer warp；240/24 寄存器额度；Q 放寄存器，K 消费后提前交还槽位。

**为何更快/选择理由：** 隐藏 K/V 搬运，减少 Q 的重复读取，使整数 MMA 更连续。使用 SM120 能力，不套用 SM100 的 TMEM/tcgen05。

**数值与适用边界：** 66,560 B shared；原 KV 遍历、online softmax、概率 U8/PV S8 量化与累加次序保留；dense 不删 attention block。

**验收证据：** S15/S17/S27：长 QKV、边界、stream、完整生成；当前 SM120 170 SM，CTA shared 上限按设备查询。

**实测收益口径：** 纳入整组 R85 对照；历史后续 dense 小改完整周期仅 −0.1245%。不能把 pure-MMA 95.3% 比值写成整机理论利用率。

**代码入口：** selected-dense；r85_lib.py；SM120 CUDA headers

## 10. o_scale 等于 1 的恒等乘法消除

**阶段：DiT｜状态：R85 固定｜来源：项目适配与优化**

**原始路径：** online softmax 归一化路径即使 o_scale==1 也执行分母和输出的对应乘法。

**具体改动：** 只对精确等于 1 的分支省略恒等乘法；其他 scale 仍用原实现。

**为何更快/选择理由：** 减少主循环末端依赖和指令。

**数值与适用边界：** 不是把相近 scale 当作 1；KV 顺序、softmax 和输出舍入保留。

**验收证据：** S17/S27：与 QKV 调度的正式队列组合，2175.599311→2165.853432 GPU s，历史四 SHA。

**实测收益口径：** 组合串行容量 +0.450%；没有把恒等乘法单独拆成这一收益。

**代码入口：** selected-dense skip-scale branch

## 11. Packed embedding 复制后释放无消费者的引用

**阶段：DiT｜状态：R85 固定｜来源：项目适配与优化**

**原始路径：** text/video/audio embedding 已打包，但原 tensor 仍被 Python 变量或容器持有。

**具体改动：** 在 packed copy 完成且没有后续消费者后释放原引用。

**为何更快/选择理由：** 减少峰值显存和 allocator 压力，为大业务形状留余量。

**数值与适用边界：** 必须等待 copy 的真实 stream 依赖；不能释放尚被某个消费者读取的 storage。

**验收证据：** S18/S19：峰值内存与完整输出验证。

**实测收益口径：** 单独速度收益不稳定，主要作为内存治理；不虚构独立降时百分比。

**代码入口：** embedding_release.py；business_fusion.py

## 12. 跨 block FC2 residual→下一层 norm/quant

**阶段：DiT｜状态：R85 固定｜来源：项目适配与优化**

**原始路径：** 本 block FC2 的 residual 先完整写出，下一 block 再读入 norm1/量化。

**具体改动：** 以单消费者载体串联相邻 block，8 steps×49 连接共 392 次跨层消费；8 次末层单独收尾。

**为何更快/选择理由：** 消除跨层中间激活写读，减少 launch。

**数值与适用边界：** 状态绑定 request/step/block/generation/owner/pointer；不能用上一 step 的值。同实例并发不满足资格，必须隔离或回退。

**验收证据：** S04/S05/S16：消费次数、owner 检查与全部业务历史四 SHA。

**实测收益口径：** 属于当前精确组合收益，尚未单独测整服务贡献。

**代码入口：** cross_quant.py；combined_gate.py；gate_adapter.py

## 13. Out-proj/FC2 INT8 GEMM gate-residual epilogue

**阶段：DiT｜状态：R85 固定｜来源：项目适配与优化**

**原始路径：** GEMM 输出反量化 BF16 branch 后，另起 gate/residual kernel 再读写。

**具体改动：** 在 epilogue 接上 gate/residual，但显式保留 INT32→FP32 scales→bias→BF16 branch→BF16 gate→FP32 FMA→BF16 的原顺序。

**为何更快/选择理由：** 省掉分支结果的额外全量写读和 kernel launch。

**数值与适用边界：** 不能删 BF16 中间舍入或改变 FMA 边界；只作为 owner 元数据保留的 buffer 不得被当成已初始化数据读取。

**验收证据：** S16：FC2 stage1/config13，真实链 27.393008→26.082304 ms；15 片段、70 次含预热完整请求四 SHA。

**实测收益口径：** 局部链 −4.783%；out-proj＋FC2 完整组合容量 +0.468%。分母不同。

**代码入口：** combined_gate.py；gate_adapter.py；runtime_int8.py

## 14. FC1 与 QKV 的 tile raster 配置

**阶段：DiT｜状态：R85 固定｜来源：项目适配与优化**

**原始路径：** 通用 GEMM 的 block 遍历和 swizzle 未必适合 Ref2VA 的长 M。

**具体改动：** FC1 采用 config3/swizzle8，QKV 使用 config5；短 M 和真实长 M 分开验证。

**为何更快/选择理由：** 改善 tile 调度与缓存复用；不改变数学运算。

**数值与适用边界：** 尺寸、align、bias、scale 和 epilogue 资格保持；不能按短 microbenchmark 选全业务默认配置。

**验收证据：** S04/S05/S17：正式完整周期与历史 SHA；已吸收到 R85 固定入口。

**实测收益口径：** 最终累计对照包含该项，未另报独立百分比。

**代码入口：** runtime_int8.py；r126_driver.py

## 15. QKV projection 后的 RMS/RoPE/Q quant/V partial-max 融合

**阶段：DiT｜状态：R85 固定｜来源：项目适配与优化**

**原始路径：** QKV projection 后多次进行拆分、RMS、RoPE、量化与 V 的统计准备。

**具体改动：** 连接投影输出和 RMS/RoPE/Q 量化/V partial-max，按原中间舍入生成后继所需数据。

**为何更快/选择理由：** 减少高带宽中间张量、独立统计扫描和 launch。

**数值与适用边界：** K 中心化依赖与 anchor 定义照旧；Q/K/V 各自 scale 与 padding 照旧。

**验收证据：** S04/S05/S15/S27：实际调用覆盖、400 次完整 block attention 和全业务输出。

**实测收益口径：** 是 R85 累计 −1.905% 组合的一部分，不能再次叠加到最终 A→B。

**代码入口：** r71_model.py；r79_lib.py；r82_lib.py

## 16. 当前输入的 9 行 K-only anchor 与 4-warp K fusion

**阶段：DiT｜状态：R85 固定｜来源：项目适配与优化**

**原始路径：** 为 anchor 所需的少量行计算完整 Q/K/V，尽管只消费 K。

**具体改动：** 每次从当前输入投影 9 行 K-only anchor；使用 4-warp K fusion，省去不被消费的 Q/V。

**为何更快/选择理由：** 按真实消费者删除无用工作，减少小路径调度开销。

**数值与适用边界：** 每个 step 重新计算当前 K；不是跨 step cache，也不替换当前 attention 结果。

**验收证据：** S04/S05：R85 所有 15 片段历史四 SHA；R126 独立重复。

**实测收益口径：** 与 QKV/末层裁行整组新增 −1.905%，R126 再现 −1.897%；两者是同一收益的验证。

**代码入口：** r85_model.py；r85_lib.py

## 17. 末层 reference prefix 无消费者行裁剪

**阶段：DiT｜状态：R85 固定｜来源：项目适配与优化**

**原始路径：** 最后一层继续为不进入最终导出的 reference prefix 计算 Q、out-proj、FC1/FC2。

**具体改动：** 根据消费者关系仅计算 live query 与后续 GEMM 行；其他 query 需要的 K/V 仍完整保留。

**为何更快/选择理由：** 消除最终输出不依赖的计算，而不减少有效 token、层数或采样次数。

**数值与适用边界：** 原 prefix residual/填充、对齐和形状语义照旧；未知输入关系不能套用。没有稀疏注意力或早停。

**验收证据：** S04/S05/S27：15 片段 60 次正式请求，4 SHA 一致。

**实测收益口径：** 纳入 R85 新增 −1.905% 组合；未单独分配速度数字。

**代码入口：** r45_model.py；r45_window.py；r54_model.py

## 18. INT8 ConvRot ViT3D decoder，保留旧 FP16 encoder

**阶段：视频 VAE｜状态：R85 固定｜来源：上游量化模型与实现，项目接入验证**

**原始路径：** 历史原生视频 VAE 权重实际为 FP16；更早生产 decode 用 TRT FP16。参考 encoder 也用旧 FP16 路径。

**具体改动：** 视频 decode 切换 .35 配套 INT8 ConvRot 权重与新版模型/helper。36 个 ViT3D block 共 144 个量化 Linear；encoder 继续旧类与旧权重。

**为何更快/选择理由：** 将 QKV、attention output 和 FFN w1/w2 主要乘法改用量化路径，降低权重带宽与计算成本。

**数值与适用边界：** 量化会改变像素；这项只主张实测差异很小，不能按位无损。模型含 F16/F32 元件；不把历史 FP16 标成 BF16。

**验收证据：** 新同 latent FP16→INT8 15 段比较单列于“VAE全15段质量”。历史 S03/S06 是 TRT FP16→INT8，不能移作原生 FP16 的新质量数字。

**实测收益口径：** 历史 TRT 阶段 16.559342→9.959952 s，−39.853%；本次原生 FP16 基线的完整收益以新的直接实测为准。

**代码入口：** latest_vae.py；vae_pr_helpers.py；INT8 ConvRot checkpoint

## 19. VAE 内部 RMS/SwiGLU 输入量化与残差 epilogue

**阶段：视频 VAE｜状态：R85 固定｜来源：上游实现与项目兼容迁移**

**原始路径：** 输入激活处理、旋转量化、Linear、残差分别读写中间张量。

**具体改动：** 新版 linear_input_act 把 RMSNorm/SwiGLU 接进 ConvRot 量化；支持的 Linear 使用 residual/residual_scale epilogue，Q/K norm 与 RoPE 融合。

**为何更快/选择理由：** 减少 36 层 decoder 内的中间张量和 launch。

**数值与适用边界：** Kitchen .33 的 8 参数 ABI 变为 .35 的 10 参数，act_weight/act_eps 必须传递；旧 wrapper 只对已验证资格截获。

**验收证据：** S02/S03/S31；Kitchen #167、ComfyUI #16187/#16332；实际导入源码/二进制 pin。

**实测收益口径：** 上游 INT8 decoder 的组成部分；没有把它与 INT8 decoder 收益重复相加。

**代码入口：** vae_pr_helpers.py；runtime_compat035；Kitchen CUDA kernels

## 20. Tile 批量、预分配 canvas 与引用释放

**阶段：视频 VAE｜状态：R85 固定｜来源：上游 canvas/tile 能力与项目输出接入**

**原始路径：** 时空 tile 解码后长期保留引用，或通过重复拼接构建输出，增加显存占用和拷贝。

**具体改动：** 按显存限制使用最多 4 个 tile，输出写预分配 canvas；旧 tile 消费后及时释放，保留原时空 overlap。

**为何更快/选择理由：** 限制峰值内存、减少拼接和重复分配，为 decoder 算子留工作空间。

**数值与适用边界：** 不改变 tile 边界、blend 权重、叠加顺序；不能在尚需 blend 的 tile 上提前转 uint8。

**验收证据：** S03/S18/S31：tile 边界、RGB 与同 latent 比较。

**实测收益口径：** 未单独隔离端到端收益；更大 tile 并未形成默认稳定优化。

**代码入口：** latest_vae.py；chunk_rgb.py

## 21. 最终 blend→GPU RGB8 转换融合

**阶段：视频 VAE｜状态：R85 固定｜来源：项目适配与优化**

**原始路径：** 原视频输出以 FP32 传到 CPU，再 ×255、clamp、转 uint8；每通道传 4 byte。

**具体改动：** 仅在原 blend 完成后，GPU 按原 FP32 std/mean、clamp、×255、truncate 顺序产生 RGB8，每通道传 1 byte。

**为何更快/选择理由：** 把 D2H 数据量降到四分之一，减少 CPU 转换与主机临时大张量。

**数值与适用边界：** 禁止改变边界结果的 FMA contraction；truncate 不能换 round；未结束的 overlap 区间保持 FP32。

**验收证据：** S18/S34 与完整 RGB SHA；Case1 传输 3,009,871,872→752,467,968 byte。

**实测收益口径：** 这项是精确输出工程；传输字节 −75% 不等于完整耗时 −75%。

**代码入口：** finalize_rgb.py；chunk_rgb.py

## 22. 双 pinned buffer 与事件驱动 D2H

**阶段：视频 VAE｜状态：R85 固定｜来源：项目适配与优化**

**原始路径：** 输出转换和 GPU→CPU 拷贝串行；buffer 复用时机如果只依赖 non_blocking=True，容易阻塞或破坏数据。

**具体改动：** compute 记录 ready event，copy stream 等待后异步 D2H；CPU 等 copied event 消费，消费完成才归还 slot。双 slot 有明确上限和背压。

**为何更快/选择理由：** 把可重叠的搬运隐藏在后续 GPU 工作中，并约束主机内存。

**数值与适用边界：** record_stream、强引用、GPU 源与 CPU 目标生命周期必须完整；CPU 仍在读时不能复用锁页 slot。

**验收证据：** S18/S34；10s 历史帧交接 1.757→0.303 s，15s 2.617→0.465 s；28 次含预热 MP4 黄金验证。

**实测收益口径：** 历史帧交接包含累计优化，不能全部归因 D2H；本次直接总耗时已包含 R85 输出路径。

**代码入口：** chunk_rgb.py；export_handoff.py

## 23. CPU x264/AAC/MP4 与下一请求 GPU 重叠

**阶段：服务工程｜状态：服务层已验证｜来源：原部署已有异步能力，项目保留并约束所有权**

**原始路径：** 若等当前 MP4 编码结束再启动下一请求，CPU 成片期间 GPU 会空闲。

**具体改动：** 请求 N 的 RGB/PCM 进入限量 CPU 编码队列，GPU 可以处理 N+1；mmap/帧缓冲由当前 export 独占。

**为何更快/选择理由：** 利用 CPU 与 GPU 的不同资源提升连续队列吞吐。

**数值与适用边界：** 以 MP4、上传和结果提交全部完成作为成功；需排空、取消、失败处理和按总字节背压。GPU 完成不等于请求成功。

**验收证据：** S01/S19/S34：9/13 真队列完成间隔 143.785→137.341 s（10s），261.766→251.404 s（15s）。

**实测收益口径：** 两臂当时已启用异步 export；这些是历史累计收益。本次单请求 A/B 没有测试跨请求重叠吞吐。

**代码入口：** export_graph；export_handoff.py；server finalizer

## 24. 本地不可变模型缓存与 Qwen 保留热页

**阶段：服务工程｜状态：服务层已验证｜来源：项目适配与优化**

**原始路径：** 共享文件系统和频繁 mark_cold 可能让相同权重被反复读盘。

**具体改动：** 将固定模型放本地不可变缓存，核对版本/SHA；Qwen 使用 mark_cold=False 保留活跃页。

**为何更快/选择理由：** 以主机内存换取共享盘读量和加载波动降低。

**数值与适用边界：** 每副本需要内存预算，不能无限复制；已缓存文件仍按身份校验；当前 Qwen 未来拆分不算本次收益。

**验收证据：** S19：一次 591 次 / 27.12 GB 权重读取后，连续 5 请求没有新增读取；主机内存约 56.49→78.58 GiB。

**实测收益口径：** 当前计时排除首次读权重与 Qwen，所以不能给本项分配这次总加速中的百分比。

**代码入口：** H3INT8Release；H3ReleaseOps；local model cache

## 25. 素材预取、请求隔离与编码排空

**阶段：服务工程｜状态：服务层已验证｜来源：项目适配与优化**

**原始路径：** 素材准备、下载、上传、编码若串在 GPU 前后，会扩大服务完成时间；无限异步队列又会耗尽内存。

**具体改动：** 在当前 GPU 请求期间准备下一请求素材；当前 CPU 编码/上传可继续；每项以版本、请求 owner 与终态关联。

**为何更快/选择理由：** 把不占 GPU 的工作放入可重叠区间，提高持续任务下的资源利用。

**数值与适用边界：** 不得按相似 prompt 直接共享 condition；冻结 condition 只是实验控制变量。失败和取消必须回收对应任务资源。

**验收证据：** S01/S19/S34：服务完成事件、drain 与业务集成记录。

**实测收益口径：** 依部署接入而定；本次固定 condition→MP4 比较不覆盖下载/上传和真实 condition。

**代码入口：** server；export_graph；task finalizer

## 26. 音频 VAE 搬 CPU 的实验与保留 GPU 的原因

**阶段：音频｜状态：未采用｜来源：项目适配与优化**

**原始路径：** 当前音频 VAE 在 GPU 上以 FP32 解码，约 0.11–0.14 s。

**具体改动：** 曾试 4 线程 CPU 以便与 GPU 视频 VAE 并行；耗时 4.49–9.44 s，PCM 也不同，因此当前继续 GPU FP32。

**为何更快/选择理由：** 并行只有在关键路径缩短时才有收益；当前音频在总周期中太短。

**数值与适用边界：** 音频权重 917 个 FP32 tensor；不通过隐式半精度改变 PCM。与视频量化比较时使用同一 PCM。

**验收证据：** S14/S15；CPU/GPU 耗时及 PCM 对照。

**实测收益口径：** 即使完全隐藏当前 GPU audio，完整周期理论可省约 0.05%；没有实测并行净收益。

**代码入口：** nodes_audio.vae_decode_audio；FP32 audio VAE

## 27. VAE 阶段静态 cast/load

**阶段：视频 VAE｜状态：后续候选｜来源：项目适配与优化**

**原始路径：** R85 的视频 VAE 仍采用动态包装，逐层 cast/load 带来重复处理。

**具体改动：** 在 VAE 阶段预先按原 dtype/scale cast/load，阶段结束再释放；不是永久同时驻留整个 DiT 与 VAE。

**为何更快/选择理由：** 减少动态包装和重复权重处理。

**数值与适用边界：** 必须验证组件切换、下一请求峰值显存和长期生命周期；候选没有并入本次冻结 R85。

**验收证据：** S07/S20：Case1/8 完整生成各 2 正式/臂；全15同 latent。

**实测收益口径：** 仅静态完整周期约 −0.20%；静态＋R207 全15视频 −6.802%，分母为 R85 动态 VAE。

**代码入口：** r197_full_vae.py；r207 VAE overlay

## 28. R234v2：VAE RMS/RoPE→Q 旋转量化

**阶段：视频 VAE｜状态：后续候选｜来源：项目适配与优化**

**原始路径：** VAE Q 完成 RMS/RoPE 后先写出，再由 ConvRot 量化读取。

**具体改动：** CTA 内处理 32 token，Q 写 shared 后按原 32-row/8 组 scale 旋转量化；K 保留原 FP16 写回和 anchor 检测。

**为何更快/选择理由：** 缩短 VAE attention 输入准备，减少 Q 中间往返。

**数值与适用边界：** 首版 FMA/倒数边界不一致被拒绝；仅 v2 保留原数值。hook 只在 VAE D64 作用域使用并可靠恢复。

**验收证据：** S08/S20：v2 全15 decode 正确性；30 预热＋60 正式。

**实测收益口径：** 与 R244 组合视频 −0.632%，相对静态＋R207；不能再与 R85 总收益相加。

**代码入口：** r234v2_vae_ropeq.cu；r245_attention_overlay.py

## 29. R244：VAE D64 cached-Q 与小 Q tile

**阶段：视频 VAE｜状态：后续候选｜来源：项目适配与优化**

**原始路径：** VAE 真实 [4,32,1797,64] shape 的原核使用较大 Q tile，196 register，约 2 CTA/SM。

**具体改动：** Q 保留寄存器，CTA_Q 128→64、WARP_Q 32→16，K tile=64；约 128 register、4 CTA/SM。

**为何更快/选择理由：** 提升该短序列 D64 shape 的并发 CTA 与 Tensor 管线活动。

**数值与适用边界：** 原 128-row 量化 scale 索引、softmax/PV 顺序保留；不能拿 VAE D64 核覆盖 DiT D128。

**验收证据：** S08/S20：Tensor active 53.42%→63.28%；全15视频14段更快，Case12 微慢约 0.021%。

**实测收益口径：** 局部 attention 约 −11%，全视频组合仅 −0.632%；完整 VAE <5s 未达到。

**代码入口：** r244_q64w16.cuh；r245_attention_overlay.py

## 30. 新 PyTorch/Triton、NVFP4 与近似计算候选

**阶段：编译与精度｜状态：未采用｜来源：项目适配与优化**

**原始路径：** 希望通过升级编译栈、更低比特或 cache/稀疏减少运算。

**具体改动：** Torch 2.14/Triton 3.8 完整 INT8 对照未获得稳定 DiT 提速且轨迹改变；CPU audio、NVFP4、近似跳算等没有进入当前无损算子基线。

**为何更快/选择理由：** 保留负结果，避免复用时把试验开关误当推荐配置。

**数值与适用边界：** 新栈并未重编全部自定义 CUDA 核；不能据此排除未来编译收益。NVFP4 不按位等价；近似方案须另做视频/音频评估。

**验收证据：** S09/S33：Case1 DiT 225.55784→225.56589 s；Case8 212.01914→212.04437 s，完整周期方向不一致。

**实测收益口径：** 当前没有可纳入默认 R85 的新增收益。详细负结果与 LoRA 质量见原章节。

**代码入口：** r360_stack_comparison.py；docs/EXPERIMENTS.md
