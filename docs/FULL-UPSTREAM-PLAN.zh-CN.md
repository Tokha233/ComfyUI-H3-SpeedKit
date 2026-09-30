# 整套加速方案的上游提交范围与 6.49% 口径

2026-09-30。结论：整套方案中仍有实质增量的部分可以继续向上游贡献；现有
#217/#218/#219 只是第一批，不能代表整套 DiT 已提交完成。完整 SpeedKit
代码已公开，与“被 Kitchen/ComfyUI 合并”是不同状态。下面是代码和证据审计，
本次没有增加 GPU 性能样本，也没有新建上游 PR。

## 先确认完整方案的收益

原始记录见 [Kitchen .36 对照](../evidence/kitchen036-comparison.json)，
完整协议见 [BENCHMARK-036](BENCHMARK-036.md)。

| 同场比较 | 基础路径 | 完整 SpeedKit | 耗时减少 |
|---|---:|---:|---:|
| **DiT，相同 INT8 权重、Larry v4、8 步** | **233.986204 s** | **218.795239 s** | **6.4922%** |
| 完整周期，两臂均为 INT8 VAE | 250.449598 s | 233.626182 s | 6.7173% |
| 完整周期，基础 FP16 VAE → 优化 INT8 VAE | 264.439498 s | 233.626182 s | 11.6523% |

DiT 每次平均节省 **15.190965 秒**。耗时减少 6.49% 对应耗时倒数增加约
6.94%，但这只是容量换算，不能称为持续服务吞吐实测。DiT 计时不含 VAE，
所以 6.49% 没有把 INT8 VAE 的收益算进去。

条件：RTX 5090 D v2，Kitchen 0.2.36，Torch 2.12.0+cu130，ComfyUI 固定
`2504e68d4d9dedb514e172692f13436623f25aed`。两个业务代表片段（243/260 帧，
packed tokens 为 87,142/90,461），每臂每片段两次正式运行，独立进程热身后
顺序运行。不是完整的五个约 30 秒业务故事，也不是随机交错的持续服务测试。
同 INT8 路径的四次优化请求均与基础路径的 video/audio latent、RGB、PCM
SHA256 相等；FP16 VAE 的 RGB 则有微小差异。

## 为什么 #219 只有约 1.7%

[#219 消费者实测](INDEXED-GATE.md)只启用 outproj/FC2 的 indexed gate
epilogue，其余 Norm、QKV、attention 和输出流程使用基础实现。
它测的是另一个公开输入：768×512、124 帧、14,850 packed tokens。
两组共 16 次正式请求得到 DiT **15.901378 → 15.633065 秒（−1.6874%）**，
完整本地文件周期 **19.183121 → 18.990520 秒（−1.0040%）**。

这与业务片段整套方案的 6.49% 不矛盾：启用的优化数量、序列长度和输入不同。
不能用 `6.49% - 1.69%` 推算其他优化的贡献，也不能把 #217/#218/#219 的
单核百分比相加。融合之间会消除相同中间结果、影响内存峰值和调度，组合收益
需要直接测量。完整 R85 已有 gate 融合，#219 的收益也不能再叠加到它上面。

## 当前三个 PR 实际覆盖了什么

| PR | 已提交的上游边界 | 与完整方案相比仍未覆盖 |
|---|---|---|
| [#217](https://github.com/Comfy-Org/comfy-kitchen/pull/217) | SM120、BF16、ConvRot256、K5376/K7168 的寄存器路径及原入口分派 | Norm/调制直接生成 INT8；RMS 的规约与量化跨算子融合 |
| [#218](https://github.com/Comfy-Org/comfy-kitchen/pull/218) | SM120 D128 dense 三槽 TMA + cached Q，有限定的 BHSD contiguous 分派 | 消费者需要的 BSHD 输出、融合 QKV/V 准备、末层 live-query 范围 |
| [#219](https://github.com/Comfy-Org/comfy-kitchen/pull/219) | 通用预量化 INT8 GEMM + indexed BF16 gate + residual API | 上游 ComfyUI 的自动调用、整条融合链；SpeedKit 中已有独立实验调用节点 |

#218 不是把完整私有 dense 文件原封不动提交：公共 API 的 dtype、mask、
尺度和布局范围更广，当前 PR 有意限定了已验证分派。#219 也是一个新增 API，
仅安装它并不会让现有 ComfyUI 自动调用。实际调用代码同样需要上游接入。

## 全部活跃优化的提交去向

“候选”表示建议的下一批范围，不表示已经实现上游接口、完成通用验证或获得
维护者认可。以下依据发布版 [runtime.py](../h3_speedkit/runtime.py) 与
[迁移说明](MIGRATION-036.md)，不是把历史 30 项实验都当作当前生效优化。

| 活跃内容 | 建议去向 | 剩余工程与验收 |
|---|---|---|
| RMSNorm → 分段 modulation → ConvRot INT8 | **Kitchen 新融合算子 + ComfyUI 调用** | 当前 K5376、固定 Torch RMS reduction；明确每次 BF16 舍入、eps、表格 stride、异常 row、极值和回退，不覆盖原有 `rms_adaln` 不同契约 |
| QKV GEMM → RMS/RoPE → Q/K INT8，生成 V partial-max | **Kitchen 新的组合算子 + ComfyUI 调用** | 当前 H3 56×128、hidden5376、partial RoPE 专用；需要公开接口、独立参考、更多 M/边界和 fallback；不能把“已存在 RMS/RoPE”重新作为独立新贡献 |
| 当前输入九行 K-only anchor、四 warp K 准备 | **与上一项作为同一候选组** | 保留每步重算、anchor 选择和 K scale 语义；不是跨步缓存。先证明与当前基础量化完全一致 |
| V partial scales + 量化转置 | **Kitchen，优先随 QKV 组合提交** | partial-max 已来自融合 QKV，单独剥离会重新扫描 V；必须分别测完整量化链与消费者收益 |
| Attention 直接 BSHD 输出，避免 layout copy | **Kitchen 布局/输出接口 + ComfyUI 消费者** | 当前公共接口分配 BHSD；明确 stride/返回布局，保留默认行为和非目标设备回退；作为 #218 后续，不临时扩大正在审查的 PR |
| SM120 dense 主循环及寄存器 ConvRot | **现有 #218/#217** | 后续根据审查补形状/设备验证；剩余 scalar 特化先比对当前源码再决定是否存在增量 |
| Outproj/FC2 gate-residual epilogue | **现有 #219 + ComfyUI 接入 PR** | 当前实验节点已证明真实调用，正式 upstream 接入需兼容 per-block attention、LoRA callbacks、hooks、compile/offload |
| FC1/QKV tile raster 与 L2 调度 | **先对现有 [#215](https://github.com/Comfy-Org/comfy-kitchen/pull/215) 做对照** | 不另提重复 FC1 调度 PR。比较 current main、#215、我们的 raster，再测能否组合；作者“约 2%”不是我们已复现的数据 |
| Packed embedding 无消费者引用释放 | **ComfyUI 小 PR** | 当前主线仍保留待审查的局部引用；先以主线对象生命周期/显存测量确认增量。兼容新 malloc scope，优先报告峰值内存，不虚构速度百分比 |
| 最后一层只计算最终输出需要的 query/projection/MLP 行 | **ComfyUI 模型优化 + 必要 Kitchen query-range 接口** | 保留全部 K/V、原量化 tile 对齐、有效输出；测试带 reference、纯生成、masked、patch/hook。不能把 H3 最后一层规则塞成通用 attention 默认行为 |
| 模型局部 clone、patch、预量化直接交接与生命周期 | **ComfyUI 消费者集成；插件保留完整接入** | 原生 API 消费后才能获得融合链收益；避免硬编码整个 block、避免依赖私有 ctypes ABI、保留用户 patch 可见性 |
| VAE 最终 blend 完成后 fused RGB8 转换 | **ComfyUI 输出接口或插件；通用算子部分可另评估 Kitchen** | 量化应发生在 blending 完成后；普通 IMAGE 后处理需要浮点，RGB8 不适合静默替换所有 VAE 输出 |
| 两个 pinned buffer、事件驱动 D2H | **ComfyUI/插件输出链** | 需要可流式消费的输出，验证 stream、owner、异常退出和内存上限；减少传输字节不等于等比例降低总耗时 |
| CPU MP4 编码与下一请求 GPU 重叠 | **插件/服务调度层** | 保留队列背压和错误传播；正常保存节点仍等文件完成。需要真实并发吞吐测试，不能从单请求时间推断 |

不能作为新增优化再次申报的内容：

- **INT8 VAE 模型及基础解码/输入激活融合来自上游**，可以贡献调用改进、回归
  数据和输出工程，不能把已有权重和算子当作我们的新实现。
- **Larry v4 来自上游**。我们贡献选择评测、合并工具和部署接入；当前速度
  对照两臂已经使用相同 baked 权重，不能再加一次预合并收益。
- **DynamicVRAM、预取流与原生分配器是共同基础**。保留这些能力不等于新增 PR。
- **CPU 音频 VAE 未采用**：历史 CPU 更慢且 PCM 改变。当前 GPU FP32 音频
  不是可开源为“无损 CPU 加速”的结果。
- 历史 residual→norm carrier、跨 block token carrier 已被最终直接 gate
  epilogue + norm/quant 路径替代，不再作为额外活跃融合重复计算或提交。
- 未进入默认方案的 NVFP4、近似缓存、稀疏路径和失败实验保留研究记录，
  不加入当前“输出一致的完整加速”声明。

## 当前 ComfyUI 主线与测试版本不同

本次审计固定的 Kitchen main 仍是
[`19ea55b`](https://github.com/Comfy-Org/comfy-kitchen/commit/19ea55b9ebdaf77942dab36223e1222009d3ce11)。
ComfyUI 主线固定到
[`8cfe5e1`](https://github.com/Comfy-Org/ComfyUI/commit/8cfe5e1ecb97512dea8deaac15e1228d7e6feeb1)，
其 [H3 源码](https://github.com/Comfy-Org/ComfyUI/blob/8cfe5e1ecb97512dea8deaac15e1228d7e6feeb1/comfy/ldm/minimax/model.py)
SHA256 为 `8e4cdda6ffd6994baa29085fb6d6a826a399ed094f0ec8e85387083877a4d518`。

与我们测试 pin 相比，当前 H3 已经：

- 去掉 `v = v.clone()`；这部分不能再次提交，也不能归因于我们的 PR。
- 使用 `ComfyAttention` / `preferred_attention`，支持 per-block attention。
- 为 sparse patches 暴露 gate_compress、layout 和 block_index 等信息。
- 更改 denoise mask 行为，增加 malloc graph 和 prefetch `malloc_scope`。

因此，**Kitchen .36 + 旧 ComfyUI 上的 6.49% 已有证据；当前 ComfyUI 主线上
还剩多少净增量，尚未重测**。把现有源码 SHA 白名单改成新值并不能完成迁移，
也不能直接删掉这些上游新接口。应在隔离环境适配后重新做 A/B。

## 推荐的下一批工作顺序

1. **固定当前主线的新基线并迁移 consumer。** 保留旧 6.49% 记录，另建 current
   baseline；先保证相同权重、采样条件、attention 和输出精度，再测全部四签名。
2. **完成两个核心融合候选组：Norm/调制/量化、QKV/RMS/RoPE/量化。**
   每组有可调用 API、独立参考、回退和真实 consumer；不用未证实的单项百分比排序。
3. **追加 BSHD 布局与已有 #219 的正式模型调用。** 上游 API 和 ComfyUI
   调用可以是相互链接的 draft；待 Kitchen 有可用发布版本后再启用默认路径。
4. **验证 #215 与我们的调度、末层 live-row、embedding 释放。**
   小范围改动独立提交；与已有 PR 重叠的调度优先提供对照证据或合作补丁。
5. **最后提交完整组合验证。** 同一机器、同一当前基础版本、同一批条件，交错
   顺序测基础、各组、全组合；保留原始样本、冷启动、首次验证、峰值内存和失败。
   对照音视频 latent 与 RGB/PCM，单独标明 VAE 量化差异。

目标是让用户通过上游库加少量原生模型接入获得完整收益，并让每一块可以独立
审阅。不是要求一个 Kitchen PR 接受整个 H3 runtime、复制的 ComfyUI 模型、
九个动态库和服务调度。Kernel 的 Apache/BSD 来源声明保留；ComfyUI GPL
模型流程的修改留在相应项目，不能直接换个文件头作为 Apache Kitchen 代码。

这份表不保证全部候选会被上游接收。即使某项更适合插件，完整可用方案仍然
在 SpeedKit 中公开；“上游合并比例”与“完整方案是否可复用”分别跟踪。
