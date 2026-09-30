# 2026-09-30 第二批上游贡献实测

本批基于当前 ComfyUI `8cfe5e1` 和 Kitchen main `19ea55b`。旧版完整方案
DiT 6.4922% 的记录保留，不把本批单项百分比与它相加。

| 改动 | 上游 PR | 当前状态 | 实测范围 |
|---|---|---|---|
| SM120 寄存器 ConvRot | [Kitchen #217](https://github.com/Comfy-Org/comfy-kitchen/pull/217) | Open | 80 passed / 3 skipped；大形状算子−7.76%～9.22% |
| SM120 dense TMA | [Kitchen #218](https://github.com/Comfy-Org/comfy-kitchen/pull/218) | Open | 9项新增测试通过；完整suite 428 passed / 594 skipped / 2 inherited failures |
| gate 回退输出布局和 wheel 许可证 | [Kitchen #219](https://github.com/Comfy-Org/comfy-kitchen/pull/219) | 已修复到 5f7290d | 38 passed / 1 skipped；实际 wheel 包含完整许可 |
| INT8 attention 直接 BSHD 输出 | [Kitchen #220](https://github.com/Comfy-Org/comfy-kitchen/pull/220) | Draft，依赖 #218 | 37 GPU tests；attention+布局链 −0.50%～4.54% |
| RMSNorm＋indexed 调制＋ConvRot INT8 | [Kitchen #221](https://github.com/Comfy-Org/comfy-kitchen/pull/221) | Draft | 34 tests；六分段准备链 −65.14%～80.50% |
| H3 QKV/RMS/RoPE/量化融合 | [Kitchen #222](https://github.com/Comfy-Org/comfy-kitchen/pull/222) | Draft | 20 tests；完整采样 −2.769% |
| float-input indexed gate 包装 | [Kitchen #223](https://github.com/Comfy-Org/comfy-kitchen/pull/223) | Draft，依赖#219 | 27 operator tests |
| H3 FC2 gate消费者 | [ComfyUI #16681](https://github.com/Comfy-Org/ComfyUI/pull/16681) | Draft，依赖#223发布；审查修复已推送 | 原GPU版15 tests、完整采样−1.384%；修订版20 CPU tests，待GPU重验 |
| 提前释放打包 embedding 临时引用 | [ComfyUI #16677](https://github.com/Comfy-Org/ComfyUI/pull/16677) | Open；CodeRabbit APPROVED | 峰值 Torch 显存减少 161.47 MiB |
| 消费 BSHD attention 接口 | [ComfyUI #16678](https://github.com/Comfy-Org/ComfyUI/pull/16678) | Draft，等 Kitchen 发布 | 完整采样 −0.410% |

目前共10个相关PR：Kitchen #217～#223，以及ComfyUI #16677/#16678/#16681。均未合并。

## 当前主线的独立完整采样

RTX 5090 D v2；Torch 2.12.0+cu130；comfy-aimdo 0.5.5（隔离安装）；公开输入
768×512、124 帧、14,850 packed tokens；同一 Larry v4 INT8、Euler/beta 8步、
seed42、CFG1、shift12/4；各组热身后四对 AB/BA 交错正式运行。
计时为完整 sampler，不含 VAE、condition 编码和文件输出。所有组都使用正常
ComfyUI sampler，只有指定的消费者改动不同。

| 独立 A/B | 基础 DiT/s | 优化 DiT/s | 减少 | 正确性 |
|---|---:|---:|---:|---|
| embedding 引用释放 | 15.798279 | 15.736245 | 0.393% | 8次正式 video/audio latent SHA 相同 |
| BSHD 输出消费者 | 15.778404 | 15.713747 | 0.410% | 8次正式 video/audio latent SHA 相同 |
| QKV 准备融合 | 15.803847 | 15.366207 | 2.769% | 400融合/请求；8次正式 SHA相同；峰值相同 |
| FC2 indexed gate | 15.802266 | 15.583501 | 1.384% | 400融合/请求；8次正式 SHA相同；峰值+153 KiB |
| Norm 生命周期修正消费者 | 15.801076 | 15.087788 | 4.514% | 800融合/请求；SHA相同；峰值减少152.27 MiB |

Norm 首版多保留 projected QKV 临时张量，峰值 Torch 分配
1833455616→2474106880 bytes，不作为推荐接入方案；修复后峰值为1673788416 bytes，正式记录单独保存。
embedding 的峰值 1833455616→1664144896 bytes，不包含 aimdo 外部分配。
低于 0.5% 的计时差异按本机小收益报告，不承诺所有输入稳定相同加速。

当前仅核验 DiT latent，未重复解码视频/音频，不把它写成新的 RGB/PCM 实测。
原有完整方案四签名验收和 FP16/INT8 VAE 质量对比仍见旧证据。

## 证据与跟踪

- [本批5组sampler重算汇总](../evidence/upstream-0930/summary.json)
- [PR状态快照](../evidence/upstream-0930/pr-status.json)
- [审查及checks快照](../evidence/upstream-0930/review-status.json)
- [本地修订与组合验证记录](../evidence/upstream-0930/local-validation.json)

- [机器可读优化清单](../evidence/upstream-campaign.json)
- [embedding 正式记录](../evidence/upstream-0930/current-embedding-result.json)
- [BSHD 正式记录](../evidence/upstream-0930/current-layout-result.json)
- [Norm 修正后正式记录](../evidence/upstream-0930/current-norm-final-result.json)
- [Norm 首版记录](../evidence/upstream-0930/current-norm-result.json)
- [attention 原始样本](../evidence/upstream-0930/layout-benchmark.json)
- [Norm 原始样本](../evidence/upstream-0930/norm-benchmark.json)

此前#217～#221已检查的CLA/Socket通过；新#222/#223最新head的CLA已通过，Build Wheels为action_required。外部贡献的 Build Wheels/ComfyUI CI
需要维护者批准执行（action_required），不能标成测试失败或已通过。
Draft 不会自动发起 CodeRabbit 审查。已手动请求#220/#221/#222/#223/#16678/#16681审查；#220两条、#221三条意见已确认修复，其他回复以PR状态为准。

## 审查修复

- #219：回退输出连续性、真实 Inductor、wheel 完整 CUTLASS 许可已修复。
- #220：新增 BSHD 非正尺度/重复/stream/graph 的独立 math-SDPA 验证；37 tests通过；benchmark输出改为固定证据目录。
- #221：只在原生分派时分配输出，拒绝原生分派后释放缓冲再回退；增加回退显存回归测试。benchmark确认Torch revision/SM120/真实入口，并记录原生实现；34 tests通过。
- #16677：在实际打补丁的 model.py 上重复执行现有两项 H3 回归，2 passed。

FC2消费者属于可审阅Draft，仍等待Kitchen依赖发布，不能直接给当前.36用户默认打开。Norm/QKV完整采样实验消费者仍需正式API集成；当前实验保留在证据中，不等同于通用ComfyUI消费者已经提交。

## 连接中断时的剩余工作

h31 在最后上传后提示“会话超过最大连接时间，断开连接”，回到堡垒机 `[Host]>`。
此前全部四对正式单项 sampler 已完成，原始 JSON 已下载。#215 的隔离编译已启动，
但未拿到完成结果，不声明它已编译成功或获得了速度收益。组合分支已在本地解决
接口冲突，尚未完成组合 CUDA 编译和完整性能测试。不会将单项百分比相加代替该测试。

还需要GPU验证的范围：组合收益、Norm/QKV消费者公共接口、outproj gate正式接入、
最后一层 live-query（量化块对齐与patch契约）、#215调度与现有raster的重叠。
VAE权重和基础INT8解码已经来自上游，不重复提交为原创；RGB8/D2H/异步CPU导出保留
在可用插件中，通用ComfyUI IMAGE契约仍返回浮点数据。

最后检查：#16681 页面显示6项工作流等待维护者批准，并要求code-owner review；
#222显示CLA/Socket成功、1项工作流等待批准，手动CodeRabbit请求因“Review rate limited”尚未完成，
不能标为已经审查通过。GitHub公共API随后达到速率限制，后续状态通过Chrome页面查看。
本地额外验证：macOS/Torch2.10的Norm CPU组10项通过、Gate wrapper CPU组3项通过。

## 23:08 前后的审查跟进

- **#16677 已获 CodeRabbit APPROVED**，仍不是维护者合并或GPU CI通过。
- **#16678 收到 CHANGES_REQUESTED**：当前 Kitchen 0.2.36 不接受新的
  `output_layout` 参数。已回复确认，保持 Draft，并将此意见保留为未解决的
  合入条件；等待 #220 的真实发布版本后再改 pin，不填虚构版本号。
- **#16681 收到 CHANGES_REQUESTED**：普通浮点/完整精度回退展开整张gate，
  会增加显存。已推送 `5552653` 修正：保留原分段，只在INT8分支生成行索引；
  普通推理回退复用新生成的linear输出，避免完整gate和额外完整结果分配。
  hooks仍能看到未加gate的linear输出，梯度路径保留可微计算。
  `251f911` 补充INT8行映射与cast/uncast测试，**真实ComfyUI CPU测试20项通过**，
  Ruff通过；这是本地回归，不是新增GPU质量/性能测量。旧−1.384%对应`3d134bd`。
  修复已回复审查线程，仍等待复核，不能标为审查通过。
- **Kitchen组合分支**已纳入#217～#223，保留每项来源与许可证，当前
  `d6cafade890d96c88f15ba14c4c917a0d120afa3`。六个相关测试模块在
  macOS/Torch2.10 CPU上 **34 passed / 90 skipped**；跳过项需要CUDA。
  默认Inductor缓存下曾发生OpenMP等待/进程异常，使用独立缓存和单线程后，
  真实Inductor用例及整组测试通过，故环境与失败尝试均保留在记录中。

组合CUDA完整重编译脚本和源码包已准备，**尚未在5090 D v2执行**；不以CPU
回归代替GPU正确性、组合加速或RGB/PCM验收。检查结果为时间点快照，未配置
无限期后台监控；后续推送、维护者反馈和CI执行后需要再次检查。
