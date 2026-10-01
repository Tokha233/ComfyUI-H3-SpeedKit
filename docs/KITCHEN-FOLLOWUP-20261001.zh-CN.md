# Kitchen 最新审计、V 量化实测与 PR 推进

2026-10-01 晚，北京时间。主线为 `12389a3`，PyPI 仍为 0.2.36。最近与 NVIDIA 相关的主线增量是 #215 的分带 Stream-K 调度和 #216/W4A8 requant；本轮检查时没有更晚的 NVIDIA 主线提交。开放 #229 是 HIP 对应移植，不是 5090 新内核。

## 本轮已完成

1. 新增正式 [Kitchen #231](https://github.com/Comfy-Org/comfy-kitchen/pull/231)：SM120 大 D128 输入的 V INT8 量化调度。提交 `6dd6f95`，代码、38 项专项测试、公开随机输入 benchmark 和完整测量记录随 PR 提交。
2. [ComfyUI #16678](https://github.com/Comfy-Org/ComfyUI/pull/16678)、[#16681](https://github.com/Comfy-Org/ComfyUI/pull/16681) 均已从 Draft 转为 Ready for review，并修订正文，把最新测试和依赖条件放在正文中。
3. 没有更换线上模型或算子；实验仅使用空闲 GPU0 隔离容器，完成后显存 0 MiB、利用率 0%。

Ready 表示可以开始正式代码审核，不是已获批准或可以立即合并。#16678 仍依赖 Kitchen #220 发布；#16681 仍依赖 #223/#219 发布。两者都须更新 Kitchen pin 并验证正式 wheel。当前 0.2.36 不支持新 API，不能静默认为依赖已满足；原依赖审查条件继续保留。

## 新 PR 的实现和实测

V 量化原先 N>256 一律 512 线程。新分派仅在 SM120、FP16/BF16、D128、N>=12288、B*H>=32、逻辑 V 至少128MiB，且为已测 BHND/BNHD/QKV stride 时使用同一 kernel 的1024线程实例。量化公式、scale、舍入、布局、padding均不变。

| BF16，B1/H56/D128 | 基础 | 新调度 | 耗时减少 |
|---|---:|---:|---:|
| V量化，QKV布局，N14850 | 0.852880 ms | 0.521000 ms | 38.91% |
| 完整attention，QKV布局，N14850 | 13.428368 ms | 13.069504 ms | 2.67% |
| V量化，QKV布局，N87142 | 6.950430 ms | 6.167594 ms | 11.26% |
| Larry v4 INT8完整8步采样 | 15.688698 s | 15.537895 s | **0.9612%** |

采样为768×512、124帧、14850 packed tokens，固定Euler/beta、seed42、CFG1、shift12/4。两臂各2个独立进程，每进程1预热+2正式，次序旧/新/新/旧。8个正式请求的video/audio latent哈希全部一致，Torch峰值均为1,833,455,616字节。单独profiling确认每请求400次1024线程V kernel；profiling不进入性能均值。

两臂都使用main Python与main attention/V源码；其余二进制对象来自相同的既有combined构建，链接时只替换V对象。因此这是隔离的V调度A/B，不是重新清洁构建全部main。性能不含condition、VAE、导出、冷加载，也不是在线QPS。**完整r85已走QKV/V准备融合，此0.96%不能直接加到r85历史收益。**

- 204组探索性shape/dtype/layout调度比较后，选择窄分派；正式36组V量化和6组完整attention对照输出exact。
- 38项专项通过、1项双卡切换测试跳过（隔离容器只见一张GPU）；包括极值/NaN/Inf、边界、不同布局、stream、CUDA Graph。
- 原attention测试加新测试：两臂均457通过、595跳过、2失败。失败同为原零/负scale问题，由已有#224处理；没有隐藏或算作通过。
- 未新增成片MOS/SSIM评测：本轮是算子调度对照，以量化输出和视频/音频latent逐位一致验收。

## PR 当前状态

| PR | 内容 | 当前状态 |
|---|---|---|
| Kitchen #217 | ConvRot寄存器 | Ready，待维护者 |
| Kitchen #218 | D128 dense TMA | Ready，待维护者 |
| Kitchen #219 | indexed gate GEMM | Ready，待维护者 |
| Kitchen #220 | BSHD输出 | Ready，待维护者 |
| Kitchen #221 | Norm/mod/ConvRot融合 | Ready，待维护者 |
| Kitchen #222 | QKV/RMS/RoPE/量化融合 | Ready，复审无新增意见 |
| Kitchen #223 | 输入量化＋gate | Ready，复审问题已修复 |
| Kitchen #224 | 非正scale修复 | Ready，待维护者 |
| Kitchen #227 | D64 attention tile | Ready，复审无新增意见 |
| Kitchen #231 | 本轮V量化调度 | **新建Ready** |
| ComfyUI #16677 | embedding临时引用释放 | **已合并**，10/1 20:04:56 |
| ComfyUI #16678 | BSHD接入 | **本轮转Ready**，发布依赖未解除 |
| ComfyUI #16681 | FC2 gate接入 | **本轮转Ready**，发布依赖未解除 |

合计1已合并、12开放，全部开放PR均Ready，无Draft。此前9个Kitchen PR的CLA与Socket检查成功；Build Wheels仍为action_required，需维护者批准外部workflow，不能当成测试通过。未发现新的人工修改意见。新#231的CLA/Socket也已通过，Build Wheels等待维护者批准，CodeRabbit正在审核。后续状态以GitHub页面为准。

## 下一步还有哪些适合上游

| 方向 | 为什么还有空间 | 提交方式与条件 |
|---|---|---|
| #219 indexed gate 接入 #215 分带调度 | 现有gate仍使用旧LeanStreamK实例；主线普通GEMM已换分带版本 | 先在FC2/outproj真实形状做同epilogue A/B。若有稳定收益，优先补#219，不重复新建#215 |
| #221/#222 的原生ComfyUI消费者 | Kitchen算子已经提交，完整组合试验有效，但主线尚未自动调用 | 分开提交Norm与QKV接入，保留hook、offload、training、attention patch契约；依赖正式API发布 |
| 最后一层live-query | r85已利用最终只输出目标token的语义 | 属于模型层，适合ComfyUI；须覆盖ref2va、mask、hook和完整K/V依赖，不能塞进通用attention默认分派 |
| VAE最终blend后RGB8＋异步D2H | 避免搬运大float图像，r85已有工程收益 | 优先保留SpeedKit节点，或提明确的输出API；普通IMAGE下游仍需float，不能静默换dtype |
| VAE GroupNorm/SiLU/padding | 主线已有融合，剩余是统计、读写与launch开销 | 保持规约和舍入顺序，先profile实际占比；没有新增实测收益前不提速率声明 |

当前最值得推进的是已有核心PR的审核与消费者接入。新V调度是小而独立、已实测的增量；不把Larry、INT8 VAE、#215等上游成果重复作为新贡献。

[脱敏原始数据](../evidence/upstream-vquant-1001/results.json) · [状态快照](../evidence/upstream-vquant-1001/pr-status.json) · [新PR协议](https://github.com/Tokha233/comfy-kitchen/blob/perf/sm120-v-quant-long/docs/benchmarks/sm120-v-quant.md)
