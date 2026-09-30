# Kitchen 上游 PR 提交与实测记录（2026-09-30）

已用 Tokha233 账号正式提交三个独立 PR，均基于 Kitchen main
`19ea55b9ebdaf77942dab36223e1222009d3ce11`（0.2.36）。代码、许可证说明、
DCO sign-off、原始计时、复现脚本和回归测试均在对应 PR 内。

| PR | 改动 | RTX 5090 D v2 实测 |
|---|---|---|
| [#217](https://github.com/Comfy-Org/comfy-kitchen/pull/217) | BF16 ConvRot256 寄存器实现 | 大矩阵算子耗时降低 7.76%–9.22%；4096 行形状降低 51.55%–57.62% |
| [#218](https://github.com/Comfy-Org/comfy-kitchen/pull/218) | SM120 D128 dense attention TMA 预取与 cached Q | 修复后四个形状降低 0.74%–4.35%，默认正 scale 输出逐位一致 |
| [#219](https://github.com/Comfy-Org/comfy-kitchen/pull/219) | INT8 GEMM indexed gate + residual epilogue | 大形状完整算子链降低 11.49%–18.70%，包含公开 API 开销 |

## 实测口径

- 单卡 RTX 5090 D v2，PyTorch 2.12.0+cu130，CUDA 13.0.88。
- 上游固定 CUTLASS `d4b4b494c3c51bf6507e7ab09fbafd1e9fa94f39`。
- ConvRot/attention 用 CUDA Graph + CUDA event；gate 公共 API 用交错执行、CUDA event，包含调用/验证开销。
- 算子收益不等于整段部署收益，不叠加宣传。没有更改 LoRA、步数、量化格式。
- 上述测试输出逐位相等。gate 测试为业务形状的合成输入，不宣称视频/音频主观质量提升。
- gate 需要消费者显式接入；以下为最初提交口径，消费者验证见文末。M=1024 两个形状走原 GEMM 链，额外 API 开销约 0.5%–0.9%。

## 完整 DiT 核对

固定公开条件，Larry v4 INT8、768×512、124 请求帧、Euler/beta 8 步、CFG1、seed42。
每个进程一次热身、三次正式采样；不包含 condition、VAE、封装。

| 分支 | 正式平均秒数 | 相比基线 |
|---|---:|---:|
| Kitchen main | 15.860485 | — |
| 单独 ConvRot PR | 15.815302 | 降低 0.285% |
| 单独 attention PR | 15.839752 | 降低 0.131% |

两路 latent 在所有轮次 SHA256 一致。这些顺序进程测试中的整段差异较小，
不能当成稳定的端到端收益；对外主要采用独立算子实测。

## 验证

- ConvRot 新增测试 + 现有 input_act：80 passed，3 skipped。
- indexed gate + 现有 residual：36 passed，1 skipped。
- attention 现有测试 + 最初新增测试：421 passed，594 skipped，2 failed。
  两个失败均在原始主线上复现：当前 Torch/CUDA 的 SDPA 对 scale=0/负数产生 NaN 参考值，
  对应 Lq=129 不会触发新 TMA 路径。保留并披露失败，没有修改断言隐藏它们。
- 扩大的 TMA stream/graph 测试（含 scale=0/负数）：6 passed。
- 三个完整扩展均构建，SM120a 独立 CMake target 构建通过；Ruff、git diff --check 通过。
- 原始样本与 source SHA256 位于各 PR 的 `docs/benchmarks/`。

## 2026-09-30 20:05 检查与修复

三个 PR 均 Open，未合并，无合并冲突。GitHub 的红色失败均来自
`cla-assistant`：日志明确为作者尚未签署 CLA。Socket 两项安全检查通过。
Build Wheels 的状态为 `action_required`：新贡献者的工作流等待维护者批准，
目前没有 GitHub 编译失败的结果，也不能说跨平台 CI 已通过。

| PR | 当前审查情况 | 本轮动作 |
|---|---|---|
| #217 | CodeRabbit 完成，无 actionable comments；未有人类批准 | 无需修改代码 |
| #218 | 原审查意见已被 CodeRabbit 标记 Addressed；新提交全面复审仍受额度限制 | 已修复并推送 `2eb97e0948180e568d722dab65ca8c2db1fa0790`，正文与审查回复已更新 |
| #219 | CodeRabbit 审查额度耗尽；success 状态不代表完成审查 | 已发 `@coderabbitai review`，机器人仍回复 rate limited |

### #218 本轮发现与修复

新增独立 FP32 math SDPA 参考，保持相同 scale、复制 GQA heads，只分块 query，
每个 query 仍看完整的 keys。强制数学后端以避免当前环境中优化 SDPA 返回 NaN。
全部 9 组覆盖默认正 scale、零 scale、负 scale、完整/不完整 tile、GQA、
重复调用、独立 stream 和 CUDA Graph 的测试通过。

独立测试揭示之前重复测试未发现的问题：原始 main 和最初 PR 对两组负 scale
不完整 tile 输入均输出全零，NRMSE 为 1.0。根因是先求 max 再乘 scale，且
padding 屏蔽值也被负数翻转。修复后的 TMA 路径对非正 scale 先缩放，再求 max
并屏蔽 padding；默认正 scale 保持原计算顺序。

| scale | 修复后 NRMSE | 验收门槛 |
|---|---:|---:|
| 默认正数 | 0.016187–0.016302 | <0.03 |
| 零 | 0.009147–0.009456 | <0.03 |
| 负数 | 0.016199–0.016305 | <0.03 |

完整 attention suite：428 passed，594 skipped，2 failed。剩余两个仍为
原主线 Lq129/Lkv8193 的非 TMA 路径；优化 SDPA 参考返回 NaN，同时原 kernel
也保留负 scale 问题。本 PR 修复新增 TMA 范围，没有改动旧测试掩盖失败。
因此此前把旧失败仅归因为参考 NaN 的说明不够完整，现已修正。

完整扩展与 CMake SM120a target 重新构建通过。默认正 scale 性能复测：

| B,Hq,Hkv,Lq,Lkv | 主线 ms | 修复后 ms | 降低耗时 | BF16 输出逐位一致 |
|---|---:|---:|---:|---|
| 1,8,2,8192,8193 | 0.589357 | 0.567477 | 3.71% | 是 |
| 1,8,8,16384,16384 | 2.130446 | 2.083659 | 2.20% | 是 |
| 1,42,42,32700,32700 | 42.821834 | 42.503345 | 0.74% | 是 |
| 2,4,2,8197,9217 | 0.713445 | 0.682398 | 4.35% | 是 |

复测完整 DiT 三次均值 15.834297 秒，音视频 latent SHA256 均与原基线一致。
这次完整 DiT 为正确性回归，不把与早先计时的微小差异当作新加速收益。
原始数值、计时样本和新的源码 SHA256 已随 #218 修复提交。
已在[对应审查线程](https://github.com/Comfy-Org/comfy-kitchen/pull/218#discussion_r4144354011)
回复实测结果，并请求维护者批准 Build Wheels 运行；PR 正文经 API 回读核对。
CodeRabbit 已在原意见中追加 `✅ Addressed in commit 2eb97e0`。
新提交的 [Build Wheels](https://github.com/Comfy-Org/comfy-kitchen/actions/runs/36711701464)
仍为 `action_required`，等待维护者批准。

### 当时的待办与后续状态

1. 上游维护者批准 Build Wheels 运行；代码作者不能代替上游批准。
2. CLA 当时未签；用户随后明确授权，20:15 已完成签署，三个 PR 均通过。
3. #218 新提交与 #219 等待 CodeRabbit 额度恢复或上游发起复审。三个 PR 仍需维护者审阅决定合并。

签署原句为：`I have read and agree to the Contributor License Agreement`。

## 复用位置

本次完整工作目录：`~/.agent-reach/h3-kitchen-upstream-prs-20260930/`。
包含三个 worktree、原始实测、构建/测试日志索引、PR 正文和 GitHub API 回读记录。
远程隔离目录：容器 `/tmp/h3-kitchen-prs-0930`；实验仅使用空闲 GPU0/3。
原 `/study` 已满，未删除历史实验或改动其他服务。

## 2026-09-30 20:23 CLA 已签署

用户明确授权后，已用 Tokha233 发表上游指定签署评论：
[签署记录](https://github.com/Comfy-Org/comfy-kitchen/pull/218#issuecomment-5911109674)，
时间 20:15:28（北京时间）。随后在 #217/#219 发出 `recheck`。
三个 PR 的 CLA 机器人均已确认 `All contributors have signed the CLA`，
此前「CLA 未签署、需要授权」状态已解除。Build Wheels 仍需上游维护者批准运行，
CLA 通过不表示代码已合并或跨平台 CI 已完成。

贡献评价：这三个 PR 是有实质技术内容、可复现的中等规模底层工程贡献。
#219 大形状算子链耗时降低 11.49%–18.70%，实际性能价值最突出，但仍需消费者接入
和完整模型 A/B；#217 改动集中且输出逐位相等，适合先推动落地；#218 技术难度高，
当前性能增益为 0.74%–4.35%，需继续证明复杂度值得维护，并保留数值正确性测试。
不能将各算子百分比相加或把整套 SpeedKit 的累计收益归到三个独立 PR。


## 2026-09-30 20:44 状态与处理时间判断

已用 GitHub API 和登录页面重新核对：三个 PR 均 Open、未合并、无冲突，
CLA 与两项 Socket 检查通过。#217 CodeRabbit 已完成；#218 原问题已标记
Addressed，但新提交全面复审与 #219 仍受机器人额度限制。尚未收到维护者人工
review 或 approval。Build Wheels 等待维护者批准，不能当成编译失败或通过。
CLA 机器人的 “ready to be merged” 只说明签署要求满足，不代表上游已经接受代码。

三个 PR 在北京时间 19:15 左右创建，当前仅约一个半小时。
以下时间来自 Kitchen 实际 PR 的创建与合并时间：

| 历史 PR | 内容 | 创建到合并 |
|---|---|---:|
| [#211](https://github.com/Comfy-Org/comfy-kitchen/pull/211) | 小范围 SM75 条件修复 | 18.7 小时 |
| [#192](https://github.com/Comfy-Org/comfy-kitchen/pull/192) | conv3d / GroupNorm 接口与布局 | 29.0 小时 |
| [#167](https://github.com/Comfy-Org/comfy-kitchen/pull/167) | H3 VAE kernels | 4.3 天 |
| [#151](https://github.com/Comfy-Org/comfy-kitchen/pull/151) | Ascend INT8 后端 | 21.4 天 |

17:57 的历史快照中，#183/#184 已等约两周，#142 已等约一个月。
最近已合并样本会遗漏这些长期开放 PR，且包含维护者自己的改动，因此不能用
“已合并 PR 的中位数”承诺新贡献者的等待时间。

规划预期：首轮人工反馈可以先按 1–3 个工作日安排，但也可能更久；
小改动可能几天，专用 CUDA 内核/API 可按 1–2 周或更久准备。这是工作安排建议，
不是维护者承诺，也不保证最终合并。没有消息时等 3–5 个工作日，再带新的
复现证据留一次简短跟进，不反复催促或无意义推送。

推进顺序：#217 改动集中，最适合先争取落地；#219 补可用的 ComfyUI 消费者和
完整 A/B，降低集成不确定性；#218 继续保留独立数学参考、清楚披露非 TMA 旧问题，
由维护者评估小幅性能收益是否值得新增内核维护成本。

## #219 真实消费者补充实测

已新增独立 ComfyUI 节点 **H3 SpeedKit · Indexed Gate (Kitchen PR 219)**，
仅替换 outproj / FC2 的 GEMM + indexed gate + residual；无需 R85 的独立动态库。
普通 Optimize DiT 节点保持原实现。两者二选一，不能把收益叠加。

公开输入：Larry v4 INT8、768×512、124 请求帧、14,850 packed tokens、8 步、
CFG1、Euler/beta、seed42。两次独立启动，每次一对热身和四对交错正式采样。
两臂使用相同 attention、视频 VAE、CPU RGB 转换、GPU FP32 音频 VAE 和 MP4 输出。

| 对比 | 基础路径 | 仅 #219 | 耗时减少 |
|---|---:|---:|---:|
| 第一组 DiT | 15.902738 s | 15.644471 s | 1.62% |
| 第一组完整本地文件周期 | 19.206741 s | 19.020499 s | 0.97% |
| 第二组 DiT | 15.900018 s | 15.621659 s | 1.75% |
| 第二组完整本地文件周期 | 19.159501 s | 18.960541 s | 1.04% |
| 两组 DiT 均值 | 15.901378 s | 15.633065 s | **1.69%** |
| 两组完整本地文件周期均值 | 19.183121 s | 18.990520 s | **1.00%** |

16 次正式请求的 video/audio latent、RGB8、PCM 全部 SHA256 一致。
最终代码逐请求确认 400 次 outproj、400 次 FC2，零回退，正式计时零首次验证。
首次遇到两种 modulation layout 时另做 100 个 block 原路径对照；热身单独保留。
更早的接入诊断曾对一个完整 8 步请求的全部 400 个 block 做逐位对照，也通过。

保留全部样本，包括一个音频 decode 波动；不宣称每次请求都快 1%。
这里只证明一个公开输入的平均收益，没有重新跑全部业务场景，也没有证明持续吞吐。
PyTorch allocator 记录的峰值增加约 152 MiB；该计数不包含外部分配，不能当总显存。
原始数据、源码 SHA、安装和复现方法见 [Indexed Gate 文档](INDEXED-GATE.md)
和[完整样本](../evidence/indexed-gate-consumer.json)。

本轮也补齐 clone 隔离、Comfy 配置复制后的 wrapper 注册、冲突 patch、无效
segment、fallback 和异常清理测试；CPU 10 项通过，真实 Comfy V3 schema 与
disabled 分支通过，完整模型走正常采样器完成输出。新节点尚未单独通过前端
画布导入或完整 PromptExecutor 工作流验收，不把旧节点的验收结果套用到它。
