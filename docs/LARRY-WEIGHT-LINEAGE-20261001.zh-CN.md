# Larry v4：INT8 合并来源、BF16 先合并候选与 PR 进展

核查日期：2026-10-01，Asia/Shanghai。本文区分已有部署权重与新的量化候选，不更新已发布的性能基线。

## 1. 当前权重到底如何生成

历史部署权重 `minimax_h3_ref2va_larry8_v4_baked_int8_convrot.safetensors` 的 SHA256 为：

```text
9eccf52e4fe6e764f4aac8cdb6045a58bc86d8458c783c35b7035edca71fec12
```

本次检查可访问机器上四个运行中的服务进程，确认其模型路径、工作流 UNETLoader、权重文件头及运行 manifest 都指向该 Larry baked 权重。没有运行时 LoRA 节点。这个结论限定于实际检查的四个实例及已固定的历史发布产物；不代表远端所有集群实例均已审计，也不代表四实例都已迁移全部 R85 算子。

文件头明确记录：

- 底模 SHA256：`9255f52b6677845ad238f20dfaafa94727053694127ab7f255c048f0f9365779`，对应未挂 LoRA 的 `minimax_h3_ref2va_pruned_int8_convrot.safetensors`。
- Larry adapter SHA256：`7098acf3ee75028fd9fcd948f50fcc8d995057fabb76f86bd3ca2c0ffc58e409`，v4 step600 EMA、pruned ComfyUI、strength=1。
- 合并方法：捕获实际 ComfyUI DynamicVRAM post-cast 有效权重，保留 INT8 ConvRot 序列化格式。
- 208 个 adapter 目标，其中 200 个 INT8 层、8 个浮点层；408 个序列化 tensor 发生变化。103 个 AdaLN tensor 逐字节保留。

因此，既有制作链路是：

```text
已量化的 pruned INT8 底模
  → 按 ConvRot 格式反量化至浮点权重
  → 合并 Larry v4
  → 重新量化，保存实际有效权重
  → 推理直接加载 baked INT8
```

这不是每次请求重新合并。推理中的 INT8 GEMM 输出缩放／反量化属于算子过程，与上述一次性模型制作不同。不能直接把 `B@A` 加到 INT8 存储字节上。

还有一个容易误判的遗留文件：基础镜像内的旧 `h3-baked-model-manifest.json` 仍记录 Turbo4。核查应以实际进程环境、工作流、模型文件头及新的 runtime manifest 交叉确认，不能仅看这个继承文件。

公共工具 [merge_lora.py](../tools/merge_lora.py) 同样支持从 INT8 底模 materialize Comfy 有效权重；其公开复验产物 SHA `422dffed…` 与上述 `9eccf52e…` 不同。两个产物不能因为都叫 Larry INT8 而视作相同数值基线。

## 2. 可以直接从 BF16 合并，并继续得到 pruned INT8

可保持现有底模结构，使用：

```text
minimax_h3_ref2va_pruned_bf16.safetensors
  + minimax_h3_turbo_v4_step600_ema_pruned_comfyui.safetensors
  → 浮点合并并保存 BF16 候选
  → 一次 ConvRot256、逐行 INT8 量化
  → pruned Larry v4 INT8 ConvRot 候选
```

原 BF16 底模已经存在，不需要把现有 INT8 转成 BF16。反量化不会恢复首次量化丢失的信息。

本次候选选择 FP32 计算 `B@A`，与原始权重在 FP32 中相加，然后舍入到该 tensor 原存储类型；INT8 步骤重新计算 scale，`stochastic_rounding=0`。Larry 文件无 alpha tensor，实际缩放为 1，不能额外除以 rank。此合并精度配方不冒充所有 ComfyUI DynamicVRAM 版本的逐位等价路径。

虽然文件名写 BF16，模型仍有原生 FP32／FP16 小模块，尤其 pruned AdaLN curve。转换保留这些模块的类型和结构。INT8 候选保留同样 200 个量化层、932 个 tensor 和逐层量化配置；不强行把所有参数转 INT8。

收益判断：

- 可以避免底模量化后再合并、再量化带来的额外误差来源。
- 同样 INT8 格式、相同层数和 8 步，不应预先承诺更快推理；目标主要是改善权重精度。
- 浮点误差更小不等同于视频、音频质量一定更好。扩散轨迹会放大变化，需要固定 condition、seed、采样和 decoder 做成片对照。
- 本轮模型构建与加载验证不能替代 15 片段生成验收。此前 PDMD 一次量化的改善不直接证明 Larry 也改善。

## 3. `pruned` 不等于 2:4 稀疏加速

底模的 curve-form/pruned 路径用较小的时间嵌入曲线基替代完整 time embedder 和全宽 AdaLN 权重，ComfyUI 原生支持该结构。

[drbaph 的模型说明](https://huggingface.co/drbaph/MiniMax-H3-Turbo-Lora-ComfyUI/blob/bb2bc497cbaca89dadd0bcf1856eed4f8275be20/README.md) 则明确解释了 **Larry LoRA 的 pruned**：

| 项目 | 原 adapter | pruned ComfyUI adapter |
|---|---:|---:|
| LoRA A/B 对 | 259 | 208 |
| Tensor | 518 | 416 |
| 被移除的 AdaLN A/B 对 | — | 51 |

删除原因是原 AdaLN adapter 的维度与 curve-form 底模不匹配，包含 50 个 block 和 final layer。保留了 attention、MLP、token-refiner 的 208 对。它不意味着这些 INT8 矩阵已经满足结构化稀疏要求，也不提供 Sparse Tensor Core 的额外吞吐。

若使用**完整非 pruned BF16 底模＋完整 Larry adapter**，会涉及另外 51 个 AdaLN 更新，属于另一模型方案。之后的 curve-form 转换需重新构建／拟合和验证，不能简单删掉同名 key 并声称与完整合并等价。

## 4. PR 当前进度

**20:35 更新：[#16677](https://github.com/Comfy-Org/ComfyUI/pull/16677) 已于 2026-10-01 20:04:56（北京时间）由 kijai 合并，merge commit `2d6b73283af2447bdd065ece4090b8c6b02a579f8`。当前 1 个已合并、11 个仍开放（9 Ready、2 Draft）。** 下表保留先前查询时的 head，16677 状态以本更新为准。

刚出现的六条失败通知对应旧提交 `20db23b` 的未批准工作流，GitHub 注释均为 `This workflow run required approval but was not approved before it expired.`；六条均无实际 job。新提交 `6349f94` 的执行测试、Unit Tests、Lint、启动、换行及 AI co-author 检查均 success，无需为这些旧通知修改代码。证据见 [PR 邮件核查](../evidence/larry-bf16-1001/pr-mail-expiry.json)。


19:23 查询时 GitHub 状态及讨论：12 个 PR 仍开放，10 个 Ready、2 个 Draft，无合并。部分详细 API 请求遇到匿名限流，已用 GitHub 页面核对相应讨论；缺少的新检查状态未用旧快照填补。

| PR | 最新已推送 head | 状态与反馈 |
|---|---|---|
| [Kitchen #217](https://github.com/Comfy-Org/comfy-kitchen/pull/217) ConvRot | `6e2006d` | Ready，待维护者审核 |
| [Kitchen #218](https://github.com/Comfy-Org/comfy-kitchen/pull/218) D128 TMA attention | `2eb97e0` | Ready，独立 FP32 reference 修复已获机器人确认 |
| [Kitchen #219](https://github.com/Comfy-Org/comfy-kitchen/pull/219) indexed gate epilogue | `5f7290d` | Ready，待维护者审核 |
| [Kitchen #220](https://github.com/Comfy-Org/comfy-kitchen/pull/220) BSHD | `13bbea6` | Ready，最新机器人复审无新增可操作意见 |
| [Kitchen #221](https://github.com/Comfy-Org/comfy-kitchen/pull/221) norm/mod/quant | `f66dda5` | Ready，已推送修复，待维护者审核 |
| [Kitchen #222](https://github.com/Comfy-Org/comfy-kitchen/pull/222) QKV 融合 | `bbfee43` | Ready；六项反馈已有代码／GPU 验证，本轮重新触发机器人复审 |
| [Kitchen #223](https://github.com/Comfy-Org/comfy-kitchen/pull/223) 输入量化＋gate | `6bf0cb0` | Ready，ConvRot 超界回退意见已标记解决 |
| [Kitchen #224](https://github.com/Comfy-Org/comfy-kitchen/pull/224) 非正 scale | `96b7062` | Ready，机器人无新增可操作意见 |
| [Kitchen #227](https://github.com/Comfy-Org/comfy-kitchen/pull/227) D64 小 query tile | `5ff2a31` | Ready；本轮复审已完成，无新增可操作意见 |
| [ComfyUI #16677](https://github.com/Comfy-Org/ComfyUI/pull/16677) embedding 生命周期 | `6349f94`（后由维护者同步 master） | **已合并**；kijai 要求 helper scope 和精简测试，两项均已修改推送；新提交 CI 通过 |
| [ComfyUI #16678](https://github.com/Comfy-Org/ComfyUI/pull/16678) BSHD consumer | `c0ac30d` | Draft，等待 #220 API 发布与依赖 pin 更新 |
| [ComfyUI #16681](https://github.com/Comfy-Org/ComfyUI/pull/16681) gate consumer | `251f911` | Draft，等待 #223／#219 API 发布与依赖 pin 更新 |

#227 的最新远端 head 也经 `git ls-remote` 确认。CLA／Socket success；Build Wheels `action_required` 表示需要上游维护者批准外部工作流，不是 GPU 测试失败。机器人通过不等于维护者批准，也不等于合并。

#227 已提交实测：代表 D64 attention 约快 4.6–5.8%；独立进程完整 VAE decode 9.962410 → 9.866619 秒，减少 0.9615%，输出 exact。这是 decoder 数据，不能写成整个视频生成提高 5%。

已有生成质量与性能数据保持原有权重基线。


## 5. 本轮实际构建结果

已在隔离的 RTX 5090 D v2 实验容器生成两份候选，输入文件保持只读，没有替换服务模型。完整文件逐 tensor 回读并校验 SHA，构建退出码 0。

| 产物 | 大小 | Tensor 数 | SHA256 |
|---|---:|---:|---|
| Larry v4 pruned BF16 | 40,225,724,720 bytes | 532 | `75b7672ddadcb4e61cbee76d889cc79fa2a3a74693d5655335dc3cbe3aa7f236` |
| Larry v4 pruned 一次 INT8 ConvRot | 20,970,380,152 bytes | 932 | `0fe2d7dad437e591373ab6912e9716eada5cd3c46d89ef0f09d6cfbb4784b055` |

208 个 adapter 完整覆盖，200 层 INT8，103 个 AdaLN tensor 与原 INT8 模板一致。BF16 底模 SHA 为 `37c0da793e20ca735272ec2be655f08a2e10f97a3ec8fdfb40f5b39a736ed6fe`。精确构建参数见 [manifest.json](../evidence/larry-bf16-1001/manifest.json)。权重文件不随代码仓库再分发。

本轮完成的是权重构建、结构及加载验证；尚未完成该 Larry 新候选的业务生成质量和速度对照，不能宣称优于当前部署。


ComfyUI 实际加载成功：200 个量化 Linear、`adaln_curve=true`、无运行时 LoRA patch；所有 key、shape、dtype 与既有 baked INT8 完全对应。

取前／中／后三个代表层，以本次 BF16 合并候选为参考，比较反量化权重的相对 L2 误差：

| 权重 | 现有 baked INT8 | BF16 先合并、一次 INT8 | 误差减少 |
|---|---:|---:|---:|
| `blocks.0.attn.qkv_proj.weight` | 1.3500% | 0.9723% | 27.98% |
| `blocks.25.mlp.fc2.weight` | 1.4028% | 1.0191% | 27.36% |
| `blocks.49.attn.out_proj.weight` | 1.5899% | 1.1913% | 25.07% |

这些是三个权重矩阵的重构误差，不是全模型误差，也不是视频 SSIM、音频相似度或主观提升百分比。逐层数据见 [validation.json](../evidence/larry-bf16-1001/validation.json)。
