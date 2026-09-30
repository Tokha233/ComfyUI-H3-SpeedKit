# 2026-10-01 PR 跟进

北京时间01:30前后检查：10个PR均Open、尚未合并。新增一条人工反馈，已修复并回复；
此前FC2显存回退问题已获CodeRabbit复核确认。状态为检查时快照。

## 新增反馈与处理

### ComfyUI #16677：kijai认可方向，建议用函数作用域释放中间结果

[kijai原话](https://github.com/Comfy-Org/ComfyUI/pull/16677#issuecomment-5914007087)：
“This makes sense”，但多条`del`不够整洁，建议把embedding和packing移到helper，
让中间引用随函数返回自然释放。

已推送`20db23b`：把原有计算顺序原样移入`_embed_and_pack`，只返回packed tensor，
移除显式`del`。没有修改分段、数值计算、attention或模型参数加载。

- 实际ComfyUI checkout：**25 CPU tests passed**，涵盖弱引用释放检查、
  FP32/BF16、参考图音频、keyframe、原始/预编码文本、梯度及原有H3/混合精度回归。
- 缩小的真实H3模型：**24组新旧`_forward`对照全部视频/音频输出逐值相等**，
  包含遮罩和不整除patch的空间输入。
- Ruff、`git diff --check`通过；已回复kijai。新版`20db23b`随后再次获得
  CodeRabbit APPROVED，无新增可操作意见；仍需代码所有者复核与CI批准。
- 历史5090 D v2减少161.47 MiB的显存数据对应`a983f80d`；新版helper的GPU复测待连接恢复。
  CPU弱引用与输出检查不代替GPU峰值测量。

### ComfyUI #16681：显存回退修复已确认

[CodeRabbit复核](https://github.com/Comfy-Org/ComfyUI/pull/16681#discussion_r4146131688)
明确确认`251f911`解决原问题，并关闭审查线程：分段gate保留紧凑表示、推理回退
复用linear输出、hook/梯度单独处理、INT8行映射及cast/uncast保留。

该确认只针对这条finding；不等于整个PR已获最终合并批准。仍为Draft，等待
Kitchen #223发布、依赖pin和GPU复测。现有20项CPU测试记录保持不变。

## 全部PR状态

| PR | 当前反馈 / 剩余条件 |
|---|---|
| [Kitchen #217](https://github.com/Comfy-Org/comfy-kitchen/pull/217) | Open；未见新人工反馈；Build Wheels待维护者批准 |
| [Kitchen #218](https://github.com/Comfy-Org/comfy-kitchen/pull/218) | Open；已提交非正scale审查修复；无新问题 |
| [Kitchen #219](https://github.com/Comfy-Org/comfy-kitchen/pull/219) | Open；回退布局/许可证修复已提交；无新问题 |
| [Kitchen #220](https://github.com/Comfy-Org/comfy-kitchen/pull/220) | Draft；依赖#218；既有两项审查意见已处理 |
| [Kitchen #221](https://github.com/Comfy-Org/comfy-kitchen/pull/221) | Draft；既有三项意见已处理；正式消费者/组合验收待完成 |
| [Kitchen #222](https://github.com/Comfy-Org/comfy-kitchen/pull/222) | Draft；冷却后重试成功启动，页面显示Review in progress |
| [Kitchen #223](https://github.com/Comfy-Org/comfy-kitchen/pull/223) | Draft；依赖#219；冷却后重试，页面仍显示审查限流 |
| [ComfyUI #16677](https://github.com/Comfy-Org/ComfyUI/pull/16677) | Open；已按kijai意见修复；新head获CodeRabbit批准，等代码所有者/CI |
| [ComfyUI #16678](https://github.com/Comfy-Org/ComfyUI/pull/16678) | Draft；依赖Kitchen #220正式发布；pin意见保留未解决 |
| [ComfyUI #16681](https://github.com/Comfy-Org/ComfyUI/pull/16681) | Draft；显存finding已确认解决；依赖发布/GPU复测仍待完成 |

检查到的CLA和Socket checks成功。按各PR当前head查询，Kitchen Build Wheels与
ComfyUI六类外部贡献工作流显示`action_required`，即等待维护者批准执行，
不是代码测试失败，也不是通过。#220本轮actions查询触发API限流，沿用之前状态，
不冒充本次已重新查询。推送后的新head以随后CI结果为准。

## 尚待实测

h31先因最大会话时长断开，随后提示空闲超时，本次无法进行GPU复测。
组合CUDA构建、全组合采样和RGB/PCM验收仍待完成，不把单项收益相加。
没有配置无限期后台监控，也没有修改现有线上部署。

- [本次状态、人工反馈及workflow证据](../evidence/upstream-1001/status.json)
- [24组小模型逐值对照](../evidence/upstream-1001/embedding-forward-equivalence.json)
- [25项CPU测试记录](../evidence/upstream-1001/embedding-validation.json)
- [上批GPU数据与完整PR清单](UPSTREAM-BATCH2.zh-CN.md)
