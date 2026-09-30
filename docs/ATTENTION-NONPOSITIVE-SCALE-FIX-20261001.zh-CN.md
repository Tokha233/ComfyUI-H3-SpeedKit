# 普通 INT8 attention 零/负 scale 异常：原因、修复与实测

记录时间：2026-10-01，Asia/Shanghai。独立修复：[Kitchen PR #224](https://github.com/Comfy-Org/comfy-kitchen/pull/224)，提交 `96b7062`，基于原版 `19ea55b`。本次未替换线上服务。

## 当前业务要做什么

当前已验证的 H3 D128 attention 使用正系数 `1/sqrt(128) ≈ 0.08838835`，不触发本次普通内核的零/负系数缺陷。维持现有 Larry v4、采样参数和部署，无需因此回滚、改 LoRA 强度或改 CFG。这里的 scale 只指 attention 的 QK 分数缩放系数，与 LoRA 强度、CFG、量化比例都不同。

自定义代码确实需要非正 scale 时，可使用本 PR 源码构建并先验证，或在该调用使用明确选择 FP32 math 的 SDPA。不要简单地将负数改为正数，这会改变 attention 语义。普通自动选择的 BF16 SDPA 在本环境也可能出现 NaN，不能当作已验证的替代方案。

## 为什么原来有两项失败

这是两个叠加的问题，原版和优化组合都能复现：

1. **参考实现异常**：原测试自动选择 BF16 SDPA，在 `scale=0` 与负 scale 的长 KV 用例返回 NaN，无法作为有效精度参考。本次改为独立 FP32 `SDPBackend.MATH`，明确断言参考结果有限；保留原来的 NRMSE < 0.03 精度门槛。
2. **普通 CUDA INT8 内核的边界错误**：为了融合运算，先对原始 QK 求最大值再乘 scale。正 scale 时成立，负 scale 时大小关系反转。同时，末尾补齐位置先填负的大数，再乘负 scale，会变成巨大的正数；有效位置的概率被压到零，补齐的 V 又为零，最终输出全零。零 scale 会将屏蔽值变成零，让补齐位置参与归一化。

原问题形状：B=1、H=4、Lq=129、Lkv=8193、D=128，BF16，seed 31。它使用普通 attention 路径；此前 #218 的长序列 TMA 路径已有非正系数处理。

## 如何修复

- Launcher 仅在无 mask 且 `scale <= 0` 时选择独立 CUDA 模板实例。
- 该实例先将 QK 转成 FP32 并乘 scale，再求最大值；补齐 mask 在缩放后施加。
- 正 scale 保留原来的计算路径与指令级运算顺序，不在其主循环里添加判断。
- 已有 custom/prepared mask 路由保持不变。
- 新增 96 组回归：FP16/BF16、D64/128/256、GQA，KV 长度 1/64/65/512/513/1024/1025/8193，零与负 scale；每组检查直接和预量化两种调用。

## 5090 D v2 实测

环境：RTX 5090 D v2，CUDA 13.0.88，PyTorch 2.12.0+cu130。独立修复重编唯一包含修改头文件的 launcher TU，其余对象与原版一致；组合版使用 CMake 依赖重编，实际重编 launcher、TMA 并重新链接。

| 原失败用例 | 原版 NRMSE | 修复后 NRMSE | 原有门槛 |
|---|---:|---:|---:|
| scale = 0 | 0.016976889 | 0.008582708 | < 0.03 |
| scale = -1/sqrt(128) | 1.000000000，输出全零 | 0.016013842 | < 0.03 |

该指标是 attention 算子相对 FP32 math 参考的误差，不是视频 SSIM 或音频相似度。

独立修复在原来六个 INT8、RMS/RoPE、GEMM bias 测试文件上连同新用例：**999 passed、598 skipped、0 failed**。组合版还运行七个优化专项测试文件：**1152 passed、599 skipped、0 failed**。这些是列出的回归集合，不代表整库所有平台都已验证。跳过项保留原有后端/能力条件，包含单卡环境不能运行的双卡项；没有跳过本次失败或降低精度阈值。

### 正系数路径是否受影响

48 组不同形状/精度的输出 SHA-256，以及四组 benchmark 的输出，在 AB/BA 四个独立进程中全部一致。

以下耗时为两轮 AB/BA、每个版本每形状 16 个记录的合并中位数；每条记录计时 10 次 CUDA Graph replay：

| 正 scale shape：B,Hq,Hkv,Lq,Lkv,D | 原版 ms | 修复版 ms | 变化 |
|---|---:|---:|---:|
| 1,4,4,129,8193,128 | 0.171090 | 0.172750 | +0.971% |
| 1,16,16,4096,4096,128 | 0.273707 | 0.271405 | -0.841% |
| 1,56,56,14850,14850,128 | 11.904987 | 11.909430 | +0.0373% |
| 1,42,42,32700,32700,128 | 42.822321 | 42.825537 | +0.0075% |

未观察到有意义的正系数性能回退，不将这些微小波动宣传为加速。

### 完整 H3 采样

Larry v4 INT8、Euler/beta 8 步、CFG=1、seed=42、shift=12/4，768×512、124 帧、14850 tokens。每个版本独立进程、1 次预热、2 次正式采样。

| 版本 | 两次正式采样秒数 | 平均秒数 |
|---|---|---:|
| Kitchen 19ea55b | 15.731729 / 15.735235 | 15.733482 |
| 原版加独立修复 96b7062 | 15.739764 / 15.739507 | 15.739635 |

差异约 +6.15 ms（+0.0391%）；本次是正确性与性能冒烟检查，样本量不足以支持吞吐结论。两版全部预热/正式结果的视频与音频 latent 哈希均一致：

- 视频：`8df41570a04e49d6e71924eacfcf5967baddab5762299589e813f6212d12abf1`
- 音频：`72e7410dc0809f58071a923c9cd8b4c80a2abf3239cafe5910060b4dfb98aefb`

本次没有重新解码视频/音频，因此以 latent 逐字节一致性为证据，不新增 SSIM/听感结论。之前同 latent 的 VAE 解码一致性记录见恢复测试报告。

### 与已有优化组合的兼容性

组合分支 [2be5d18](https://github.com/Tokha233/comfy-kitchen/tree/2be5d18) 已包含本修复。依赖重编后的 13 个测试文件共 **1152 passed、599 skipped、0 failed**。完整 H3 采样 1 次预热、2 次正式结果全部保持同样的视频/音频 latent 哈希，正式耗时 14.200106 / 14.205494 s，均值 **14.202800 s**。这与此前约 14.20 s 的组合结果一致；没有将本次兼容性样本混入原先 8 次正式 AB/BA 统计。

- [组合回归日志](../evidence/upstream-1001/combined-nonpositive-regression.log)
- [组合完整采样](../evidence/upstream-1001/nonpositive-sampler-combined-6.json)
- [CMake 依赖重编日志](../evidence/upstream-1001/build-combined-nonpositive.log)

## 证据与边界

- [独立回归日志](../evidence/upstream-1001/nonpositive-regression.log)
- [原失败用例与形状检查：原版](../evidence/upstream-1001/nonpositive-probe-base-0.json)
- [原失败用例与形状检查：修复版](../evidence/upstream-1001/nonpositive-probe-negative-scale-0.json)
- [反向顺序原版记录](../evidence/upstream-1001/nonpositive-probe-base-1.json)
- [反向顺序修复版记录](../evidence/upstream-1001/nonpositive-probe-negative-scale-1.json)
- [原版完整采样](../evidence/upstream-1001/nonpositive-sampler-base-0.json)
- [修复后完整采样](../evidence/upstream-1001/nonpositive-sampler-negative-scale-0.json)
- [编译 manifest](../evidence/upstream-1001/nonpositive-build-manifest.json)
- [复现脚本目录](../experiments/upstream-1001/)

PR #224 已创建为 Open；页面显示外部贡献 workflow 等待维护者批准运行。该状态是审核权限要求，不是本地回归失败，也不表示 GitHub 全平台 CI 已通过。其他 GPU 架构尚未运行验证。此前 [9.7633% 组合加速](UPSTREAM-RECOVERY-TESTS-20261001.zh-CN.md) 是另外一套正式 AB/BA 数据，本修复不产生额外加速收益，也不能叠加计算。
