# 2026-10-01：连接恢复后的完整实测与 PR 推进

在 RTX 5090 D v2 的隔离实验容器中，重新编译并组合 Kitchen #217–#223 与 H3 消费者优化，完成 4 组 AB/BA 交错对照、每组每个版本 2 次正式采样。基础版完整 8 步采样均值 **15.737106 s → 14.200641 s，耗时减少 9.7633%**。16 次正式采样的视频、音频 latent 哈希一致；随后分别解码两版保存的 latent，最终 RGB8 帧与 FP32 PCM 哈希也一致。

本轮推进：**#222 六项审查问题和 #223 一项问题已修复、实测、推送并回复；#16677、#16681 已补充最新提交的 GPU 回归证据。** 目前仍无 PR 合并，外部贡献 CI 需要维护者批准执行。

## 1. 对照口径

| 项目 | 配置 |
|---|---|
| GPU | NVIDIA GeForce RTX 5090 D v2，24 GB |
| Kitchen 基础版 | `19ea55b9ebdaf77942dab36223e1222009d3ce11` |
| ComfyUI 基础版 | `8cfe5e1ecb97512dea8deaac15e1228d7e6feeb1` |
| 运行环境 | Torch 2.12.0+cu130，CUDA 13.0.88，comfy-aimdo 0.5.5 |
| 模型与采样 | Larry v4 INT8，Euler / beta，8 步，seed=42，CFG=1，shift=12/4 |
| 用例 | 768×512，124 帧，14,850 packed tokens；已保存的公开参考条件 |
| 组合代码 | [test/combined-upstream，8710121](https://github.com/Tokha233/comfy-kitchen/tree/8710121) |
| 构建 | CMake/Ninja 完整清洁构建；新增审查修复后按依赖重新编译 |
| 正式样本 | 4 组 AB/BA；每个独立进程先预热，再测 2 次；每版共 8 次 |
| 计时范围 | 完整 `common_ksampler`，包含 8 步 DiT；不包含 condition、VAE、文件导出、模型首次加载 |

这不是历史 30 秒业务用例的 233.99→218.80 s，也不是在历史 r85 上又减少 9.76%。本轮基础版是上述 Kitchen/ComfyUI 固定提交。不同基线和单项百分比不能相加。

## 2. 全组合性能与质量

| 指标 | 基础版 | 全组合 | 变化 |
|---|---:|---:|---:|
| 正式采样均值 | 15.737106 s | 14.200641 s | **−9.7633%** |
| 中位数 | 15.737869 s | 14.200591 s | −9.7680% |
| 最快–最慢 | 15.727525–15.742399 s | 14.190002–14.208853 s | 组内波动较小 |
| Torch 峰值分配 | 1,833,455,616 B | 1,664,301,568 B | **−161.32 MiB** |
| 视频 latent | 固定 SHA256 | 相同 | 逐字节相同 |
| 音频 latent | 固定 SHA256 | 相同 | 逐字节相同 |
| RGB8 帧 | 固定 SHA256 | 相同 | 逐字节相同 |
| FP32 PCM | 固定 SHA256 | 相同 | 逐字节相同 |

按采样时间倒数换算，串行 DiT 阶段的处理能力提高 **10.8197%**。这不是在线并发 QPS 实测；服务端吞吐还受 VAE、排队、调度和输入处理影响。Torch 峰值不包含 aimdo 在 Torch 之外的权重存储。

正式四组测量包含 QKV 的最新修复。随后又加入 #223 的 K/共享内存回退检查，并额外跑一组优化版完整采样，仍约 14.20 s、相同 latent 哈希；额外这组不混入正式 A/B 均值。

解码验收使用两版相同的原生 FP16 视频 VAE 与 FP32 音频 VAE。没有将视频 VAE 误标为 BF16，也没有用本轮数据重新声称 INT8 VAE 的量化误差。解码顺序是基础版先、组合版后，前者包含冷启动，因此 JSON 中的解码耗时不能拿来计算 VAE 加速比。

四类 SHA256：

```text
video latent  8df41570a04e49d6e71924eacfcf5967baddab5762299589e813f6212d12abf1
audio latent  72e7410dc0809f58071a923c9cd8b4c80a2abf3239cafe5910060b4dfb98aefb
RGB8          1e6be78c89d28ed601d29365ae0fbb658ec4eea06a978c4a57b17babcf6155d5
PCM FP32      24d2fce7641d923c17e8ba07fac3701d0c43ff094110f37a56c97116cbfc3e25
```

## 3. 组合里具体运行了什么

| 优化 | 作用 | 每次正式请求的消费者调用 |
|---|---|---:|
| ConvRot 寄存器路径（#217） | 降低支持形状的旋转与量化开销 | 由 shared INT8 backend 分派 |
| SM120 dense attention TMA（#218） | 改进 Q/K/V 搬运与 dense 主循环 | 使用相同 dense INT8 attention 策略 |
| BSHD 输出（#220） | 输出直接采用后续线性层需要的连续布局 | 400 |
| indexed Norm + modulation + ConvRot（#221） | 合并 norm、scale/shift、旋转和量化 | 800 |
| H3 QKV + RMS/RoPE + INT8 preparation（#222） | 合并投影与 Q/K/V 准备，保留全部 token | 400 |
| indexed gate epilogue（#219、#223） | out-proj 与 FC2 的 gate/residual 在 GEMM 收尾完成 | 800 |
| embedding helper（ComfyUI #16677） | embedding 与 packing 结束后自然释放中间结果 | 每步一次 |

采样脚本明确固定 Kitchen INT8 attention 后端，使用实验消费者连接各 API；它不是通用的 ComfyUI attention 分派实现。正式上游消费者仍需保留 ComfyUI 的 patch/hook/训练与 offload 契约。实验脚本中的私有绑定不能直接拷进 ComfyUI core。

## 4. 审查问题与修复

### Kitchen #222：六项问题全部提交修复

提交：[bbfee43](https://github.com/Tokha233/comfy-kitchen/commit/bbfee43)。

1. **SM120 设备不等于二进制含可用 kernel**：实际探测各阶段的 CUDA function attributes；缺少设备镜像时在启动前返回 fallback。保留真实 CUDA 错误的报错，避免误归因为当前阶段。
2. **非连续长序列内存暴涨**：先连续化输入，使其可进入 native；真正不能走 CUTLASS 时，按最多 1024 行处理 INT32/FP32 临时投影，复用 BF16 输出。
3. **fallback 设备上下文**：使用输入所在 device 的作用域调用 CUTLASS。
4. **cta_k 重复规则**：fake shape、元数据和 native eligibility 共用 attention selector。
5. **sdist 漏 CUDA 文件**：补 MANIFEST，实际创建 source distribution，检查 65 个 CUDA 项均在包内。
6. **benchmark 输出名**：修正为文档指向的 `docs/benchmarks/h3-qkv-benchmark.json`。

验证：**24 passed，1 skipped**；跳过双 GPU current-device 测试，因为隔离容器只暴露一张 GPU。SM80-only SASS、无 CUTLASS 两种真实测试构建均返回 unavailable，随后 CUDA 操作成功。不能将此写成已经在其他卡型执行过完整 H3 推理。

额外长序列检查：

| token 数 | 非连续输入额外峰值 | 含参考输出的测试总峰值 | 六类输出一致 | 输入未修改 |
|---:|---:|---:|---|---|
| 87,142 | 5.70 GiB | 8.46 GiB | 是 | 是 |
| 200,000 | 13.07 GiB | 19.28 GiB | 是 | 是 |

修复后 QKV 微基准：87,142 rows **50.770336→42.275801 ms，−16.7313%**；90,461 rows **52.687628→43.912533 ms，−16.6549%**。包含投影与完整 Q/K/V 准备，不包含 dense attention；全部六个 packed tensor 相同。

### Kitchen #223：宽输入与共享内存回退

提交：[6bf0cb0](https://github.com/Tokha233/comfy-kitchen/commit/6bf0cb0)。原 wrapper 接受大 K，却无条件调用有 K/共享内存限制的融合量化器。现在先复用现有 eligibility 判断，不支持时保持 `int8_linear` + indexed gate。

**33 passed**：包含 CPU/CUDA K=16,640、带/不带 SwiGLU、torch.compile 与强制共享内存不足回退。仍依赖 #219。

### ComfyUI #16677、#16681

- #16677 的 helper 提交 `20db23b`：最新 GPU 输出一致，峰值减少 **161.47 MiB**。两轮单项复测都出现约 0.25 s 的孤立高耗时，保留全部记录，暂不宣传该小改动的延迟收益。其主要收益是显存生命周期。
- #16681 的提交 `251f911`：20 项实际容器回归通过；完整 8 步对照的 latent 全部相同。独立 FC2 时间均值受孤立高耗时影响，本轮第二次测量 **15.722069→15.630107 s，−0.5849%**，不覆盖此前另一轮的 1.3844% 数据。
- 两项均已在 PR 中补充真实 GPU 证据；#16677 已有 CodeRabbit APPROVED；#16681 的旧 finding 已确认解决，但整个 PR 仍需依赖发布与维护者审核。

## 5. 扩大历史回归与继承问题

除了 153 项新增/专项回归，还在基础版和组合版分别运行原有 INT8 linear、residual、input activation、attention、RMS/RoPE、GEMM bias 六个测试文件：**两版均为 901 passed、598 skipped、2 failed**，失败项完全相同。

两项失败发生在 Lq=129、Lkv=8193 的普通非 TMA attention，scale=0 或负数：测试的自动 BF16 SDPA 参考返回 NaN。临时改为独立 FP32 `SDPBackend.MATH` 参考，并保留 NRMSE<0.03 门槛，结果变为两版均 **902 passed、598 skipped、1 failed**。零 scale 验证通过，负 scale 实际输出全零、NRMSE=1.0，说明它还存在继承的普通内核数值问题。

诊断没有跳过失败，也没有放宽阈值。临时测试修改保存为证据 patch，未混入性能 PR；已恢复代码。#218 新 TMA 路径自身的零/负 scale 独立参考测试通过。当前 H3 用例使用正常正 scale，不触发上述普通负 scale 路径，但不能将库级回归称为全绿。该继承缺陷需要单独的正确性修复，不纳入本次 9.76% 的速度宣传。

原始日志与独立参考日志均保留在 `evidence/upstream-1001/`。

## 6. 借鉴现有上游 #215 的独立测试

对上游 GEMM raster 调度方案 #215 做两组 AB/BA 进程对照；每个形状 8 个 timing interval、每次 4 次 GEMM，所有输出哈希相同。

| M×N×K | 基础 ms | #215 ms | 耗时减少 |
|---|---:|---:|---:|
| 8192×21504×5376 | 3.379796 | 3.380470 | −0.02% |
| 14850×21504×5376 | 6.144162 | 6.152232 | −0.13% |
| 32700×21504×5376 | 13.661868 | 13.362028 | **2.19%** |
| 87142×21504×5376 | 36.377665 | 35.437028 | **2.59%** |
| 14850×28672×5376 | 8.166766 | 8.167760 | −0.01% |
| 87142×28672×5376 | 47.965631 | 47.186445 | **1.62%** |
| 14850×5376×14336 | 4.148160 | 4.141428 | 0.16% |

结论：值得在业务长序列的 FC1 等剩余普通 GEMM 上继续测试；对当前 14,850-token 公共用例未显示明显优势。它没有并入上述 9.7633% 组合结果，也没有重复提交相同优化 PR。

## 7. 当前可推进程度

| PR | 当前进展 | 剩余条件 |
|---|---|---|
| Kitchen #217 | Open，数据与测试已提交 | 维护者审核、CI 执行批准 |
| Kitchen #218 | Open，已处理审查意见 | 维护者审核、CI 执行批准 |
| Kitchen #219 | Open，已有回归证据 | 维护者审核、CI 执行批准 |
| Kitchen #220 | Draft，组合布局测试通过 | #218 依赖与维护者审核 |
| Kitchen #221 | Draft，153 项组合回归及完整采样已验证 | 正式消费者接口、维护者审核 |
| Kitchen #222 | Draft，六项修复已推送，长序列/源码包/组合验证通过 | 新一轮审查、正式消费者边界 |
| Kitchen #223 | Draft，一项修复已推送 | #219 依赖、新一轮审查 |
| ComfyUI #16677 | Open，已按 kijai 建议改 helper，GPU 数据已回复 | 代码所有者审核、CI 批准 |
| ComfyUI #16678 | Draft | Kitchen #220 发布后 pin 依赖 |
| ComfyUI #16681 | Draft，最新 GPU 数据已回复 | Kitchen #223 发布后 pin 依赖 |

本次查询时 CLA、Socket 成功；Kitchen Build Wheels 和 ComfyUI 外部贡献 workflows 为 `action_required`。这表示维护者需要允许执行，不是代码 CI 失败，也不是测试已通过。10 个 PR 仍 Open，均未合并。推送后的新提交需以随后 CI 为准，部分 API 查询再次遭遇限流，已用页面核对已发布的修复回复。

## 8. 复现与证据

- [全部样本与均值](../evidence/upstream-1001/combined-summary-1001.json)
- [RGB/PCM 解码验收](../evidence/upstream-1001/combined-decode-1001.json)
- [153 项组合回归](../evidence/upstream-1001/combined-final-tests-1001.log)
- [代码版本与证据 SHA256](../evidence/upstream-1001/recovery-validation-manifest.json)
- [隔离实验脚本](../experiments/upstream-1001/README.md)
- [全部 PR 清单与此前单项实验](UPSTREAM-BATCH2.zh-CN.md)

仍需完成业务五例与并发服务验收，验证更长序列下的组合性能；正式消费者应继续分成可审阅的独立 PR。此次没有替换线上容器，也没有占用其他服务的 GPU。
