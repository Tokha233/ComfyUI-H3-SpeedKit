# 2026-09-30 上游调研与可借鉴方案

来源均为本次直接抓取的 GitHub/Hugging Face 主仓库或官方 ComfyUI 文档。搜索服务用于发现，结论优先依据原始文件。时间按 UTC 保存，表中日期换算为北京时间。抓取发生在 9 月 30 日凌晨；不代表当天后续没有更新。

## Comfy Kitchen：应当立即纳入发布前基线比较

`pyproject.toml`：0.2.36；main head `888b13e2c0e721f6576fe351a2ad79894b1c451f`。不是根据 tag、镜像 metadata 或 pip list 推断。GitHub release 列表本次为空，不应据此说它没有发布渠道；README 使用 PyPI。

| 时间（北京） | 主来源 | 核实到的变化 | 对 H3 SpeedKit 的意义 |
|---|---|---|---|
| 9/27 17:11 | [#208](https://github.com/Comfy-Org/comfy-kitchen/pull/208) | CUDA Q cached helper；8-column PV fragments 流水；Q tile 选择；HIP/mask 路径及测试也有改动 | 对照自己的 dense、Q cache 与流水；不能把整个 PR 的所有改动都归为 SM120 H3 收益 |
| 9/28 05:59 | [d73ec16](https://github.com/Comfy-Org/comfy-kitchen/commit/d73ec16853489c0c9831f4b57051609c42039c48) | commit 标题为修复性能回退 | 提醒 shape 范围内验证的重要性；本轮未逐行审计该提交 |
| 9/28 13:43 | [0432be9](https://github.com/Comfy-Org/comfy-kitchen/commit/0432be9a3ac8193e4c516ad167560adabab3f146) | SM120、D128、Q/K 4096–4608、head≤32 分支；30–32 heads 查询 SM 数，比较 Q64/Q128 的 CTA rounds | 真实长 Ref2VA 的 Q 长度常远大于此范围，不能直接套 PR 的 tile 优势 |
| 9/29 04:13 | [b54f851](https://github.com/Comfy-Org/comfy-kitchen/commit/b54f85128ee4246cb42b9888d1e0ae1a0d829049) | batch/head 基址支持 64-bit，按需选择 offset 类型；head 内仍有索引限制 | 可减少长序列尺寸限制；主要是覆盖/正确性变化，不等于一定提速 |
| 9/29 08:16 | [888b13e](https://github.com/Comfy-Org/comfy-kitchen/commit/888b13e2c0e721f6576fe351a2ad79894b1c451f) | 版本更新至 0.2.36 | 正式 benchmark 需要锁定这个或后续明确 commit |

README 明确 CUDA wheel 支持 Linux x86_64 与 Windows x64，另有 pure-Python wheel。Python 3.12+ 使用 Stable ABI；这只是 Python 绑定 ABI，不能据此声称任意 CUDA/驱动/Torch/模型都兼容。值得参考 nanobind/DLPack、能力分派与源码构建，而不是直接复制它的整个支持矩阵。

## H3-Optimizations：直接竞品，也可能是集成伙伴

[仓库](https://github.com/Zironic/H3-Optimizations)，当前 0.2.45，main `862774944a331bc1a66cee1cf805b94bd138ebbe`。这次核实到的最新 commit 仍为 9 月 25 日凌晨北京时的 FastH3 VSA 发布，不是 9 月 30 日新增一个版本。

它已经提供 Memory Optimization、Sparse Attention、Sparse Advanced、FastH3 VSA、AIMDO Residency Limiter。通用 Sparse 预算默认 15%，牺牲部分视频连接；文字、参考、音频及边界保留 dense。预算设为 1 也可能仍走节点自有 backend，所以做原生对照要 bypass 节点。

新版说明包含 native Kitchen 路线、4K chunk QKV、MLP 内存优化、early release 和冲突回退。FastH3 的 learned coarse branch 与 tile routing 要求训练权重；不是 Larry 直接开开关即可等效。它对部分二进制目标明确写了只做过某一型号的 live run，这种区分值得采用。

可借鉴：当前 checkpoint contract 检查、首次 backend 数值检查、失败时明确回退、advanced 与普通界面分开、发布状态诚实。不能直接复制：未审计许可的 Python 代码、与权重无关地开启 VSA、把编译目标当性能支持。

许可核实：仓库根 LICENSE 未找到，GitHub API license 为 null；完整 tree 存在 `native/LICENSE`（Apache-2.0）、`native/NOTICE.upstream` 和其他子目录许可。结论是**native 有明确许可，不能自动覆盖整仓所有文件**，不是“全仓无许可”。

## Nunchaku：降低安装摩擦的参考

[插件](https://github.com/nunchux-ai/ComfyUI-nunchaku)和[核心](https://github.com/nunchux-ai/nunchaku)原 `nunchaku-tech` 地址现在跳转至 `nunchux-ai`。本次计数使用跳转后的标准地址。

官方 [安装文档](https://nunchaku.tech/docs/ComfyUI-nunchaku/get_started/installation.html)分开安装插件和后端，并有安装 wheel 的工作流。Release 资产名字实际编码了 CUDA、Torch、Python 和平台，Linux/Windows 均有，证明 wheel 兼容矩阵本身就是交付工作的一部分。

我们建议借鉴兼容选择器与错误提示，但生成节点执行时不自动 pip install；诊断入口只输出建议。新项目不需要一开始复制它成熟之后的双仓库结构。

## WaveSpeed / TeaCache：学习用户入口，不混用性能分母

[WaveSpeed](https://github.com/chengzeyi/Comfy-WaveSpeed)采用 First Block Cache 与 Compile Model+ 节点；[TeaCache](https://github.com/welltop-cn/ComfyUI-TeaCache)模型 patch 易插入现有图。这证明“最小工作流改动”是实用交付方式，但并不证明某一节点对所有采样器无损。

[H3 TeaCache 示例](https://github.com/Icyoung/ComfyUI-MiniMaxH3-TeaCache)的 README 报告 306.22→102.08 秒；配置为 FL2VA INT8 ConvRot、20-step、res_multistep/simple、1024×576、124 frames、CMP170HX SM80 64GB。它与我们的 Larry8 Ref2VA、5090 D v2、dense 精确路径不同，不能拿 3× 和 6.47% 直接排优劣。

开源评测表应标记改变步数、删算/缓存、量化与数学等价的类别。每个方案必须有同 seed、相同任务配置和完整音视频质量口径，单项速度不能脱离输出。

## 官方 ComfyUI 插件机制

1. [V3 migration](https://docs.comfy.org/custom-nodes/v3_migration)：以 `ComfyExtension`、`comfy_entrypoint`、`io.Schema` 组织节点；`latest` 是开发中的 API 别名，不应误称永久稳定版本。发布时固定经过实际测试的最低 ComfyUI/API 组合。
2. [ModelPatcher 源码](https://github.com/Comfy-Org/ComfyUI/blob/master/comfy/model_patcher.py)：默认 `get_clone_model_override` 返回同一 underlying model；`clone()` 拷贝 options/patch metadata 不意味着复制所有底层模块。直接修改 `.model.forward` 会有跨图污染风险。
3. `add_wrapper_with_key`、attachments、injections 提供集成入口，但 wrapper 覆盖范围是否够我们的跨层载体，要用具体版本验证。仅选了官方 API 不代表自动线程安全。
4. [Registry publishing](https://docs.comfy.org/registry/publishing)：注册 publisher 和项目后可供 Manager 安装；ID 不可随意更改；`comfy node publish` 是真正外部发布动作。
5. [Registry specifications](https://docs.comfy.org/registry/specifications)：`pyproject.toml`、版本、Repository、PublisherId、requires-comfyui、`.comfyignore` 等控制分发。不能把当前未注册的工作名假装已可 `comfy node install`。

本仓库已经实现诊断 CLI 和 V3 诊断节点源文件；没有伪造完整生成 workflow，也没有设置自动 publish。首个真正的加速节点须通过干净 ComfyUI 启动、保存/重载、bypass、重入、异常取消及与其他 patch 冲突的验证。

## 本轮研究限制

没有取得统一真实硬件环境下的最新竞品 A/B，没有重新运行 GPU，也没有证明 .36 比 R85 更快或更慢。对同类项目报告的收益均视为作者报告。GitHub star 只描述抓取时状态；许可证以实际路径文件为依据，API 标签只用于发现。
