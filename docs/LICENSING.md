# Licensing and attribution / 许可证与来源

The combined plugin is **GPL-3.0-or-later** because it contains derived ComfyUI model/VAE implementations. The original kernel adaptations, diagnostics, export tools and documentation are also available under Apache-2.0 where their headers or the file index specify it. Existing third-party copyright and conditions are preserved. This source release includes CUDA/C++ and Python implementations, but no model weights or prebuilt binaries.

- `runtime.py`, `video.py`, `vendor/`, `tools/merge_lora.py`: GPL-3.0-or-later integration/derived implementation.
- `kernels/`: Apache-2.0, with PyTorch-derived normalization additionally subject to BSD-3-Clause; see [file index](../evidence/kernel-provenance.json).
- `ops/`, `export.py`, `service.py`, diagnostics, build/verification tools and new documentation: Apache-2.0 unless marked otherwise.
- CUTLASS C++ is downloaded separately for builds; its BSD license is included. CuTe DSL is not bundled or used.

ComfyUI model source: commit `2504e68d4d9dedb514e172692f13436623f25aed`; decoder/helpers: `b2e31e89412a01a67be599571cc57ff74b242a82`. The forward copy deletes dead references; the VAE combines the old encoder with the newer decoder/helpers. Kernel source ancestry and modifications are recorded per file. [NOTICE](../NOTICE) and [third-party texts](../third_party/README.md) accompany the release.

## 已核实的来源

| 项目/范围 | 核实到的许可 | 发布处理 |
|---|---|---|
| [Comfy Kitchen](https://github.com/Comfy-Org/comfy-kitchen/blob/main/LICENSE) | Apache-2.0 | 按固定 commit 引用；复制/改动保留许可与 NOTICE |
| [SageAttention](https://github.com/thu-ml/SageAttention/blob/main/LICENSE) | Apache-2.0 | 原作者、原始文件及改动链清楚记录 |
| [ComfyUI](https://github.com/Comfy-Org/ComfyUI/blob/master/LICENSE) | GPL v3 文本 | 复制模型实现或构成衍生作品的部分按对应义务处理；全包分发时一起评估兼容性，不以改目录名规避 |
| [CUTLASS C++](https://github.com/NVIDIA/cutlass/blob/main/LICENSE.txt) | BSD-3-Clause | 保留版权/免责声明；当前文件明确 `python/CuTeDSL` 另受 NVIDIA EULA，不泛称整个 CUTLASS 都 BSD |
| [H3-Optimizations native](https://github.com/Zironic/H3-Optimizations/blob/862774944a331bc1a66cee1cf805b94bd138ebbe/native/LICENSE) | Apache-2.0；有 upstream NOTICE | 可按具体文件审计；根 LICENSE 缺失，不能自动覆盖其全部 Python 文件 |
| [learn-cuda 历史 commit](https://github.com/gau-nernst/learn-cuda/tree/8c4d1b887a25727b320bc3ace19b63e2db6f8b44) | 完整 tree 未发现 LICENSE/NOTICE/COPYING | 只作研究参考；不要搬用其源码。若已衍生，先查清授权或以合法来源独立实现 |
| [Kablex VSA LICENSE](https://github.com/Kablex/ComfyUI-Ref2VA-VSA/blob/main/LICENSE) | 文件自称 Apache 2.0，但文本与标准全文不一致；API 为 NOASSERTION | 不依赖 API 空标签推论可任意复制；首版不 vendor，真要引用再确认具体授权 |
| [Larry LoRA 模型卡](https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora) | `license: apache-2.0` | 保留作者与文件 revision；底座模型许可仍需单独遵守 |
| [Comfy-Org H3 模型卡](https://huggingface.co/Comfy-Org/MiniMax-H3) | `license: other`，链接 MiniMax H3 Community License | INT8 权重也不变成 Apache；本仓库仅记录身份和原地址 |

仅通过 ComfyUI 公开 API 自写 adapter 与复制其实现不是同一情况；具体衍生关系需要以实际提取内容判断。不能预先承诺未来所有 GPU 文件都将是 Apache。若需要整体按 GPL 兼容分发，应在相应 release 明确范围。

## H3 模型许可的实质条件

本次直接读取 [MiniMaxAI/MiniMax-H3 LICENSE](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE)，文本日期 2026-08-02，抓取 SHA `59b99642b95ea21630e311198ddbfffbfe05aadba0c2f5d884cbdf4efcc90f44`。

- I.5 原文列出 excluded territories：“the European Union, the United Kingdom, the Republic of Korea and the United States of America”。II 和 V.4 将授权及使用范围限制在适用地区。对全球用户传播时不能隐藏这一点。
- III 涉及再分发附带许可、修改声明、NOTICE 等要求；输出不是定义中的 Model Derivatives，但其他输出/使用约束仍在文本中。
- IV.1 写明商业产品/服务年收入超过 2,000 万美元时，需要另行书面授权；IV.2 有商业界面展示模型名称要求。
- V 含用途、下游及产品/服务义务。以原始完整文本为准，README 的简短摘要不能替代它。

所以建议**发布独立工具与算子源码，权重仍由用户按上游授权取得**。这并不自动豁免模型使用者，也不意味着原作者允许随意再分发 baked checkpoint。插件、内核、模型三者的许可证范围需分别说明。

## 每个待公开 kernel 的来源记录

每条记录至少有：本仓库路径、上游 URL/commit/原路径、原始 SHA、许可证路径、自有改动概述、构建输入、对应二进制 SHA 和验收记录。对已被上游吸收的优化记录 upstream PR，避免重复宣称发明。

没有查清的历史源码保持在内部复用材料，不复制到 public Git；这不妨碍公开分析、规格与合法重写的自有实现。本地 Git 中也不存业务媒体、条件 tensor、生产 manifest、账号、内部域名或私有镜像操作脚本。

发布内容按上述文件范围执行；模型许可不由本仓库的软件许可替代。
