> 历史 R85 证据说明，来源于 2026-09-30 交接资料；不代表本仓库已交付 GPU 后端。旧资料中的内部路径索引、runner 和黄金素材不在本仓库。当前开源方案见 [开源策略](OPEN-SOURCE-STRATEGY.zh-CN.md)。

# 上游来源、版本与开源代码拆分

本文不进行新的“全网最新版本”排名。版本和PR状态按各已完成实验的固定快照记录；主分支后来更新，不自动改变本项目的验收结果。

## 1. 采用与测试版本

| 组件 | 固定/测试身份 | 取舍 |
|---|---|---|
| Comfy Kitchen | 实际运行0.2.35；旧基础镜像metadata另有0.2.33 | 固定已验证源码树，不按pip list误判 |
| PyTorch/Triton | 2.12.0+cu130 / 3.7.0 | 默认 |
| 新运行栈 | 2.14.0+cu130 / 3.8.0 / cuDNN9.24 | 完整测试无稳定加速、输出改变，未采用 |
| cuDNN | 默认9.20 | 新库不自动重写自定义INT8 CUDA核 |
| CUTLASS | 自定义核依赖冻结构建；4.5/4.8同源同调度比较 | 升级本身无可靠收益，不逐包强升 |
| CUDA编译器 | 固定CUDA13.0构建；另筛选ptxas13.4.59 | 新编译结果更慢，未整体替换 |
| FlashInfer | 各PR固定head、0.7.0基座等分别留档 | #5493/#5472等属候选测试，不是R85自动依赖升级 |
| H3-Optimizations | 0.2.45 FastH3 VSA候选 | 需要训练权重；不直接给Larry开启 |
| CPU导出 | 原PyAV/FFmpeg/x264/AAC链路，历史镜像PyAV18.1.0 | 实际发布应补录完整二进制/编译选项，不只记Python包 |

原镜像127个Python分发包的盘点是某时间点静态快照，不等于所有生产实例实时状态。移植首先固定实际import路径、commit、source SHA与.so，不能用“最新”取代可复现版本。

## 2. 核心上游来源与归属

| 来源 | 本项目关系 |
|---|---|
| [Comfy Kitchen](https://github.com/Comfy-Org/comfy-kitchen) | ConvRot INT8、GEMM、dense/VAE融合基础；项目做适配、调度与验证 |
| [ComfyUI](https://github.com/Comfy-Org/ComfyUI) | H3模型/采样/模型管理/VAE接入及动态显存等基础 |
| [Kitchen #167](https://github.com/Comfy-Org/comfy-kitchen/pull/167) | H3 VAE输入激活、量化和融合路径 |
| [ComfyUI #16187](https://github.com/Comfy-Org/ComfyUI/pull/16187) | 新VAE接入 |
| [ComfyUI #16332](https://github.com/Comfy-Org/ComfyUI/pull/16332) | tile引用释放和canvas内存优化 |
| [SageAttention](https://github.com/thu-ml/SageAttention) | INT8注意力/在线softmax等基础路径与思想；自定义SM120实现需保留来源 |
| [CUTLASS](https://github.com/NVIDIA/cutlass) | GEMM模板、MMA/epilogue基础 |
| [FlashInfer #5493](https://github.com/flashinfer-ai/flashinfer/pull/5493) | FP8 QKV大融合实测；head 2c5fe4dc323c8c5ebcfe30344d0b0ed51f134983；未替换INT8 |
| [FlashInfer #5472](https://github.com/flashinfer-ai/flashinfer/pull/5472) | SM120 BF16 dense实测；测试head 2653034125b095a78ce15ff3adbabd5fe75a96db；慢于当前INT8 |
| [LightX2V #1557](https://github.com/ModelTC/LightX2V/pull/1557) | DPCache DP/Taylor代码移植，head 99dabd7e4010261403294cd82a282e0b37445233；近似分支未采用 |
| [FlashAttention #2599](https://github.com/Dao-AILab/flash-attention/pull/2599) | SM120 TMA/warp分工的研究参考；不将作者BF16收益归为本项目 |
| [niw/mmh3](https://github.com/niw/mmh3) | commit 67763f64的TMA GEMM实验参考；48真实VAE解码未稳定胜出 |
| [learn-cuda SM120](https://github.com/gau-nernst/learn-cuda/tree/8c4d1b887a25727b320bc3ace19b63e2db6f8b44/02_matmul_sm120) | TMA/producer实验参考；历史LICENSE读取未取得，不能自动声明可按MIT再分发 |
| [NVIDIA Blackwell tuning guide](https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html) | 硬件资料；最终资源限制使用5090 D v2设备查询与实测 |

vLLM/SGLang在本项目主要提供批处理、权重驻留、请求调度、缓存分层与生命周期的研究借鉴，以及独立Qwen编码服务实验。R85不是部署在vLLM/SGLang上的整套H3模型；不能将它们其他模型的PR benchmark算成本项目收益。

## 3. 权重来源

| 权重 | 来源/身份 | 发布注意 |
|---|---|---|
| H3基础/量化模型 | 原项目固定模型manifest | 本资料仅记录SHA，未重新分发 |
| Larry v4 | [larryvrh](https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora) | 保留作者、训练版本和许可证 |
| Larry pruned ComfyUI转换 | [drbaph](https://huggingface.co/drbaph/MiniMax-H3-Turbo-Lora-ComfyUI) | 保留转换来源与精确文件SHA |
| INT8 VAE | [Comfy-Org固定revision](https://huggingface.co/Comfy-Org/MiniMax-H3/blob/7e75982b97cd5a41d2dcfa1904ee88d0686d6fd1/vae/minimax_h3_video_vae_int8_convrot.safetensors) | 文件SHA 52a2c8c73583c86e4f41cdcce3a6ad0ea562987bc0bf3d60a0cef5f5c8e60c0e |
| HyperFlow | [videorebirth/hyperflow](https://huggingface.co/videorebirth/hyperflow) | 已测候选，非默认依赖 |
| LightX2V Turbo | [lightx2v/Minimax-h3-Turbo](https://huggingface.co/lightx2v/Minimax-h3-Turbo) | 旧4-step、Ref2VA8、FL2VA8需按具体文件区分 |

模型权重、LoRA、第三方源文件、CUDA二进制和本项目新代码可能具有不同许可。准备开源应按文件保留NOTICE/LICENSE和修改记录；不能给整个历史研究目录套一个宽泛许可证。本次没有替用户选择项目许可证，也没有向外发布。

## 4. 从研究代码拆成可复用库

下面是建议结构，尚未声称已实现成独立安装包：

```text
h3_sm120/
  config.py                 # 模型与数值模式，默认固定dense INT8
  identities.py             # source/binary/weights/runtime身份
  dispatch.py               # dtype/shape/stride/layout/quant资格与回退
  kernels/
    dense_sm120/            # INT8 attention，不带服务全局变量
    norm_convrot/           # norm/mod/residual/quant融合
    gemm_epilogue/          # outproj/FC2与QKV接入
    vae_d64/                # 可选R234v2/R244后端
  runtime/
    ownership.py            # carrier单消费者、stream与generation
    vae_loading.py          # dynamic默认，static候选显式选择
    rgb8_transfer.py        # blend后转换、双pinned队列
    cpu_export.py           # 有界x264/AAC/MP4任务与排空
  integration/comfyui.py    # 模型结构适配，不硬编码业务case
benchmarks/
  operator.py               # 真实capture和边界/poison/stream
  full_pipeline.py          # 各臂独立预热、配对完整生成
  service_queue.py          # 真实condition与完成事件
  metrics.py                # 固定评估版本和分母
tests/
  numerical_contracts/      # 舍入/量化/所有权/回退语义
third_party/                # 来源与许可证，按需下载
```

现有研究runner依赖多套历史目录挂载和动态源码替换；公开版本应把实验choice整理为明确的后端接口。未知shape回退原Kernel；stateful跨block carrier不能共用到多请求并发。新VAE D64只作用于decoder，不污染DiT或condition。

权重建议提供转换/下载清单及SHA校验，而不是附带私有baked产物。首次转换要对选定输入验证与旧post-cast有效权重一致。CPU导出应有独立可替换接口，不带内部任务队列或对象存储凭据。

## 5. 本次可公开资料与内部复用材料

公开草稿包含技术说明、匿名case指标、源文件SHA、精简基线身份、本地复算脚本与审阅网页；没有第三方大段源码或权重再分发。

内部目录另存原文件路径索引、代码入口清单与未改动R85固定包副本，供后续接手者快速定位。内部包仍有原环境路径，不能直接推到公开仓库。需要公开实际实现时，按上节拆分代码并完成干净环境构建与统一全链路对照，才可称为可安装开源推理方案。
