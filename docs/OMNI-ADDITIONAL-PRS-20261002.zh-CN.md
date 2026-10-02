# H3 优化上游扩展：2026-10-02

## 本轮结果

此前并非只有两个 PR：相关仓库原有 17 个未合并的正式 PR。本轮新增两个独立 PR，目前合计 19 个 open；另有 ComfyUI #16677 已于 10 月 1 日合入。

| 去向 | 本轮提交 | 实测 | 当前边界 |
|---|---|---|---|
| [vLLM-Omni #8416](https://github.com/vllm-project/vllm-omni/pull/8416) | H3 固定时间步位置提前计算，标量赋值改 index_fill | 32,320 行准备耗时 0.229254 → 0.029448 ms，减少 87.15%；每次 nonzero 4→0、流同步 8→0 | 准备阶段收益；8 步约省 1.6 ms，不是 DiT 整体快 87% |
| [SGLang #42186](https://github.com/sgl-project/sglang/pull/42186) | SM120 BF16 INT8 linear 调整分块默认值 | 32700×5376×16128 的公开复测约 12.484 → 10.557 ms，减少 15.44% | 完整 Larry8 中位数 33.563→33.396 s，约 0.50%，基线波动大，尚不能证明稳定端到端加速 |
| [vLLM-Omni #8414](https://github.com/vllm-project/vllm-omni/pull/8414) | 补充既有 VAE PR 的逐项消融及 profile | 见下节 | 不是新增的第三个 PR |

两项新 PR 均已正式提交、不是 draft。#8416 的 DCO、pre-commit、Python3.11/3.12 构建通过；#8416 文档构建尚在等待；#42186 的 lint 通过，GPU CI 门禁缺少维护者 run-ci 标签，已留言申请。此表是时间点快照，不代表已获人工批准或合入。

## SGLang：只改变已测范围的策略

旧默认每 8192 行分块，目的是避开 4090 上开销大的 Stream-K 路径。5090 D v2 的部分矩阵不适用这条经验，额外的 kernel launch、结果分配和回写反而更慢。

新默认仅限 SM120、BF16 输入、K≤8192、8192≤N≤24832。其他 dtype、其他 GPU、大 K、大 N 保留原行为；用户显式设置任一分块环境变量时也保留用户配置。能力检测按实际输入设备缓存。

未采取全局关闭分块：M14850/K7168/N28672 反而慢约 11.7%；更大 K 的收益很弱或略退化。30 组矩阵对照全部 bit-exact，14 项策略测试通过，公开 benchmark 已重新执行。

完整采样两臂使用原版 Kitchen 0.2.36、相同 Larry v4 INT8 权重、相同加载修复 #42121；768×512、124 帧、8 步 Euler/beta、CFG1、seed42、shift12/4、SDPA、无常驻层。A/B/B/A 四个独立进程，每进程 1 次热身、2 次正式请求。共 12 次采样的音视频 latent 哈希完全一致。

| 组别 | 四次正式请求 / ms | 中位数 / ms |
|---|---|---:|
| 旧分块策略 | 33960.216、33967.192、33140.687、33165.233 | 33562.724 |
| 新策略 | 33365.072、33370.351、33421.806、33426.852 | 33396.079 |

基线末轮比候选还快，不能仅凭 0.50% 宣传稳定整段加速。没有测并发服务 QPS，也没有重算视频/音频主观质量；latent 逐位相同证明本次策略没有改变生成数值。正常 shutdown 后两臂均有 ComfyUI 析构告警，进程返回 0，告警在计时之外，未声明已修复。

## vLLM：消除固定位置的重复筛选

每个请求的图像/音频 target 和 condition 行位置在去噪期间固定。原来 fill_timesteps 每步用 boolean indexing 筛选，CUDA 会执行 nonzero 并同步。现在在 branch 初始化时生成四张位置表，uniform timestep 使用 index_fill；编辑区域自定义时间步仍按原逻辑赋值。

| packed rows | 基线 / ms | 新版 / ms | 耗时减少 |
|---|---:|---:|---:|
| 4672 | 0.166366 | 0.022796 | 86.30% |
| 15040 | 0.228388 | 0.029313 | 87.17% |
| 32320 | 0.229254 | 0.029448 | 87.15% |

ABBA，每组 20 次热身、30 组×8 次填充；计时与 profiler 分开。每次 FP32 时间步 tensor 精确相同。请求/step 批处理、1/8/50 步等价、交错参考位置、locked audio、latent edit 和非法目标长度等共 45 项测试通过。新增位置表约每个 AV 行 8 字节，随请求释放，不缓存模型输出。

## 已有 VAE PR 的进一步证据

同一提交 ce3039e、相同真实 latent，六个配置正序＋反序，每组 1 次热身＋2 次计时；全部 36 次 RGB8 输出逐位一致。预转换匹配参考 autocast 精度，不是新增量化。

| 配置 | 中位数 / ms | 范围 / ms | 峰值分配 / GB |
|---|---:|---:|---:|
| 参考路径 | 6204.470 | 6195.725–6217.621 | 10.661 |
| 仅预转换权重 | 5375.281 | 5359.920–5389.491 | 5.825 |
| 预转换＋FF融合 | 5206.627 | 5191.183–5221.735 | 5.825 |
| 预转换＋QK融合 | 4878.474 | 4867.504–4890.755 | 5.825 |
| 预转换＋残差融合 | 5303.970 | 5288.718–5320.151 | 5.825 |
| 全部优化 | 4641.391 | 4626.332–4651.800 | 5.825 |

分项百分比不能相加。另做的 PyTorch profile 观察到 copy_ 43111→6823、RMSNorm 12096→6048、cat 12585→489，GEMM 次数保持 12180。该 trace 有 profiler 开销和 command-buffer stall，仅用于确认删除了哪些计算，不用于速度/利用率结论。

消融首版脚本的循环变量保留了一个 block，造成后续峰值显存多约 133 MB；修正释放后所有配置已完整重跑，上表为修正结果。原 #8414 完整 decode ABBA 没有这个循环变量。

## 没有为数量而重复提交的方向

- SGLang-Omni 接收缓冲免清零与单 tensor 拷贝已有 #2368；原 #2481 是互补的多 tensor GPU→CPU 打包。
- SGLang VAE QK/RMS/RoPE 已有 #41906。基础 INT8 VAE、Larry v4 权重均来自上游，不作为新原创重复贡献。
- Residual kernel 的多种 row/flat 调度没有稳定整体收益；不以某一个形状的微小变化单独发 PR。
- vLLM VAE 非支持输入 fallback 会重复 QKV，但官方默认输入不走该回退；没有真实受益场景和充分契约验证，不先改远程模型接口。
- 去噪循环里的 clone 有回调快照/输入所有权语义，未盲目删除。
- ComfyUI #16678 仍依赖 Kitchen #220 的 BSHD API 发布。当前 pin 0.2.36 不支持它；没有填写不存在的 release，也没有把未解决的依赖写成已通过。

## 复现资料

- [逐项审查与否决理由](../evidence/omni-migration-1002/candidate-precheck.md)
- [矩阵完整原始记录](../evidence/omni-migration-1002/kitchen-split-stock.json)
- [矩阵边界探测](../evidence/omni-migration-1002/kitchen-split-edge.json)
- [完整采样 ABBA](../evidence/omni-migration-1002/sg-sampler-abba.json)
- [时间步准备及同步计数](../evidence/omni-migration-1002/timestep-portable.json)
- [测试日志](../evidence/omni-migration-1002/candidate-tests.log)
- [公开实验脚本](../experiments/omni-migration-1002/)

所有 GPU 实验限定在原实验容器的 GPU0，未更改生产部署或其他卡服务。
