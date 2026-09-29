> 历史 R85 证据说明，来源于 2026-09-30 交接资料；不代表本仓库已交付 GPU 后端。旧资料中的内部路径索引、runner 和黄金素材不在本仓库。当前开源方案见 [开源策略](OPEN-SOURCE-STRATEGY.zh-CN.md)。

# 复现协议与统一累计补测

本页包含可执行的证据校验、已完成的新GPU直接对照与仍待扩展的服务测试。原生FP16视频VAE基础版→R85的两片段完整对照已完成；全15同latent VAE质量另完成。原R85与新对照runner保留在内部原环境资料中，不是通用安装包。

## 1. 无GPU复算现有结果

解压本资料包，在包根目录执行：

```bash
python3 tools/verify_metrics.py
python3 tools/verify_stock_comparison.py
python3 tools/verify_native_comparison.py
python3 tools/check_repository.py
```

第一个命令从附带标量/逐组数据重新计算早期完整服务收益、Kitchen回归、INT8 VAE、R85、VAE候选、新运行库计时和旧LoRA逐故事指标。它检查R85 15个case的各重复四SHA一致、每请求400次dense。

第二、三个命令分别复算固定 INT8 VAE 与原生 FP16 VAE 对照；最后一个命令检查当前仓库的 JSON、相对链接和发布范围。历史源文件 SHA 保留在 evidence/import-provenance.json；离线复算不等同重新生成视频或复核权重正确性。

公共包只提供经字段筛选的指标，不含业务prompt、参考图、视频、latent或内部连接方式。由源文件SHA追溯完整证据，需内部资料或使用获授权的公开输入重新测量。Q2的31配方表来自原报告的舍入数字，不能当高精度raw timing。

## 2. 固定基线身份

| 项 | 值 |
|---|---|
| ID | H3-5090Dv2-LarryV4-8step-R85-INT8Decode-20260926-v1 |
| 原manifest SHA | 903aa049d5a68442ed90d1d6f570300f814ceb6d46ef9ec0021a642b0586993c |
| 普通runner展开源码SHA | 1fcc6c165a749a4b4695402e167e8a91bc3ad33344fa2a4057a627059fb5e963 |
| 镜像身份 | sha256:1ecd3c0d53eacdbed1dfe66b7eb8b8005364748d23bee09b0e4eb6d227ff1f15 |
| DiT baked权重SHA | 9eccf52e4fe6e764f4aac8cdb6045a58bc86d8458c783c35b7035edca71fec12 |
| INT8视频VAE SHA | 52a2c8c73583c86e4f41cdcce3a6ad0ea562987bc0bf3d60a0cef5f5c8e60c0e |
| 旧FP16 VAE SHA | 7c1f131492e7eddacaac9069a61b81bdd39de5cc96561e677c5eab1cdce5e522 |
| 旧TRT engine SHA | c22469eef744367e1ba253811b894ed37983ad12dc81b9cd779b4b2a31815cc8 |

原固定包包括42个study文件、29个外部pin，另有53个实际Kitchen文件及输入身份。包名/镜像tag不能替代实际导入源码和二进制SHA。旧脚本某些protocol文案曾继承旧实验，判断采用实际参数、展开代码、调用计数和输出签名。

在原固定包中 `python3 -B verify.py` 已于本次本地复核通过。`--external` 需要真实挂载环境；本次 GPU runner 已通过 42 study＋29 external pins 的启动/收尾校验；公开摘要的 `baseline-identity.json` 是脱离私有路径的索引，不是原manifest的字节副本。

## 3. 统一测试需要分清两个问题

**问题A：在同一Larry8配方上，所有工程改动累计快多少？** 这是新的主基准，用同一权重、condition、step、seed、分辨率和CPU预算，比较stock与R85。

**问题B：从最早实际部署到今天，延迟和质量如何变化？** 最早旧Turbo5与Larry8不是相同模型方案，需作为产品档位对照，同时报告质量收益与计算增加。不能用问题B数据归因内核改造。

如果真正的原版是Kitchen .33，应恢复该版本的源码、VAE和原部署入口。若只能完成.35 stock回放，名称必须写“.35 stock”，不得冒充.33部署复现。

## 4. 推荐测试臂

| 臂 | 配置 | 用途 | 状态 |
|---|---|---|---|
| A0 | Kitchen .33原部署，旧Turbo5 | 历史产品配置复现 | 历史已有数据；新统一测试待执行 |
| A-FP16 | Kitchen .35 stock，Larry8 baked，原生 FP16 视频 VAE，旧encoder，FP32 CPU输出 | 用户要求的历史高精度VAE主基准；实际精度已核验 | 两片段8正式＋4预热；全15同latent质量/60正式decode已完成 |
| A1 | Kitchen .35 stock，Larry8 baked，TRT，旧encoder，统一导出 | 隔离相同模型的工程对照 | 待实现并验证没有泄漏项目补丁 |
| A2 | Kitchen .35 stock，Larry8 baked，INT8 VAE，旧encoder，原生 FP32 CPU 输出 | 固定 INT8 VAE 的整体工程对照 | 9/29 两片段、每臂每 case 2 正式已完成 |
| B | 冻结R85，动态INT8 VAE，旧encoder | 当前固定基线 | 已有完整固定runner和历史验收 |
| C | R85＋静态VAE＋R234v2/R244，旧encoder | 最新完整集成候选 | 仅VAE阶段已验收，完整driver待实现 |

A1与B的比较同时含TRT→INT8，应报告其微小像素变化；A2与B可在固定INT8 VAE上评估工程输出是否一致。若关闭项目kernel本身会改变输出，也应单独报告，不强行用“无损”概括全部差异。

统一导出是为了公平隔离GPU计算；要测RGB8/异步工程本身，需要另加原FP32交接/原服务导出臂。每种消融明确包含和排除什么，最终主数字必须来自两端整套方案直接比较。

## 5. 实施顺序

### 阶段一：身份与数值门槛

1. 只使用实时检查通过的授权实验GPU，隔离目录和环境，不覆盖固定代码/二进制/权重。
2. 对每臂记录GPU型号/SM/UUID、driver、runtime、实际库路径、编译器和flags、权重/source/binary SHA。
3. 分别独立预热；核对输入/参考顺序/condition/seed、NFE、每层实际kernel调用和fallback计数。
4. B要恢复历史四SHA；C要先通过两个代表case，再扩全15个保存latent和完整DiT输出。
5. A1 stock入口必须核实旧monkey-patch未残留：不能只设一个enabled=false，就假定其他后装wrapper也关闭。
6. 对C在VAE前后严格启用/恢复D64 hook；验证下一请求DiT未受影响，检查静态权重释放和峰值显存。

### 阶段二：同卡性能

先做case1/8，各臂独立1次预热、至少4次正式，ABBA与反序BAAB跨两个时间块。所有候选使用同一GPU、同一功耗/频率策略，保留全部样本；不以profile计时作为正式成绩。

确认无明显回退后，扩全15段，同卡配对，每臂每case至少2次正式；若亚百分比收益且波动大，追加独立时间块再判断。保存所有失败、重试和未完成记录，不用挑最短值代替分布。

输出每case median/mean/min/max、配对差值；合计用各同卡工作组的等量秒数之和。批内时间相关，不能把数百帧当作数百个独立性能或质量样本。

### 阶段三：完整服务和吞吐

用真实参考encoder/Qwen condition，并按固定输入顺序提交backlog。单卡GPU串行，CPU成片与下一请求重叠。至少跑完整5故事/15段的固定队列，双向顺序，所有任务排空到MP4/结果成功。

计时分别记录：素材获取、condition、DiT、video/audio VAE、RGB/D2H、CPU编码、上传、结果提交、queue wait。condition内部reference encode不重复相加。没有上传服务的实验明确排除上传。

稳态吞吐用实际完成间隔或固定工作量总完成时间；冷启动/模型读盘/JIT单列。长时间测试报告成功率、失败/重试、P50/P95、主机内存/锁页/本地盘/显存水位和GPU功耗，不从两次正式样本推断P95。

## 6. 质量指标协议

精确工程比较输出四SHA，且对照历史黄金而非只有新A/B彼此一致。换VAE/LoRA/运行栈时另外报告原始RGB误差与成片指标，统一视频色彩空间、fps、帧数与音频容器。

BF16/50参考必须保持原文件SHA；原始PCM不存在时明确说明参考来自AAC。使用全帧SSIM/PSNR、2fps SqueezeNet LPIPS、8k频谱和32k立体声STFT；从t0直接对齐。若另做最优时间对齐，只能作为额外诊断，不覆盖主表。

预先约定质量门槛。新模型增加多seed，人工盲听/盲看与台词/口型评分单列；不因一次SSIM上升就宣传质量提升。

## 7. 结果记录格式

每个请求至少记录以下信息；这是接口示意，不是已生成的实验数据：

```json
{
  "experiment_id": "unique-new-id",
  "baseline_id": "frozen-id",
  "arm": "B",
  "case_id": 1,
  "phase": "formal",
  "repeat": 0,
  "hardware_identity": "verified-device-record",
  "input_sha256": "computed-not-placeholder-in-real-run",
  "weights_manifest_sha256": "computed",
  "runtime_manifest_sha256": "computed",
  "actual_nfe": 8,
  "dense_calls": 400,
  "timing_scope": "frozen condition to local MP4",
  "condition_seconds": null,
  "dit_seconds": null,
  "video_vae_seconds": null,
  "audio_vae_seconds": null,
  "cpu_export_seconds": null,
  "worker_done_monotonic": null,
  "video_latent_sha256": "computed",
  "audio_latent_sha256": "computed",
  "rgb8_sha256": "computed",
  "pcm_sha256": "computed",
  "fallback_count": 0,
  "status": "pending"
}
```

正式结果不能把null/placeholder当0或成功。图回放、sanitizer、profile与正式计时用不同phase，避免误混统计。

## 8. 本次进展与当前缺口

9/29 新 GPU 同场实测：相同 Larry v4 8 步、同一 INT8 VAE，Kitchen 0.2.35 基础实现到固定 R85，两个代表性片段的总生成耗时减少 **6.685%**，串行容量折算 **+7.164%**。共 8 正式＋8 预热，所有正式请求的 video latent、audio latent、RGB8、PCM 与历史黄金逐位一致。 详细记录见 [STOCK-VS-R85.md](R85-STOCK-VS-R85.md)。

GPU 0/3 各自同卡交叉，其他服务核查未变。基础臂直接不导入 teacher/attention/量化/epilogue 等项目适配器，避免只关闭一个开关却遗留融合；两臂 8 NFE/400 次 dense 的入口和输出全部审计。原生整体 attention 入口与分拆入口不同，首次审计误选导致的基础预热失败已保留并排除。

当前完成的是相同 INT8 VAE 下的两片段直接比较，每臂每 case 2 正式，少于上文推荐的强化验证 4 次。尚未完成全 15 新对照、TRT 基础臂、最新静态 VAE 的完整集成、真实 condition 与长时间服务队列；不得将小样本串行容量折算称为稳态吞吐。


## 9. 原生FP16基线新增完成项

新同场实测：**Kitchen 0.2.35 基础 Larry v4 8 步＋原生 FP16 视频 VAE → 固定 R85＋INT8 视频 VAE，两个代表片段的总生成耗时减少 11.591%，串行容量折算增加 13.111%**。每请求平均节省 30.66 秒。完整计时从冻结 condition 到本地 MP4，排除 Qwen/参考处理、HTTP/排队/上传与首次读权重。

相同 latent 的全部 15 片段，INT8 VAE 对原生 FP16：成片 SSIM 均值 **0.989070**，LPIPS **0.007589**，原始 RGB8 PSNR 均值 **58.67 dB**；音频 PCM **15/15 逐字节一致**。这支持“当前测试中的画面差异很小、音频不变”，不等于逐像素无损，也不是多 seed 人工盲评结论。

历史“BF16 满血版”实际为 BF16 DiT/Qwen＋FP16 视频 VAE＋FP32 音频 VAE。本次读取 safetensors 头和运行 dtype，视频原权重 562 个张量均为 FP16，运行也是 torch.float16。因此主速度基线准确标为“原生 FP16 视频 VAE”；单独 BF16 VAE 尚未测试。

全15的“同latent解码测试”不等于全15“完整DiT请求对照”。本次保留动态VAE；静态R234/R244尚未计入。
