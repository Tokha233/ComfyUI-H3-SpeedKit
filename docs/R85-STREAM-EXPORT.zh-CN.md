# R85 增量：解码同时编码 MP4

## 2026-10-08：补充含 DiT 的短队列验证

固定现网 Larry v4 8-step、全 R85、R234v2/R244 INT8 VAE 和旧 encoder 条件；Torch2.12/cu130、基线 patched Kitchen0.2.35。每运行 1 次 Case1 预热，再生成 Case1/Case8 两片段（约10/11秒，768×1344，243/260帧），GPU 串行、CPU 导出与下一条 DiT 重叠。每容器独占一张5090 D v2，CPU quota16核，内存80GiB。计时从固定 condition 到本地完整 MP4，含模型切换、DiT、VAE、CPU导出，排除Qwen/参考编码、网络、冷权重加载和哈希。验收用latent CPU拷贝同样计入两臂。

| 同卡顺序 | 基线两请求秒 | stream秒 | 队列降时 | 短队列完成率变化 |
| --- | ---: | ---: | ---: | ---: |
| GPU0：每次启动子进程导出 → stream | 461.182019 | 457.121530 | 0.880% | +0.888% |
| GPU1：stream → 每次启动子进程导出 | 461.942499 | 457.663665 | 0.926% | +0.935% |
| GPU2：常驻同进程导出 → stream | 461.568986 | 458.661966 | 0.630% | +0.634% |

**更接近常驻服务的口径是 GPU2：省2.907秒 / 两请求。** 它通过同进程 executor 直接调用相同 `export_chunks`，避免把每请求启动子进程的时间算作重叠收益。该臂目前一组同卡配对；不能承诺稳定收益率。所有运行共12次正式、6次预热，视频latent、音频latent、RGB8、PCM和MP4哈希逐case与历史R85一致。Case8未单独暖机，各臂协议相同。

两请求GPU完成窗口：GPU0 456.585→456.657秒，GPU1 457.306→457.193秒，GPU2 458.265→458.194秒，基本持平。GPU2最后文件尾部3.304→0.468秒；**收益主要是最后文件更早完成，没有证明持续backlog下GPU吞吐提高**。不能把输出阶段19–20%降时写成整套推理或稳态QPS快20%。

本轮使用隔离sink overlay把本PR的编码实现接到冻结生产镜像的原decoder，未替换GPU算子、未改线上服务。这个临时overlay不作为并发安全的生产patch发布。正式节点/CLI的复现入口、异常与取消验证仍见本文其余章节。没有重测五个完整30秒故事，也没有声称对BF16教师质量更高。

[每请求原始数据与输出哈希](../evidence/r85-pipeline-1008/summary.json)。

这是 **R85 之后的输出调度改动**。Larry v4 八步、INT8 DiT、INT8 VAE、旧 encoder 和 GPU 算子保持原样。它减少的是同一请求末尾的 CPU 编码等待，尚无完整在线服务 QPS 或 P95 的新增实测。

## 使用

在 `H3 SpeedKit/Experimental` 中选择 **H3 SpeedKit · Decode and Save Video**：

- `samples`：KSampler 的 H3 joint latent。
- `vae`：H3 SpeedKit Video VAE Loader 的视频 VAE。
- `audio`：原 FP32 GPU audio VAE 的 AUDIO 输出，可选。音频应先完成；没有迁移到 CPU 解码。
- `fps`、`crf`、文件名前缀：与原 Save Video 保持一致。

这个节点直接返回已完整写好的 MP4 文件名。如果需要对整片 IMAGE 做调色、插帧、超分，继续使用原 Decode to RGB8/Save Video 或其他相应节点。新节点需 NumPy、PyAV；安装依赖与原导出路径相同。

普通 Decode to RGB8 不开启后台编码。服务已有的 `ExportQueue` 保持跨请求导出用途，不要给同一输出同时启动两个编码器。

## 实现和一致性

1. temporal blend 完成后，沿用原 FP32 像素归一化与 RGB8 截断融合核。
2. 两个 pinned 槽在专用 CUDA stream 上 D2H；记录 producer/consumer 顺序和 CUDA allocator 生命周期。
3. 单个后台线程等待对应 event，把完成的槽复制到请求独占的 CPU 输出，再交给编码线程。复用槽之前必须等待该槽的任务完成。
4. 编码队列容量为 2；只交付最终、不可变的 CPU 切片，不把仍会被覆盖的 pinned 槽交给 x264。
5. RGB 全片 CPU backing buffer 仍然保留，约为帧数×高×宽×3 字节；这是固定于该请求大小的内存，不能宣称内存只占两个帧块。
6. 一次性与分块导出共用编码实现。H264/x264 4 线程、CRF、BT.709/sRGB transfer、limited YUV420P、AAC、音频裁剪、video/audio mux 与 flush 顺序均不变。用临时文件完成编码再原子替换正式文件。

取消/解码异常会终止队列消费、等待线程退出并清理 partial 文件；编码异常会反馈给生产者，即使队列已满。取消会等待当前原生 codec 调用返回，不提供强制杀死卡住的 codec 或 CUDA driver 的硬超时保证。

## 测量范围

RTX 5090 D v2 单卡；冻结 R85 环境，Torch 2.12.0+cu130，PyAV 18.1.0，CPU intra-op 8 线程，x264 4 线程。输入是两个业务片段保存的 R85 latent，case1 243 帧、case8 260 帧，画布尺寸为 768×1344（width×height）；音频 32kHz stereo。未重新采样五个完整故事。

计时从视频 decode 到本地 MP4 关闭/原子命名完成，不包括 DiT、condition、音频 decode、权重加载、哈希与网络。每组 1 次预热、2 次正式；case1 使用 BAAB，case8 ABBA。每次核验原始 RGB 和 PCM 历史签名，以及已知历史 MP4 SHA256。

| 样本 / 顺序 | R85 decode→MP4 | 候选 decode→MP4 | 节省 | 该阶段减少 |
| --- | ---: | ---: | ---: | ---: |
| case1 / BAAB | 11.508872s | 9.310951s | 2.197921s | 19.10% |
| case8 / ABBA | 12.473738s | 9.965145s | 2.508594s | 20.11% |

中位数，每个样本每臂 4 次正式，另有每臂 2 次预热。上述最终实现共 24 次完整 decode/encode，其中 16 次正式；所有 RGB、PCM、MP4 均匹配冻结历史签名。完整请求仍含 DiT 等阶段，不能称总推理快约 20%。

此外完成 36 次六臂筛选与 12 次第一版节点对照。筛选中单独 NumPy copy 或后台回收没有收益；只有后台回收与分块编码的组合保留。最终 VAE decode 基本持平，收益主要来自编码提前启动与尾部缩短。

[全部原始记录和统计](../evidence/r85-stream-1002/summary.json)。

## 如何复测

保存当前采样结果为 `torch.save({'parts': [video_latent, audio_latent]}, 'latent.pt')`。仅加载自己的可信文件。然后在已配置的 ComfyUI/SpeedKit CUDA 环境运行：

```bash
python benchmarks/stream_export.py \
  --comfyui /path/to/ComfyUI \
  --video-vae /path/to/h3-int8-video-vae.safetensors \
  --audio-vae /path/to/h3-audio-vae.safetensors \
  --latent /path/to/latent.pt \
  --order serial,stream,stream,serial --repeats 2 \
  --output results/stream-abba
python benchmarks/check_stream_sink.py --comfyui /path/to/ComfyUI
python -m unittest discover -s tests -v
```

公开 benchmark 比较同一 SpeedKit decoder 的 serial/stream；这里的历史基线实验另有独立冻结 R85 runner，避免只用新代码自比。依赖齐全时单元测试会真正编码小视频；缺少 PyAV/Torch 的离线 CI 会明确 skip codec 测试。

## 对吞吐和上游 PR 的判断

这是单请求输出延迟优化。若原 `ExportQueue` 已将 export(N) 完全覆盖在 DiT(N+1) 下，不能把后处理的百分比换算为服务吞吐提升。仍需在真实服务固定 backlog/CPU 配额下测 QPS 和 P95；本改动不默认替换线上。

思路参考 SGLang 的 [#41819](https://github.com/sgl-project/sglang/pull/41819)，该 PR 已有 H3/Wan 分块编码，不重复向 SGLang 提交相同功能。本项目实现保留 R85 的 PyAV 输出合同和双 pinned 槽，进一步将槽回收及 CPU copy 放到后台。适合先作为 SpeedKit 的可选节点审阅。Kitchen 属于算子库，不应接收文件编码/Comfy 节点调度；ComfyUI 核心若要接收通用能力，还需要有实际消费者且保持 IMAGE/API 合同，不能只提交无调用方回调。

此次没有发现经过验证、还能叠加到 R85 的新 dense INT8 主循环。Kitchen 最新提交仍主要是 HIP/Ascend，现有上游 PR 的参考基线也不是冻结 R85，不把它们再算一次增益。

## 检查与现有 PR 最新进展

16 项测试在服务器全部通过；CUDA 另测 5 种分块边界、2 个并发 sink、回调失败传播和线程退出。最终源码通过完整 helper 与实际 ComfyUI 输出节点，两者生成历史 SHA 一致的 case1 MP4；中断后无 partial 文件或工作线程残留。[检查清单](../evidence/r85-stream-1002/validation.json)。

- SGLang [#42121](https://github.com/sgl-project/sglang/pull/42121) 于 10/2 14:51 获维护者 niehen6174 APPROVED，随后加 `run-ci`。四个 NPU 任务在下载 Triton-Ascend wheel 时 HTTP 403，测试尚未执行；Extra 仍缺 `run-ci-extra`。已向维护者[回报具体日志](https://github.com/sgl-project/sglang/pull/42121#issuecomment-5947059125)，没有为依赖服务器错误改动 loader。
- vLLM-Omni [#8414](https://github.com/vllm-project/vllm-omni/pull/8414)、[#8416](https://github.com/vllm-project/vllm-omni/pull/8416) 的五项检查全部通过，等待审核。
- Kitchen 当前已有十个开放的相关 PR；现有核心算子不重复拆 PR。ComfyUI #16678 仍须等 Kitchen #220 的接口发行与 pin。

这些状态是本次检查的快照，不代表已合并或持续自动监控。

公开复现 CLI 也已在 case8 独立进程 BAAB 跑完 12 次（8 正式）：同一 SpeedKit decoder 的 serial **12.395756s** → stream **9.949752s**，该阶段减少 **19.73%**，三项 SHA 仍与历史一致。[原始记录](../evidence/r85-stream-1002/public-cli-c8-baab.json)。这组同时验证最终提交源码和可公开执行的 CLI；未计入上面的冻结 runner 表格。
