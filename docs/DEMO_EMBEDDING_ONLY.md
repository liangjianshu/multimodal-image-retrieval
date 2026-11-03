> 此页记录纯 Embedding 对照。当前展示已恢复 Qwen 意图识别与类别粗筛，见 [当前指南](DEMO.md)。

# EmbeddingGemma 2 原图检索演示

[当前在线演示（已更换流程）](https://liangjianshu.github.io/multimodal-image-retrieval/) · [原始结果](demo/baseline-embeddinggemma2/results.json) · [旧 Qwen LoRA / Taiyi-CLIP 流程](DEMO_TAIYI_CLIP.md)

此历史版本在 12 张 CC0 图片上运行 7 组固定查询：文字 3 组、参考图 3 组、图文联合 1 组。结果由模型实际计算，没有人工重排。在线页面和离线 `docs/demo/index.html` 都是结果回放，不是实时推理服务；离线文件已内嵌图片。

历史页面及截图可在 [5c33666 版本](https://github.com/liangjianshu/multimodal-image-retrieval/tree/5c33666/docs/demo) 查看；当前网页已切换为 Qwen 意图与类别粗筛流程。

## 展示阈值

演示默认统一使用 **最低余弦相似度 0.66**，在每组原始 Top-3 中保留达到阈值的结果，不足三张就少展示。滑块可调整，点击“查看未过滤结果”恢复原始前三名。所有示例共用同一个设置，没有按图片 ID 或类别删除特定结果。

花纹杯查询原始分数为 0.802449、0.721974、0.653903：前两张杯子保留，第三张椅子被阈值过滤。白色自行车查询也会从三张减少到一张；这说明阈值确实对所有示例生效。阈值在当前小图库示例上选定，未经过独立验证；提高阈值也可能漏掉相关结果，不能将它表述为模型准确率提高或通用最优阈值。

[原始 JSON](demo/baseline-embeddinggemma2/results.json) 与下方对照表保留未过滤的模型排名和分数。网页设置只影响展示，CLI 默认仍返回未设阈值的 Top-K；使用 `--min-score 0.66` 可应用同一规则。页面构建可用 `build_page.py --min-score 0.66` 改变默认值。

## 编码流程

1. 校验 [图库来源清单](../examples/showcase/sources.json) 的图片 SHA-256，直接将原图输入 EmbeddingGemma 2，生成归一化的 768 维向量。
2. 文字查询保留用户原文，添加官方搜索前缀 `task: search result | query: ` 后编码；图片查询直接编码像素。
3. 图文查询把完整文字、`<|image|>` 标记和参考图作为一个多模态输入，单次前向生成查询向量。没有 Qwen 摘要覆盖，也没有 50/50 向量平均。
4. 在全部图库图像向量上做余弦排序，排除参考图自身，取前三名。没有类别粗筛或描述向量排序。
5. 页面图片说明沿用原微调 Qwen 生成的元数据，仅为阅读提供上下文，不参与查询编码、建库或排序。

当前编码器使用公开预训练权重，未进行本项目领域微调；原 Qwen LoRA 不是这个编码器的适配器。两条流程分别保存，不能混用向量或把旧微调的收益算到新编码器上。

## 实测配置

| 项目 | 值 |
| --- | --- |
| 模型 | `google/embeddinggemma-2` |
| 固定版本 | `f9972ac22e696a105c482202646979e33e4b5a1e` |
| 权重 SHA-256 | `197a32965d4b1105faf060417baa899e193fb73cd401f42ec9295234d5553d79` |
| 加载模块 | 文字 + 视觉，关闭音频；438,760,448 参数 |
| 维度 / 图像预算 / 批量 | 768 / 280 tokens / 4 |
| 设备与精度 | 单张 Ascend 910B2，`npu:0`，BF16，eager attention |
| PyTorch / torch_npu | 2.7.1 / 2.7.1 |
| Transformers / Sentence Transformers | 5.22.1 / 6.2.0 |

官方模型及多模态输入方式见 [Google 模型卡](https://huggingface.co/google/embeddinggemma-2) 与 [官方 Sentence Transformers 指南](https://ai.google.dev/gemma/docs/embeddinggemma/multimodal-embeddinggemma-with-sentence-transformers)。运行时验证官方固定版本的权重哈希。首次使用需下载完整 checkpoint；本次只加载文字和视觉模块。

## 耗时与旧结果对照

| 查询 | 旧 Taiyi/CLIP Top-3 | EmbeddingGemma 2 Top-3 | 编码中位数 |
| --- | --- | --- | --- |
| 白色自行车 | image-09, image-12, image-10 | image-09, image-12, image-10 | 90.0 ms |
| 红色的椅子 | image-01, image-04, image-02 | image-01, image-03, image-04 | 90.0 ms |
| 咖啡杯 | image-08, image-06, image-07 | image-07, image-08, image-06 | 90.7 ms |
| 一张参考图，找到更多杯子 | image-08, image-06, image-05 | image-08, image-06, image-02 | 173.2 ms |
| 给一张椅子图，找相似素材 | image-04, image-03, image-01 | image-03, image-01, image-04 | 187.1 ms |
| 用咖啡杯照片找相似图片 | image-05, image-07, image-08 | image-07, image-08, image-05 | 175.3 ms |
| 草地上的自行车 | image-10, image-12, image-11 | image-10, image-12, image-11 | 174.6 ms |

图库 12 张图片批量编码：**1.45 秒**。首次模型初始化（含权重哈希校验）：28.59 秒；首次单图预热：10.18 秒。这两个启动数字各测一次，不作为稳定启动延迟。

每个查询先预热，再测 5 次，表中为中位数。编码耗时包含预处理、设备传输、模型前向、归一化和返回 CPU 向量；不含加载、预热、网络与索引扫描。12 张图的 NumPy 扫描中位数均低于 0.15 ms。没有测并发、网络服务或大图库吞吐，不能线性外推生产速度。当前 CLI 每次调用都会重新加载模型；服务集成时应复用 `GemmaEncoder` 实例。

**不能宣称新模型全面胜出。** 黄色杂草自行车 `image-10` 在新旧两版都排第一，新版与第二名分差很小。花纹杯查询中，新版第三名为椅子 `image-02`，原类别粗筛版本则保留了杯子类结果。新方案取消了类别粗筛，因此比较的是完整流程变化，不是仅替换一个编码器的严格消融。跨模型余弦分数不能直接比较大小来判断优劣。

图库只有 12 张，查询未建立完整的独立相关性标注；此处只报告排名与耗时，不报告准确率提升。严格同款、材质、否定要求和背景修改仍需专门评测。

## 复跑与换图库

在独立环境按 [README](../README.md) 安装 `.[gemma]`，然后：

```bash
python examples/showcase/run_gemma.py \
  --config configs/embeddinggemma2.json --output artifacts/showcase-gemma2 --repeats 5
python examples/showcase/build_page.py \
  --results artifacts/showcase-gemma2/results.json --output artifacts/showcase-gemma2/index.html
python examples/showcase/build_page.py \
  --results artifacts/showcase-gemma2/results.json --image-mode external --output docs/index.html
```

本地已有权重可传 `--model-path /path/to/embeddinggemma-2`；Ascend 需传 `--device npu:0`。运行脚本输出持久化索引和含环境、每次耗时、排名的结果 JSON；HTML 生成器只渲染，不改变排序。

Ascend 请先安装厂商匹配的 torch / torch_npu / torchvision 和 CANN，再在独立环境安装 `transformers==5.22.1 sentence-transformers==6.2.0` 及本项目基础依赖，检查 pip 解析结果不要替换已有 torch。不要在原 Qwen 的 Transformers 4.57.1 环境中直接升级。

自己的图库只需图片文件和包含 `id`、`file_path` 的 JSONL；类别/描述可以为空。按 [自有数据指南](CUSTOM_DATA.md) 直接执行 `index`，随后 `search` 或 `evaluate`。每次切换模型、权重、向量维度或图像预算必须重建索引；代码拒绝查询端与索引的配置混用。

历史原文保留版完整结果见 [baseline-taiyi-clip](demo/baseline-taiyi-clip/results.json)；更早的丢失场景版本见 [baseline-lora-intent](demo/baseline-lora-intent/results.json)，3B 基座版本见 [baseline-3b](demo/baseline-3b/results.json)。
