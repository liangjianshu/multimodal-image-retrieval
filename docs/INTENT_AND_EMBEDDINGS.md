# 查询条件保留与图文向量对齐

## 两条路线共享 Qwen 意图识别

默认路线 A 为 `qwen-semantic`：Qwen 生成图库类别和描述，类别向量粗筛后按描述向量排序。路线 B 为 `qwen-embedding`：EmbeddingGemma 2 直接编码原图，不进行 Qwen 类别筛选。两者每次查询均调用 Qwen 判断是否检索和返回数量，纯图片也经过意图判断。非检索请求在向量编码前返回。实现见 [routes.py](../mmretrieval/routes.py)。

两条路线使用 [intent.py](../mmretrieval/intent.py) 中的 `ROUTE_INTENT_PROMPT`，保留原始用户文字；Qwen 摘要不覆盖它。路线 A 分别编码目标类别和完整查询，避免把颜色等条件直接塞进类别粗筛。路线 B 把原文与参考图共同输入多模态编码器。完整输入不丢失，不等于所有属性都能被准确满足。

旧 `qwen-gemma2` 混合版（Qwen 类别筛选＋原图向量排序）可通过 `configs/showcase-hybrid.json` 独立复跑，其类别名称归并规则仍在 [hybrid.py](../mmretrieval/hybrid.py)。它与当前 A/B 路线是不同实验，不混记效果。

## Prompt 工程：从自然语言到可执行查询

本项目的 Prompt 工程包含任务拆分、字段定义、输出约束与示例引导，并由程序保留输入、校验结果。它与 LoRA 权重训练分别负责不同环节。

| 设计点 | 项目中的具体做法 |
|---|---|
| 意图边界 | 区分图片检索、知识问答与只描述图片；结合参考图判断相似图片检索。 |
| 结构化输出 | 定义 `needs_retrieval`、`target_category`、`description`、`constraints`、`top_k`，要求 JSON，并在代码中校验字段。 |
| 数量消歧 | `top_k` 只表示返回几张图片；“三张有两只猫的图片”应返回 3 张，两只猫属于内容条件。 |
| 条件保留 | 保留颜色、材质、场景、否定和修改要求；不把完整需求缩成一个类别。程序始终优先编码用户原文。 |
| 图文冲突处理 | 用户要求的文字条件优先于参考图背景，例如参考图靠蓝墙，查询“草地上的自行车”仍保留草地条件。 |
| 阶段专用提示词 | 类别提示词用于生成类别列表，描述提示词覆盖物体类型、颜色、材质、形状和场景；不把所有任务合并成同一个输出。 |

例如，“找三张有两只猫的图片”的字段设计示意为：

```json
{"needs_retrieval":true,"target_category":"猫","description":"有两只猫的图片","constraints":["两只猫"],"top_k":3}
```

这是格式说明，不是新增模型实测结果。提示词约束不能保证模型始终遵循，也不能保证检索满足全部属性，因此保留了[条件保留与输出校验测试](../tests/test_intent_preservation.py)、非检索示例和[格式归一化对照](QWEN_CONTROLS.md)。前期意图识别 95%／91% 是独立历史实测，不能全部归因于本次提示词整理。

源码入口：[`intent.py`](../mmretrieval/intent.py) 中的 `ROUTE_INTENT_PROMPT`（当前双路线）和 `PRESERVE_INTENT_PROMPT`（条件保留实现），以及 [`historical_prompts.py`](../mmretrieval/historical_prompts.py) 中的类别／描述提示词。新类别训练使用 [`build_restored_study.py`](../examples/training/build_restored_study.py) 的 `CATEGORY_PROMPT`，各实验的提示词口径分别记录。

## 原提示词为什么会丢细节

原意图提示词保存在 [`historical_prompts.py`](../mmretrieval/historical_prompts.py)。它要求“用简短的语句描述用户想要检索的目标物体”，并把“和这张图片风格相似的椅子图片”示例简化为“椅子”。这会鼓励摘要化；旧代码随后用摘要替换了用户原文。

旧版真实记录中，输入“草地上的自行车”变成了“自行车”。图库中黄色车的描述已经包含“杂草丛生”，问题发生在查询端。修复前结果保存在 [baseline-lora-intent](demo/baseline-lora-intent/results.json)。

## 历史 Qwen 分支的规则

完整可执行提示词见 [`intent.py`](../mmretrieval/intent.py)。它分别提取主要类别、完整目标描述、用户约束和返回数量，要求：

- 保留颜色、材质、形状、场景、位置、关系、否定和修改要求。
- 用户文字优先于参考图；不能用图里的颜色、背景覆盖用户要求修改的属性。
- “找三张有两只猫的图片”中，返回数量是三，物体数量是两；未要求返回数量时 `top_k=null`。
- 只有参考图时，根据可见主体和主要外观生成目标描述。

提示词之外，还有代码保障：有用户文字时，`resolve_search_text` 始终把用户原文交给编码器，不使用模型摘要替换它。即使模型只返回“自行车”，原文“草地上的自行车”仍然进入编码。只有图片、没有文字时，直接根据图片生成类别与描述，并由代码固定为图片检索；不再让模型判断是否需要检索，也不把图中观察到的属性当成用户明确提出的限制。CLI 和演示脚本共用此逻辑，并在结果中保存 `search_text`。

超过当前文本编码器 512 token 上限时明确报错，不再静默截掉尾部条件。这个保障只针对输入不被删除；它不保证向量相似度能正确理解每个否定词或满足每个属性。`constraints` 当前用于审计，并未实施逐条硬过滤。

`intent_profile=preserve` 为默认；`historical` 仅用于重跑旧方法。图库的类别和描述提示词仍由 `prompt_profile` 控制，两者可以独立选择。旧流程复跑配置：`configs/showcase-lora-historical-intent.json`。

## 历史 Qwen / Taiyi-CLIP 分支编码了什么

| 数据 | 编码方式 | 在主展示中的用途 |
| --- | --- | --- |
| 图库类别、详细描述 | 微调 Qwen 生成文字，Taiyi 编码 | 类别粗筛、描述相似度排序 |
| 用户文字 | Taiyi 直接编码原文 | 查询文字分支 |
| 参考图片 | CLIP 直接编码像素 | 查询图像分支，与文字向量各占 50% 后归一化 |
| 图库原图 | CLIP 编码并保存 | 可用于纯 CLIP 基线；当前语义模式未用于候选的最终打分 |

因此并非“参考图只转成文字再检索”。参考图的直接视觉特征已经参与；候选图片的最终表示仍主要依靠描述，可能遗漏细微纹理、轮廓、局部结构等信息。

## 融合需要共享向量空间，不需要共享词表

文字 tokenizer 的词表把字符串转换成 token ID；图像编码器处理图片 patch，不需要和中文词表一一对应。跨模态比较依赖训练得到的共同输出空间。维度相同只是计算上可相加，不代表语义坐标对齐。

当前组合并非随意拼接：Taiyi 官方说明它冻结 CLIP ViT-L/14 视觉编码器、训练中文文本侧，并提供 `IDEA-CCNL/Taiyi-CLIP-Roberta-large-326M-Chinese` 与 `openai/clip-vit-large-patch14` 配套的示例。当前代码使用的正是这组模型。[官方模型卡](https://huggingface.co/IDEA-CCNL/Taiyi-CLIP-Roberta-large-326M-Chinese)

共同空间允许跨模态比较，但不保证简单平均能表达“保留这辆车的主体、把蓝墙背景改成草地”。文字条件与参考图属性冲突时，需要单独评估组合查询的效果。

## 编码器选择与后续评测

建议对照 **EmbeddingGemma 2** 与 **Qwen3-VL-Embedding-2B**，不根据参数量或发布时间预判质量。

Google 于 2026-10-06 发布的 `google/embeddinggemma-2` 是原生多模态模型，完整模型 740M 参数，可选择加载视觉和音频模块；文字、图片及混合输入映射到统一的 768 维空间。第一代 EmbeddingGemma 是纯文本模型，应区分版本。对于本项目，第二代可以直接编码商品图片和图文联合查询，适合作为轻量候选先评估。[官方模型卡](https://ai.google.dev/gemma/docs/embeddinggemma/model_card_2?hl=en) · [发布日期](https://blog.google/innovation-and-ai/technology/developers-tools/embeddinggemma-2/)

**Qwen3-VL-Embedding-2B** 可作为另一组对照。官方支持文字、图片及图文混合输入，并映射到统一空间，最高 2048 维。可以让查询端直接接收“参考图＋完整修改文字”，图库端直接编码原图，或原图加商品字段。[官方模型卡](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B)

EmbeddingGemma 2 已接入并完成 12 张图库实测，作为当前原图编码器；Qwen3-VL 方案尚未接入。新旧排名并非全部改善，具体对照见 [演示指南](DEMO.md)。切换时应同时重建查询和图库的编码流程与全部索引，不能把新的向量与旧 Taiyi/CLIP 向量直接混用。即使 EmbeddingGemma 2 和旧索引同为 768 维，其学习到的坐标空间也不同，仍须重建全部向量。现有 Qwen2.5-VL-7B LoRA 对应不同基座，不能直接加载到这些 Embedding 模型；可以继续保留它负责图库描述和属性提取。

评估应覆盖同类别难负例、颜色和材质差异、否定要求、背景修改、相似款与同款，并比较 Recall@K、排序指标、延迟和内存，而非只看 12 张演示图。若需要逐候选评估查询与商品图片的相关性，也可独立评估配套的 [Qwen3-VL-Reranker-2B](https://huggingface.co/Qwen/Qwen3-VL-Reranker-2B)；本次尚未接入。
