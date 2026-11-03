# 路线 B：原图 Embedding 微调

两条路线都由 Qwen 解析意图。路线 B 的 Qwen 只决定是否检索和返回数量；检索向量使用完整原文与参考图，**不读取 Qwen 类别来筛候选**。

已完成一次 64 步类别对比学习试训，并重新加载保存的权重完成固定测试图库评测，见 [试训报告](EMBEDDING_EXPERIMENT.md)。公开配置默认仍使用基座；微调权重需指定保存后的配置。

## 准备监督数据

对比学习训练的是相对匹配关系。例如查询“杯子”，正图是一只杯子，负图是一只容易混淆的碗。对查询与候选的余弦相似度做 softmax，令正图的匹配概率为 `p+`，最小化 `-log(p+)`。正图分数相对负图越低，损失越大。这是可微损失与反向传播，不需要强化学习的奖励模型；相似度或训练候选集内的 softmax 概率也不等于上线后的正确概率。

类别标签能帮助构造这些关系，但正负关系由查询决定：对于“杯子”，白杯也是正例；只有查询明确要求红色且颜色标注可靠时，白杯才可作为负例。不能只因另一个图片 ID 不同就把它当负样本。

每行一个 JSON 对象，图片路径相对于 `--image-root`：

```json
{"id":"red-cup-01","text":"红色陶瓷杯","positive":"train/red-cup.jpg","negative":"train/white-cup.jpg"}
{"id":"similar-cup-01","image":"train/cup-reference.jpg","positive":"train/similar-cup.jpg","negative":"train/confusable-bowl.jpg"}
{"id":"grass-bike-01","text":"草地上的自行车","image":"train/bike-reference.jpg","positive":"train/grass-bike.jpg","negative":"train/wall-bike.jpg"}
```

这些只是格式示例，不是仓库已提供的训练图片。正负关系必须针对查询核对；同类但颜色或场景不符的图片可以作为困难负例。一个查询有多个相关结果时，不应把另一个正确结果标为负例。

先按图片内容划分训练、验证和测试集，再生成问答/三元组。入口按 SHA-256 检查训练与验证图片重叠、正负图片相同、参考图等于正样本等问题。测试集不要参与训练或阈值选择。演示图库只用于展示，不用于拟合模型。

## 训练与保存

安装 `.[hybrid]`，先配置适配硬件的 PyTorch。使用同一 EmbeddingGemma 2 文字/视觉模型进行全参数对比学习；Qwen 不参与损失或反向传播。

```bash
python -m mmretrieval --config configs/route-b-embedding.json train-embedding \
  --train data/triplets/train.jsonl --val data/triplets/val.jsonl \
  --image-root data/images --output checkpoints/embedding-run-01 \
  --epochs 1 --learning-rate 0.000002 --grad-accum 4 --seed 42
```

损失为 `MultipleNegativesRankingLoss`，微批次为一个三元组，使用显式负例；梯度累积不会产生跨微批次负例。`--max-steps` 可限制优化器更新步数。学习率与轮数是起始配置，需在验证集上选择。

训练使用 FP32 参数与优化器状态，并在 NPU/CUDA 上用 BF16 自动混合精度计算，避免小学习率更新直接舍入到 BF16 参数上。保存时恢复推理精度。

## 从旧图片恢复类别监督

`examples/training/build_category_triplets.py` 可读取历史 Qwen 数据文件，仅从 ImageNet 图片路径恢复类别，**不使用旧 assistant 回答**。它按解码后的 RGB 内容去重、排除同图不同标签的冲突，再按类别划分训练、验证、测试。COCO 的多物体图片不在此入口中使用，避免把包含查询目标的图片误标为负例。

```bash
python examples/training/build_category_triplets.py \
  --manifest private/finetune_dataset.json --image-root private/dataset \
  --output data/category-triplets --seed 42
PYTHONPATH=. python examples/training/evaluate_categories.py \
  --config configs/route-b-embedding.json \
  --catalog data/category-triplets/test-catalog.jsonl \
  --image-root private/dataset --output artifacts/baseline-category-test.json
```

评测固定同一测试图库，分别计算文字和图片查询的 Hit@1、Precision@5 和 mAP；图片查询排除自身。同类别的其他图片在这套标注中均为相关结果，因此只能衡量类别检索，不能证明颜色、款式或场景约束的效果。旧 Qwen 可能已见过这些图片，不应使用此划分与旧 Qwen 做公平性声明。解码哈希只排除完全相同图片，不保证排除所有裁剪、近重复或预训练数据重叠。

输出包括 `final/` 权重、`training_report.json`（数据摘要、实际步数、损失及训练前后验证结果）和 `inference-config.json`。验证指标为正样本分数高于显式负样本的比例，**不等同于全图库 Recall@K**。

```bash
python -m mmretrieval --config checkpoints/embedding-run-01/inference-config.json index \
  --manifest data/catalog.jsonl --image-root data/images --output artifacts/index-b-finetuned
python -m mmretrieval --config checkpoints/embedding-run-01/inference-config.json search \
  --index artifacts/index-b-finetuned --image-root data/images --text '请找两张杯子图片'
```

更换权重后必须重建图库向量。配置包含微调权重校验和；不要让新查询向量匹配旧模型的图库索引。若使用自己的 Qwen LoRA，还需在推理配置中填写 `adapter`。训练数据、原始图片与权重不随代码提交到 GitHub。

## 与路线 A 比较

固定测试图库、查询、相关性标注、Qwen 意图输出和返回数量，比较路线 A、路线 B 微调前与微调后。分别报告全图库检索指标、跨类别错误、细粒度条件和延迟。训练的收益必须由路线 B 的前后对照衡量，不能把架构差异全部算成微调收益。

实现参考 [Google 官方多模态微调指南](https://ai.google.dev/gemma/docs/embeddinggemma/fine-tuning-multimodal-embeddinggemma-with-sentence-transformers)。
