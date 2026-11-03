# 原始图库恢复与多参数微调实验

状态：原始 2,000 张图库已完整恢复，正在开展正式参数实验，尚无本轮正式测试结论。早先的 297 张划分结果属于独立的小规模试验，不能替代这里的原始 2,000 张评测。

## 数据来源与身份

原清单为 `legacy/project_one/imagenet_json.json` 对应的服务器原件，SHA-256：
`1ff13a0b4e74bb69d7140ea38a14fdf4b2adf63fd9c6da787f8fdbd2600e7e79`。

仓库提供去掉旧机器绝对路径的可移植清单：[`original_test_manifest.json`](../examples/training/original_test_manifest.json)。全部 2,000 个原文件名、类别、ID 及分组保留不变；该导出文件自身 SHA-256 为 `735c43b4547e4b8a0a3ab2e6e74a09baa74a9bf172ed16b7253abf4924b6ec0f`。图片内容不随清单分发。

原脚本从 ImageNet synset tar 包抽取图片，保留原文件名。图库含 5 个目标类各 40 张、5 个混淆类及 5 个其他类各 180 张，共 2,000 张；每类 normal/fine 各半。fine 是原目标检测面积规则下的小目标分组，不是颜色或材质属性标注。

已有归档恢复了 1,280 张。原先缺少的甜甜圈、键盘、橙子、披萨共 720 张，已按原 ImageNet ID 从公开副本 [gmongaras/Imagenet21K](https://huggingface.co/datasets/gmongaras/Imagenet21K) 恢复，固定 revision：
`a23c2bbe64852ae38c386ec5fdb64767b233f4e0`。

该副本保存 PNG。5 张可与本地旧归档交叉验证的杯子图片，解码后的 RGB 像素 SHA-256 全部相同。原清单没有旧 JPEG 文件哈希，因此不能声称新 PNG 与旧 JPEG 的文件字节相同。恢复记录同时保存原文件名、新实际路径、来源分片、新文件及像素哈希。

不重新随机抽取 2,000 张同类图片来冒充原图库。完整恢复检查失败时，训练流水线不会进入正式实验。

## 训练与验证

从相同 15 类的归档及公开副本收集到 21,477 张有效图片，排除测试重叠与重复后剩余 19,205 张。使用真实 synset 标签，弃用旧 SFT 中把负类图片回答为随机正类的答案。

- 先排除原测试图片 ID、相同解码像素以及 dHash 汉明距离不超过 2 的潜在重复。
- 在剩余数据中先做重复分组；跨标签冲突组剔除，同类组保留一张，再划分训练与验证。
- 实际训练每类 300 张，共 4,500 张；验证为目标类各 20 张、其他类各 90 张，共 1,000 张；另有 13,705 张干净图片暂不参与本轮训练。
- 原测试图库保持 2,000 张及原 normal/fine 分组。另审计旧训练集与测试集的潜在重叠。
- dHash 是保守排除规则，不是证明两张图完全相同的依据。

Qwen 新监督任务为图像类别 JSON 输出。它学习正确的类别及混淆类边界；没有据此声称训练了颜色、材质或完整图像描述。

Embedding 每张训练图构建文本查询及另一张同类参考图查询，配一个显式异类负样本；相邻混淆类别优先。批内同类候选都作为正例，避免把它们误当负例。该监督仍是类别级相关性，不代表实例或细节属性的检索标注。

## 实验矩阵与选择规则

| 模型 | 主学习率 | 检查点 epoch | 追加比较 |
|---|---|---|---|
| EmbeddingGemma 2，文本及视觉全参数 | 1e-6、2e-6、5e-6 | 0.5、1、2 | 最佳验证学习率下 temperature 0.05 与 0.1 |
| Qwen2.5-VL-7B，语言注意力 LoRA | 5e-6、1e-5、2e-5 | 0.5、1、2 | 最佳验证学习率下增加视觉 merger 训练 |

初始 Qwen LoRA rank 64、alpha 128，与保留的旧 adapter 实际作用范围对齐。旧权重并不含训练后保存的 merger 参数，不能将它描述成已经验证的 MLP+LLM 权重。新 Qwen 实验将所有可训练参数保留为 FP32 主参数，包含 merger，避免小学习率更新在 BF16 参数中被舍入；训练及推理使用混合精度。

固定 seed 42；训练用 AdamW、梯度裁剪及前 5% 步数线性预热。Embedding 使用 FP32 主参数和 BF16 autocast，梯度检查点降低显存；验证从保存文件重新加载 BF16 推理模型。梯度累积不被描述为扩大对比损失的批内负样本池。

Embedding 按验证集目标五类的文搜图与图搜图 mAP 均值选择。Qwen 按验证集目标五类粗筛 F1 选择。所有模型及轮数选择冻结后才运行测试。粗筛验证分数与完整链路检索分数分别报告。

## 固定测试比较

比较原始 Embedding、最佳新 Embedding、Qwen 基座、旧 LoRA、最佳新 Qwen。Qwen 完整链路使用模型类别粗筛、Taiyi 描述向量排序及 CLIP 参考图片向量；Embedding 使用原图/文本共享空间，不加 Qwen 类别筛选。意图识别仍属于共同的 Qwen 上游能力，本次固定类别查询评测不冒充自由自然语言意图评测。每类固定三种中文查询：类别名、“给我看看{类别}”、“这是一张{类别}的照片”。不将英文直接送入原中文编码器来制造不公平对比。

统计 mAP、Precision@1/5/10/20、Recall@K、验证集选阈值后的 F1、每类及 normal/fine 分组结果，以及训练耗时、参数量、逐查询输出。文搜图按固定查询在验证集校准阈值；图搜图用验证集目标五类的查询统一校准一个阈值，测试时不重新选择。图搜图排除参考图片自身。粗筛漏掉的正例仍保留在召回率和 AP 的分母中。

旧训练清单的 3,996 个独立文件已完成图像重叠审计：1 张原测试图与其解码像素相同，合计 149 张存在潜在重叠，正式报告另列剔除这些已检出重叠后的结果。其中 3 张旧 COCO 文件只下载了一部分；已从其原始 COCO URL 找回完整文件，并验证旧字节是完整文件的精确前缀。补回文件单独保存，旧文件未覆盖；这不代表旧模型当时训练过完整图片。当前旧清单也不是当年训练输入的不可变快照，因此审计不被描述为完全证明旧模型的训练成员。

旧脚本曾在同一测试集上遍历阈值取最高 F1。若复算这套历史口径，将明确标注为测试集最优阈值诊断，不与验证集校准后的测试 F1 混称。

## 运行与审计文件

入口：`examples/training/run_restored_study.py`。恢复脚本支持按需读取 Parquet 字节范围，不下载整个多 TB 数据集。所有 HTTP Range 响应都核对偏移与长度。

缺失四类的固定分片映射保存在 [`public_recovery_shards.json`](../examples/training/public_recovery_shards.json)。例如，在安装 `huggingface_hub`、`pyarrow`、`requests` 和 `Pillow` 的环境中补齐这四类：

```bash
python examples/training/recover_hf_imagenet.py \
  --shards examples/training/public_recovery_shards.json \
  --output data/hf-class-pool --parallel-ranges 8 \
  --classes n07639069 n03614007 n07747607 n07873807
python examples/training/restore_original_gallery.py \
  --manifest examples/training/original_test_manifest.json \
  --archives /path/to/original-synset-archives \
  --public-pool data/hf-class-pool --output data/original-eval
```

其余十一类仍从原 synset 归档恢复；上面的四类补图命令不等于独立下载完整图库。恢复脚本会明确列出缺失 ID，不会悄悄换成其他图片。

实验目录包含：

- `data/original-eval/recovery-report.json`：原始图片恢复计数与来源。
- `data/study-v1/data-report.json`：划分、排除原因、清单哈希。
- `legacy-overlap.json`：旧训练与原测试的重叠审计。
- `trials/*/trial.json`：每组参数、训练损失、验证指标与检查点。
- `hyperparameters.csv`：各参数与轮数汇总。
- `selection.json`：在测试前冻结的模型选择。
- `evaluation/*/test-results.json`：完整测试结果及逐查询记录。
- `study-status.json` 和 `logs/`：运行状态、命令、耗时和失败记录。

报告入口：`examples/training/report_restored_study.py --study-root /path/to/study --output /path/to/report`。它要求 8 组完整实验及 24 个检查点、5 组最终评测全部完成，核对选择记录、数据清单、权重哈希和验证集阈值后，输出 HTML、JSON 和 CSV。逐查询差值保留退化案例，不只展示改善。

本基准用固定 15 类别名映射粗筛，以及不混入生成描述的纯参考图查询。网页使用开放类别的语义粗筛及图文融合，因此网页案例单独验收；基准指标不能直接当作整个网页自由查询的准确率。

测试图片、训练图片及权重不作为代码仓库附件上传。最终结果尚未完成时，不以准备阶段试跑分数作为正式效果指标。
