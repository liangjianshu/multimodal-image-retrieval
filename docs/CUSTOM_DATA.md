# 换成自己的数据

## 路线 A：只做素材检索

1. 图片保留在运行机器上，准备 `file_path` 相对同一个 `image-root` 的清单。
2. 默认 `qwen-gemma2` 的 `index` 会调用 Qwen 生成类别、归并类别名称，再用 EmbeddingGemma 2 编码原图；原始类别和描述可以为空。
3. 建立新索引，使用 `search`。这条路线不需要训练模型。

建议目录：

```text
data/
├── images/                 image-root
│   ├── catalog/001.jpg
│   └── queries/reference.jpg
├── catalog.jsonl           仅包含待检索的图库图片，不要把独立查询图片加入
└── queries.test.jsonl
```

对应图库清单：

```json
{"id":"item-001","file_path":"catalog/001.jpg","categories":["椅子"],"description":"红色木质椅子"}
```

参考图可以位于 `queries/`；只有写进 `catalog.jsonl` 的图片会入库。如果通过 `prepare` 递归扫描目录，目录内全部支持的图片都会入库，因此请在扫描后添加查询图片，或自行生成只包含图库图片的清单。

## 路线 B：增加领域微调

此路线针对独立的 Qwen / Taiyi-CLIP 历史分支（`configs/taiyi-clip.json`、`.[models]` 环境），不是 EmbeddingGemma 2 微调。在路线 A 的数据基础上，核对图片的主要物体标签。`make-sft` 使用 `categories` 生成训练答案，不会把 `retrieval_categories` 预测值直接当作真值。

类别映射与难负类别对需要换成你的领域。例如鞋类：

```json
{"运动鞋":["休闲鞋","登山鞋"],"休闲鞋":["运动鞋"],"登山鞋":["运动鞋"]}
```

它表达“哪些类别容易混淆”，不改变图片真实标签。多物体图片要明确“主要物体”的标注范围；标签不完整会产生错误的否定问答，应先人工核对。当前自动构造器主要生成类别识别与类别核验任务；属性训练需要自己准备有依据的属性问答，不能把它称为已完成属性监督。

`make-sft` 会：

- 校验图片文件、唯一 ID、路径、标签。
- 按图片内容 SHA-256 分组，重复图片的不同路径不会进入不同划分。
- 在每组标签内固定随机种子划分，再扩展多条问答。
- 对每个类别保留训练图片；类别仅有一张独立图片时，该类别没有验证样本。

内容哈希只能排除文件完全相同的重复项；视频相邻帧、不同压缩版本和同一商品连拍仍需按来源分组去重。严肃实验应另留独立检索测试图库/查询，不用训练验证集替代最终检索测试集。

## 字段说明

默认混合链路使用 Qwen 生成的 `retrieval_categories` 和 `retrieval_category_groups` 进行粗筛，EmbeddingGemma 2 直接编码原图。纯原图基线不使用类别，历史 `taiyi-clip` 还使用描述向量排序。

| 字段 | 来源 | 用途 |
| --- | --- | --- |
| `id` | 用户 | 索引和评测关联 |
| `file_path` | 用户 | 相对 image-root 的文件位置 |
| `categories` | 人工标签 / 分类目录 | SFT 真值；无模型元数据时也可用于素材检索 |
| `description` | 人工描述，可为空 | 无生成描述时用于索引 |
| `retrieval_categories` | Qwen 预测 | 原类别与类别归并的来源 |
| `retrieval_category_groups` | Qwen 归并结果 | 当前混合检索的类别粗筛 |
| `retrieval_description` | Qwen 预测 | 当前仅供展示；历史分支用于描述向量排序 |
| `retrieval_source` | `describe` 自动记录 | 模型和适配器来源追踪 |

`examples/catalog.jsonl` 是格式示例，其中图片不随仓库分发。不能直接对示例文件执行真实建库，必须将它替换成现有图片对应的清单。

## 已有 COCO 格式标注

无需重新下载图片。以下示例导入本地 `instances` 标注，产生相对路径清单。请根据任务人工处理多物体/背景标签；默认混合检索使用 Qwen 预测类别，不使用标注类别；评测历史语义流程时，先 `describe` 生成预测字段，以免把标注类别当作模型预测能力。

```python
import json
from collections import defaultdict
from pathlib import Path

annotation = json.loads(Path("data/instances.json").read_text())
names = {x["id"]: x["name"] for x in annotation["categories"]}
labels = defaultdict(set)
for item in annotation["annotations"]:
    labels[item["image_id"]].add(names[item["category_id"]])
with open("data/catalog.jsonl", "w", encoding="utf-8") as out:
    for item in annotation["images"]:
        row = {"id": str(item["id"]), "file_path": item["file_name"],
               "categories": sorted(labels[item["id"]]), "description": ""}
        out.write(json.dumps(row, ensure_ascii=False) + "\n")
```

中文检索建议将英文类别映射为准确的中文标签。ImageNet wnid 目录同样需要 wnid → 类别名映射，不能直接把 `n03001627` 当作有意义的中文类别词。

## 评测查询

```json
{"id":"q1","text":"红色椅子","relevant_ids":["item-001","item-005"]}
{"id":"q2","image":"queries/reference.jpg","relevant_ids":["item-008"]}
{"id":"q3","text":"类似款式的红色鞋","image":"queries/shoe.jpg","relevant_ids":["item-010"],"exclude_ids":["item-011"]}
```

`relevant_ids` 是图库中所有满足该查询的图片 ID，必须非空且存在于索引。参考图属于图库时自动排除自身，相关性标注也必须排除它。多个相关图片在 Top-K 截断下可能导致 Recall/F1 低于 1，不能将检索 F1 当作分类准确率。

## 每次换数据时检查

1. `validate` 成功，图片路径不再依赖原服务器。
2. 自定义类别、描述、相似类别对均与自己的任务一致。
3. 更换模型、图片或视觉预算后，在新目录重建索引；新旧编码空间不能混用。
4. 查询 ID 与新图库对应；训练/验证/测试和近重复图片隔离。
5. 在验证集确定视觉 token 预算、向量维度与返回阈值，再固定设置跑测试。旧分支还需选择类别阈值与融合权重。

图库规模较大时，全量内存索引和重建成本会成为瓶颈。请先小规模验证，原图编码已有可配置批量；增量索引或 FAISS / 向量数据库尚未接入。
