# 真实检索演示与复现

这组演示于 2026-11-03 使用真实预训练模型运行。图库是 12 张 CC0 图片，包含椅凳、杯子和自行车；查询包含中文文字、参考图片与图文融合，并保留方法对照和表现不理想的案例。

## 直接看效果

下载仓库后，用浏览器打开 [`demo/index.html`](demo/index.html)。点击查询方式或示例按钮可以切换已记录结果，点击结果图片可以查看完整的模型描述。页面可离线浏览，不需要下载模型。

**这是已记录结果的交互回放，不是在线推理服务。** 所有展示排名来自 [results.json](demo/results.json)，没有手工调整图片顺序。无法在 GitHub 文件预览中执行 HTML 时，请下载仓库后打开。

![中文文搜图](demo/text-search.png)

![参考图搜图](demo/image-search.png)

![图文融合](demo/fusion-search.png)

## 实际运行了什么

1. 从 Wikimedia Commons 获取 12 张 CC0 图片，保存源链接、作者和 SHA-256。见 [图片来源](../examples/showcase/SOURCES.md)。
2. 输入清单的人工 `categories` / `description` 均为空。Qwen2.5-VL-3B 根据图片生成 [检索类别与描述](demo/catalog.described.jsonl)，没有手工修改这些输出。
3. Taiyi-CLIP 文本侧编码类别、描述和中文查询；CLIP ViT-L/14 编码图片。用真实向量建立索引。
4. 文字及图文融合采用类别粗筛、描述向量排序；纯图片查询同时展示 CLIP 图像向量排序和语义检索对照。两种实现都来自仓库现有 `Index.search`。
5. 固定 `category_ratio=0.85`、`image_weight=0.5`、`top_k=3`；参考图在图库内时排除自身。返回不足 3 张时不补入被粗筛排除的图片。

本次没有运行意图解析，也没有加载领域 LoRA。`configs/showcase.json` 使用 3B 视觉基座，与默认 7B 配置区分。

## 自己运行一遍

在仓库根目录，先按 [README](../README.md) 安装与设备匹配的模型依赖，再运行：

```bash
python examples/showcase/run.py --output artifacts/showcase
```

它会校验随仓库提供的图片、生成描述、建立索引、运行全部查询，最终保存：

```text
artifacts/showcase/
├── catalog.described.jsonl
├── index/                     # 图片清单、真实特征向量与索引元数据
└── results.json               # 所有查询、排名、分数、配置与环境版本
```

如果描述阶段已完成，可追加 `--reuse-descriptions` 复用对应输出，避免重复调用视觉模型。本次发布的结果使用了这一选项，`description_seconds` 因而仅表示读取已有描述的时间，**不是生成 12 张图片描述的耗时**。记录的耗时未经重复测量，不用作性能基准。

用新结果生成自己的离线页面：

```bash
python examples/showcase/build_page.py \
  --results artifacts/showcase/results.json \
  --output artifacts/showcase/index.html
```

脚本只渲染结果，不执行检索或改变排名。页面会引用仓库中的图片；分享时需要一并保留图片目录及相对路径。

如果模型已下载到本地，可用 `--model-root /path/to/models`，目录应包含 `taiyi/`、`clip/`、`qwen-vl-3b/`。这三个目录必须分别对应配置中的 Taiyi、CLIP 与 Qwen2.5-VL-3B 模型，不能随意替换为不同模型后继续沿用名称。

## 本次环境

| 项目 | 设置 |
| --- | --- |
| 硬件 | 单张 Ascend 910B2，显式 `npu:0` |
| Python | 3.11.15 |
| PyTorch / torch_npu | 2.7.1 / 2.7.1 |
| Torchvision | 0.22.1 |
| Transformers | 4.57.1 |
| qwen-vl-utils | 0.0.14 |
| Qwen 精度 | bfloat16；贪心生成 |
| 输入限制 | `max_pixels=262144`、`max_new_tokens=256` |
| 描述模型 | Qwen/Qwen2.5-VL-3B-Instruct，无适配器 |
| 文本编码器 | IDEA-CCNL/Taiyi-CLIP-Roberta-large-326M-Chinese |
| 图像编码器 | openai/clip-vit-large-patch14 |

下载后的模型文件哈希见 [model-files.json](demo/model-files.json)，模型权重不随仓库分发。不同硬件、精度或处理器版本可能产生不同描述与排序。

Ascend 需要先准备与本机 CANN 配套的 `torch` / `torch_npu` / `torchvision`，不要在这个环境中直接安装会替换 PyTorch 的 `.[models]` 依赖组。安装基础包和所需推理依赖后，可以显式指定：

```bash
pip install -e .
# 在已经配好 Ascend PyTorch 的环境中，补充对应推理依赖
pip install transformers==4.57.1 qwen-vl-utils==0.0.14
ASCEND_RT_VISIBLE_DEVICES=0 python examples/showcase/run.py --device npu:0
```

仓库的 `auto` 仍然优先 CUDA，否则选择 CPU。此次验证范围是上述 Ascend 环境上的描述生成与检索；没有验证 NPU 训练。

## 结果应该怎样理解

- **颜色约束不是硬过滤**：“红色的椅子”案例中，黑色椅子排在红色凳子之前。当前描述向量排序没有保证属性条件逐项成立。
- **类别粒度会影响召回**：模型可能把相似物体分别描述为“椅子”“办公椅”或“家具”。语义粗筛会排除部分外观相关图片；纯图搜图可查看 CLIP 图像向量方法的对照。
- **CLIP 也可能跨类别误召回**：参考黑色椅子时，CLIP 的结果中出现了杯子；它的相似度并不等同于业务上的同类或同款。页面保留了这组结果。
- **描述可能含推测或错误**：生成文本按原样保留，不能当作人工标注真值。
- **小图库有明显局限**：12 张图片只能检查流程和展示个案，不足以证明实际商品库的准确率；结果也不证明同品牌同型号识别能力。

如果接入自己的数据，按照 [自有数据指南](CUSTOM_DATA.md) 重建清单、描述与索引，并准备独立查询和相关性标注后评测。这里没有宣称历史文搜图 / 图搜图 F1 提升已经复现。
