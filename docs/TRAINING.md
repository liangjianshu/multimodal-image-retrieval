# 训练范围、数据和权重使用

## 三种模式

| `--mode` | 更新哪些参数 | 保存形式 | 推理配置 |
| --- | --- | --- | --- |
| `llm-lora` | 语言注意力 q/k/v/o 的 LoRA | 适配器 | 基座 `vl_model` + `adapter` |
| `mlp-llm` | 上述 LoRA + `visual.merger` 原始参数 | 适配器，含 `modules_to_save` 的 merger | 基座 `vl_model` + `adapter` |
| `mlp-vision` | 视觉编码器和 merger 原始参数，冻结语言侧 | 完整模型 | `vl_model` 指向 final 目录，`adapter: null` |

MLP＋Vision 不是整个 VLM 全量微调，也不是视觉 LoRA。完整模型的磁盘与训练开销明显高于语言侧 LoRA，请自行准备资源。

这些模式是整理版的明确实现。历史代码中的 `tune_mm_*` 开关与 LoRA 分支曾有不同实现，不能仅凭历史目录名判断当时哪些参数实际更新并保存。

## 输入

推荐使用 `make-sft` 生成 JSONL，每行：

```json
{"id":"sample-1","image_id":"image-1","file_path":"chair/001.jpg","prompt":"请输出主要物体类别列表。","answer":"[\"椅子\"]"}
```

单条样本只含一张图片，路径相对 `--image-root`。自定义属性问答也采用相同格式，必须有真实标注。训练前校验所有图片，并按文件内容检测训练/验证重叠。

损失仅计算 assistant 的答案；user 输入和 padding 设为 -100。训练入口核对 chat template 的 token 前缀，不会盲目按字符串长度遮罩。超长样本报错，不截断视觉占位 token；应降低 `max_pixels` 或缩短文本。

## 一次训练

```bash
python -m mmretrieval --config configs/default.json train \
  --train data/sft/train.jsonl --val data/sft/val.jsonl \
  --image-root data/images --output checkpoints/mlp-llm \
  --mode mlp-llm --rank 64 --alpha 128 --epochs 0.5 \
  --grad-accum 4 --learning-rate 0.00002 --seed 42
```

每设备 batch size 固定 1，主要为正确处理不同大小图片的 grid 与 token；有效 batch 还取决于梯度累积。0.5 epoch 只是示例设置，不保证对你的数据最佳。当前入口面向单进程训练，没有声称支持已验证的多卡或中断续训工作流。

输出包括 `training_scope.json`（完整可训练参数名、数量和配置）、Trainer 检查点、`trainer_state.json` 和 `final/`。不要提交模型权重到普通 Git 仓库。

## 将训练结果用于检索

对 `mlp-llm` / `llm-lora`，复制默认配置并修改：

```json
{"vl_model":"Qwen/Qwen2.5-VL-7B-Instruct","adapter":"checkpoints/mlp-llm/final"}
```

本项目配置允许只填写要覆盖的字段。基座必须与训练一致。对 `mlp-vision`：

```json
{"vl_model":"checkpoints/mlp-vision/final","adapter":null}
```

随后运行：

```bash
python -m mmretrieval --config configs/finetuned.json describe \
  --manifest data/catalog.jsonl --image-root data/images \
  --output data/catalog.finetuned.jsonl
python -m mmretrieval --config configs/finetuned.json index \
  --manifest data/catalog.finetuned.jsonl --image-root data/images \
  --output artifacts/index-finetuned
```

请先把上面的 JSON 保存为 `configs/finetuned.json`。只替换检索时的 adapter、却复用旧模型生成的描述和索引，不能检验离线语义建库的微调收益。

## 做模块对比时

- 固定基座、数据划分、图片预处理、查询集合和评测口径。
- 明确报告不同模式的可训练参数量与学习率；视觉全参数和语言 LoRA 的训练成本不同。
- 用验证集选择超参数，最终测试保持独立；至少记录随机种子，资源允许时重复实验。
- 保存并重新加载权重后再做推理验证，确认 merger 的修改确实存在。
- 比较基础模型与微调模型生成的新索引；输出逐查询结果，检查提升集中在哪些类别。

本仓库的微型随机模型测试验证参数范围、训练反向传播和 merger 保存恢复；它不能替代 3B/7B 模型的完整收敛与效果实验。
