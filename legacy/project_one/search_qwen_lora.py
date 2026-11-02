# Historical source snapshot. See legacy/README.md; use mmretrieval for supported commands.
"""
Qwen2.5-VL-7B LoRA 微调脚本
目标：利用 imagenet_json.json 中的 2000 张图片，微调模型使其更准确地识别图中物体类别。

训练数据格式：
  输入 = 图片 + prompt（"根据图片内容，输出图中包含的物品类别列表"）
  标签 = 该图片对应的中文类别，如 ["椅子"]

依赖：
  pip install peft accelerate bitsandbytes
  （qwen_vl_utils 已在项目中使用）
"""

import os
import json
import torch
import random
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from modelscope import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from peft import LoraConfig, get_peft_model, TaskType
from transformers import get_cosine_schedule_with_warmup

# ===================== 配置区 =====================

# 模型路径（和 search_qwen.py 保持一致）
MODEL_PATH = "/path/to/workspace/multimodel_project/model-Qwen2.5-VL-7B-Instruct"

# 数据集路径
DATASET_JSON = "/path/to/workspace/multimodel_project/project_one/imagenet_json.json"

# LoRA 输出目录
OUTPUT_DIR = "/path/to/workspace/multimodel_project/project_one/qwen_lora_output"

# 训练超参数
EPOCHS = 3                # 训练轮数（2000张小数据集，3轮足够收敛）
BATCH_SIZE = 5            # 开启 gradient checkpointing 后可以提高
GRAD_ACCUM_STEPS = 2      # 等效 batch_size = 8
LEARNING_RATE = 2e-4      # LoRA 常用学习率
WARMUP_RATIO = 0.1        # 预热比例
MAX_LENGTH = 2048         # 最大 token 长度（需容纳图片 vision token 展开）
SEED = 42
VAL_RATIO = 0.1           # 10% 数据做验证集

# LoRA 超参数
LORA_RANK = 16            # 秩，越大拟合能力越强但参数越多
LORA_ALPHA = 32           # 缩放因子，通常设为 2*rank
LORA_DROPOUT = 0.05       # 防过拟合

# ===================== 数据集定义 =====================

# 训练用的 prompt（和推理时保持一致，这样微调效果才能迁移）
TRAIN_PROMPT = """根据图片内容，输出图中包含的物品类别列表。
严格按照以下列表格式输出：
["类别1","类别2","..."]"""


class QwenVLFineTuneDataset(Dataset):
    """
    将 imagenet_json.json 转换为 Qwen2.5-VL 的训练格式。
    每条样本：
      - 输入：图片 + prompt
      - 标签：该图片的中文类别列表，如 '["椅子"]'
    """

    def __init__(self, samples, processor):
        self.samples = samples
        self.processor = processor

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image_path = sample["file_path"]
        label_cn = sample["label_cn"]  # 如 "椅子"

        # 构造标签文本：模型应该输出的内容
        # 格式和推理时期望的输出一致
        target_text = json.dumps([label_cn], ensure_ascii=False)  # '["椅子"]'

        # 构造 Qwen VL 的对话格式
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": TRAIN_PROMPT},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": target_text},
                ],
            },
        ]

        # 用 processor 的 chat template 把对话转成文本
        # add_generation_prompt=False 因为我们已经包含了 assistant 的回复
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

        # 处理图片信息
        image_inputs, _ = process_vision_info(messages)

        # tokenize
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            padding="max_length",
            max_length=MAX_LENGTH,
            truncation=True,
            return_tensors="pt",
        )

        # 构造 labels：只对 assistant 回复部分计算 loss
        # 先获取完整的 input_ids
        input_ids = inputs["input_ids"].squeeze(0)

        # 找到 assistant 回复的起始位置
        # Qwen 的 chat template 中，assistant 回复前有特殊标记
        # 我们用只包含 user 部分的文本来定位
        user_only_messages = [messages[0]]  # 只有 user 消息
        user_text = self.processor.apply_chat_template(
            user_only_messages, tokenize=False, add_generation_prompt=True
        )
        user_inputs = self.processor(
            text=[user_text],
            images=image_inputs,
            padding=False,
            return_tensors="pt",
        )
        user_len = user_inputs["input_ids"].shape[1]

        # labels: prompt 部分设为 -100（不计算 loss），只对回复部分计算 loss
        labels = input_ids.clone()
        labels[:user_len] = -100

        # padding 部分也设为 -100
        if "attention_mask" in inputs:
            pad_mask = inputs["attention_mask"].squeeze(0) == 0
            labels[pad_mask] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "labels": labels,
            # 图像字段也 squeeze 掉 batch 维度
            **{k: v.squeeze(0) if isinstance(v, torch.Tensor) else v
               for k, v in inputs.items()
               if k not in ("input_ids", "attention_mask")},
        }


# ===================== 工具函数 =====================

def load_and_split_data(json_path, val_ratio=0.1, seed=42):
    """加载数据集并划分训练/验证集"""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    samples = data["images"]
    print(f"数据集总量: {len(samples)} 张图片")

    # 按类别分层划分，保证每个类别在训练/验证集中都有代表
    from collections import defaultdict
    by_label = defaultdict(list)
    for s in samples:
        by_label[s["label"]].append(s)

    train_samples, val_samples = [], []
    random.seed(seed)
    for label, items in by_label.items():
        random.shuffle(items)
        n_val = max(1, int(len(items) * val_ratio))
        val_samples.extend(items[:n_val])
        train_samples.extend(items[n_val:])

    random.shuffle(train_samples)
    random.shuffle(val_samples)

    print(f"训练集: {len(train_samples)} | 验证集: {len(val_samples)}")
    return train_samples, val_samples


def print_trainable_params(model):
    """打印可训练参数量"""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"可训练参数: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")


# ===================== 自定义 collate_fn =====================
def collate_fn(batch):
    """
    处理不同图片尺寸导致的 pixel_values 维度不一致问题。
    对 input_ids / attention_mask / labels 做 stack，
    对 pixel_values / image_grid_thw 做 cat（沿 batch 维拼接）。
    """
    keys = batch[0].keys()
    collated = {}

    for key in keys:
        tensors = [item[key] for item in batch]
        if key in ("input_ids", "attention_mask", "labels"):
            collated[key] = torch.stack(tensors)
        elif key == "pixel_values":
            # 不同图片 pixel_values 数不同，用 cat 拼接
            collated[key] = torch.cat(tensors, dim=0)
        elif key == "image_grid_thw":
            # 每个样本 (3,)，stack 成 (batch, 3)
            collated[key] = torch.stack(tensors)
        elif isinstance(tensors[0], torch.Tensor):
            try:
                collated[key] = torch.stack(tensors)
            except RuntimeError:
                collated[key] = torch.cat(tensors, dim=0)
        else:
            collated[key] = tensors

    return collated


# ===================== 训练主流程 =====================

def train():
    random.seed(SEED)
    torch.manual_seed(SEED)

    # ---------- 1. 加载模型和 processor ----------
    print("=" * 60)
    print("  加载 Qwen2.5-VL-7B 模型...")
    print("=" * 60)

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        min_pixels=256 * 28 * 28,    # 默认值
        max_pixels=1024 * 28 * 28,    # 降低最大分辨率，减少 vision token 数量
    )

    # # 开启 gradient checkpointing 节省显存
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    # ---------- 2. 配置 LoRA ----------
    # 对 Qwen2.5-VL 的注意力层施加 LoRA
    # target_modules 选择 q_proj / v_proj 是经典做法，
    # 也可以加上 k_proj / o_proj 来增强拟合能力
    print("\n配置 LoRA...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_RANK,                    # LoRA 秩
        lora_alpha=LORA_ALPHA,          # 缩放因子 alpha/r 控制 LoRA 的影响强度
        lora_dropout=LORA_DROPOUT,      # dropout 防过拟合
        target_modules=[                # 对哪些层施加 LoRA
            "q_proj", "k_proj",         # Query 和 Key 投影
            "v_proj", "o_proj",         # Value 和 Output 投影
        ],
        bias="none",                    # 不训练 bias，减少参数量
    )

    # 将 LoRA 注入模型
    model = get_peft_model(model, lora_config)
    print_trainable_params(model)

    # 冻结视觉编码器（只微调语言模型部分，保留视觉特征提取能力）
    for name, param in model.named_parameters():
        if "visual" in name and param.requires_grad:
            param.requires_grad = False
    print("已冻结视觉编码器")
    print_trainable_params(model)

    # ---------- 3. 准备数据 ----------
    print("\n加载数据集...")
    train_samples, val_samples = load_and_split_data(DATASET_JSON, VAL_RATIO, SEED)

    train_dataset = QwenVLFineTuneDataset(train_samples, processor)
    val_dataset = QwenVLFineTuneDataset(val_samples, processor)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
    )

    # ---------- 4. 优化器和学习率调度 ----------
    total_steps = len(train_loader) * EPOCHS // GRAD_ACCUM_STEPS
    warmup_steps = int(total_steps * WARMUP_RATIO)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
        weight_decay=0.01,
    )

    # 余弦退火调度器：先 warmup 再逐步降低学习率
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    print(f"\n总训练步数: {total_steps} | 预热步数: {warmup_steps}")
    print(f"等效 batch_size: {BATCH_SIZE * GRAD_ACCUM_STEPS}")

    # ---------- 5. 训练循环 ----------
    print("\n" + "=" * 60)
    print("  开始训练")
    print("=" * 60)

    model.train()
    best_val_loss = float("inf")
    global_step = 0

    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            # 将数据移到 GPU
            input_ids = batch["input_ids"].to("cuda")
            attention_mask = batch["attention_mask"].to("cuda")
            labels = batch["labels"].to("cuda")

            # 处理图像相关输入
            model_inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            }
            # 传递图像 tensor（pixel_values 等）
            for k, v in batch.items():
                if k not in ("input_ids", "attention_mask", "labels"):
                    if isinstance(v, torch.Tensor):
                        model_inputs[k] = v.to("cuda")
                    elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], torch.Tensor):
                        model_inputs[k] = [t.to("cuda") for t in v]

            # 前向传播
            with torch.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(**model_inputs)
                loss = outputs.loss / GRAD_ACCUM_STEPS  # 梯度累积需要除以累积步数

            # 反向传播
            loss.backward()
            epoch_loss += loss.item() * GRAD_ACCUM_STEPS

            # 梯度累积：每 GRAD_ACCUM_STEPS 步更新一次参数
            if (step + 1) % GRAD_ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % 10 == 0:
                    avg_loss = epoch_loss / (step + 1)
                    lr = scheduler.get_last_lr()[0]
                    print(f"  Epoch {epoch+1}/{EPOCHS} | Step {global_step} | "
                          f"Loss: {avg_loss:.4f} | LR: {lr:.2e}")

        # 处理最后不足 GRAD_ACCUM_STEPS 的剩余梯度
        if (step + 1) % GRAD_ACCUM_STEPS != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        avg_train_loss = epoch_loss / len(train_loader)
        print(f"\n  Epoch {epoch+1} 训练完成 | 平均 Loss: {avg_train_loss:.4f}")

        # ---------- 6. 验证 ----------
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to("cuda")
                attention_mask = batch["attention_mask"].to("cuda")
                labels = batch["labels"].to("cuda")

                model_inputs = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels,
                }
                for k, v in batch.items():
                    if k not in ("input_ids", "attention_mask", "labels"):
                        if isinstance(v, torch.Tensor):
                            model_inputs[k] = v.to("cuda")
                        elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], torch.Tensor):
                            model_inputs[k] = [t.to("cuda") for t in v]

                with torch.autocast("cuda", dtype=torch.bfloat16):
                    outputs = model(**model_inputs)
                    val_loss += outputs.loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"  验证 Loss: {avg_val_loss:.4f}")

        # 保存最优模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = os.path.join(OUTPUT_DIR, "adapter_config")
            model.save_pretrained(save_path)
            processor.save_pretrained(save_path)
            print(f"  保存最优模型 -> {save_path}")

        model.train()
        print()

    # ---------- 7. 保存最终模型 ----------
    final_path = os.path.join(OUTPUT_DIR, "final")
    model.save_pretrained(final_path)
    processor.save_pretrained(final_path)
    print(f"\n训练完成！最终模型 -> {final_path}")
    print(f"最优验证 Loss: {best_val_loss:.4f}")

    # ---------- 8. 打印使用说明 ----------
    print("\n" + "=" * 60)
    print("  使用微调后的模型（在 search_qwen.py 中取消注释）：")
    print("=" * 60)
    print(f"""
# 加载 LoRA 微调后的模型
lorapath = '{os.path.join(OUTPUT_DIR, "best")}'
model = PeftModel.from_pretrained(
    model,
    lorapath   # 含 adapter_config.json 的目录
).merge_and_unload()
""")


if __name__ == "__main__":
    train()
