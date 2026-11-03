"""Qwen2.5-VL SFT with explicit, saveable training scopes. CUDA recommended."""
import hashlib
import json
import math
from pathlib import Path

from .data import image_path, read_jsonl


def configure_training(model, mode, rank=64, alpha=128):
    if rank < 1 or alpha < 1:
        raise ValueError("LoRA rank and alpha must be positive")
    for parameter in model.parameters():
        parameter.requires_grad = False
    if mode in {"llm-lora", "mlp-llm"}:
        from peft import LoraConfig, TaskType, get_peft_model
        model = get_peft_model(model, LoraConfig(
            task_type=TaskType.CAUSAL_LM, r=rank, lora_alpha=alpha, lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            modules_to_save=["merger"] if mode == "mlp-llm" else None,
            bias="none"))
    elif mode == "mlp-vision":
        # Full visual tower + merger tuning; save a full model, not a language LoRA adapter.
        for name, parameter in model.named_parameters():
            if "visual." in name:
                parameter.requires_grad = True
    else:
        raise ValueError(f"Unknown training mode: {mode}")
    # Low-rate updates need FP32 master parameters, including modules_to_save.
    # Autocast still performs eligible training operations in BF16.
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.data=parameter.data.float()
    trainable = {name: p.numel() for name, p in model.named_parameters() if p.requires_grad}
    if not trainable:
        raise ValueError("No trainable parameters matched this model")
    if mode == "mlp-llm" and not any("merger" in name for name in trainable):
        raise ValueError("MLP mode selected but no merger parameters are trainable")
    if mode == "mlp-vision" and any("visual." not in name for name in trainable):
        raise ValueError("Unexpected language parameters in vision-only training")
    return model, trainable


def validate_sft(rows, root):
    seen, hashes = set(), set()
    for row in rows:
        if not isinstance(row.get("id"), str) or row["id"] in seen:
            raise ValueError("SFT records require unique string ids")
        seen.add(row["id"])
        for key in ("prompt", "answer"):
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError(f"SFT record {row['id']} requires nonempty {key}")
        hashes.add(hashlib.sha256(image_path(root, row.get("file_path")).read_bytes()).hexdigest())
    return hashes


class SFTCollator:
    """Use the processor's batched image-grid ordering and mask every prompt."""
    def __init__(self, processor, root, max_length=4096):
        self.processor, self.root, self.max_length = processor, root, max_length

    def __call__(self, rows):
        import torch
        from qwen_vl_utils import process_vision_info
        if not rows:raise ValueError('Empty training batch')
        users=[];prefixes=[];full=[]
        for row in rows:
            user = {"role": "user", "content": [
                {"type": "image", "image": str(image_path(self.root, row["file_path"]))},
                {"type": "text", "text": row["prompt"]}]}
            messages = [user, {"role": "assistant", "content": [{"type": "text", "text": row["answer"]}]}]
            users.append([user])
            prefixes.append(self.processor.apply_chat_template([user], tokenize=False, add_generation_prompt=True))
            full.append(self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False))
        images, _ = process_vision_info(users)
        self.processor.tokenizer.padding_side='right'
        inputs = self.processor(text=full, images=images, padding=True, return_tensors="pt")
        prefix_inputs = self.processor(text=prefixes, images=images, padding=True, return_tensors="pt")
        if inputs["input_ids"].shape[1] > self.max_length:
            raise ValueError("Sample exceeds max-length; reduce max-pixels or shorten text (image tokens are not truncated)")
        labels = inputs["input_ids"].clone()
        for i in range(len(rows)):
            size=int(prefix_inputs['attention_mask'][i].sum())
            if not torch.equal(inputs['input_ids'][i,:size],prefix_inputs['input_ids'][i,:size]):
                raise ValueError("Chat template prefix mismatch; cannot safely mask prompt loss")
            labels[i,:size]=-100
        labels[inputs["attention_mask"] == 0] = -100
        if ((labels != -100).sum(dim=1)==0).any().item():
            raise ValueError("No assistant target tokens remain")
        inputs["labels"] = labels
        return dict(inputs)


def run_training(args, config):
    train_rows = read_jsonl(args.train)
    val_rows = read_jsonl(args.val) if args.val else []
    if not train_rows:
        raise ValueError("Training split is empty")
    train_hashes = validate_sft(train_rows, args.image_root)
    val_hashes = validate_sft(val_rows, args.image_root)
    if train_hashes & val_hashes:
        raise ValueError("Train/validation image content overlaps; split by image before expanding VQA")
    if not math.isfinite(args.epochs) or not math.isfinite(args.learning_rate) or args.epochs <= 0 or args.grad_accum < 1 or args.learning_rate <= 0 or (getattr(args,'max_steps',None) is not None and args.max_steps<1):
        raise ValueError("epochs, grad-accum and learning-rate must be positive")
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, Trainer, TrainingArguments, set_seed
    import torch
    from .models import resolve_device
    device=resolve_device(config['device'])
    set_seed(args.seed)
    dtype = torch.bfloat16 if device.startswith('npu') or (device.startswith('cuda') and torch.cuda.is_bf16_supported()) else torch.float32
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        config["vl_model"], torch_dtype=dtype, attn_implementation="eager" if device.startswith('npu') else "sdpa")
    processor = AutoProcessor.from_pretrained(config["vl_model"], max_pixels=config["max_pixels"])
    model, trainable = configure_training(model, args.mode, args.rank, args.alpha)
    model.config.use_cache = False
    model.enable_input_require_grads()
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use a new output directory; existing training artifacts are not overwritten')
    output.mkdir(parents=True, exist_ok=True)
    (output / "training_scope.json").write_text(json.dumps({
        "mode": args.mode, "model": config["vl_model"], "rank": args.rank, "alpha": args.alpha,
        "epochs": args.epochs, "per_device_batch_size": 1, "gradient_accumulation_steps": args.grad_accum,
        "seed": args.seed, "learning_rate": args.learning_rate, "max_pixels": config["max_pixels"],
        "device":device, "max_steps":getattr(args,'max_steps',None),
        "train_manifest_sha256":hashlib.sha256(Path(args.train).read_bytes()).hexdigest(),
        "val_manifest_sha256":hashlib.sha256(Path(args.val).read_bytes()).hexdigest() if args.val else None,
        "train_image_count": len(train_hashes), "val_image_count": len(val_hashes),
        "trainable_parameter_count": sum(trainable.values()), "trainable_parameters": trainable,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    trainer = Trainer(model=model, args=TrainingArguments(
        output_dir=str(output), num_train_epochs=args.epochs, learning_rate=args.learning_rate,
        max_steps=getattr(args,'max_steps',None) or -1,
        per_device_train_batch_size=1, per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum, bf16=dtype == torch.bfloat16,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False, report_to="none", seed=args.seed,
        logging_steps=10, save_strategy="epoch", eval_strategy="epoch" if val_rows else "no",
        save_total_limit=2), train_dataset=train_rows, eval_dataset=val_rows or None,
        data_collator=SFTCollator(processor, args.image_root, args.max_length), processing_class=processor)
    trainer.train()
    model.config.use_cache = True
    trainer.save_model(str(output / "final"))
    processor.save_pretrained(output / "final")
    trainer.save_state()
    return {"output": str(output / "final"), "mode": args.mode,
            "trainable_parameters": sum(trainable.values()), "training_records": len(train_rows)}
