import argparse
import json
from pathlib import Path

import numpy as np

from .data import (folder_catalog, image_path, make_sft, read_jsonl,
                   retrieval_metrics, validate_catalog, write_jsonl)
from .models import DEFAULTS
from .retrieval import Index, fuse_query


def load_config(path):
    supplied = json.loads(Path(path).read_text(encoding="utf-8")) if path else {}
    if not isinstance(supplied, dict) or set(supplied) - set(DEFAULTS):
        raise ValueError("Unknown config keys; see configs/default.json")
    config = {**DEFAULTS, **supplied}
    if config["prompt_profile"] not in ("standard", "historical"):
        raise ValueError("prompt_profile must be standard or historical")
    return config


def encode_query(encoder, text, image, weight):
    return fuse_query(encoder.text(text) if text else None,
                      encoder.image(image) if image else None, weight)


def checked_index(directory, config):
    index = Index.load(directory)
    if index.metadata.get("demo"):
        raise ValueError("This is a synthetic smoke-test index. Build a model index for real queries.")
    for key in ("text_encoder", "image_encoder"):
        if index.metadata.get(key) != config[key]:
            raise ValueError(f"Index was built with a different {key}; rebuild it or restore config")
    return index


def demo(output):
    """Six owned geometric images + predefined vectors, NOT a model quality demo."""
    from PIL import Image, ImageDraw
    root = Path(output)
    rows, features = [], []
    for i, (name, color) in enumerate([("red-circle", "red"), ("blue-square", "blue"), ("green-triangle", "green")]):
        for variant in range(2):
            relative = Path("images") / name / f"{variant}.png"
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            image = Image.new("RGB", (96, 96), "white")
            draw = ImageDraw.Draw(image)
            offset = variant * 5
            if i == 0:
                draw.ellipse((10 + offset, 10, 75, 75), fill=color)
            elif i == 1:
                draw.rectangle((10 + offset, 10, 75, 75), fill=color)
            else:
                draw.polygon([(48, 10 + offset), (10, 75), (80, 75)], fill=color)
            image.save(path)
            rows.append({"id": f"{name}-{variant}", "file_path": relative.as_posix(),
                         "categories": [name], "description": name})
            vector = np.zeros(3)
            vector[i] = 1
            features.append(vector)
    validate_catalog(rows, root)
    write_jsonl(root / "catalog.jsonl", rows)
    index = Index(rows, features, features, features, {"demo": True, "version": 1})
    index.save(root / "index")
    reloaded = Index.load(root / "index")
    results = reloaded.search([1, 0, 0], top_k=2)
    metrics = retrieval_metrics([r["id"] for r in results], ["red-circle-0", "red-circle-1"])
    split = make_sft(rows, root)
    write_jsonl(root / "sft" / "train.jsonl", split["train"])
    write_jsonl(root / "sft" / "val.jsonl", split["val"])
    return {"notice": "Synthetic vectors validate plumbing only; these are NOT model evaluation results.",
            "output": str(root), "image_count": len(rows), "results": results, "synthetic_metrics": metrics}


def execute(args):
    config = load_config(args.config)
    if args.command == "demo":
        return demo(args.output)
    if args.command == "prepare":
        mapping = json.loads(Path(args.category_map).read_text(encoding="utf-8")) if args.category_map else None
        rows = folder_catalog(args.image_root, mapping)
        write_jsonl(args.output, rows)
        return {"output": args.output, "images": len(rows)}
    if args.command == "validate":
        rows = validate_catalog(read_jsonl(args.manifest), args.image_root)
        return {"valid": True, "images": len(rows), "categories": sorted({c for r in rows for c in r.get("categories", [])})}
    if args.command == "make-sft":
        negatives = json.loads(Path(args.hard_negatives).read_text(encoding="utf-8")) if args.hard_negatives else None
        result = make_sft(read_jsonl(args.manifest), args.image_root, args.val_ratio, args.seed, negatives)
        for split, rows in result.items():
            write_jsonl(Path(args.output) / f"{split}.jsonl", rows)
        return {"output": args.output, "records": {k: len(v) for k, v in result.items()},
                "note": "Verify category labels; empty validation means insufficient distinct images per class."}
    if args.command == "train":
        from .training import run_training
        return run_training(args, config)
    if args.command == "describe":
        from .models import Qwen
        rows = validate_catalog(read_jsonl(args.manifest), args.image_root)
        model = Qwen(config)
        for row in rows:
            description = model.describe(image_path(args.image_root, row["file_path"]))
            # Predictions are stored separately from human labels, which evaluation/SFT may use.
            row["retrieval_categories"] = description["categories"]
            row["retrieval_description"] = description["description"]
            row["retrieval_source"] = {"model": config["vl_model"], "adapter": config["adapter"]}
        write_jsonl(args.output, rows)
        return {"output": args.output, "images": len(rows)}
    if args.command == "index":
        from .models import Encoders
        rows = validate_catalog(read_jsonl(args.manifest), args.image_root)
        semantic_texts = []
        for row in rows:
            categories = row.get("retrieval_categories", row.get("categories", []))
            description = row.get("retrieval_description", row.get("description", ""))
            if not categories or not isinstance(categories, list) or any(not isinstance(c, str) or not c.strip() for c in categories):
                raise ValueError(f"{row['id']}: run describe or supply valid category metadata first")
            if not isinstance(description, str):
                raise ValueError("Description must be text")
            category_text = (", " if config["prompt_profile"] == "historical" else "、").join(categories)
            semantic_texts.append((category_text, description or category_text))
        encoder = Encoders(config)
        category, description, images = [], [], []
        for row, (cat_text, desc_text) in zip(rows, semantic_texts):
            category.append(encoder.text(cat_text))
            description.append(encoder.text(desc_text))
            images.append(encoder.image(image_path(args.image_root, row["file_path"])))
        index = Index(rows, category, description, images, {
            "version": 1, "demo": False, "text_encoder": config["text_encoder"],
            "image_encoder": config["image_encoder"],
            "rows_with_model_descriptions": sum("retrieval_source" in r for r in rows),
            "rows_with_manual_metadata": sum("retrieval_source" not in r for r in rows)})
        index.save(args.output)
        return {"output": args.output, "images": len(rows), "dimension": index.category.shape[1]}
    if args.command == "search":
        from .models import Encoders, Qwen
        index = checked_index(args.index, config)
        text = args.text
        reference = image_path(args.image_root, args.image) if args.image else None
        if not text and reference is None:
            raise ValueError("search requires --text or --image")
        intent = None
        k = None if args.all_matches else args.top_k
        if args.intent:
            router = Qwen(config, vision=reference is not None)
            intent = router.intent(text or "查找相似图片", reference)
            if not intent["needs_retrieval"]:
                return {"intent": intent, "results": []}
            text = intent.get("description") or text
            if not args.all_matches and intent.get("top_k") is not None:
                k = intent["top_k"]
            del router
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        encoder = Encoders(config)
        query = encode_query(encoder, text, reference, config["image_weight"])
        excluded = set(args.exclude_id)
        if reference:
            root = Path(args.image_root).resolve()
            excluded.update(r["id"] for r in index.rows if (root / r["file_path"]).resolve() == reference)
        results = index.search(query, top_k=k, category_ratio=config["category_ratio"],
                               min_score=args.min_score, mode=args.mode, exclude_ids=excluded)
        return {"intent": intent, "results": results}
    if args.command == "evaluate":
        from .models import Encoders
        index = checked_index(args.index, config)
        queries = read_jsonl(args.queries)
        if not queries:
            raise ValueError("Evaluation queries are empty")
        ids, query_ids = {r["id"] for r in index.rows}, set()
        for q in queries:
            if not isinstance(q.get("id"), str) or q["id"] in query_ids:
                raise ValueError("Evaluation queries require unique string ids")
            query_ids.add(q["id"])
            if not isinstance(q.get("relevant_ids"), list) or not q["relevant_ids"] or not set(q["relevant_ids"]) <= ids:
                raise ValueError("Each query needs nonempty relevant_ids present in the index")
        encoder = Encoders(config)
        details = []
        for q in queries:
            reference = image_path(args.image_root, q["image"]) if q.get("image") else None
            excluded = set(q.get("exclude_ids", []))
            if reference:
                excluded.update(r["id"] for r in index.rows if (Path(args.image_root) / r["file_path"]).resolve() == reference)
            if excluded & set(q["relevant_ids"]):
                raise ValueError("Relevant ids include excluded/reference image; fix evaluation labels")
            vector = encode_query(encoder, q.get("text"), reference, config["image_weight"])
            results = index.search(vector, top_k=None if args.all_matches else args.top_k,
                                   category_ratio=config["category_ratio"], min_score=args.min_score,
                                   mode=args.mode, exclude_ids=excluded)
            predicted = [r["id"] for r in results]
            details.append({"id": q["id"], "predicted_ids": predicted, **retrieval_metrics(predicted, q["relevant_ids"])})
        report = {"query_count": len(details), "mode": args.mode,
                  "top_k": None if args.all_matches else args.top_k, "min_score": args.min_score,
                  "category_ratio": config["category_ratio"], "image_weight": config["image_weight"],
                  "index_metadata": index.metadata,
                  "macro": {key: float(np.mean([r[key] for r in details])) for key in ("precision", "recall", "f1")},
                  "queries": details}
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return {k: v for k, v in report.items() if k != "queries"}
    raise ValueError("Unknown command")


def parser():
    p = argparse.ArgumentParser(description="Chinese multimodal image retrieval")
    p.add_argument("--config", help="JSON configuration; put this BEFORE the command")
    sub = p.add_subparsers(dest="command", required=True)
    demo_p = sub.add_parser("demo", help="No-model synthetic smoke demo")
    demo_p.add_argument("--output", default="artifacts/demo")
    prepare = sub.add_parser("prepare", help="Build a manifest from class folders")
    prepare.add_argument("--image-root", required=True)
    prepare.add_argument("--category-map")
    prepare.add_argument("--output", required=True)
    for name in ("validate", "describe", "index", "make-sft"):
        command = sub.add_parser(name)
        command.add_argument("--manifest", required=True)
        command.add_argument("--image-root", required=True)
        if name != "validate":
            command.add_argument("--output", required=True)
        if name == "make-sft":
            command.add_argument("--hard-negatives")
            command.add_argument("--val-ratio", type=float, default=0.2)
            command.add_argument("--seed", type=int, default=42)
    for name in ("search", "evaluate"):
        command = sub.add_parser(name)
        command.add_argument("--index", required=True)
        command.add_argument("--image-root", required=True)
        command.add_argument("--top-k", type=int, default=5)
        command.add_argument("--all-matches", action="store_true")
        command.add_argument("--min-score", type=float)
        command.add_argument("--mode", choices=["semantic", "clip"], default="semantic")
        if name == "search":
            command.add_argument("--text")
            command.add_argument("--image", help="Path relative to image-root")
            command.add_argument("--intent", action="store_true")
            command.add_argument("--exclude-id", action="append", default=[])
        else:
            command.add_argument("--queries", required=True)
            command.add_argument("--output", required=True)
    train = sub.add_parser("train")
    train.add_argument("--train", required=True)
    train.add_argument("--val")
    train.add_argument("--image-root", required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--mode", choices=["llm-lora", "mlp-llm", "mlp-vision"], default="mlp-llm")
    train.add_argument("--rank", type=int, default=64)
    train.add_argument("--alpha", type=int, default=128)
    train.add_argument("--epochs", type=float, default=0.5)
    train.add_argument("--grad-accum", type=int, default=4)
    train.add_argument("--learning-rate", type=float, default=2e-5)
    train.add_argument("--max-length", type=int, default=4096)
    train.add_argument("--seed", type=int, default=42)
    return p


def main():
    p = parser()
    args = p.parse_args()
    try:
        result = execute(args)
    except (ValueError, FileNotFoundError, ImportError) as exc:
        p.exit(2, f"Error: {exc}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
