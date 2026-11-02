"""Portable manifests and correctly labelled supervised examples."""
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def read_jsonl(path):
    rows = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected an object")
            rows.append(row)
    return rows


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def image_path(root, relative):
    root = Path(root).expanduser().resolve()
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError("file_path must be a nonempty path relative to image-root")
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Image escapes image-root: {relative}")
    if not path.is_file():
        raise ValueError(f"Image does not exist: {path}")
    return path


def validate_catalog(rows, root, verify_images=True):
    if not rows:
        raise ValueError("Catalog is empty")
    ids, paths = set(), set()
    for row in rows:
        item_id = row.get("id")
        if not isinstance(item_id, str) or not item_id.strip() or item_id in ids:
            raise ValueError(f"Missing or duplicate string id: {item_id!r}")
        ids.add(item_id)
        path = image_path(root, row.get("file_path"))
        if path in paths:
            raise ValueError(f"Duplicate image path: {row['file_path']}")
        paths.add(path)
        categories = row.get("categories", [])
        if not isinstance(categories, list) or any(not isinstance(c, str) or not c.strip() for c in categories):
            raise ValueError(f"{item_id}: categories must be a list of nonempty strings")
        if not isinstance(row.get("description", ""), str):
            raise ValueError(f"{item_id}: description must be a string")
        if verify_images:
            with Image.open(path) as image:
                image.verify()
    return rows


def folder_catalog(root, category_map=None):
    """First directory component is a user-provided category, not a prediction."""
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"Not an image directory: {root}")
    category_map = category_map or {}
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            relative = path.relative_to(root)
            category = category_map.get(relative.parts[0], relative.parts[0]) if len(relative.parts) > 1 else None
            rows.append({"id": hashlib.sha256(relative.as_posix().encode()).hexdigest()[:16],
                         "file_path": relative.as_posix(),
                         "categories": [category] if category else [], "description": ""})
    return validate_catalog(rows, root)


def make_sft(rows, root, val_ratio=0.2, seed=42, hard_negatives=None):
    """Split image content BEFORE question expansion; negatives get truthful answers."""
    validate_catalog(rows, root)
    if not 0 < val_ratio < 1:
        raise ValueError("val-ratio must be between 0 and 1")
    hard_negatives = hard_negatives or {}
    if not isinstance(hard_negatives, dict) or any(
        not isinstance(k, str) or not isinstance(v, list) or any(not isinstance(x, str) or not x for x in v)
        for k, v in hard_negatives.items()
    ):
        raise ValueError("hard-negatives must map category strings to lists of strings")
    groups, by_category = {}, defaultdict(list)
    for row in rows:
        if not row.get("categories"):
            raise ValueError(f"{row['id']}: SFT requires human-verified categories")
        if row.get("categories_source") == "model":
            raise ValueError(f"{row['id']}: verify model-generated categories before SFT")
        digest = hashlib.sha256(image_path(root, row["file_path"]).read_bytes()).hexdigest()
        label = tuple(sorted(set(row["categories"])))
        if digest in groups and groups[digest]["label"] != label:
            raise ValueError("Identical image bytes have conflicting category labels")
        groups.setdefault(digest, {"label": label, "rows": []})["rows"].append(row)
    for digest, group in groups.items():
        by_category[group["label"]].append(digest)
    rng = random.Random(seed)
    validation = set()
    for label in sorted(by_category):
        hashes = sorted(by_category[label])
        rng.shuffle(hashes)
        count = min(len(hashes) - 1, max(1, round(len(hashes) * val_ratio))) if len(hashes) > 1 else 0
        validation.update(hashes[:count])
    output = {"train": [], "val": []}
    for digest, group in groups.items():
        split = "val" if digest in validation else "train"
        for row in group["rows"]:
            categories = sorted(set(row["categories"]))
            base = {"image_id": row["id"], "file_path": row["file_path"], "image_sha256": digest}
            output[split].append({**base, "id": row["id"] + ":category",
                                  "prompt": "请输出图片中主要物体的类别列表，只返回 JSON 数组。",
                                  "answer": json.dumps(categories, ensure_ascii=False)})
            candidates = set(categories)
            for category in categories:
                candidates.update(hard_negatives.get(category, []))
            for number, candidate in enumerate(sorted(candidates)):
                answer = "是。" if candidate in categories else "否。主要物体类别为：" + "、".join(categories) + "。"
                output[split].append({**base, "id": f"{row['id']}:verify:{number}",
                                      "prompt": f"图片中的主要物体类别包含“{candidate}”吗？", "answer": answer})
    return output


def retrieval_metrics(predictions, relevant):
    predicted, truth = set(predictions), set(relevant)
    if not truth:
        raise ValueError("Each evaluation query must have at least one relevant id")
    tp = len(predicted & truth)
    precision = tp / len(predicted) if predicted else 0.0
    recall = tp / len(truth)
    return {"precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "tp": tp, "fp": len(predicted) - tp, "fn": len(truth) - tp}
