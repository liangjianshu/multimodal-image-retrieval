"""Category filtering followed by description ranking; no external vector database."""
import json
from pathlib import Path

import numpy as np

from .data import read_jsonl, write_jsonl


def normalized(values):
    array = np.asarray(values, dtype=np.float32)
    if array.ndim not in (1, 2) or not np.isfinite(array).all():
        raise ValueError("Expected finite vectors")
    norms = np.linalg.norm(array, axis=-1, keepdims=True)
    if (norms < 1e-10).any():
        raise ValueError("Zero embedding cannot be normalized")
    return array / norms


def fuse_query(text=None, image=None, image_weight=0.5):
    if not 0 <= image_weight <= 1:
        raise ValueError("image-weight must be in [0, 1]")
    if text is None and image is None:
        raise ValueError("Provide text or a reference image")
    if text is None:
        return normalized(image)
    if image is None:
        return normalized(text)
    if np.shape(text) != np.shape(image):
        raise ValueError("Text and image encoders must share the same embedding space and dimension")
    return normalized((1 - image_weight) * normalized(text) + image_weight * normalized(image))


def relative_cutoff(best, ratio):
    if not 0 <= ratio <= 1:
        raise ValueError("category-ratio must be in [0, 1]")
    # Unlike best * ratio, this keeps the best candidate when all cosine scores are negative.
    return best - (1 - ratio) * abs(best)


class Index:
    def __init__(self, rows, category, description, image, metadata):
        self.rows = rows
        self.category, self.description, self.image = map(normalized, (category, description, image))
        if not rows or len({r["id"] for r in rows}) != len(rows):
            raise ValueError("Index requires nonempty, unique ids")
        if any(a.ndim != 2 or a.shape[0] != len(rows) for a in (self.category, self.description, self.image)):
            raise ValueError("Index row count does not match feature matrices")
        if len({a.shape[1] for a in (self.category, self.description, self.image)}) != 1:
            raise ValueError("Index embedding dimensions differ")
        self.metadata = metadata

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        write_jsonl(directory / "catalog.jsonl", self.rows)
        np.savez_compressed(directory / "features.npz", category=self.category,
                            description=self.description, image=self.image)
        (directory / "metadata.json").write_text(json.dumps(self.metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        with np.load(directory / "features.npz", allow_pickle=False) as vectors:
            return cls(read_jsonl(directory / "catalog.jsonl"), vectors["category"],
                       vectors["description"], vectors["image"],
                       json.loads((directory / "metadata.json").read_text(encoding="utf-8")))

    def search(self, query, *, top_k=5, category_ratio=0.85, min_score=None, mode="semantic", exclude_ids=()):
        query = normalized(query)
        if query.ndim != 1 or query.shape[0] != self.category.shape[1]:
            raise ValueError("Query has wrong embedding dimension")
        if top_k is not None and top_k < 1:
            raise ValueError("top-k must be positive")
        if mode not in {"semantic", "clip"}:
            raise ValueError("mode must be semantic or clip")
        excluded = set(exclude_ids)
        candidates = np.array([i for i, row in enumerate(self.rows) if row["id"] not in excluded], dtype=int)
        if not len(candidates):
            return []
        if mode == "semantic":
            scores = self.category[candidates] @ query
            candidates = candidates[scores >= relative_cutoff(float(scores.max()), category_ratio)]
            scores = self.description[candidates] @ query
        else:
            scores = self.image[candidates] @ query
        order = sorted(range(len(candidates)), key=lambda j: (-float(scores[j]), self.rows[candidates[j]]["id"]))
        results = [{**self.rows[candidates[j]], "score": float(scores[j])} for j in order
                   if min_score is None or float(scores[j]) >= min_score]
        return results if top_k is None else results[:top_k]
