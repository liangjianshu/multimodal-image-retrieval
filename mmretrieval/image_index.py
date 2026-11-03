"""Direct image retrieval: one embedding per image, no caption-based filtering."""
import json
from pathlib import Path

import numpy as np

from .data import read_jsonl, write_jsonl
from .retrieval import normalized


class ImageIndex:
    def __init__(self, rows, embeddings, metadata):
        self.rows, self.metadata = rows, metadata
        self.embeddings = normalized(embeddings)
        if not rows or len({row["id"] for row in rows}) != len(rows):
            raise ValueError("Image index requires nonempty unique ids")
        if self.embeddings.shape != (len(rows), metadata["dimension"]):
            raise ValueError("Image embedding dimensions do not match the index metadata")

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        meta_path = directory / "metadata.json"
        if meta_path.exists() and json.loads(meta_path.read_text()).get("backend") != "embeddinggemma2":
            raise ValueError("Choose a new output directory; this contains a legacy index")
        write_jsonl(directory / "catalog.jsonl", self.rows)
        np.savez_compressed(directory / "features.npz", embeddings=self.embeddings)
        meta_path.write_text(json.dumps(self.metadata, ensure_ascii=False, indent=2) + "\n")

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        metadata = json.loads((directory / "metadata.json").read_text())
        if metadata.get("backend") != "embeddinggemma2":
            raise ValueError("This is not an EmbeddingGemma 2 image index; rebuild it")
        with np.load(directory / "features.npz", allow_pickle=False) as saved:
            return cls(read_jsonl(directory / "catalog.jsonl"), saved["embeddings"], metadata)

    def verify(self, expected):
        for key, value in expected.items():
            if self.metadata.get(key) != value:
                raise ValueError(f"Index {key} differs from the query encoder; rebuild the index")

    def search(self, query, top_k=5, exclude_ids=(), min_score=None, include_ids=None):
        query = normalized(query)
        if query.shape != (self.embeddings.shape[1],):
            raise ValueError("Query dimensions differ from image embeddings")
        if top_k is not None and (type(top_k) is not int or top_k < 1):
            raise ValueError("top_k must be a positive integer or None")
        excluded = set(exclude_ids)
        included = None if include_ids is None else set(include_ids)
        candidates = [i for i, row in enumerate(self.rows) if row["id"] not in excluded
                      and (included is None or row["id"] in included)]
        scores = self.embeddings[candidates] @ query
        order = sorted(range(len(candidates)), key=lambda j: (-float(scores[j]), self.rows[candidates[j]]["id"]))
        results = [{**self.rows[candidates[j]], "score": float(scores[j])} for j in order
                   if min_score is None or scores[j] >= min_score]
        return results if top_k is None else results[:top_k]
