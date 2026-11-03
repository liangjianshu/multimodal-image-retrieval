"""Qwen intent/category routing followed by native image embedding retrieval."""
import json
from pathlib import Path
import time

import numpy as np

from .data import image_path, read_jsonl, retrieval_metrics, validate_catalog
from .gemma import GemmaEncoder, signature
from .image_index import ImageIndex
from .models import Qwen


def category_labels(row):
    return row.get("retrieval_category_groups", row.get("retrieval_categories"))


def gallery_categories(rows):
    for row in rows:
        labels = category_labels(row)
        if not isinstance(labels, list) or not labels or any(not isinstance(c, str) or not c.strip() for c in labels):
            raise ValueError("Hybrid retrieval needs Qwen-generated retrieval_categories; rebuild its index")
    return sorted({c for row in rows for c in category_labels(row)})


def group_gallery_categories(rows, router):
    raw_labels = sorted({c for row in rows for c in row["retrieval_categories"]})
    mapping = router.category_groups(raw_labels)
    for row in rows:
        row["retrieval_category_groups"] = sorted({mapping[c] for c in row["retrieval_categories"]})
    return mapping


class HybridRetriever:
    def __init__(self, index, router, encoder_factory):
        self.index, self.router, self.encoder_factory = index, router, encoder_factory
        self.encoder = None
        self.categories = gallery_categories(index.rows)

    def search(self, text=None, image=None, *, top_k=5, exclude_ids=(), min_score=None):
        if not (text and text.strip()) and image is None:
            raise ValueError("Provide query text or a reference image")
        start = time.perf_counter()
        routing = {"available_categories": self.categories}
        if self.index.metadata.get("category_groups"):
            routing["category_aliases"] = self.index.metadata["category_groups"]
        intent = self.router.intent(text or "", image, **routing)
        route_seconds = time.perf_counter() - start
        record = {"backend": "qwen-gemma2", "intent": intent, "search_text": text,
                  "image": str(image) if image else None, "retrieval_executed": False,
                  "exclude_ids": sorted(set(exclude_ids)), "results": [], "candidate_count": 0,
                  "top_k": intent.get("top_k") if top_k is not None and intent.get("top_k") else top_k,
                  "timing": {"intent_seconds": route_seconds, "encoder_load_seconds": 0.0,
                             "encoding_seconds": 0.0, "ranking_seconds": 0.0}}
        if not intent["needs_retrieval"]:
            return {**record, "status": "not_retrieval", "elapsed_seconds": time.perf_counter() - start}
        selected = set(intent["candidate_categories"])
        candidates = [row["id"] for row in self.index.rows if row["id"] not in exclude_ids
                      and (intent["category_filter"] == "all" or selected.intersection(category_labels(row)))]
        record["candidate_count"], record["candidate_ids"] = len(candidates), candidates
        if not candidates:
            return {**record, "status": "no_category_match", "elapsed_seconds": time.perf_counter() - start}
        if self.encoder is None:
            loaded = time.perf_counter()
            self.encoder = self.encoder_factory()
            self.index.verify(self.encoder.fingerprint)
            record["timing"]["encoder_load_seconds"] = time.perf_counter() - loaded
        encoded = time.perf_counter()
        # Qwen routes; its summary never replaces the original text or image pixels.
        vector = self.encoder.query(text, image)
        record["timing"]["encoding_seconds"] = time.perf_counter() - encoded
        ranked = time.perf_counter()
        results = self.index.search(vector, record["top_k"], exclude_ids, min_score, include_ids=candidates)
        record["timing"]["ranking_seconds"] = time.perf_counter() - ranked
        return {**record, "status": "retrieved", "retrieval_executed": True, "results": results,
                "elapsed_seconds": time.perf_counter() - start}


def execute_hybrid(args, config):
    if config["intent_profile"] != "preserve":
        raise ValueError("Hybrid routing requires the condition-preserving intent profile")
    if args.command == "index":
        rows = validate_catalog(read_jsonl(args.manifest), args.image_root)
        router = Qwen(config, vision=True)
        for row in rows:
            generated = router.describe(image_path(args.image_root, row["file_path"]))
            row.update(retrieval_categories=generated["categories"], retrieval_description=generated["description"],
                       retrieval_source={"model": config["vl_model"], "adapter": config["adapter"]})
        groups = group_gallery_categories(rows, router)
        encoder = GemmaEncoder(config)
        vectors = encoder.images([image_path(args.image_root, r["file_path"]) for r in rows])
        index = ImageIndex(rows, vectors, {**encoder.fingerprint, "version": 1,
            "category_source": "qwen-predictions", "category_model": config["vl_model"],
            "category_adapter": config["adapter"], "category_groups": groups})
        index.save(args.output)
        return {"backend": "qwen-gemma2", "output": args.output, "images": len(rows),
                "categories": gallery_categories(rows), "dimension": vectors.shape[1]}
    if args.mode not in ("auto", "embedding"):
        raise ValueError("Hybrid retrieval uses embedding mode after Qwen routing")
    index = ImageIndex.load(args.index)
    index.verify(signature(config))
    gallery_categories(index.rows)
    router = Qwen(config, vision=True)  # The same VL model/adapter handles text and images.
    pipeline = HybridRetriever(index, router, lambda: GemmaEncoder(config))

    def search(text, relative_image, excluded):
        reference = image_path(args.image_root, relative_image) if relative_image else None
        if reference:
            excluded.update(r["id"] for r in index.rows if image_path(args.image_root, r["file_path"]) == reference)
        result = pipeline.search(text, reference, top_k=None if args.all_matches else args.top_k,
                                 exclude_ids=excluded, min_score=args.min_score)
        result["image"] = relative_image
        return result

    if args.command == "search":
        return search(args.text, args.image, set(args.exclude_id))
    queries, seen, details = read_jsonl(args.queries), set(), []
    ids = {r["id"] for r in index.rows}
    for query in queries:
        if not isinstance(query.get("id"), str) or query["id"] in seen:
            raise ValueError("Evaluation query ids must be unique strings")
        seen.add(query["id"])
        relevant = query.get("relevant_ids")
        if not isinstance(relevant, list) or not relevant or not set(relevant) <= ids:
            raise ValueError("Retrieval evaluation requires relevant_ids; evaluate routing separately")
        result = search(query.get("text"), query.get("image"), set(query.get("exclude_ids", [])))
        if set(result["exclude_ids"]) & set(relevant):
            raise ValueError("Relevant ids include an excluded image")
        predicted = [r["id"] for r in result["results"]]
        details.append({"id": query["id"], "predicted_ids": predicted, "intent": result["intent"],
                        **retrieval_metrics(predicted, relevant)})
    if not details:
        raise ValueError("Evaluation queries are empty")
    report = {"backend": "qwen-gemma2", "query_count": len(details), "queries": details,
              "macro": {key: float(np.mean([r[key] for r in details])) for key in ("precision", "recall", "f1")}}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return {k: v for k, v in report.items() if k != "queries"}
