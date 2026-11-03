"""CLI operations for one shared native multimodal encoder."""
import json
from pathlib import Path

import numpy as np

from .data import image_path, read_jsonl, retrieval_metrics, validate_catalog
from .gemma import GemmaEncoder, signature
from .image_index import ImageIndex


def execute_direct(args, config):
    if args.command == "index":
        rows = validate_catalog(read_jsonl(args.manifest), args.image_root)
        encoder = GemmaEncoder(config)
        embeddings = encoder.images([image_path(args.image_root, r["file_path"]) for r in rows])
        index = ImageIndex(rows, embeddings, {**encoder.fingerprint, "version": 1})
        index.save(args.output)
        return {"output": args.output, "images": len(rows), "dimension": embeddings.shape[1],
                "backend": "embeddinggemma2", "corpus_input": "image-pixels"}
    if args.mode not in ("auto", "embedding"):
        raise ValueError("EmbeddingGemma 2 uses --mode embedding (or auto); use the legacy config for semantic/clip")
    if getattr(args, "intent", False):
        raise ValueError("Native joint encoding uses the original input directly; omit --intent")
    index = ImageIndex.load(args.index)
    index.verify(signature(config))
    encoder = GemmaEncoder(config)
    index.verify(encoder.fingerprint)
    k = None if args.all_matches else args.top_k

    def search(text, relative_image, excluded):
        reference = image_path(args.image_root, relative_image) if relative_image else None
        if reference:
            excluded.update(r["id"] for r in index.rows if image_path(args.image_root, r["file_path"]) == reference)
        vector = encoder.query(text, reference)
        return index.search(vector, k, excluded, args.min_score), excluded

    if args.command == "search":
        results, excluded = search(args.text, args.image, set(args.exclude_id))
        return {"backend": "embeddinggemma2", "search_text": args.text, "image": args.image,
                "fusion": "native-joint", "exclude_ids": sorted(excluded), "results": results}
    queries = read_jsonl(args.queries)
    ids, seen, details = {r["id"] for r in index.rows}, set(), []
    if not queries:
        raise ValueError("Evaluation queries are empty")
    for query in queries:
        if not isinstance(query.get("id"), str) or query["id"] in seen:
            raise ValueError("Evaluation query ids must be unique strings")
        seen.add(query["id"])
        relevant = query.get("relevant_ids")
        if not isinstance(relevant, list) or not relevant or not set(relevant) <= ids:
            raise ValueError("Each query needs relevant_ids present in the index")
        results, excluded = search(query.get("text"), query.get("image"), set(query.get("exclude_ids", [])))
        if excluded & set(relevant):
            raise ValueError("Relevant ids include the reference/excluded image")
        predicted = [r["id"] for r in results]
        details.append({"id": query["id"], "predicted_ids": predicted, **retrieval_metrics(predicted, relevant)})
    report = {"backend": "embeddinggemma2", "query_count": len(details), "index_metadata": index.metadata,
              "macro": {key: float(np.mean([r[key] for r in details])) for key in ("precision", "recall", "f1")},
              "queries": details}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return {k: v for k, v in report.items() if k != "queries"}
