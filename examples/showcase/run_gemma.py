"""Rebuild the CC0 gallery from pixels with EmbeddingGemma 2 and time real queries."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from mmretrieval.cli import load_config
from mmretrieval.data import image_path, read_jsonl, validate_catalog
from mmretrieval.gemma import GemmaEncoder
from mmretrieval.image_index import ImageIndex


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/embeddinggemma2.json")
    parser.add_argument("--output", default="artifacts/showcase-gemma2")
    parser.add_argument("--model-path", help="Override local weight location, omitted from public report")
    parser.add_argument("--device")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--display-metadata", default="docs/demo/catalog.described.jsonl",
                        help="Optional captions for display only; they are never embedded")
    args = parser.parse_args()
    if args.repeats < 1:
        raise ValueError("repeats must be positive")
    import torch
    torch.set_num_threads(4)
    torch.manual_seed(42)
    public = load_config(args.config)
    if public["backend"] != "embeddinggemma2":
        raise ValueError("Use an EmbeddingGemma configuration")
    if args.device:
        public["device"] = args.device
    config = dict(public)
    if args.model_path:
        config["embedding_model_path"] = args.model_path
    output, gallery = Path(args.output), ROOT / "examples/showcase"
    sources = json.loads((gallery / "sources.json").read_text())
    for source in sources:
        if hashlib.sha256((gallery / source["file_path"]).read_bytes()).hexdigest() != source["sha256"]:
            raise ValueError("Gallery checksum mismatch")
    rows = validate_catalog(read_jsonl(gallery / "catalog.jsonl"), gallery)
    display = {r["id"]: r for r in read_jsonl(args.display_metadata)} if Path(args.display_metadata).exists() else {}
    for row in rows:
        caption = display.get(row["id"], {})
        if caption and caption["file_path"] != row["file_path"]:
            raise ValueError("Display metadata image path mismatch")
        row["retrieval_categories"] = caption.get("retrieval_categories", [row["id"]])
        row["retrieval_description"] = caption.get("retrieval_description", "")
        row["display_metadata_only"] = True
    started = time.perf_counter()
    encoder = GemmaEncoder(config)
    encoder.sync()
    load_seconds = time.perf_counter() - started
    paths = [image_path(gallery, row["file_path"]) for row in rows]
    started = time.perf_counter()
    encoder.images(paths[:1])
    warmup_seconds = time.perf_counter() - started
    started = time.perf_counter()
    vectors = encoder.images(paths)
    encoding_seconds = time.perf_counter() - started
    index = ImageIndex(rows, vectors, {**encoder.fingerprint, "version": 1})
    index.save(output / "index")
    records = []
    queries_path = gallery / "queries-lora.json"
    for query in json.loads(queries_path.read_text()):
        reference = image_path(gallery, query["image"]) if query.get("image") else None
        excluded = [row["id"] for row in rows if reference and image_path(gallery, row["file_path"]) == reference]
        encoder.query(query.get("text"), reference)  # Per-query warmup excluded from timing.
        encoding_ms, search_ms = [], []
        for _ in range(args.repeats):
            encoder.sync()
            started = time.perf_counter()
            vector = encoder.query(query.get("text"), reference)
            encoding_ms.append((time.perf_counter() - started) * 1000)
            started = time.perf_counter()
            ranked = index.search(vector, top_k=None, exclude_ids=excluded)
            search_ms.append((time.perf_counter() - started) * 1000)
        records.append({**query, "mode": "embedding", "top_k": 3, "exclude_ids": excluded,
            "intent": None, "search_text": query.get("text"), "candidate_count": len(ranked),
            "encoding_ms": encoding_ms, "median_encoding_ms": statistics.median(encoding_ms),
            "median_search_ms": statistics.median(search_ms), "results": ranked[:3]})
        print(query["id"], [(r["id"], round(r["score"], 4)) for r in ranked[:3]],
              "median_ms", round(statistics.median(encoding_ms), 2), flush=True)
    packages = {name: importlib.metadata.version(name) for name in
                ("torch", "transformers", "sentence-transformers", "numpy", "Pillow")}
    if config["device"].startswith("npu"):
        packages["torch-npu"] = importlib.metadata.version("torch-npu")
    report = {"created_at": datetime.now(timezone.utc).isoformat(), "kind": "real-model-qualitative-showcase",
        "notice": "Native EmbeddingGemma 2 image retrieval. Captions shown for display only. No manual ranking.",
        "backend": "embeddinggemma2", "gallery_size": len(rows),
        "config": {k: v for k, v in public.items() if k.startswith("embedding_") or k in ("backend", "vision_tokens", "device")},
        "encoder": encoder.fingerprint, "resolved_device": encoder.device,
        "parameters_loaded": sum(p.numel() for p in encoder.model.parameters()),
        "dtype": str(next(encoder.model.parameters()).dtype), "python": platform.python_version(),
        "packages": packages, "query_policy": "original-user-text-and-reference-pixels",
        "adapter_verification": None, "query_file_sha256": hashlib.sha256(queries_path.read_bytes()).hexdigest(),
        "timing": {"load_seconds": load_seconds, "warmup_seconds": warmup_seconds,
                   "gallery_encoding_seconds": encoding_seconds, "query_repeats": args.repeats,
                   "query_timing_includes": "preprocessing, device transfer, forward, pooling, normalization, CPU result",
                   "query_timing_excludes": "model loading, per-query warmup, network, index scan"},
        "queries": records}
    write_json(output / "results.json", report)
    print("Saved", output / "results.json", flush=True)


if __name__ == "__main__":
    main()
