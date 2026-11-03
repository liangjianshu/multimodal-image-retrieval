"""Rerun Qwen + original LoRA routing/category generation, then Gemma image retrieval."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from mmretrieval.cli import load_config
from mmretrieval.data import image_path, read_jsonl, validate_catalog, write_jsonl
from mmretrieval.gemma import GemmaEncoder
from mmretrieval.hybrid import HybridRetriever, gallery_categories, group_gallery_categories
from mmretrieval.image_index import ImageIndex
from mmretrieval.intent import PRESERVE_INTENT_PROMPT, CATEGORY_ROUTING_PROMPT, CATEGORY_GROUP_PROMPT
from mmretrieval.models import Qwen
from examples.showcase.run import audit_adapter


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/showcase-hybrid.json")
    parser.add_argument("--output", default="artifacts/showcase-hybrid")
    parser.add_argument("--device")
    parser.add_argument("--qwen-path")
    parser.add_argument("--gemma-path")
    parser.add_argument("--adapter-path")
    parser.add_argument("--reuse-descriptions", action="store_true")
    args = parser.parse_args()
    import torch
    torch.set_num_threads(4)
    torch.manual_seed(42)
    public = load_config(args.config)
    if public["backend"] != "qwen-gemma2":
        raise ValueError("Use the Qwen/Gemma hybrid config")
    if args.device:
        public["device"] = args.device
    config = dict(public)
    for arg, key in ((args.qwen_path, "vl_model"), (args.gemma_path, "embedding_model_path"), (args.adapter_path, "adapter")):
        if arg:
            config[key] = arg
    output, gallery = Path(args.output), ROOT / "examples/showcase"
    output.mkdir(parents=True, exist_ok=True)
    for source in json.loads((gallery / "sources.json").read_text()):
        if hashlib.sha256((gallery / source["file_path"]).read_bytes()).hexdigest() != source["sha256"]:
            raise ValueError("Gallery checksum mismatch")
    rows = validate_catalog(read_jsonl(gallery / "catalog.jsonl"), gallery)
    start = time.perf_counter()
    router = Qwen(config, vision=True)
    qwen_load_seconds = time.perf_counter() - start
    audit = audit_adapter(router, config["adapter"]) if config["adapter"] else None
    write_json(output / "adapter-verification.json", audit)
    print("Qwen loaded; adapter audit", audit, flush=True)
    source = {"model": public["vl_model"], "adapter": public["adapter"], "prompt_profile": public["prompt_profile"]}
    start = time.perf_counter()
    if args.reuse_descriptions:
        saved = validate_catalog(read_jsonl(output / "catalog.described.jsonl"), gallery)
        if [(r["id"], r["file_path"]) for r in saved] != [(r["id"], r["file_path"]) for r in rows]:
            raise ValueError("Description cache does not match gallery")
        if any(r.get("retrieval_source") != source for r in saved):
            raise ValueError("Description cache model/prompt differs")
        rows = saved
    else:
        for row in rows:
            prediction = router.describe(image_path(gallery, row["file_path"]))
            row.update(retrieval_categories=prediction["categories"], retrieval_description=prediction["description"],
                       retrieval_source=source)
            print("Described", row["id"], prediction, flush=True)
        write_jsonl(output / "catalog.described.jsonl", rows)
    description_seconds = time.perf_counter() - start
    start = time.perf_counter()
    category_groups = group_gallery_categories(rows, router)
    category_group_seconds = time.perf_counter() - start
    write_json(output / "category-groups.json", category_groups)
    write_jsonl(output / "catalog.described.jsonl", rows)
    print("Category groups", category_groups, flush=True)
    print("Gallery categories", gallery_categories(rows), flush=True)
    start = time.perf_counter()
    encoder = GemmaEncoder(config)
    gemma_load_seconds = time.perf_counter() - start
    start = time.perf_counter()
    vectors = encoder.images([image_path(gallery, r["file_path"]) for r in rows])
    gallery_encoding_seconds = time.perf_counter() - start
    index = ImageIndex(rows, vectors, {**encoder.fingerprint, "version": 1,
        "category_source": "qwen-predictions", "category_model": public["vl_model"],
        "category_adapter_sha256": audit["sha256"] if audit else None, "category_groups": category_groups})
    index.save(output / "index")
    pipeline = HybridRetriever(index, router, lambda: encoder)
    records = []
    queries_path = gallery / "queries-hybrid.json"
    for query in json.loads(queries_path.read_text()):
        reference = image_path(gallery, query["image"]) if query.get("image") else None
        excluded = [r["id"] for r in rows if reference and image_path(gallery, r["file_path"]) == reference]
        result = pipeline.search(query.get("text"), reference, top_k=3, exclude_ids=excluded)
        record = {**query, **result, "image": query.get("image"), "mode": "hybrid"}
        record["median_encoding_ms"] = None  # Per-query single run, not the old repeated timing.
        if "expected_retrieval" in query:
            record["routing_check_passed"] = result["intent"]["needs_retrieval"] == query["expected_retrieval"]
        if "expected_top_k" in query:
            record["quantity_check_passed"] = result["intent"]["top_k"] == query["expected_top_k"]
        records.append(record)
        write_json(output / "queries.partial.json", records)
        print("Query", query["id"], result["intent"], "candidates", result["candidate_count"],
              [(r["id"], round(r["score"], 4)) for r in result["results"]],
              "seconds", round(result["elapsed_seconds"], 3), flush=True)
    write_json(output / "intents.json", {r["id"]: r["intent"] for r in records})
    report = {"created_at": datetime.now(timezone.utc).isoformat(), "kind": "real-model-qualitative-showcase",
        "backend": "qwen-gemma2", "gallery_size": len(rows), "gallery_categories": gallery_categories(rows),
        "category_groups": category_groups,
        "notice": "Qwen original LoRA generates categories and routes every query; Gemma ranks original pixels. No manual ranking.",
        "config": {k: v for k, v in public.items() if k.startswith("embedding_") or k in (
            "backend", "vision_tokens", "device", "vl_model", "adapter", "max_pixels", "max_new_tokens", "prompt_profile", "intent_profile")},
        "encoder": encoder.fingerprint, "adapter_verification": audit, "resolved_device": encoder.device,
        "query_policy": "qwen-routing-original-user-text-and-reference-pixels", "seed": 42,
        "prompt_sha256": hashlib.sha256((PRESERVE_INTENT_PROMPT + CATEGORY_ROUTING_PROMPT).encode()).hexdigest(),
        "category_group_prompt_sha256": hashlib.sha256(CATEGORY_GROUP_PROMPT.encode()).hexdigest(),
        "query_file_sha256": hashlib.sha256(queries_path.read_bytes()).hexdigest(),
        "python": platform.python_version(), "packages": {name: importlib.metadata.version(name) for name in (
            "torch", "transformers", "sentence-transformers", "peft", "qwen-vl-utils", "numpy", "Pillow")},
        "timing": {"qwen_load_seconds": qwen_load_seconds, "gemma_load_seconds": gemma_load_seconds,
            "description_seconds": description_seconds, "descriptions_reused": args.reuse_descriptions,
            "category_group_seconds": category_group_seconds,
            "gallery_encoding_seconds": gallery_encoding_seconds,
            "query_note": "Single run per query, model loads excluded, includes Qwen generation; not a repeated latency benchmark."},
        "queries": records}
    write_json(output / "results.json", report)
    print("Saved", output / "results.json", flush=True)
    if any(r.get("routing_check_passed") is False or r.get("quantity_check_passed") is False for r in records):
        raise RuntimeError("Routing/quantity check failed; see saved raw model outputs")


if __name__ == "__main__":
    main()
