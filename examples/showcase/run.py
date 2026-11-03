"""Run real pretrained models on the bundled CC0 gallery; no synthetic vectors."""
import argparse
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from mmretrieval.cli import encode_query, load_config
from mmretrieval.data import image_path, read_jsonl, validate_catalog, write_jsonl
from mmretrieval.models import Encoders, Qwen
from mmretrieval.retrieval import Index


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=str(ROOT / "configs/showcase.json"))
    p.add_argument("--output", default="artifacts/showcase")
    p.add_argument("--device", help="Override auto; e.g. cuda:0, cpu, or npu:0")
    p.add_argument("--model-root", help="Optional local weights: taiyi/, clip/, qwen-vl-3b/")
    p.add_argument("--reuse-descriptions", action="store_true", help="Reuse this run's completed description stage")
    args = p.parse_args()
    import torch
    torch.set_num_threads(4)
    torch.manual_seed(42)
    gallery, output = ROOT / "examples/showcase", Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    public_config = load_config(args.config)
    if args.device:
        public_config["device"] = args.device
    config = dict(public_config)
    if args.model_root:
        for key, name in (("text_encoder", "taiyi"), ("image_encoder", "clip"), ("vl_model", "qwen-vl-3b")):
            config[key] = str(Path(args.model_root).resolve() / name)
    sources = json.loads((gallery / "sources.json").read_text(encoding="utf-8"))
    for source in sources:
        actual = hashlib.sha256((gallery / source["file_path"]).read_bytes()).hexdigest()
        if actual != source["sha256"]:
            raise ValueError(f"Gallery checksum mismatch: {source['id']}")
    rows = validate_catalog(read_jsonl(gallery / "catalog.jsonl"), gallery)
    started = time.perf_counter()
    described_path = output / "catalog.described.jsonl"
    if args.reuse_descriptions:
        described = validate_catalog(read_jsonl(described_path), gallery)
        if [r["id"] for r in described] != [r["id"] for r in rows]:
            raise ValueError("Saved descriptions do not match the gallery")
        if any(r.get("retrieval_source") != {"model": public_config["vl_model"], "adapter": public_config["adapter"]} for r in described):
            raise ValueError("Saved descriptions use a different model/adapter")
        rows = described
    else:
        model = Qwen(config)
        for row in rows:
            prediction = model.describe(image_path(gallery, row["file_path"]))
            row.update(retrieval_categories=prediction["categories"], retrieval_description=prediction["description"],
                       retrieval_source={"model": public_config["vl_model"], "adapter": public_config["adapter"]})
            print("Described", row["id"], prediction, flush=True)
        write_jsonl(described_path, rows)
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if hasattr(torch, "npu") and config["device"].startswith("npu"):
            torch.npu.empty_cache()
    description_seconds = time.perf_counter() - started
    started = time.perf_counter()
    encoder = Encoders(config)
    category, description, images = [], [], []
    for row in rows:
        category.append(encoder.text("、".join(row["retrieval_categories"])))
        description.append(encoder.text(row["retrieval_description"]))
        images.append(encoder.image(image_path(gallery, row["file_path"])))
        print("Encoded", row["id"], flush=True)
    index = Index(rows, category, description, images, {
        "version": 1, "demo": False, "text_encoder": public_config["text_encoder"],
        "image_encoder": public_config["image_encoder"], "rows_with_model_descriptions": len(rows),
        "rows_with_manual_metadata": 0})
    index.save(output / "index")
    index_seconds = time.perf_counter() - started
    queries = json.loads((gallery / "queries.json").read_text(encoding="utf-8"))
    records = []
    for query in queries:
        started = time.perf_counter()
        reference = image_path(gallery, query["image"]) if query.get("image") else None
        vector = encode_query(encoder, query.get("text"), reference, config["image_weight"])
        excluded = [r["id"] for r in rows if reference and image_path(gallery, r["file_path"]) == reference]
        mode = query.get("mode", "semantic")
        ranked = index.search(vector, top_k=None, category_ratio=config["category_ratio"], exclude_ids=excluded, mode=mode)
        results = ranked[:3]
        records.append({**query, "mode": mode, "top_k": 3, "exclude_ids": excluded,
            "candidate_count": len(ranked), "elapsed_seconds": round(time.perf_counter() - started, 4), "results": results})
        print("Query", query["id"], [(r["id"], round(r["score"], 4)) for r in results], flush=True)
    packages = {}
    for name in ("torch", "torch-npu", "torchvision", "transformers", "qwen-vl-utils", "numpy", "Pillow"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    report = {"created_at": datetime.now(timezone.utc).isoformat(), "kind": "real-model-qualitative-showcase",
        "notice": "Recorded model outputs on 12 CC0 images. No human metadata used for indexing. Not a benchmark or fine-tuning comparison.",
        "gallery_size": len(rows), "seed": 42, "config": public_config, "resolved_device": encoder.device,
        "python": sys.version.split()[0], "packages": packages,
        "description_seconds": round(description_seconds, 3), "descriptions_reused": args.reuse_descriptions,
        "index_seconds": round(index_seconds, 3), "queries": records}
    write_json(output / "results.json", report)
    print("Saved", output / "results.json", flush=True)


if __name__ == "__main__":
    main()
