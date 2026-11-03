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
from mmretrieval.intent import resolve_search_text
from mmretrieval.retrieval import Index


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def audit_adapter(model, adapter_path):
    """Check the actual loaded LoRA tensors against the supplied checkpoint."""
    from safetensors import safe_open
    from peft import get_peft_model_state_dict
    import torch
    path = Path(adapter_path)
    weights = path / "adapter_model.safetensors"
    # PEFT's export mapping covers both LoRA and modules_to_save (visual merger).
    # The latter has a different live parameter name from its checkpoint key.
    params = get_peft_model_state_dict(model.model, adapter_name="default")
    checked, nonzero_b = 0, 0
    with safe_open(weights, framework="pt", device="cpu") as saved:
        for name in saved.keys():
            if name not in params:
                raise ValueError(f"Adapter tensor not loaded: {name}")
            actual = params[name].detach().cpu()
            expected = saved.get_tensor(name).to(actual.dtype)
            if not torch.equal(actual, expected):
                raise ValueError(f"Loaded adapter tensor differs from checkpoint: {name}")
            checked += 1
            nonzero_b += int(".lora_B." in name and bool(torch.count_nonzero(actual)))
    if not checked or not nonzero_b or "default" not in model.model.active_adapters:
        raise ValueError("Expected a nonempty, active LoRA adapter")
    cfg = json.loads((path / "adapter_config.json").read_text())
    return {"sha256": hashlib.sha256(weights.read_bytes()).hexdigest(), "bytes": weights.stat().st_size,
        "base_model": cfg["base_model_name_or_path"], "rank": cfg["r"], "alpha": cfg["lora_alpha"],
        "verified_tensors": checked, "nonzero_lora_b_tensors": nonzero_b,
        "active_adapters": list(model.model.active_adapters)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=str(ROOT / "configs/showcase.json"))
    p.add_argument("--output", default="artifacts/showcase")
    p.add_argument("--device", help="Override auto; e.g. cuda:0, cpu, or npu:0")
    p.add_argument("--model-root", help="Optional local weights: taiyi/, clip/, qwen-vl-3b/ or qwen-vl-7b/")
    p.add_argument("--adapter-path", help="Local adapter location; the report retains the public config alias")
    p.add_argument("--queries", help="Optional query JSON; historical profile uses queries-lora.json")
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
    historical = config["prompt_profile"] == "historical"
    retrieval_source = {"model": public_config["vl_model"], "adapter": public_config["adapter"],
                        "prompt_profile": public_config["prompt_profile"]}
    if args.model_root:
        vl_name = "qwen-vl-7b" if config["vl_model"] == "Qwen/Qwen2.5-VL-7B-Instruct" else "qwen-vl-3b"
        for key, name in (("text_encoder", "taiyi"), ("image_encoder", "clip"), ("vl_model", vl_name)):
            config[key] = str(Path(args.model_root).resolve() / name)
    if args.adapter_path:
        if not public_config["adapter"]:
            raise ValueError("Set the public adapter alias in the config before using --adapter-path")
        config["adapter"] = str(Path(args.adapter_path).resolve())
    queries_path = Path(args.queries) if args.queries else gallery / ("queries-lora.json" if historical else "queries.json")
    queries = json.loads(queries_path.read_text(encoding="utf-8"))
    model = None
    adapter_audit = None
    if not args.reuse_descriptions or historical:
        model = Qwen(config)
        if config["adapter"]:
            adapter_audit = audit_adapter(model, config["adapter"])
            write_json(output / "adapter-verification.json", adapter_audit)
            print("Adapter verified", adapter_audit, flush=True)
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
        if any(r.get("retrieval_source") != retrieval_source for r in described):
            raise ValueError("Saved descriptions use a different model/adapter")
        rows = described
    else:
        for row in rows:
            prediction = model.describe(image_path(gallery, row["file_path"]))
            row.update(retrieval_categories=prediction["categories"], retrieval_description=prediction["description"],
                       retrieval_source=retrieval_source)
            print("Described", row["id"], prediction, flush=True)
        write_jsonl(described_path, rows)
    intents = {}
    if historical:
        for query in queries:
            if query.get("image"):
                fallback = "帮我找三张和参考图片同类的图片。" if config["intent_profile"] == "historical" else ""
                intent = model.intent(query.get("text") or fallback,
                                      image_path(gallery, query["image"]))
                intents[query["id"]] = intent
                print("Intent", query["id"], intent, flush=True)
        write_json(output / "intents.json", intents)
    if model is not None:
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
        category.append(encoder.text((", " if historical else "、").join(row["retrieval_categories"])))
        description.append(encoder.text(row["retrieval_description"]))
        images.append(encoder.image(image_path(gallery, row["file_path"])))
        print("Encoded", row["id"], flush=True)
    index = Index(rows, category, description, images, {
        "version": 1, "demo": False, "text_encoder": public_config["text_encoder"],
        "image_encoder": public_config["image_encoder"], "rows_with_model_descriptions": len(rows),
        "rows_with_manual_metadata": 0})
    index.save(output / "index")
    index_seconds = time.perf_counter() - started
    records = []
    for query in queries:
        started = time.perf_counter()
        reference = image_path(gallery, query["image"]) if query.get("image") else None
        intent = intents.get(query["id"])
        search_text = resolve_search_text(query.get("text"), intent, config["intent_profile"]) if intent and intent["needs_retrieval"] else query.get("text")
        vector = encode_query(encoder, search_text, reference, config["image_weight"])
        excluded = [r["id"] for r in rows if reference and image_path(gallery, r["file_path"]) == reference]
        mode = query.get("mode", "semantic")
        ranked = [] if intent and not intent["needs_retrieval"] else index.search(
            vector, top_k=None, category_ratio=config["category_ratio"], exclude_ids=excluded, mode=mode)
        results = ranked[:3]
        records.append({**query, "mode": mode, "top_k": 3, "exclude_ids": excluded,
            "intent": intent, "search_text": search_text,
            "candidate_count": len(ranked), "elapsed_seconds": round(time.perf_counter() - started, 4), "results": results})
        print("Query", query["id"], [(r["id"], round(r["score"], 4)) for r in results], flush=True)
    packages = {}
    for name in ("torch", "torch-npu", "torchvision", "transformers", "peft", "qwen-vl-utils", "numpy", "Pillow"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    report = {"created_at": datetime.now(timezone.utc).isoformat(), "kind": "real-model-qualitative-showcase",
        "notice": "Recorded model outputs on 12 CC0 images. No human metadata used for indexing. Not a benchmark or fine-tuning comparison.",
        "gallery_size": len(rows), "seed": 42, "config": public_config, "resolved_device": encoder.device,
        "adapter_verification": adapter_audit, "query_file_sha256": hashlib.sha256(queries_path.read_bytes()).hexdigest(),
        "query_policy": "original-user-text" if config["intent_profile"] == "preserve" else "generated-summary",
        "python": sys.version.split()[0], "packages": packages,
        "description_seconds": round(description_seconds, 3), "descriptions_reused": args.reuse_descriptions,
        "index_seconds": round(index_seconds, 3), "queries": records}
    write_json(output / "results.json", report)
    print("Saved", output / "results.json", flush=True)


if __name__ == "__main__":
    main()
