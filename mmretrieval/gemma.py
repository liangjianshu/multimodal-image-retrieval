"""Native text/image/joint embeddings; isolated from the legacy Qwen dependencies."""
import hashlib
import json
from pathlib import Path

import numpy as np

from .models import resolve_device
from .retrieval import normalized

MODEL_ID = "google/embeddinggemma-2"
REVISION = "f9972ac22e696a105c482202646979e33e4b5a1e"
WEIGHTS_SHA256 = "197a32965d4b1105faf060417baa899e193fb73cd401f42ec9295234d5553d79"
QUERY_PREFIX = "task: search result | query: "


def signature(config):
    """Same vector width is insufficient: model, revision and preprocessing must match."""
    return {"backend": "embeddinggemma2", "model": config["embedding_model"],
            "revision": config["embedding_revision"], "dimension": config["embedding_dimension"],
            "vision_tokens": config["vision_tokens"], "query_prefix": QUERY_PREFIX,
            "corpus_input": "image-pixels", "fusion": "native-joint", "normalized": True}


def query_payload(text=None, image=None):
    text = text if text and text.strip() else None
    if text is None and image is None:
        raise ValueError("Provide query text or a reference image")
    if text and any(token in text for token in ("<|image|>", "<|audio|>", "<|video|>")):
        raise ValueError("Query text must not contain reserved media placeholder tokens")
    if text is None:
        return {"image": str(image)}
    if image is None:
        return QUERY_PREFIX + text
    return {"text": QUERY_PREFIX + text + " <|image|>", "image": str(image)}


class GemmaEncoder:
    def __init__(self, config):
        import torch
        from huggingface_hub import snapshot_download
        from sentence_transformers import SentenceTransformer
        self.config, self.torch = config, torch
        self.device = resolve_device(config["device"])
        location = config.get("embedding_model_path") or snapshot_download(
            config["embedding_model"], revision=config["embedding_revision"])
        self.location = Path(location)
        if json.loads((self.location / "config.json").read_text())["model_type"] != "embedding_gemma2":
            raise ValueError("Expected an EmbeddingGemma 2 checkpoint")
        digest = hashlib.sha256()
        with (self.location / "model.safetensors").open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
        self.weight_sha256 = digest.hexdigest()
        if config["embedding_model"] == MODEL_ID and config["embedding_revision"] == REVISION:
            if self.weight_sha256 != WEIGHTS_SHA256:
                raise ValueError("Local weights do not match the pinned official checkpoint")
        if config.get("embedding_weights_sha256") and self.weight_sha256 != config["embedding_weights_sha256"]:
            raise ValueError("Embedding checkpoint hash differs from configuration")
        dtype = torch.bfloat16 if self.device.startswith("npu") or (
            self.device.startswith("cuda") and torch.cuda.is_bf16_supported()) else torch.float32
        self.model = SentenceTransformer(str(self.location), device=self.device,
            config_kwargs={"audio_config": None},
            model_kwargs={"dtype": dtype, "attn_implementation": "eager"})
        self.model.eval()
        self.fingerprint = {**signature(config), "weight_sha256": self.weight_sha256}
        self.processing = {"image": {"max_soft_tokens": config["vision_tokens"]}}

    def sync(self):
        if self.device.startswith("npu"):
            self.torch.npu.synchronize()
        elif self.device.startswith("cuda"):
            self.torch.cuda.synchronize()
        elif self.device == "mps":
            self.torch.mps.synchronize()

    def _encode(self, payloads):
        vectors = self.model.encode(payloads, batch_size=self.config["embedding_batch_size"],
            show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True,
            truncate_dim=self.config["embedding_dimension"], processing_kwargs=self.processing)
        self.sync()
        vectors = normalized(vectors)
        if vectors.shape != (len(payloads), self.config["embedding_dimension"]):
            raise ValueError("Unexpected embedding shape")
        return vectors

    def images(self, paths):
        return self._encode([{"image": str(path)} for path in paths])

    def query(self, text=None, image=None):
        payload = query_payload(text, image)
        if text:
            # Reserve space for media markers and the selected visual token budget.
            tokens = self.model.tokenizer.encode(QUERY_PREFIX + text, add_special_tokens=True)
            if len(tokens) + (self.config["vision_tokens"] if image else 0) + 32 > 8192:
                raise ValueError("Query exceeds the 8192-token budget; shorten it explicitly")
        return self._encode([payload])[0]
