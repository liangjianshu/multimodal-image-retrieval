import json
from unittest.mock import Mock

import numpy as np
import pytest
from PIL import Image

from mmretrieval.cli import execute, parser, load_config, checked_index
from mmretrieval.data import write_jsonl
from mmretrieval.gemma import GemmaEncoder, QUERY_PREFIX, query_payload, signature
from mmretrieval.image_index import ImageIndex


def test_joint_query_uses_one_payload_and_preserves_all_text():
    text = "像这辆车，但在草地上，不要蓝墙背景"
    payload = query_payload(text, "reference.jpg")
    assert payload == {"text": QUERY_PREFIX + text + " <|image|>", "image": "reference.jpg"}
    encoder = object.__new__(GemmaEncoder)
    encoder.config = {"vision_tokens": 280}
    encoder.model = Mock()
    encoder.model.tokenizer.encode.return_value = [1, 2]
    encoder._encode = Mock(return_value=np.array([[1., 0.]]))
    encoder.query(text, "reference.jpg")
    encoder._encode.assert_called_once_with([payload])
    assert query_payload(None, "ref.jpg") == {"image": "ref.jpg"}
    with pytest.raises(ValueError):
        query_payload("<|image|>", "ref.jpg")


def test_same_width_different_model_or_preprocessing_cannot_mix(tmp_path):
    config = load_config(None)
    meta = {**signature(config), "weight_sha256": "abc"}
    index = ImageIndex([{"id": "a", "file_path": "a.jpg"}], np.ones((1, 768)), meta)
    index.save(tmp_path)
    loaded = ImageIndex.load(tmp_path)
    loaded.verify(meta)
    for field, wrong in [("model", "another-model"), ("revision", "other"), ("vision_tokens", 560), ("weight_sha256", "wrong")]:
        with pytest.raises(ValueError, match=field):
            loaded.verify({**meta, field: wrong})
    with pytest.raises(ValueError, match="EmbeddingGemma"):
        checked_index(tmp_path, {"backend": "taiyi-clip"})


def test_direct_image_ranking_ignores_caption_labels_and_excludes_reference():
    rows = [{"id": "ref", "categories": ["杯子"]}, {"id": "correct", "categories": ["wrong"]},
            {"id": "wrong", "categories": ["自行车"]}]
    index = ImageIndex(rows, [[1, 0], [.9, .1], [0, 1]], {"dimension": 2})
    assert [r["id"] for r in index.search([1, 0], exclude_ids=["ref"])] == ["correct", "wrong"]


def test_pure_cli_builds_from_pixels_without_description_or_qwen(tmp_path, monkeypatch):
    config = tmp_path / "pure.json"
    config.write_text('{"backend":"embeddinggemma2"}')
    prefix = ["--config", str(config)]
    Image.new("RGB", (16, 16)).save(tmp_path / "a.jpg")
    manifest = tmp_path / "catalog.jsonl"
    write_jsonl(manifest, [{"id": "a", "file_path": "a.jpg", "categories": [], "description": ""}])
    fake = Mock()
    fake.fingerprint = signature(load_config(None))
    fake.images.return_value = np.ones((1, 768))
    monkeypatch.setattr("mmretrieval.direct.GemmaEncoder", lambda *_: fake)
    output = tmp_path / "index"
    execute(parser().parse_args(prefix + ["index", "--manifest", str(manifest), "--image-root", str(tmp_path), "--output", str(output)]))
    fake.images.assert_called_once_with([tmp_path / "a.jpg"])
    assert ImageIndex.load(output).metadata["corpus_input"] == "image-pixels"
    fake.query.return_value = np.ones(768)
    result = execute(parser().parse_args(prefix + ["search", "--index", str(output), "--image-root", str(tmp_path),
                                         "--text", "不要蓝墙背景的椅子", "--top-k", "1"]))
    fake.query.assert_called_with("不要蓝墙背景的椅子", None)
    assert result["results"][0]["id"] == "a"
    queries = tmp_path / "queries.jsonl"
    write_jsonl(queries, [{"id": "q", "text": "椅子", "relevant_ids": ["a"]}])
    report = execute(parser().parse_args(prefix + ["evaluate", "--index", str(output), "--image-root", str(tmp_path),
        "--queries", str(queries), "--top-k", "1", "--output", str(tmp_path / "evaluation.json")]))
    assert report["macro"]["f1"] == 1
    with pytest.raises(ValueError, match="target Qwen"):
        execute(parser().parse_args(prefix + ["describe", "--manifest", str(manifest), "--image-root", str(tmp_path),
                                     "--output", str(tmp_path / "described.jsonl")]))
