"""Regression: a lossy model summary must never erase explicit query conditions."""
import json
from unittest.mock import Mock

import numpy as np
import pytest
from PIL import Image

from mmretrieval.cli import execute, parser
from mmretrieval.intent import resolve_search_text
from mmretrieval.models import Encoders, Qwen
from mmretrieval.retrieval import Index


def router(description="自行车", constraints=None, top_k=None):
    model = object.__new__(Qwen)
    model.config = {"intent_profile": "preserve"}
    model.generate = Mock(return_value=json.dumps({
        "needs_retrieval": True, "target_category": "自行车", "description": description,
        "constraints": constraints if constraints is not None else [], "top_k": top_k,
    }, ensure_ascii=False))
    return model


@pytest.mark.parametrize("original", [
    "草地上的自行车", "红色自行车，不要蓝色的", "像参考图一样，但要放在草地上，不要靠墙",
])
def test_cli_encodes_user_conditions_even_when_model_drops_them(tmp_path, monkeypatch, original):
    Image.new("RGB", (16, 16)).save(tmp_path / "ref.jpg")
    idx = Index([{"id": "candidate", "file_path": "candidate.jpg"}], [[1., 0.]], [[1., 0.]], [[1., 0.]], {})
    enc = Mock()
    enc.text.return_value = np.array([1., 0.])
    enc.image.return_value = np.array([1., 0.])
    model = router()
    monkeypatch.setattr("mmretrieval.cli.checked_index", lambda *_: idx)
    monkeypatch.setattr("mmretrieval.models.Encoders", lambda *_: enc)
    monkeypatch.setattr("mmretrieval.models.Qwen", lambda *_, **kw: model)
    config = tmp_path / "legacy.json"
    config.write_text('{"backend":"taiyi-clip"}')
    args = parser().parse_args(["--config", str(config), "search", "--index", "unused", "--image-root", str(tmp_path),
                               "--image", "ref.jpg", "--text", original, "--intent"])
    result = execute(args)
    assert result["intent"]["description"] == "自行车"  # Deliberately lossy model output.
    assert result["search_text"] == original
    enc.text.assert_called_once_with(original)
    enc.image.assert_called_once()


def test_image_only_search_uses_visual_description_but_cannot_be_empty():
    assert resolve_search_text(None, {"description": "红色陶瓷杯"}) == "红色陶瓷杯"
    with pytest.raises(ValueError, match="Image-only"):
        resolve_search_text(None, {"description": ""})


def test_image_only_operation_is_known_without_a_model_classification():
    model = router()
    model.describe = Mock(return_value={"categories": ["椅子"], "description": "黑色椅子，米色坐垫"})
    result = model.intent("", "chair.jpg")
    assert result["needs_retrieval"] is True
    assert result["description"] == "黑色椅子，米色坐垫"
    assert result["constraints"] == [] and result["top_k"] is None
    model.generate.assert_not_called()
    model.describe.assert_called_once_with("chair.jpg")


def test_legacy_summary_replacement_is_explicitly_opt_in():
    assert resolve_search_text("草地上的自行车", {"description": "自行车"}, "historical") == "自行车"


def test_new_schema_keeps_negation_and_rejects_invalid_fields():
    result = router(constraints=["不要蓝色"], top_k=3).intent("找三张不要蓝色的自行车图片")
    assert result["constraints"] == ["不要蓝色"] and result["top_k"] == 3
    with pytest.raises(ValueError, match="constraints"):
        router(constraints="不要蓝色").intent("找图")
    with pytest.raises(ValueError, match="top_k"):
        router(top_k=True).intent("找图")


def test_overlength_text_fails_before_silently_truncating_conditions():
    encoder = object.__new__(Encoders)
    encoder.text_tokenizer = Mock(return_value={"input_ids": np.zeros((1, 513))})
    encoder.text_model = Mock()
    with pytest.raises(ValueError, match="512-token"):
        encoder.text("很长的条件文本")
    encoder.text_model.assert_not_called()
