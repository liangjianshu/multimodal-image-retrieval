import json
from unittest.mock import Mock

import numpy as np
import pytest

from mmretrieval.hybrid import HybridRetriever
from mmretrieval.image_index import ImageIndex
from mmretrieval.models import Qwen
from mmretrieval.cli import execute, parser, load_config
from mmretrieval.gemma import signature


def setup_pipeline(intent):
    rows = [{"id": "cup", "retrieval_categories": ["杯子"]},
            {"id": "tea", "retrieval_categories": ["茶杯"]},
            {"id": "chair", "retrieval_categories": ["椅子"]}]
    # The chair would rank first without category routing.
    index = ImageIndex(rows, [[.8, .6], [.7, .7], [1., 0.]], {"dimension": 2})
    router, encoder = Mock(), Mock()
    router.intent.return_value = intent
    encoder.fingerprint = {"dimension": 2}
    encoder.query.return_value = np.array([1., 0.])
    factory = Mock(return_value=encoder)
    return HybridRetriever(index, router, factory), router, encoder, factory


def intent(needed=True, mode="selected", categories=None, top_k=None):
    return {"needs_retrieval": needed, "target_category": "杯子" if needed and mode != "all" else "",
            "description": "杯子" if needed else "", "constraints": [], "top_k": top_k,
            "category_filter": mode, "candidate_categories": categories or []}


def test_nonretrieval_calls_qwen_but_never_loads_encoder_or_searches_index():
    pipeline, router, encoder, factory = setup_pipeline(intent(False, "none"))
    pipeline.index.search = Mock(side_effect=AssertionError("Must not search"))
    result = pipeline.search("只描述图片，不要检索", "cup.jpg")
    router.intent.assert_called_once_with("只描述图片，不要检索", "cup.jpg", available_categories=["杯子", "椅子", "茶杯"])
    factory.assert_not_called()
    encoder.query.assert_not_called()
    assert result["status"] == "not_retrieval" and not result["retrieval_executed"] and result["results"] == []


def test_category_prefilter_keeps_subtypes_and_raw_input_despite_lossy_summary():
    pipeline, router, encoder, _ = setup_pipeline(intent(categories=["杯子", "茶杯"]))
    original = "像参考图的杯子，但要红色，不要蓝色"
    result = pipeline.search(original, "ref.jpg", top_k=3)
    assert [r["id"] for r in result["results"]] == ["cup", "tea"]
    assert result["candidate_count"] == 2 and result["search_text"] == original
    encoder.query.assert_called_once_with(original, "ref.jpg")


def test_image_only_still_uses_qwen_then_encodes_original_pixels():
    pipeline, router, encoder, _ = setup_pipeline(intent(categories=["杯子", "茶杯"]))
    pipeline.search(None, "ref.jpg")
    assert router.intent.call_args.args == ("", "ref.jpg")
    encoder.query.assert_called_once_with(None, "ref.jpg")


def test_ambiguous_category_explicitly_allows_full_search_and_respects_quantity():
    pipeline, _, _, _ = setup_pipeline(intent(mode="all", top_k=2))
    result = pipeline.search("找两张相似风格的图片", top_k=5, exclude_ids=["tea"])
    assert result["candidate_count"] == 2 and result["top_k"] == 2
    assert [r["id"] for r in result["results"]] == ["chair", "cup"]


def test_missing_category_returns_empty_without_forcing_unrelated_results():
    pipeline, _, _, factory = setup_pipeline(intent(categories=[]))
    result = pipeline.search("找飞机")
    factory.assert_not_called()
    assert result["status"] == "no_category_match" and result["results"] == []


def test_qwen_schema_rejects_invented_categories_and_routes_image_only():
    model = object.__new__(Qwen)
    model.config = {"intent_profile": "preserve"}
    model.generate = Mock(return_value=json.dumps(intent(categories=["茶杯"]), ensure_ascii=False))
    assert model.intent("", "ref.jpg", available_categories=["杯子", "茶杯"])["candidate_categories"] == ["茶杯"]
    assert "available_categories" in model.generate.call_args.args[0]
    assert model.generate.call_args.args[1] == "ref.jpg"
    with pytest.raises(ValueError, match="unknown gallery"):
        model.intent("找图", available_categories=["椅子"])


def test_nonretrieval_missing_optional_routing_fields_still_stops():
    model = object.__new__(Qwen)
    model.config = {"intent_profile": "preserve"}
    response = intent(False, "none")
    del response["category_filter"], response["candidate_categories"]
    model.generate = Mock(return_value=json.dumps(response))
    result = model.intent("只描述图片", "ref.jpg", available_categories=["杯子"])
    assert result["needs_retrieval"] is False and result["category_filter"] == "none"
    assert result["candidate_categories"] == [] and result["raw_response"] == response
    model.generate.return_value = '{"needs_retrieval":false}'
    minimal = model.intent("你好", available_categories=["杯子"])
    assert minimal["category_filter"] == "none" and minimal["top_k"] is None
    assert minimal["raw_response"] == {"needs_retrieval": False}


def test_known_qwen_target_recovers_missing_extra_category_fields_without_inventing_labels():
    model = object.__new__(Qwen)
    model.config = {"intent_profile": "preserve"}
    response = intent(categories=["杯子"])
    del response["category_filter"], response["candidate_categories"]
    model.generate = Mock(return_value=json.dumps(response))
    result = model.intent("找杯子", available_categories=["杯子", "椅子"])
    assert result["category_filter"] == "selected" and result["candidate_categories"] == ["杯子"]
    assert result["raw_response"] == response
    with pytest.raises(ValueError, match="requires category_filter"):
        model.intent("找杯子", available_categories=["椅子"])
    response["target_category"] = "茶杯"
    model.generate.return_value = json.dumps(response)
    result = model.intent("找茶杯", available_categories=["杯子"], category_aliases={"茶杯": "杯子"})
    assert result["candidate_categories"] == ["杯子"] and result["raw_response"]["target_category"] == "茶杯"


def test_hybrid_cli_routes_nonretrieval_before_gemma_load(tmp_path, monkeypatch):
    config = load_config("configs/showcase-hybrid.json")
    assert config["backend"] == "qwen-gemma2"
    index = ImageIndex([{"id": "cup", "file_path": "a.jpg", "retrieval_categories": ["杯子"]}],
                       np.ones((1, 768)), {**signature(config), "version": 1})
    index.save(tmp_path / "index")
    model = Mock()
    model.intent.return_value = intent(False, "none")
    monkeypatch.setattr("mmretrieval.hybrid.Qwen", lambda *args, **kwargs: model)
    factory = Mock(side_effect=AssertionError("Gemma must not load for nonretrieval"))
    monkeypatch.setattr("mmretrieval.hybrid.GemmaEncoder", factory)
    result = execute(parser().parse_args(["--config", "configs/showcase-hybrid.json", "search", "--index", str(tmp_path / "index"),
                      "--image-root", str(tmp_path), "--text", "你好"]))
    factory.assert_not_called()
    assert result["status"] == "not_retrieval"


def test_qwen_category_normalization_must_cover_labels_without_cycles():
    model = object.__new__(Qwen)
    model.generate = Mock(return_value='{"茶杯":"杯子","杯子":"杯子","椅子":"椅子"}')
    assert model.category_groups(["茶杯", "杯子", "椅子"])["茶杯"] == "杯子"
    model.generate.return_value = '{"茶杯":"杯子","杯子":"茶杯"}'
    with pytest.raises(ValueError, match="themselves"):
        model.category_groups(["茶杯", "杯子"])
