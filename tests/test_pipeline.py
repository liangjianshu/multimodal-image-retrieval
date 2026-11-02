import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from mmretrieval.cli import demo, main
from mmretrieval.data import (folder_catalog, image_path, make_sft,
                              read_jsonl, retrieval_metrics, validate_catalog)
from mmretrieval.models import extract_json
from mmretrieval.retrieval import Index, fuse_query


def gallery(tmp_path):
    for category, color in [("cup", "red"), ("bowl", "blue")]:
        (tmp_path / category).mkdir()
        for i in range(3):
            image = Image.new("RGB", (16 + i, 16), color)
            image.save(tmp_path / category / f"{i}.png")
    return folder_catalog(tmp_path, {"cup": "杯子", "bowl": "碗"})


def test_own_folder_to_truthful_sft_without_image_leakage(tmp_path):
    rows = gallery(tmp_path)
    split = make_sft(rows, tmp_path, hard_negatives={"杯子": ["碗"], "碗": ["杯子"]})
    hashes = {name: {r["image_sha256"] for r in items} for name, items in split.items()}
    assert hashes["train"] and hashes["val"] and hashes["train"].isdisjoint(hashes["val"])
    cup_id = next(r["id"] for r in rows if r["categories"] == ["杯子"])
    records = split["train"] + split["val"]
    question = next(r for r in records if r["image_id"] == cup_id and "“碗”" in r["prompt"])
    assert question["answer"].startswith("否") and "杯子" in question["answer"]
    assert split == make_sft(rows, tmp_path, hard_negatives={"杯子": ["碗"], "碗": ["杯子"]})


def test_same_image_different_paths_kept_in_one_split(tmp_path):
    rows = gallery(tmp_path)
    original = tmp_path / rows[0]["file_path"]
    duplicate = original.with_name("duplicate.png")
    duplicate.write_bytes(original.read_bytes())
    rows.append({**rows[0], "id": "duplicate", "file_path": duplicate.relative_to(tmp_path).as_posix()})
    split = make_sft(rows, tmp_path)
    memberships = [{r["image_id"] for r in records} for records in split.values()]
    assert any({rows[0]["id"], "duplicate"} <= members for members in memberships)
    rows[-1]["categories"] = ["错误类别"]
    with pytest.raises(ValueError, match="conflicting"):
        make_sft(rows, tmp_path)


def test_catalog_rejects_duplicate_ids_paths_and_escape(tmp_path):
    rows = gallery(tmp_path)
    with pytest.raises(ValueError, match="duplicate"):
        validate_catalog([rows[0], rows[0]], tmp_path)
    with pytest.raises(ValueError, match="Duplicate image"):
        validate_catalog([rows[0], {**rows[0], "id": "different"}], tmp_path)
    with pytest.raises(ValueError, match="escapes"):
        image_path(tmp_path, "../outside.jpg")
    with pytest.raises(ValueError, match="relative"):
        image_path(tmp_path, str(tmp_path / "cup/0.png"))


def test_model_generated_categories_are_not_automatic_ground_truth(tmp_path):
    rows = gallery(tmp_path)
    rows[0]["categories_source"] = "model"
    with pytest.raises(ValueError, match="verify"):
        make_sft(rows, tmp_path)


def test_two_stage_ranking_roundtrip_and_reference_exclusion(tmp_path):
    rows = [{"id": str(i), "file_path": f"{i}.jpg"} for i in range(3)]
    category = [[1, 0], [1, 0], [0, 1]]
    description = [[0.8, 0.6], [1, 0], [1, 0]]
    index = Index(rows, category, description, description, {"version": 1})
    index.save(tmp_path / "index")
    loaded = Index.load(tmp_path / "index")
    assert [r["id"] for r in loaded.search([1, 0])] == ["1", "0"]
    assert [r["id"] for r in loaded.search([1, 0], exclude_ids=["1"])] == ["0"]
    assert [r["id"] for r in loaded.search([1, 0], min_score=0.95)] == ["1"]
    assert [r["id"] for r in loaded.search([1, 0], mode="clip")] == ["1", "2", "0"]


def test_negative_scores_do_not_discard_best_candidate():
    index = Index([{"id": "a"}, {"id": "b"}], [[-1, 0], [-0.5, 0.5]],
                  [[-1, 0], [-0.5, 0.5]], [[-1, 0], [-0.5, 0.5]], {})
    assert index.search([1, 0])[0]["id"] == "b"


def test_embeddings_fail_on_invalid_vectors():
    with pytest.raises(ValueError, match="Zero"):
        fuse_query(text=[0, 0])
    with pytest.raises(ValueError, match="dimension"):
        fuse_query(text=[1, 0], image=[1, 0, 0])
    with pytest.raises(ValueError, match="finite"):
        fuse_query(text=[np.nan, 1])


def test_metrics_count_sets_not_duplicate_predictions():
    metrics = retrieval_metrics(["a", "a", "b"], ["a", "c"])
    assert metrics["precision"] == metrics["recall"] == metrics["f1"] == 0.5
    assert retrieval_metrics([], ["a"])["f1"] == 0
    with pytest.raises(ValueError):
        retrieval_metrics(["a"], [])


def test_json_parser_never_evaluates_python():
    assert extract_json('```json\n{"ok": true}\n```') == {"ok": True}
    with pytest.raises(ValueError):
        extract_json("__import__('os').system('echo unsafe')")


def test_no_model_demo_is_complete_and_explicitly_synthetic(tmp_path):
    result = demo(tmp_path)
    assert "NOT model" in result["notice"]
    assert result["image_count"] == 6
    assert result["synthetic_metrics"]["f1"] == 1
    assert read_jsonl(tmp_path / "sft/train.jsonl")
    assert read_jsonl(tmp_path / "sft/val.jsonl")


def test_cli_prepare_validate_and_split(tmp_path, monkeypatch, capsys):
    gallery(tmp_path)
    manifest = tmp_path / "catalog.jsonl"
    for command in [
        ["prepare", "--image-root", str(tmp_path), "--output", str(manifest)],
        ["validate", "--image-root", str(tmp_path), "--manifest", str(manifest)],
        ["make-sft", "--image-root", str(tmp_path), "--manifest", str(manifest), "--output", str(tmp_path / "sft")],
    ]:
        monkeypatch.setattr("sys.argv", ["mm-retrieval", *command])
        main()
        assert isinstance(json.loads(capsys.readouterr().out), dict)
