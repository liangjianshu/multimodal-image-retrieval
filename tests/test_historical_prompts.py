"""Original checkpoint prompts and response schema must survive refactoring."""
import ast
import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from mmretrieval.cli import load_config
from mmretrieval.historical_prompts import CATEGORY_PROMPT, DESCRIPTION_PROMPT, INTENT_PROMPT
from mmretrieval.models import Qwen

ROOT = Path(__file__).resolve().parents[1]


def literal(path, name):
    for node in ast.walk(ast.parse((ROOT / path).read_text())):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError(f"Missing original prompt: {name}")


def test_prompts_preserve_original_checkpoint_inputs():
    assert CATEGORY_PROMPT == literal("legacy/project_one/search_qwen.py", "category_prompt")
    assert DESCRIPTION_PROMPT == literal("legacy/project_one/search_qwen.py", "prompt2")
    assert INTENT_PROMPT == literal("legacy/project_three/multimodel_search.py", "INTENT_PROMPT")


def stub(*responses):
    model = object.__new__(Qwen)
    model.config = {"prompt_profile": "historical"}
    model.generate = Mock(side_effect=responses)
    return model


def test_historical_description_keeps_separate_category_and_attribute_calls():
    model = stub('```json\n["椅子"]\n```', ' 一把红色塑料椅子。 ')
    assert model.describe("chair.jpg") == {"categories": ["椅子"], "description": "一把红色塑料椅子。"}
    assert [call.args for call in model.generate.call_args_list] == [
        (CATEGORY_PROMPT, "chair.jpg"), (DESCRIPTION_PROMPT, "chair.jpg")]


@pytest.mark.parametrize("needed", [1, [1], "1", ["1"]])
def test_original_intent_schema_preserves_image_target_and_quantity(needed):
    raw = {"是否需要图像检索": needed, "检索描述": ["黑色椅子"], "检索方式": ["TopK", 3]}
    model = stub(json.dumps(raw))
    intent = model.intent("找三张相似图片", "chair.jpg")
    assert intent == {"needs_retrieval": True, "description": "黑色椅子", "top_k": 3, "raw_response": raw}
    assert model.generate.call_args.args[1] == "chair.jpg"


def test_nonretrieval_and_invalid_quantity_are_not_silent_searches():
    raw = {"是否需要图像检索": 0, "检索描述": [""], "检索方式": [""]}
    assert not stub(json.dumps(raw)).intent("你好")["needs_retrieval"]
    raw.update({"是否需要图像检索": 1, "检索方式": ["TopK", -1]})
    with pytest.raises(ValueError, match="top_k"):
        stub(json.dumps(raw)).intent("找图")


def test_unknown_profile_cannot_silently_use_untrained_prompts(tmp_path):
    path = tmp_path / "config.json"
    path.write_text('{"prompt_profile":"historial"}')
    with pytest.raises(ValueError, match="prompt_profile"):
        load_config(path)
