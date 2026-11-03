"""The selected category model must receive its own training prompt."""
import pytest
from examples.showcase.run_restored_winners import describe_selected, CATEGORY_PROMPT, DESCRIPTION_PROMPT


def test_selected_model_uses_training_category_prompt_and_original_description_prompt():
    calls = []
    class Model:
        def generate(self, prompt, path):
            calls.append((prompt, path))
            return '["杯子"]' if len(calls) == 1 else '一个带花纹的杯子。'
    result = describe_selected(Model(), 'cup.jpg')
    assert calls == [(CATEGORY_PROMPT, 'cup.jpg'), (DESCRIPTION_PROMPT, 'cup.jpg')]
    assert result['categories'] == ['杯子']
    assert result['description'] == '一个带花纹的杯子。'


@pytest.mark.parametrize('response', ['[]', '{}', '[null]', '[""]', 'not json'])
def test_invalid_generated_categories_are_not_replaced_with_gallery_labels(response):
    class Model:
        def generate(self, prompt, path):
            return response
    with pytest.raises(ValueError):
        describe_selected(Model(), 'cup.jpg')
