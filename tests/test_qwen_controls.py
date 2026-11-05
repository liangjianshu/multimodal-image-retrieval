"""Guard against label leakage and over-eager recovery in post-hoc controls."""
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    'qwen_controls', Path(__file__).parents[1] / 'examples/training/diagnose_qwen_controls.py')
controls = importlib.util.module_from_spec(spec)
spec.loader.exec_module(controls)


@pytest.mark.parametrize('response,expected', [
    ('[{"类别":"杯子"}]', ['cup']),
    ('```json\n{"categories":["杯子"]}\n```', ['cup']),
    ('这不是杯子，是一个未支持类别。', []),
    ('["杯子"', []),
    ('{"类别":"未知物体"}', []),
])
def test_normalization_does_not_infer_labels_from_truth_or_free_text(response, expected):
    for truth in ('cup', 'chair', 'oven'):
        row = {'valid_json': False, 'raw_response': response,
               'matched_classes': [], 'truth': truth}
        assert controls.normalized_categories(row) == expected


def test_accepted_prediction_is_not_reclassified():
    row = {'valid_json': True, 'raw_response': '["杯子"]',
           'matched_classes': ['cup'], 'truth': 'chair'}
    assert controls.normalized_categories(row) == ['cup']
