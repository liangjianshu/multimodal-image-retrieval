import numpy as np
import pytest

from mmretrieval.study_metrics import evaluate_vectors
from examples.training.export_study_cases import ranked_queries, category_errors


def fixture(categories=None):
    rows = [{'id': 'cup1', 'label': 'cup'}, {'id': 'cup2', 'label': 'cup'},
            {'id': 'chair1', 'label': 'chair'}, {'id': 'chair2', 'label': 'chair'}]
    vectors = np.array([[1., 0.], [.99, .01], [0., 1.], [.01, .99]])
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    text = np.eye(2)
    queries = [{'id': 'cup-0', 'label': 'cup', 'text': '杯子'}, {'id': 'chair-0', 'label': 'chair', 'text': '椅子'}]
    thresholds = {'cup-0': .8, 'chair-0': .8, '__image_global__': .8}
    result = evaluate_vectors(rows, vectors, queries, text, thresholds, candidate_categories=categories)
    return rows, {'images': vectors, 'text': text}, result


def test_reconstructed_rankings_match_metrics_and_exclude_reference():
    rows, arrays, result = fixture()
    cases = list(ranked_queries(rows, arrays, result, top_k=3))
    assert len(cases) == 6
    assert [r['id'] for r in cases[0]['top_k'][:2]] == ['cup1', 'cup2']
    assert cases[2]['query_id'] == 'cup1'
    assert cases[2]['top_k'][0]['id'] == 'cup2'
    assert all(r['id'] != 'cup1' for r in cases[2]['top_k'])
    assert cases[2]['relevant_total'] == 1


def test_validation_replay_is_explicit_about_calibration():
    rows, arrays, _ = fixture()
    queries = [{'id': 'cup-0', 'label': 'cup'}, {'id': 'chair-0', 'label': 'chair'}]
    result = evaluate_vectors(rows, arrays['images'], queries, arrays['text'])
    cases = list(ranked_queries(rows, arrays, result))
    assert all(c['threshold_fitted_on_this_query_set'] for c in cases)


def test_coarse_empty_and_missed_positive_are_preserved():
    categories = [['chair'], [], ['chair'], ['chair']]
    rows, arrays, result = fixture(categories)
    cases = list(ranked_queries(rows, arrays, result, categories))
    assert cases[0]['top_k'] == [] and cases[0]['relevant_total'] == 2
    assert cases[3]['query_id'] == 'cup2' and cases[3]['candidate_count'] == 0


def test_stale_vectors_or_query_order_cannot_export_plausible_wrong_cases():
    rows, arrays, result = fixture()
    arrays['images'] = arrays['images'][::-1]
    with pytest.raises(ValueError, match='differs from recorded'):
        list(ranked_queries(rows, arrays, result))
    with pytest.raises(ValueError, match='ordering'):
        list(ranked_queries(rows[::-1], arrays, result))


def test_category_error_export_distinguishes_missing_extra_and_invalid():
    records = [dict(id='a', truth='cup', matched_classes=['cup'], valid_json=True, raw_response='ok'),
               dict(id='b', truth='cup', matched_classes=['cup', 'chair'], valid_json=True, raw_response='multi'),
               dict(id='c', truth='cup', matched_classes=[], valid_json=False, raw_response='invalid')]
    errors = list(category_errors('qwen-best', records))
    assert len(errors) == 2
    assert errors[0]['extra_categories'] == 'chair' and not errors[0]['missing_truth']
    assert errors[1]['missing_truth'] and not errors[1]['valid_json']
