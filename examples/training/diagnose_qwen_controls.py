"""Post-hoc controls using frozen Qwen predictions, never a replacement test run.

Recalibrate each control on validation only, then evaluate the unchanged test set.
No generation, weight updates, checkpoint selection, or test-tuned thresholds.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from mmretrieval.models import extract_json
from mmretrieval.study_categories import matched_categories
from mmretrieval.study_metrics import PRIMARY, evaluate_vectors


def string_values(value):
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [s for child in value for s in string_values(child)]
    if isinstance(value, dict):
        return [s for child in value.values() for s in string_values(child)]
    return []


def normalized_categories(prediction):
    # Preserve accepted strict outputs; only recover strings from JSON structures.
    # No truth labels, fuzzy matching, model calls, or free-text interpretation.
    if prediction['valid_json']:
        return prediction['matched_classes']
    try:
        return matched_categories(string_values(extract_json(prediction['raw_response'])))
    except (ValueError, TypeError):
        return []


def compact(result):
    return {mode: {
        'f1': result['metrics'][mode]['primary_five_calibrated_f1'],
        'map': result['metrics'][mode]['primary_five_macro']['average_precision'],
    } for mode in ('text', 'image')}


def coarse(rows, cats):
    values = []
    for label in PRIMARY:
        truth = np.array([r['label'] == label for r in rows])
        pred = np.array([label in c for c in cats])
        tp = int((truth & pred).sum())
        fp = int((~truth & pred).sum())
        fn = int((truth & ~pred).sum())
        values.append(2 * tp / (2 * tp + fp + fn))
    return float(np.mean(values))


def run(study, output):
    output.mkdir(parents=True, exist_ok=True)
    hashes = {}

    def read(path):
        hashes[str(path.relative_to(study))] = hashlib.sha256(path.read_bytes()).hexdigest()
        return json.loads(path.read_text())

    rows = {}
    for split in ('val', 'test'):
        path = study / 'data/study-v1' / f'{split}-catalog.jsonl'
        hashes[str(path.relative_to(study))] = hashlib.sha256(path.read_bytes()).hexdigest()
        rows[split] = [json.loads(x) for x in path.read_text().splitlines()]
    data = {}
    configs = {}
    for name in ('qwen-base', 'qwen-old', 'qwen-best'):
        data[name] = {}
        metadata = read(study / 'evaluation' / name / 'metadata.json')
        configs[name] = {k: v for k, v in metadata['config'].items() if k != 'adapter'}
        for split in rows:
            folder = study / 'evaluation' / name
            cats = read(folder / f'{split}-categories.json')
            official = read(folder / f'{split}-results.json')
            path = folder / f'{split}-vectors.npz'
            hashes[str(path.relative_to(study))] = hashlib.sha256(path.read_bytes()).hexdigest()
            vectors = dict(np.load(path))
            assert [p['id'] for p in cats['predictions']] == [r['id'] for r in rows[split]]
            data[name][split] = {
                'cats': cats, 'official': official, 'vectors': vectors,
                'strict': [p['matched_classes'] for p in cats['predictions']],
                'normalized': [normalized_categories(p) for p in cats['predictions']],
            }
    for name in data:
        assert configs[name] == configs['qwen-base']
        for split in rows:
            base = data['qwen-base'][split]
            item = data[name][split]
            assert item['cats']['prompt'] == base['cats']['prompt']
            assert item['cats']['aliases'] == base['cats']['aliases']
            for key in ('text', 'reference_images'):
                assert np.array_equal(item['vectors'][key], base['vectors'][key])

    results = []
    # Frozen base predictions provide a deployable common filter; oracle is
    # explicitly truth-assisted and serves only as a non-deployable control.
    policies = ('strict', 'normalized', 'no_filter', 'fixed_base_normalized', 'oracle')
    for name in data:
        for policy in policies:
            thresholds = None
            for split in rows:
                item = data[name][split]
                cats = (None if policy == 'no_filter' else
                        [[r['label']] for r in rows[split]] if policy == 'oracle' else
                        data['qwen-base'][split]['normalized'] if policy == 'fixed_base_normalized' else
                        item[policy])
                qs = [q for q in item['official']['per_query'] if q['mode'] == 'text']
                v = item['vectors']
                evaluated = evaluate_vectors(rows[split], v['images'], qs, v['text'], thresholds,
                                             candidate_categories=cats,
                                             image_query_vectors=v['reference_images'])
                if split == 'val':
                    thresholds = evaluated['validation_thresholds']
                    continue
                scores = compact(evaluated)
                if policy == 'strict':
                    expected = compact(item['official'])
                    for mode in scores:
                        for key in scores[mode]:
                            assert abs(scores[mode][key] - expected[mode][key]) < 1e-9
                results.append({'variant': name, 'control': policy, 'metrics': scores,
                                'validation_thresholds': thresholds,
                                'primary_coarse_f1': coarse(rows[split], cats) if cats is not None else None})
            print(f'{name}: {policy} complete', flush=True)
    protocol = {
        'kind': 'post-hoc diagnostic; not a new frozen test or checkpoint-selection result',
        'shared_prompt': data['qwen-base']['test']['cats']['prompt'],
        'shared_aliases_verified': True, 'shared_query_vectors_verified': True,
        'non_adapter_model_configs_identical': True,
        'thresholds': 'recalibrated separately for every control on validation only',
        'normalization': 'JSON string values only; no ground truth or free-text label inference',
        'oracle': 'uses ground-truth labels on query and gallery; NOT a deployable retrieval result',
        'frozen_base_filter': 'all descriptor models use the same normalized base category predictions',
        'scope': 'primary five categories, full 2000-image test gallery, category relevance',
        'generation': 'reuse archived raw responses and vectors; no new model inference',
        'limitations': ['Post-hoc analysis after observing test results.',
                       'Does not measure closed-vocabulary prompting or constrained decoding.',
                       'Cannot isolate pure visual representation learning from instruction adaptation.',
                       'No multi-seed significance claim.'],
    }
    summary = {'protocol': protocol, 'source_sha256': hashes, 'results': results}
    (output / 'qwen-controls.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    with (output / 'qwen-controls.csv').open('w') as file:
        writer = csv.DictWriter(file, fieldnames=['variant', 'control', 'mode', 'f1', 'map', 'primary_coarse_f1'])
        writer.writeheader()
        for row in results:
            for mode, score in row['metrics'].items():
                writer.writerow({'variant': row['variant'], 'control': row['control'], 'mode': mode,
                                 **score, 'primary_coarse_f1': row['primary_coarse_f1']})
    print(json.dumps({'protocol': protocol, 'results': [
        {k: v for k, v in row.items() if k != 'validation_thresholds'} for row in results
    ]}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.study_root, args.output)
