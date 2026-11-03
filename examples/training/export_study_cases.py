"""Recover auditable top-K lists from saved vectors, without running models again."""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from mmretrieval.data import read_jsonl
from mmretrieval.study_metrics import ranking_metrics
from examples.training.report_restored_study import collect, VARIANTS


def ranked_queries(rows, arrays, result, categories=None, top_k=20):
    """Reconstruct the evaluator's exact score/mask/tie/self-exclusion protocol."""
    if top_k < 1:
        raise ValueError('top_k must be positive')
    vectors = arrays['images']; text = arrays['text']
    references = arrays['reference_images'] if 'reference_images' in arrays else vectors
    labels = np.array([r['label'] for r in rows])
    text_records = [r for r in result['per_query'] if r['mode'] == 'text']
    image_records = [r for r in result['per_query'] if r['mode'] == 'image']
    if vectors.shape[0] != len(rows) or references.shape != vectors.shape or len(text) != len(text_records):
        raise ValueError('Cached vectors do not match catalog or queries')
    if [r['id'] for r in image_records] != [r['id'] for r in rows]:
        raise ValueError('Image query ordering differs from saved gallery ordering')
    if categories is not None and len(categories) != len(rows):
        raise ValueError('Category predictions differ from gallery size')
    image_scores = vectors @ references.T
    np.fill_diagonal(image_scores, -np.inf)
    for mode, records in [('text', text_records), ('image', image_records)]:
        for index, query in enumerate(records):
            scores = vectors @ text[index] if mode == 'text' else image_scores[:, index].copy()
            if categories is not None:
                allowed = ([query['label'] in c for c in categories] if mode == 'text' else
                           [bool(set(categories[index]).intersection(c)) for c in categories])
                scores = np.where(allowed, scores, -np.inf)
            eligible = np.arange(len(rows)) if mode == 'text' else np.delete(np.arange(len(rows)), index)
            truth = labels == query['label']
            actual = ranking_metrics(scores, truth, eligible)
            for key in ['average_precision', 'precision_at_1', 'precision_at_5', 'precision_at_10', 'precision_at_20']:
                if not np.isclose(actual[key], query[key], rtol=0, atol=1e-9):
                    raise ValueError(f'Reconstructed {mode}/{query["id"]} differs from recorded {key}')
            candidates = eligible[np.isfinite(scores[eligible])]
            order = candidates[np.argsort(-scores[candidates], kind='stable')][:top_k]
            calibration = query.get('calibrated_f1', query.get('validation_best_f1'))
            if calibration is None:
                raise ValueError('Recorded threshold is missing')
            threshold = calibration['threshold']
            yield {'query_id': query['id'], 'mode': mode, 'label': query['label'],
                'text': query.get('text'), 'average_precision': actual['average_precision'],
                'relevant_total': actual['relevant'], 'candidate_count': len(candidates),
                'validation_threshold': threshold,
                'threshold_fitted_on_this_query_set': 'validation_best_f1' in query,
                'above_threshold_count': int(np.sum(scores[candidates] >= threshold)),
                'top_k': [{'id': rows[i]['id'], 'label': rows[i]['label'], 'score': float(scores[i]),
                           'relevant_by_class': bool(truth[i]), 'above_threshold': bool(scores[i] >= threshold)} for i in order]}


def category_errors(variant, predictions):
    for row in predictions:
        matched = set(row['matched_classes']); extras = matched - {row['truth']}
        missing = row['truth'] not in matched
        if not row['valid_json'] or missing or extras:
            yield {'variant': variant, 'image_id': row['id'], 'truth': row['truth'],
                'valid_json': row['valid_json'], 'missing_truth': missing,
                'extra_categories': '|'.join(sorted(extras)),
                'predicted_categories': '|'.join(sorted(matched)), 'raw_response': row['raw_response']}


def run(study, output, top_k=20):
    collect(study)  # Same provenance gate as the final report; no partial experiments.
    output.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(study / 'data/study-v1/test-catalog.jsonl')
    manifest = {'top_k': top_k, 'gallery_size': len(rows), 'variants': {},
                'relevance': 'Single ImageNet synset label; not instance or attribute relevance',
                'selection_sha256': hashlib.sha256((study / 'selection.json').read_bytes()).hexdigest()}
    coarse_rows = []; error_count = 0
    error_fields = ['variant', 'image_id', 'truth', 'valid_json', 'missing_truth', 'extra_categories', 'predicted_categories', 'raw_response']
    with (output / 'category-errors.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=error_fields); writer.writeheader()
        for variant in VARIANTS:
            root = study / 'evaluation' / variant
            result = json.loads((root / 'test-results.json').read_text())
            categories = None
            if variant.startswith('qwen'):
                coarse = json.loads((root / 'test-categories.json').read_text())
                lookup = {r['id']: r for r in coarse['predictions']}
                if set(lookup) != {r['id'] for r in rows}:
                    raise ValueError('Category prediction IDs differ from fixed test gallery')
                categories = [lookup[r['id']]['matched_classes'] for r in rows]
                for row in category_errors(variant, coarse['predictions']):
                    writer.writerow(row); error_count += 1
                for label, values in coarse['per_class'].items():
                    coarse_rows.append({'variant': variant, 'label': label, **values})
            target = output / f'{variant}-top{top_k}.jsonl.gz'
            count = 0
            with np.load(root / 'test-vectors.npz', allow_pickle=False) as arrays, gzip.open(target, 'wt', encoding='utf-8') as stream:
                for query in ranked_queries(rows, arrays, result, categories, top_k):
                    stream.write(json.dumps(query, ensure_ascii=False) + '\n'); count += 1
            manifest['variants'][variant] = {'queries': count, 'filename': target.name,
                'sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
                'source_vectors_sha256': hashlib.sha256((root / 'test-vectors.npz').read_bytes()).hexdigest(),
                'source_results_sha256': hashlib.sha256((root / 'test-results.json').read_bytes()).hexdigest()}
    with (output / 'category-per-class.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(coarse_rows[0])); writer.writeheader(); writer.writerows(coarse_rows)
    manifest['category_error_rows'] = error_count
    (output / 'cases-manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'output': str(output), 'variants': len(manifest['variants']), 'category_error_rows': error_count}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--top-k', type=int, default=20)
    args = parser.parse_args()
    run(args.study_root, args.output, args.top_k)
