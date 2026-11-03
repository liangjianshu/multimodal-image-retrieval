"""Protect the final report from incomplete sweeps and test-selected winners."""
import copy
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('study_report', Path(__file__).parents[1] / 'examples/training/report_restored_study.py')
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


def completed_matrix():
    trials = []
    for family, rates in [('embedding', [1e-6, 2e-6, 5e-6, 5e-6]), ('qwen', [5e-6, 1e-5, 2e-5, 2e-5])]:
        for index, rate in enumerate(rates):
            args = {'learning_rate': rate}
            if family == 'embedding':
                args['temperature'] = .1 if index == 3 else .05
            else:
                args['mode'] = 'mlp-llm' if index == 3 else 'llm-lora'
            checkpoints = [{'epoch_requested': epoch, 'selection_score': .1 * index + .01 * epoch,
                            'path': f'{family}/{index}/{epoch}'} for epoch in [.5, 1, 2]]
            trials.append({'status': 'complete', 'args': args, 'checkpoints': checkpoints, 'best': checkpoints[-1]})
    selection = {'selected': {family: max([r for r in trials if report.kind(r) == family],
                    key=lambda r: r['best']['selection_score'])['best'] for family in ['embedding', 'qwen']}}
    return trials, selection


def test_report_accepts_complete_validation_selected_matrix():
    trials, selection = completed_matrix()
    report.validate_trials(trials, selection)


@pytest.mark.parametrize('fault', ['missing', 'smoke', 'missing_epoch', 'wrong_selection', 'wrong_extra_lr', 'nan'])
def test_report_rejects_incomplete_or_invalid_selection(fault):
    trials, selection = copy.deepcopy(completed_matrix())
    if fault == 'missing':
        trials.pop()
    elif fault == 'smoke':
        trials[0]['args']['max_steps'] = 1
    elif fault == 'missing_epoch':
        trials[0]['checkpoints'].pop(0)
    elif fault == 'wrong_selection':
        selection['selected']['embedding'] = trials[0]['best']
    elif fault == 'wrong_extra_lr':
        trials[3]['args']['learning_rate'] = 1e-6
    else:
        trials[0]['checkpoints'][0]['selection_score'] = float('nan')
    with pytest.raises(ValueError):
        report.validate_trials(trials, selection)


def test_paired_changes_align_ids_and_preserve_regressions():
    one = {'mode': 'image', 'id': 'a', 'label': 'cup', 'average_precision': .8, 'precision_at_5': .6}
    two = {**one, 'id': 'b', 'average_precision': .2}
    after = [{**two, 'average_precision': .5}, {**one, 'average_precision': .3}]
    changes = report.paired_changes({'per_query': [one, two]}, {'per_query': after}, 'base -> best')
    assert [r['ap_delta'] for r in changes] == pytest.approx([-.5, .3])
    after[0]['id'] = 'different-gallery-query'
    with pytest.raises(ValueError, match='query IDs'):
        report.paired_changes({'per_query': [one, two]}, {'per_query': after}, 'base -> best')


def test_complete_report_export_and_stale_threshold_rejection(tmp_path):
    """Synthetic records exercise the export contract without pretending to be results."""
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
    data = tmp_path / 'data/study-v1'
    data.mkdir(parents=True)
    for filename in ['val-catalog.jsonl', 'test-catalog.jsonl', 'train-embedding.jsonl', 'train-qwen.jsonl']:
        (data / filename).write_text('{}\n')
    hashes = {s: report.digest(data / f'{s}-catalog.jsonl') for s in ['val', 'test']}
    write(data / 'data-report.json', {'splits': {s: {'manifest_sha256': hashes[s], 'images': 2000 if s == 'test' else 1000} for s in hashes}})
    trials, selection = completed_matrix()
    for trial in trials:
        family = report.kind(trial)
        trial.update(train_sha256=report.digest(data / f'train-{family}.jsonl'), val_sha256=hashes['val'],
                     trainable_parameters=100, seconds=60, training_step_seconds=50, peak_allocated_gib=1)
        for checkpoint in trial['checkpoints']:
            checkpoint.update(weight_sha256=f'synthetic-{checkpoint["path"]}', actual_epoch=checkpoint['epoch_requested'])
    selection.update(frozen_before_test=True, data_report_sha256=report.digest(data / 'data-report.json'))
    write(tmp_path / 'selection.json', selection); write(tmp_path / 'trials-summary.json', trials)
    write(tmp_path / 'legacy-overlap.json', {'read_failures': [], 'potential_test_ids': ['overlap']})
    metric = {'average_precision': .5, 'precision_at_1': .5, 'precision_at_5': .5, 'recall_at_20': .5}
    for variant in report.VARIANTS:
        root = tmp_path / 'evaluation' / variant
        metadata = {'catalog_sha256': hashes, 'overlap_audit_sha256': report.digest(tmp_path / 'legacy-overlap.json')}
        if variant == 'qwen-best':
            metadata['adapter_sha256'] = selection['selected']['qwen']['weight_sha256']
        if variant == 'embedding-best':
            metadata['encoder_fingerprint'] = {'weight_sha256': selection['selected']['embedding']['weight_sha256']}
        write(root / 'metadata.json', metadata)
        write(root / 'evaluation-complete.json', {'catalog_sha256': hashes, 'seconds': 10})
        write(root / 'val-results.json', {'validation_thresholds': {'cup-0': .2, '__image_global__': .3}})
        queries = [{'id': 'cup-0' if mode == 'text' else 'image1', 'mode': mode, 'label': 'cup',
                    **metric, 'calibrated_f1': {'threshold': .2 if mode == 'text' else .3}} for mode in ['text', 'image']]
        for scope, filename in report.SCOPES.items():
            write(root / filename, {'metadata': metadata, 'gallery_size': {'full': 2000, 'normal': 1000, 'fine': 1000, 'overlap-excluded': 1999}[scope],
                'per_query': queries, 'metrics': {m: {'queries': 1, 'primary_five_macro': metric, 'macro': metric,
                'primary_five_calibrated_f1': .4} for m in ['text', 'image']}})
        if variant.startswith('qwen'):
            write(root / 'test-categories.json', dict(primary_five_macro_f1=.4, macro_f1=.4, valid_json_rate=1.))
    output = tmp_path / 'synthetic-report'
    report.run(tmp_path, output)
    assert (output / 'index.html').exists()
    assert len((output / 'comparison.csv').read_text().splitlines()) == 41
    assert len((output / 'hyperparameters.csv').read_text().splitlines()) == 25
    path = tmp_path / 'evaluation/qwen-best/test-fine-gallery.json'
    broken = report.read(path); broken['per_query'][0]['calibrated_f1']['threshold'] = .9; write(path, broken)
    with pytest.raises(ValueError, match='threshold was not frozen'):
        report.collect(tmp_path)
