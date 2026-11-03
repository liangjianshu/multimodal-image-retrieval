"""Build a report only from complete, provenance-checked formal experiments.

No model loading or test-time model selection happens in this script.
"""
import argparse
import csv
import hashlib
import html
import json
import math
from pathlib import Path

VARIANTS = ('embedding-base', 'embedding-best', 'qwen-base', 'qwen-old', 'qwen-best')
SCOPES = {'full': 'test-results.json', 'normal': 'test-normal-gallery.json',
          'fine': 'test-fine-gallery.json', 'overlap-excluded': 'test-without-legacy-overlap.json'}


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def kind(trial):
    return 'qwen' if 'mode' in trial['args'] else 'embedding'


def validate_trials(trials, selection):
    require(len(trials) == 8, 'Expected exactly eight completed formal configurations')
    for trial in trials:
        require(trial['status'] == 'complete' and not trial['args'].get('max_steps'),
                'Smoke or incomplete trial cannot enter the final report')
        checkpoints = trial['checkpoints']
        require(len(checkpoints) == 3 and {c['epoch_requested'] for c in checkpoints} == {.5, 1, 2},
                'Each formal trial must contain all three requested epochs')
        require(all(math.isfinite(c['selection_score']) for c in checkpoints), 'Nonfinite validation score')
        require(trial['best'] == max(checkpoints, key=lambda c: c['selection_score']),
                'Trial best checkpoint is inconsistent with validation scores')
    for family, rates, discriminator, base_value, extra_value in (
        ('embedding', {1e-6, 2e-6, 5e-6}, 'temperature', .05, .1),
        ('qwen', {5e-6, 1e-5, 2e-5}, 'mode', 'llm-lora', 'mlp-llm'),
    ):
        records = [r for r in trials if kind(r) == family]
        baseline = [r for r in records if r['args'][discriminator] == base_value]
        extra = [r for r in records if r['args'][discriminator] == extra_value]
        require(len(records) == 4 and len(baseline) == 3 and len(extra) == 1,
                f'Incomplete {family} experiment matrix')
        require({r['args']['learning_rate'] for r in baseline} == rates, f'Unexpected {family} learning rates')
        winner = max(baseline, key=lambda r: r['best']['selection_score'])
        require(extra[0]['args']['learning_rate'] == winner['args']['learning_rate'],
                f'{family} extra experiment did not use validation-selected learning rate')
        best = max(records, key=lambda r: r['best']['selection_score'])['best']
        require(selection['selected'][family] == best, f'{family} selection changed after validation')


def paired_changes(before, after, comparison):
    left = {(r['mode'], r['id']): r for r in before['per_query']}
    right = {(r['mode'], r['id']): r for r in after['per_query']}
    require(left.keys() == right.keys(), 'Paired comparison has different query IDs')
    changes = []
    for key, old in left.items():
        new = right[key]
        require(old['label'] == new['label'], 'Paired query label changed')
        changes.append({'comparison': comparison, 'mode': key[0], 'query_id': key[1],
                        'label': old['label'], 'before_ap': old['average_precision'],
                        'after_ap': new['average_precision'],
                        'ap_delta': new['average_precision'] - old['average_precision'],
                        'p5_delta': new['precision_at_5'] - old['precision_at_5']})
    return changes


def collect(study):
    data_root = study / 'data/study-v1'
    data = read(data_root / 'data-report.json')
    selection = read(study / 'selection.json')
    require(selection.get('frozen_before_test') is True, 'Missing pre-test selection freeze')
    require(selection['data_report_sha256'] == digest(data_root / 'data-report.json'), 'Data report changed after selection')
    hashes = {s: digest(data_root / f'{s}-catalog.jsonl') for s in ('val', 'test')}
    for split, value in hashes.items():
        require(value == data['splits'][split]['manifest_sha256'], f'{split} catalog hash mismatch')
    require(data['splits']['test']['images'] == 2000, 'Original full test gallery is incomplete')
    trials = read(study / 'trials-summary.json')
    validate_trials(trials, selection)
    audit = read(study / 'legacy-overlap.json')
    require(not audit['read_failures'], 'Legacy overlap audit is incomplete')
    expected_sizes = {'full': 2000, 'normal': 1000, 'fine': 1000,
                      'overlap-excluded': 2000 - len(set(audit['potential_test_ids']))}
    hyperparameters, comparison, results, coarse = [], [], {}, []
    for trial in trials:
        args = trial['args']; family = kind(trial)
        train_hash = trial.get('train_sha256', trial.get('train_manifest_sha256'))
        require(train_hash == digest(data_root / f'train-{family}.jsonl'), f'{family} training manifest changed')
        require(trial.get('val_sha256', trial.get('val_manifest_sha256')) == hashes['val'], 'Trial validation catalog differs')
        for checkpoint in trial['checkpoints']:
            hyperparameters.append({'family': family, 'learning_rate': args['learning_rate'],
                'temperature': args.get('temperature', ''), 'mode': args.get('mode', 'full-text-and-vision'),
                'epoch_requested': checkpoint['epoch_requested'], 'actual_epoch': checkpoint['actual_epoch'],
                'validation_score': checkpoint['selection_score'],
                'validation_metric': 'primary_five_coarse_f1' if family == 'qwen' else 'primary_five_text_image_map_mean',
                'selected': checkpoint == selection['selected'][family], 'weight_sha256': checkpoint['weight_sha256'],
                'trainable_parameters': trial['trainable_parameters'], 'trial_seconds': trial['seconds'],
                'training_step_seconds': trial['training_step_seconds'], 'peak_allocated_gib': trial['peak_allocated_gib'],
                'checkpoint': checkpoint['path']})
    for variant in VARIANTS:
        root = study / 'evaluation' / variant
        complete = read(root / 'evaluation-complete.json')
        metadata = read(root / 'metadata.json')
        require(complete['catalog_sha256'] == hashes == metadata['catalog_sha256'], f'{variant}: different evaluation data')
        require(metadata['overlap_audit_sha256'] == digest(study / 'legacy-overlap.json'), f'{variant}: stale overlap audit')
        validation = read(root / 'val-results.json')
        results[variant] = read(root / 'test-results.json')
        full = results[variant]
        require(full['metadata']['catalog_sha256'] == hashes, f'{variant}: result metadata mismatch')
        if variant.endswith('-best'):
            family = variant.split('-')[0]
            actual = (full['metadata']['encoder_fingerprint']['weight_sha256'] if family == 'embedding'
                      else metadata['adapter_sha256'])
            require(actual == selection['selected'][family]['weight_sha256'], f'{variant}: evaluated wrong checkpoint')
        thresholds = validation['validation_thresholds']
        for scope, filename in SCOPES.items():
            result = read(root / filename)
            require(result['gallery_size'] == expected_sizes[scope], f'{variant}/{scope}: unexpected gallery size')
            for query in result['per_query']:
                threshold_key = query['id'] if query['mode'] == 'text' else '__image_global__'
                require(query['calibrated_f1']['threshold'] == thresholds[threshold_key],
                        f'{variant}/{scope}: test threshold was not frozen on validation')
            for mode in ('text', 'image'):
                metric = result['metrics'][mode]; primary = metric['primary_five_macro']
                comparison.append({'variant': variant, 'scope': scope, 'mode': mode,
                    'gallery_size': result['gallery_size'], 'queries': metric['queries'],
                    'primary_five_map': primary['average_precision'],
                    'primary_five_p1': primary['precision_at_1'], 'primary_five_p5': primary['precision_at_5'],
                    'primary_five_r20': primary['recall_at_20'],
                    'primary_five_calibrated_f1': metric['primary_five_calibrated_f1'],
                    'all_class_macro_map': metric['macro']['average_precision'],
                    'evaluation_seconds': complete['seconds']})
        if variant.startswith('qwen'):
            categories = read(root / 'test-categories.json')
            coarse.append({'variant': variant, **{k: categories[k] for k in
                          ('primary_five_macro_f1', 'macro_f1', 'valid_json_rate')}})
    changes = []
    for old, new in (('embedding-base', 'embedding-best'), ('qwen-base', 'qwen-best'), ('qwen-old', 'qwen-best')):
        changes.extend(paired_changes(results[old], results[new], f'{old} -> {new}'))
    return {'data': data, 'selection': selection, 'hyperparameters': hyperparameters,
            'comparison': comparison, 'coarse': coarse, 'paired_changes': changes,
            'training_wall_hours': sum(r['seconds'] for r in trials) / 3600,
            'trial_count': len(trials), 'checkpoint_count': len(hyperparameters)}


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def table(rows, columns):
    def cell(value):
        return f'{value:.6f}' if isinstance(value, float) else str(value)
    return '<table><thead><tr>' + ''.join(f'<th>{html.escape(c)}</th>' for c in columns) + '</tr></thead><tbody>' + ''.join(
        '<tr>' + ''.join(f'<td>{html.escape(cell(row[c]))}</td>' for c in columns) + '</tr>' for row in rows) + '</tbody></table>'


def run(study, output):
    report = collect(study)  # Refuse incomplete or mismatched results before writing a report.
    output.mkdir(parents=True, exist_ok=True)
    for key in ('hyperparameters', 'comparison', 'coarse', 'paired_changes'):
        write_csv(output / f'{key}.csv', report[key])
    (output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    body = '<h1>原始 2,000 张图库：微调对照实验</h1>'
    body += f'<p>正式实验 {report["trial_count"]} 组，检查点 {report["checkpoint_count"]} 个。训练及验证合计 {report["training_wall_hours"]:.2f} 小时；表中试验耗时会按检查点重复展示，不应重复相加。</p>'
    body += '<p>所有模型由验证集选定。下表分数为 0–1；F1 阈值只在验证集确定。仅衡量类别级相关性，不证明颜色、材质、意图识别或实例检索质量。</p>'
    body += '<h2>固定完整测试图库</h2>' + table([r for r in report['comparison'] if r['scope'] == 'full'],
        ['variant', 'mode', 'primary_five_map', 'primary_five_p1', 'primary_five_p5', 'primary_five_calibrated_f1', 'all_class_macro_map'])
    body += '<h2>粗筛与输出格式</h2>' + table(report['coarse'], list(report['coarse'][0]))
    body += '<h2>超参数及轮数</h2><p>两类模型的 validation_score 定义不同，不横向比较该列。</p>' + table(report['hyperparameters'],
        ['family', 'learning_rate', 'temperature', 'mode', 'epoch_requested', 'actual_epoch', 'validation_score', 'selected'])
    body += '<h2>分组与旧训练重叠敏感性分析</h2><p>fine 是小目标分组。排除已检出的旧训练潜在重叠不等于证明模型从未见过图片。</p>' + table(
        [r for r in report['comparison'] if r['scope'] != 'full'], ['variant', 'scope', 'mode', 'gallery_size', 'primary_five_map', 'primary_five_calibrated_f1'])
    body += '<h2>逐查询增益与退化</h2><p>完整逐查询差值见 paired_changes.csv；负值照常保留。只有单次种子实验，不据此声称统计显著。</p>'
    for comparison in sorted({r['comparison'] for r in report['paired_changes']}):
        rows = sorted([r for r in report['paired_changes'] if r['comparison'] == comparison], key=lambda r: r['ap_delta'])
        body += f'<h3>{html.escape(comparison)}</h3><p>最大退化与最大提升各 5 个查询：</p>' + table(rows[:5] + rows[-5:],
            ['mode', 'query_id', 'label', 'before_ap', 'after_ap', 'ap_delta'])
    body += '<p>基准采用固定 15 类映射粗筛及纯参考图查询；网页的开放类别语义筛选、意图识别和图文融合应另行验收。新指标不冒充旧脚本在测试集选阈值得到的历史数字。</p>'
    document = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>恢复图库实验报告</title><style>body{font:16px/1.65 system-ui;max-width:1200px;margin:40px auto;padding:0 24px;color:#172b35}table{border-collapse:collapse;width:100%;font-size:13px;margin:20px 0;display:block;overflow:auto}td,th{padding:9px;border:1px solid #dbe2e5;text-align:left}th{background:#edf4f5}h2{margin-top:40px}p{max-width:950px}</style>' + body + '</html>'
    (output / 'index.html').write_text(document)
    print(json.dumps({'output': str(output), 'trials': report['trial_count'], 'checkpoints': report['checkpoint_count']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--study-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.study_root, args.output)
