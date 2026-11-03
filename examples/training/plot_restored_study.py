"""Export standalone figures from the verified final report (requires matplotlib)."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path


def run(report_path, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    report = json.loads(report_path.read_text())
    if report['trial_count'] != 8 or report['checkpoint_count'] != 24:
        raise ValueError('Figures require the complete formal report')
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                         'savefig.dpi': 180, 'figure.facecolor': 'white'})
    groups = defaultdict(list)
    for row in report['hyperparameters']:
        groups[(row['family'], row['learning_rate'], row['temperature'], row['mode'])].append(row)
    if len(groups) != 8 or any(len(rows) != 3 for rows in groups.values()):
        raise ValueError('Unexpected hyperparameter matrix')
    paths = []

    def save(fig, name):
        for suffix in ('png', 'pdf'):
            path = output / f'{name}.{suffix}'
            fig.savefig(path, bbox_inches='tight'); paths.append(path)
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), layout='constrained')
    for ax, family in zip(axes, ['embedding', 'qwen']):
        for (kind, lr, temperature, mode), rows in groups.items():
            if kind != family:
                continue
            rows = sorted(rows, key=lambda r: r['actual_epoch'])
            label = f'lr={lr:g}, ' + (f'temp={temperature:g}' if family == 'embedding' else mode)
            ax.plot([r['actual_epoch'] for r in rows], [r['validation_score'] for r in rows], marker='o', label=label)
            for row in rows:
                if row['selected']:
                    ax.scatter([row['actual_epoch']], [row['validation_score']], marker='*', s=180,
                               color='#111111', zorder=5, label='Validation-selected checkpoint')
        ax.set(title='Embedding: validation retrieval mAP' if family == 'embedding' else 'Qwen: validation category F1',
               xlabel='Actual epoch', ylabel='Primary-five score', ylim=(0, 1), xticks=[.5, 1, 2])
        ax.grid(alpha=.2); ax.legend(fontsize=7, loc='upper center', bbox_to_anchor=(.5, -.18))
    fig.suptitle('Model and epoch selection uses validation only; the two panels measure different tasks')
    save(fig, 'validation-selection')

    variants = ['embedding-base', 'embedding-best', 'qwen-base', 'qwen-old', 'qwen-best']
    names = ['Embedding\nbase', 'Embedding\nselected', 'Qwen\nbase', 'Qwen\nold LoRA', 'Qwen\nselected']
    colors = ['#9ba8b1', '#008b7d', '#9ba8b1', '#5277ad', '#008b7d']
    rows = {(r['variant'], r['mode']): r for r in report['comparison'] if r['scope'] == 'full'}
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
    for row_axes, mode in zip(axes, ['text', 'image']):
        for ax, metric, title in zip(row_axes, ['primary_five_map', 'primary_five_calibrated_f1'], ['mAP', 'Validation-calibrated F1']):
            values = [rows[(v, mode)][metric] for v in variants]
            bars = ax.bar(names, values, color=colors)
            ax.bar_label(bars, labels=[f'{v:.3f}' for v in values], padding=3, fontsize=9)
            ax.set(title=f'{mode.title()} to image: {title}', ylabel='Primary-five macro score', ylim=(0, 1.08))
            ax.grid(axis='y', alpha=.2); ax.set_axisbelow(True)
    fig.suptitle('Fixed original 2,000-image gallery | category relevance | seed 42 | no test-time threshold tuning')
    save(fig, 'fixed-test-comparison')

    fig, ax = plt.subplots(figsize=(11, 5), layout='constrained')
    names, train, other = [], [], []
    for (family, lr, temperature, mode), rows in groups.items():
        row = rows[0]
        names.append(f'{family} lr={lr:g} ' + (f't={temperature:g}' if family == 'embedding' else mode))
        train.append(row['training_step_seconds'] / 3600)
        other.append(max(0, row['trial_seconds'] - row['training_step_seconds']) / 3600)
    y = np.arange(len(names))
    ax.barh(y, train, label='Recorded training-step time', color='#008b7d')
    ax.barh(y, other, left=train, label='Validation, checkpoint I/O and other trial time', color='#a6bbc6')
    ax.set(yticks=y, yticklabels=names, xlabel='Wall-clock hours', title='One accelerator; each configuration counted once')
    ax.invert_yaxis(); ax.legend(fontsize=8, loc='upper center', bbox_to_anchor=(.5, -.15), ncol=2)
    ax.grid(axis='x', alpha=.2); ax.set_axisbelow(True)
    save(fig, 'training-cost')
    manifest = {'source_report_sha256': hashlib.sha256(report_path.read_bytes()).hexdigest(),
                'matplotlib_version': matplotlib.__version__,
                'figures': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
                'note': 'Single-seed descriptive comparison; no significance or confidence-interval claim.'}
    (output / 'plots-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'output': str(output), 'figures': len(paths)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.report, args.output)
