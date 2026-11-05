"""Build the project overview from published measurements and unchanged demo ranks."""
import csv
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def build():
    report = json.loads((ROOT / 'docs/route-b/results.json').read_text())
    controls = json.loads((ROOT / 'docs/study/controls/qwen-controls.json').read_text())
    rows = list(csv.DictReader((ROOT / 'docs/study/comparison.csv').open()))
    fixed = {(r['variant'], r['mode']): float(r['primary_five_calibrated_f1'])
             for r in rows if r['scope'] == 'full'}
    norms = {r['variant']: {m: r['metrics'][m]['f1'] for m in ('text', 'image')}
             for r in controls['results'] if r['control'] == 'normalized'}
    queries = [next(q for q in report['queries'] if q['id'] == name)
               for name in ('explicit-two-cups', 'similar-cup', 'no-retrieval-qa')]
    # Only display fields, no local paths, runtime environment or private inputs.
    queries = [{k: q.get(k) for k in ('id', 'text', 'image', 'results', 'retrieval_executed')} for q in queries]
    for q in queries:
        q['results'] = [{k: r[k] for k in ('id', 'file_path', 'retrieval_categories', 'score')} for r in q['results']]
    def pair(family):
        return {stage: {m: fixed[(family + '-' + version, m)] for m in ('text', 'image')}
                for stage, version in [('base', 'base'), ('tuned', 'best')]}
    data = {'queries': queries, 'qwen_strict': pair('qwen'), 'embedding': pair('embedding'),
            'qwen_normalized': {'base': norms['qwen-base'], 'tuned': norms['qwen-best']}}
    initial = ''.join('<figure class="photo"><img src="../route-b/' + html.escape(r['file_path']) +
                      '" alt="杯子检索结果"><figcaption><span>' + f'{i+1:02d} · {r["id"]}' +
                      '</span><b>' + f'{r["score"]:.3f}' + '</b></figcaption></figure>'
                      for i, r in enumerate(queries[0]['results']))
    metric_rows = ''
    for label, key in [('Qwen 类别 + 描述', 'qwen_normalized'), ('Embedding 对比学习', 'embedding')]:
        m = data[key]
        metric_rows += '<tr><td>' + label + '</td>' + ''.join(
            f'<td>{m["base"][mode]*100:.2f}% → <strong>{m["tuned"][mode]*100:.2f}%</strong></td>'
            for mode in ('text', 'image')) + '</tr>'
    page = (ROOT / 'examples/showcase/overview.html').read_text()
    page = page.replace('__INITIAL_CARDS__', initial).replace('__METRIC_ROWS__', metric_rows)
    page = page.replace('__DATA__', json.dumps(data, ensure_ascii=False).replace('</', '<\\/'))
    (ROOT / 'docs/overview').mkdir(exist_ok=True)
    (ROOT / 'docs/overview/index.html').write_text(page)
    print('Built project overview from recorded results; no ranking or metrics changed.')


if __name__ == '__main__':
    build()
