"""Full-gallery held-out category retrieval, independent of Qwen routing."""
import argparse
import json
from pathlib import Path

import numpy as np

from mmretrieval.cli import load_config
from mmretrieval.data import read_jsonl
from mmretrieval.gemma import GemmaEncoder


def evaluate(config, catalog, root, output):
    rows=read_jsonl(catalog);encoder=GemmaEncoder(load_config(config))
    vectors=encoder.images([Path(root)/r['image'] for r in rows])
    labels=np.array([r['label'] for r in rows]);metrics={}
    details=[]
    def score(query,vec,label,exclude=None):
        similarities=vectors@vec
        ids=np.arange(len(rows))
        if exclude is not None: ids=ids[ids!=exclude]
        order=ids[np.argsort(-similarities[ids],kind='stable')]
        relevant=(labels[order]==label).astype(float)
        cumulative=np.cumsum(relevant)/np.arange(1,len(order)+1)
        result={'query':query,'label':label,'hit_at_1':float(relevant[0]),
                'precision_at_5':float(relevant[:5].mean()),
                'average_precision':float((cumulative*relevant).sum()/relevant.sum())}
        details.append(result);return result
    for mode in ['text','image']:
        values=[]
        if mode=='text':
            # Held-out wording; class labels are shared as in standard supervised learning.
            names={r['label']:r['label_cn'] for r in rows}
            for label,cn in sorted(names.items()):
                for query in [cn,f'给我看看{cn}',f'a photo of a {label}']:
                    values.append(score(query,encoder.query(query),label))
        else:
            for i,row in enumerate(rows):
                values.append(score(row['id'],vectors[i],row['label'],i))
        metrics[mode]={'queries':len(values)}
        for key in ['hit_at_1','precision_at_5','average_precision']:
            metrics[mode][key]=float(np.mean([v[key] for v in values]))
            per_class=[np.mean([v[key] for v in values if v['label']==c]) for c in sorted(set(labels))]
            metrics[mode]['macro_'+key]=float(np.mean(per_class))
    report={'encoder':encoder.fingerprint,'gallery_size':len(rows),
            'scope':'category retrieval; same-category images are relevant; image query itself excluded',
            'metrics':metrics,'per_query':details}
    Path(output).write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(metrics,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',required=True);parser.add_argument('--catalog',required=True)
    parser.add_argument('--image-root',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();evaluate(args.config,args.catalog,args.image_root,args.output)
