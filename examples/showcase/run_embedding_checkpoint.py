"""Replay previously generated Qwen decisions with a different image encoder."""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from mmretrieval.cli import load_config
from mmretrieval.data import read_jsonl
from mmretrieval.gemma import GemmaEncoder
from mmretrieval.image_index import ImageIndex
from mmretrieval.routes import RoutedRetriever


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True);p.add_argument('--baseline-results',required=True)
    p.add_argument('--descriptions',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();config=load_config(args.config)
    base=json.loads(Path(args.baseline_results).read_text())
    if base['backend']!='qwen-embedding':raise ValueError('Expected route B baseline')
    raw=Path(args.descriptions).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=base['description_cache_sha256']:raise ValueError('Description cache changed')
    gallery=ROOT/'examples/showcase';rows=read_jsonl(args.descriptions)
    sources=json.loads((gallery/'sources.json').read_text())
    for source in sources:
        if hashlib.sha256((gallery/source['file_path']).read_bytes()).hexdigest()!=source['sha256']:
            raise ValueError('Gallery image changed')
    encoder=GemmaEncoder(config)
    index=ImageIndex(rows,encoder.images([gallery/r['file_path'] for r in rows]),{**encoder.fingerprint,'version':1})
    pipeline=RoutedRetriever('qwen-embedding',index,None,lambda:encoder,config)
    report=copy.deepcopy(base);report['queries']=[]
    for q in base['queries']:
        r=pipeline.search(q.get('text'),gallery/q['image'] if q.get('image') else None,
                         top_k=3,exclude_ids=q.get('exclude_ids',[]),parsed_intent=q['intent'])
        # Cached intent decisions were originally produced by Qwen; no new timing is fabricated.
        record={**q,**r,'image':q.get('image')}
        record['timing']['cached_intent_original_seconds']=q['timing']['intent_seconds']
        report['queries'].append(record)
        print(q['id'],[(x['id'],round(x['score'],3)) for x in r['results']],flush=True)
    report.update(created_at=datetime.now(timezone.utc).isoformat(),encoder=encoder.fingerprint,
                  training_state='finetuned',notice='Category contrastive pilot. Cached actual Qwen intents shared with baseline. No manual reranking.')
    for key in ('embedding_model','embedding_revision','embedding_weights_sha256'):
        report['config'][key]=config[key]
    target=Path(args.output);target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':main()
