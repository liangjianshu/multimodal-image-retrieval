"""Run both retrieval routes with identical, actually generated Qwen intents."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from mmretrieval.cli import load_config
from mmretrieval.data import read_jsonl, image_path
from mmretrieval.models import Qwen, Encoders
from mmretrieval.gemma import GemmaEncoder
from mmretrieval.retrieval import Index
from mmretrieval.image_index import ImageIndex
from mmretrieval.routes import RoutedRetriever
from mmretrieval.intent import ROUTE_INTENT_PROMPT
from examples.showcase.run import audit_adapter, write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-root',required=True);p.add_argument('--adapter-path',required=True)
    p.add_argument('--descriptions',required=True);p.add_argument('--output',required=True)
    p.add_argument('--device',default='npu:0');p.add_argument('--embedding-config')
    args=p.parse_args()
    import torch
    torch.set_num_threads(4);torch.manual_seed(42)
    gallery=ROOT/'examples/showcase';out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    public=load_config('configs/route-a-qwen-semantic.json');public['adapter']='models/retrieval-lora'
    config={**public,'device':args.device,'vl_model':str(Path(args.model_root)/'qwen-vl-7b'),
            'adapter':args.adapter_path,'text_encoder':str(Path(args.model_root)/'taiyi'),
            'image_encoder':str(Path(args.model_root)/'clip'),'embedding_model_path':str(Path(args.model_root)/'embeddinggemma-2')}
    qwen=Qwen(config,vision=True);audit=audit_adapter(qwen,args.adapter_path)
    audit['base_model']=public['vl_model']
    rows=read_jsonl(args.descriptions)
    sources=json.loads((gallery/'sources.json').read_text())
    if {r['id'] for r in rows}!={s['id'] for s in sources}:raise ValueError('Description gallery differs')
    for s in sources:
        if hashlib.sha256((gallery/s['file_path']).read_bytes()).hexdigest()!=s['sha256']:raise ValueError('Gallery hash differs')
    queries=json.loads((gallery/'queries-hybrid.json').read_text());intents={};timings={}
    for q in queries:
        t=time.perf_counter()
        intents[q['id']]=qwen.intent(q.get('text') or '',image_path(gallery,q['image']) if q.get('image') else None,parse_image_intent=True)
        timings[q['id']]=time.perf_counter()-t
        print(q['id'],intents[q['id']],flush=True)
        if intents[q['id']]['needs_retrieval']!=q.get('expected_retrieval',True):raise ValueError('Intent check failed')
        if 'expected_top_k' in q and intents[q['id']]['top_k']!=q['expected_top_k']:raise ValueError('Quantity check failed')
    write_json(out/'shared-intents.json',intents)
    enc=Encoders(config)
    a=Index(rows,[enc.text(', '.join(r['retrieval_categories'])) for r in rows],
            [enc.text(r['retrieval_description']) for r in rows],
            [enc.image(image_path(gallery,r['file_path'])) for r in rows],
            {'version':1,'text_encoder':public['text_encoder'],'image_encoder':public['image_encoder']})
    a.save(out/'route-a/index')
    gemma=GemmaEncoder(config)
    b=ImageIndex(rows,gemma.images([image_path(gallery,r['file_path']) for r in rows]),{**gemma.fingerprint,'version':1})
    b.save(out/'route-b/index')
    paths=[('route-a','qwen-semantic',a,enc,'qwen-original-lora',public),
           ('route-b','qwen-embedding',b,gemma,'pretrained-baseline',public)]
    if args.embedding_config:
        ftcfg=load_config(args.embedding_config);ftcfg['device']=args.device
        ft=GemmaEncoder(ftcfg)
        fi=ImageIndex(rows,ft.images([image_path(gallery,r['file_path']) for r in rows]),{**ft.fingerprint,'version':1})
        fi.save(out/'route-b-finetuned/index')
        pubft={**public,'embedding_model':ftcfg['embedding_model'],'embedding_revision':ftcfg['embedding_revision']}
        paths.append(('route-b-finetuned','qwen-embedding',fi,ft,'finetuned',pubft))
    for folder,backend,index,encoder,state,pub in paths:
        pipeline=RoutedRetriever(backend,index,qwen,lambda e=encoder:e,config);records=[]
        for query in queries:
            reference=image_path(gallery,query['image']) if query.get('image') else None
            excluded=[r['id'] for r in rows if reference and image_path(gallery,r['file_path'])==reference]
            r=pipeline.search(query.get('text'),reference,top_k=3,exclude_ids=excluded,parsed_intent=intents[query['id']])
            r['timing']['intent_seconds']=timings[query['id']]
            r['elapsed_seconds']+=timings[query['id']]
            records.append({**query,**r,'image':query.get('image'),'mode':'semantic' if backend=='qwen-semantic' else 'embedding'})
            print(folder,query['id'],[(x['id'],round(x['score'],3)) for x in r['results']],flush=True)
        report={'created_at':datetime.now(timezone.utc).isoformat(),'backend':backend,'gallery_size':len(rows),
                'config':{**pub,'backend':backend},'training_state':state,'adapter_verification':audit,
                'encoder':getattr(encoder,'fingerprint',{'text_encoder':public['text_encoder'],'image_encoder':public['image_encoder']}),
                'shared_intents':True,'intent_prompt_sha256':hashlib.sha256(ROUTE_INTENT_PROMPT.encode()).hexdigest(),'description_cache_sha256':hashlib.sha256(Path(args.descriptions).read_bytes()).hexdigest(),
                'notice':'Real recorded inference. Identical Qwen intent decisions for both retrieval routes. Small qualitative gallery.',
                'queries':records}
        write_json(out/folder/'results.json',report)
    print('Completed both routes',flush=True)

if __name__=='__main__': main()
