"""Final fixed-gallery comparison. Run only after validation selection is frozen."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from mmretrieval.cli import load_config
from mmretrieval.data import read_jsonl,image_path
from mmretrieval.gemma import GemmaEncoder,query_payload
from mmretrieval.models import Encoders,Qwen
from mmretrieval.retrieval import normalized
from mmretrieval.study_metrics import evaluate_vectors
from run_embedding_trial import queries
from run_qwen_trial import predict


def write(path,data):path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')


def describe(model,processor,rows,root,device,output,batch_size):
    import torch
    from qwen_vl_utils import process_vision_info
    from mmretrieval.historical_prompts import DESCRIPTION_PROMPT
    stored={r['id']:r for r in read_jsonl(output)} if output.exists() else {}
    pending=[r for r in rows if r['id'] not in stored]
    processor.tokenizer.padding_side='left';model.eval()
    for start in range(0,len(pending),batch_size):
        batch=pending[start:start+batch_size]
        messages=[[{'role':'user','content':[{'type':'image','image':str(image_path(root,r['image']))},
                                            {'type':'text','text':DESCRIPTION_PROMPT}]}] for r in batch]
        texts=[processor.apply_chat_template(m,tokenize=False,add_generation_prompt=True) for m in messages]
        images,_=process_vision_info(messages)
        inputs=processor(text=texts,images=images,padding=True,return_tensors='pt').to(device)
        with torch.inference_mode(),torch.autocast(device_type=device.split(':')[0],dtype=torch.bfloat16):
            generated=model.generate(**inputs,max_new_tokens=128,do_sample=False,use_cache=True)
        responses=processor.batch_decode(generated[:,inputs['input_ids'].shape[1]:],skip_special_tokens=True)
        for row,response in zip(batch,responses):
            record={'id':row['id'],'description':response.strip()}
            if not record['description']:raise ValueError('Empty generated description')
            with output.open('a') as file:file.write(json.dumps(record,ensure_ascii=False)+'\n')
            stored[row['id']]=record
        if start%200==0:print(json.dumps({'descriptions':len(stored),'total':len(rows)}),flush=True)
    return [stored[r['id']]['description'] for r in rows]


def run(args):
    started=time.monotonic();out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    config=load_config(args.config)
    if args.adapter:config['adapter']=args.adapter
    elif args.backend=='qwen':config['adapter']=None
    rows={s:read_jsonl(getattr(args,s+'_catalog')) for s in ('val','test')}
    expected={s:hashlib.sha256(Path(getattr(args,s+'_catalog')).read_bytes()).hexdigest() for s in rows}
    metadata={'backend':args.backend,'config':config,'catalog_sha256':expected}
    if args.overlap_audit:
        metadata['overlap_audit_sha256']=hashlib.sha256(Path(args.overlap_audit).read_bytes()).hexdigest()
    if args.backend=='qwen' and config.get('adapter'):
        metadata['adapter_sha256']=hashlib.sha256((Path(config['adapter'])/'adapter_model.safetensors').read_bytes()).hexdigest()
    if (out/'metadata.json').exists() and json.loads((out/'metadata.json').read_text())!=metadata:
        raise ValueError('Existing evaluation belongs to different weights or data')
    write(out/'metadata.json',metadata)
    if (out/'evaluation-complete.json').exists():return
    vectors={};text={};image_queries={};categories={}
    if args.backend=='embedding':
        encoder=GemmaEncoder(config)
        for split,records in rows.items():
            cache=out/f'{split}-vectors.npz'
            if cache.exists():
                data=np.load(cache);vectors[split]=data['images'];text[split]=data['text']
            else:
                vectors[split]=encoder.images([image_path(args.image_root,r['image']) for r in records])
                text[split]=encoder._encode([query_payload(r['text']) for r in queries(records)])
                np.savez_compressed(cache,images=vectors[split],text=text[split])
        metadata['encoder_fingerprint']=encoder.fingerprint
    else:
        import torch
        from transformers import AutoProcessor,Qwen2_5_VLForConditionalGeneration
        from mmretrieval.models import resolve_device
        device=resolve_device(config['device']);torch.set_num_threads(4)
        model=Qwen2_5_VLForConditionalGeneration.from_pretrained(config['vl_model'],dtype=torch.bfloat16,
                    attn_implementation='eager' if device.startswith('npu') else 'sdpa').to(device)
        if config['adapter']:
            from peft import PeftModel
            model=PeftModel.from_pretrained(model,config['adapter'])
        processor=AutoProcessor.from_pretrained(config['vl_model'],max_pixels=config['max_pixels'])
        descriptions={}
        for split,records in rows.items():
            catpath=out/f'{split}-categories.json'
            if catpath.exists():result=json.loads(catpath.read_text())
            else:result=predict(model,processor,records,args.image_root,device,args.batch_size);write(catpath,result)
            lookup={r['id']:r for r in result['predictions']}
            categories[split]=[lookup[r['id']]['matched_classes'] for r in records]
            descriptions[split]=describe(model,processor,records,args.image_root,device,out/f'{split}-descriptions.jsonl',args.batch_size)
        del model,processor;gc.collect()
        if device.startswith('npu'):torch.npu.empty_cache()
        encoders=Encoders(config)
        for split,records in rows.items():
            cache=out/f'{split}-vectors.npz'
            if cache.exists():
                data=np.load(cache);vectors[split]=data['images'];text[split]=data['text'];image_queries[split]=data['reference_images']
            else:
                vectors[split]=normalized(np.stack([encoders.text(d) for d in descriptions[split]]))
                text[split]=normalized(np.stack([encoders.text(q['text']) for q in queries(records)]))
                image_queries[split]=normalized(np.stack([encoders.image(image_path(args.image_root,r['image'])) for r in records]))
                np.savez_compressed(cache,images=vectors[split],text=text[split],reference_images=image_queries[split])
    results={};thresholds=None
    for split,records in rows.items():
        result=evaluate_vectors(records,vectors[split],queries(records),text[split],thresholds,
                    candidate_categories=categories.get(split),image_query_vectors=image_queries.get(split))
        result['protocol']={'queries':'fixed category queries; no claim about free-form intent or attribute relevance',
                           'route':'Qwen category filter + Taiyi description ranking + CLIP reference-image query' if args.backend=='qwen' else 'native image/text embeddings; no category filtering',
                           'image_query_self_excluded':True,'threshold_source':'validation only',
                           'test_model_selection':False}
        result['metadata']=metadata
        if split=='val':thresholds=result['validation_thresholds']
        else:result['total_seconds']=time.monotonic()-started
        results[split]=result;write(out/f'{split}-results.json',result)
    for kind in ('normal','fine'):
        keep=[i for i,r in enumerate(rows['test']) if r.get('type')==kind]
        if not keep:continue
        subset=[rows['test'][i] for i in keep]
        result=evaluate_vectors(subset,vectors['test'][keep],queries(subset),text['test'],thresholds,
             candidate_categories=[categories['test'][i] for i in keep] if 'test' in categories else None,
             image_query_vectors=image_queries['test'][keep] if 'test' in image_queries else None)
        result['scope']=f'Historical {kind} gallery only; original validation thresholds retained'
        write(out/f'test-{kind}-gallery.json',result)
    if args.overlap_audit:
        audit=json.loads(Path(args.overlap_audit).read_text())
        if audit['read_failures']:raise ValueError('Cannot label an unseen subset with an incomplete legacy audit')
        excluded=set(audit['potential_test_ids'])
        keep=[i for i,r in enumerate(rows['test']) if r['id'] not in excluded]
        clean=[rows['test'][i] for i in keep]
        clean_queries=queries(clean);original_queries=queries(rows['test']);qids={r['id']:i for i,r in enumerate(original_queries)}
        result=evaluate_vectors(clean,vectors['test'][keep],clean_queries,text['test'][[qids[r['id']] for r in clean_queries]],thresholds,
             candidate_categories=[categories['test'][i] for i in keep] if 'test' in categories else None,
             image_query_vectors=image_queries['test'][keep] if 'test' in image_queries else None)
        result['scope']='Sensitivity analysis excluding potential old-training overlaps; not proof of pretraining novelty'
        result['excluded_images']=len(excluded);write(out/'test-without-legacy-overlap.json',result)
    print(json.dumps(results['test']['metrics'],ensure_ascii=False,indent=2),flush=True)
    write(out/'evaluation-complete.json',{'seconds':time.monotonic()-started,'catalog_sha256':expected})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--backend',choices=['qwen','embedding'],required=True)
    p.add_argument('--config',required=True);p.add_argument('--adapter');p.add_argument('--val-catalog',required=True)
    p.add_argument('--test-catalog',required=True);p.add_argument('--image-root',required=True);p.add_argument('--output',required=True)
    p.add_argument('--batch-size',type=int,default=8);p.add_argument('--overlap-audit');run(p.parse_args())
