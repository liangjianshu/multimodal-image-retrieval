"""One reproducible learning-rate trial; save and score several training epochs."""
import argparse
from contextlib import nullcontext
import hashlib
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
from sentence_transformers.util import batch_to_device

from mmretrieval.cli import load_config
from mmretrieval.data import read_jsonl,image_path
from mmretrieval.gemma import GemmaEncoder,query_payload
from mmretrieval.study_metrics import class_contrastive_loss,evaluate_vectors


def queries(rows):
    names={r['label']:r['label_cn'] for r in rows}
    return [{'id':label+'-'+str(i),'label':label,'text':text} for label,cn in sorted(names.items())
            for i,text in enumerate([cn,f'给我看看{cn}',f'a photo of a {label}'])]


def assess(encoder,rows,root,output):
    vectors=encoder.images([image_path(root,r['image']) for r in rows]);q=queries(rows)
    text=encoder._encode([query_payload(r['text']) for r in q])
    result=evaluate_vectors(rows,vectors,q,text)
    np.savez_compressed(output.with_suffix('.npz'),images=vectors,text=text)
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    score=sum(result['metrics'][m]['primary_five_macro']['average_precision'] for m in ('text','image'))/2
    return score,result['metrics']


def train(args):
    torch.set_num_threads(4);torch.manual_seed(args.seed);rng=random.Random(args.seed)
    config=load_config(args.config);encoder=GemmaEncoder(config);model=encoder.model
    out=Path(args.output)
    if out.exists() and any(out.iterdir()):raise ValueError('Trial output is not empty')
    out.mkdir(parents=True)
    train=read_jsonl(args.train);val=read_jsonl(args.val_catalog)
    device_type=encoder.device.split(':')[0];use_amp=device_type in ('npu','cuda')
    def amp():return torch.autocast(device_type=device_type,dtype=torch.bfloat16) if use_amp else nullcontext()
    report={'status':'running','implementation_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'args':vars(args),'base_encoder':encoder.fingerprint,
            'loss':'multi-positive class InfoNCE; same-class in-batch examples are positive',
            'train_triplets':len(train),'val_images':len(val),'train_sha256':hashlib.sha256(Path(args.train).read_bytes()).hexdigest(),
            'val_sha256':hashlib.sha256(Path(args.val_catalog).read_bytes()).hexdigest(),'checkpoints':[],'steps':[]}
    def save():(out/'trial.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    started=time.monotonic()
    score,metrics=assess(encoder,val,args.image_root,out/'validation-base.json')
    report['baseline']={'selection_score':score,'metrics':metrics};save()
    model.float()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    for parameter in model.parameters():parameter.requires_grad_(True)
    report['trainable_parameters']=sum(p.numel() for p in model.parameters())
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.learning_rate,weight_decay=.01)
    batches=[]
    for epoch in range(math.ceil(max(args.epochs))):
        streams=[]
        for mode in ('text','image'):
            records=[r for r in train if ('image' in r)==(mode=='image')];rng.shuffle(records)
            streams.append([records[i:i+args.batch_size] for i in range(0,len(records),args.batch_size)])
        for i in range(max(map(len,streams))):
            for stream in streams:
                if i<len(stream):batches.append(stream[i])
    remaining=math.ceil(max(args.epochs)*len(train));limited=[]
    for batch in batches:
        if remaining<=0:break
        limited.append(batch[:remaining]);remaining-=len(limited[-1])
    batches=limited
    target_records=[math.ceil(e*len(train)) for e in args.epochs];check_index=0;seen=0
    total_updates=math.ceil(len(batches)/args.grad_accum);warmup=max(1,int(total_updates*.05))
    model.train();optimizer.zero_grad(set_to_none=True);running=0.;step_started=time.monotonic()
    for micro,rows in enumerate(batches):
        query=[query_payload(r.get('text'),image_path(args.image_root,r['image']) if r.get('image') else None) for r in rows]
        candidates=[{'image':str(image_path(args.image_root,r[key]))} for key in ('positive','negative') for r in rows]
        labels=[r['label'] for r in rows];candidate_labels=labels+[r['negative_label'] for r in rows]
        qf=batch_to_device(model.tokenize(query,processing_kwargs=encoder.processing),encoder.device)
        imf=batch_to_device(model.tokenize(candidates,processing_kwargs=encoder.processing),encoder.device)
        with amp():
            qv=model(qf)['sentence_embedding'];iv=model(imf)['sentence_embedding']
            loss=class_contrastive_loss(qv,iv,labels,candidate_labels,args.temperature)
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
        divisor=min(args.grad_accum,len(batches)-(micro//args.grad_accum)*args.grad_accum)
        (loss/divisor).backward();running+=loss.detach().float().item()/divisor;seen+=len(rows)
        if (micro+1)%args.grad_accum==0 or micro+1==len(batches):
            step=len(report['steps'])+1
            if step==1:
                groups={}
                for name,parameter in model.named_parameters():
                    if parameter.grad is None:continue
                    group='vision' if 'vision' in name else 'text_and_projection'
                    groups.setdefault(group,[]).append(parameter.grad.detach().float().norm())
                report['first_step_gradient_norms']={key:torch.stack(values).norm().item() for key,values in groups.items()}
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            if not torch.isfinite(norm):raise RuntimeError('Nonfinite gradient')
            for group in optimizer.param_groups:group['lr']=args.learning_rate*min(1.,step/warmup)
            optimizer.step();optimizer.zero_grad(set_to_none=True)
            entry={'step':step,'loss':running,'gradient_norm':norm.item(),'examples_seen':seen,
                   'epoch':seen/len(train),'seconds':time.monotonic()-step_started,'learning_rate':optimizer.param_groups[0]['lr']}
            report['steps'].append(entry);running=0.;print(json.dumps(entry),flush=True);save();step_started=time.monotonic()
            checkpoint_due=check_index<len(target_records) and seen>=target_records[check_index]
            if checkpoint_due:
                epoch=args.epochs[check_index];target=out/f'epoch-{epoch:g}';model.eval()
                model.save_pretrained(str(target))
                digest=hashlib.sha256((target/'model.safetensors').read_bytes()).hexdigest()
                inference={**config,'embedding_model':'local/embeddinggemma2-study','embedding_revision':digest,
                           'embedding_weights_sha256':digest,'embedding_model_path':str(target.resolve())}
                (target/'inference-config.json').write_text(json.dumps(inference,indent=2)+'\n')
                restored=GemmaEncoder(inference)
                score,metrics=assess(restored,val,args.image_root,out/f'validation-{epoch:g}.json')
                del restored
                if device_type=='npu':torch.npu.empty_cache()
                checkpoint={'epoch_requested':epoch,'actual_epoch':seen/len(train),'step':step,'path':str(target),
                            'selection_score':score,'metrics':metrics,'weight_sha256':digest}
                report['checkpoints'].append(checkpoint);check_index+=1;save();print(json.dumps({'checkpoint':{k:v for k,v in checkpoint.items() if k!='metrics'}}),flush=True);model.train();step_started=time.monotonic()
            if args.max_steps and step>=args.max_steps:break
    report['status']='smoke-complete' if args.max_steps else 'complete';report['seconds']=time.monotonic()-started;report['training_step_seconds']=sum(s['seconds'] for s in report['steps'])
    report['peak_allocated_gib']=torch.npu.max_memory_allocated()/1024**3 if hasattr(torch,'npu') else (torch.cuda.max_memory_allocated()/1024**3 if torch.cuda.is_available() else None)
    if report['checkpoints']:report['best']=max(report['checkpoints'],key=lambda x:x['selection_score'])
    save()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--train',required=True)
    p.add_argument('--val-catalog',required=True);p.add_argument('--image-root',required=True);p.add_argument('--output',required=True)
    p.add_argument('--learning-rate',type=float,required=True);p.add_argument('--epochs',type=float,nargs='+',default=[.5,1,2])
    p.add_argument('--batch-size',type=int,default=2);p.add_argument('--grad-accum',type=int,default=8)
    p.add_argument('--temperature',type=float,default=.05);p.add_argument('--seed',type=int,default=42);p.add_argument('--max-steps',type=int)
    train(p.parse_args())
