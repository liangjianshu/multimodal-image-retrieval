"""Category-grounded Qwen LoRA trial, selected on held-out validation only."""
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
from transformers import AutoProcessor,Qwen2_5_VLForConditionalGeneration

from mmretrieval.cli import load_config
from mmretrieval.data import read_jsonl,image_path
from mmretrieval.models import resolve_device,extract_json
from mmretrieval.training import configure_training,SFTCollator
from mmretrieval.study_metrics import PRIMARY
from build_restored_study import CATEGORY_PROMPT,LABELS

# Fixed before test inference. Subtype names map to their parent category.
ALIASES={
 'chair':['椅','凳'], 'cup':['杯'], 'bicycle':['自行车','单车','脚踏车'],
 'skateboard':['滑板'], 'oven':['烤箱','烤炉'], 'couch':['沙发'], 'bowl':['碗'],
 'motorcycle':['摩托','电动车'], 'snowboard':['滑雪板'], 'microwave':['微波炉'],
 'cat':['猫'], 'donut':['甜甜圈','甜圈','多纳圈'], 'keyboard':['键盘'],
 'orange':['橙','橘','桔'], 'pizza':['披萨','比萨'],
}


def predict(model,processor,rows,root,device,batch_size=8):
    from qwen_vl_utils import process_vision_info
    model.eval();processor.tokenizer.padding_side='left';predictions=[]
    for start in range(0,len(rows),batch_size):
        batch=rows[start:start+batch_size];messages=[]
        for row in batch:
            messages.append([{'role':'user','content':[{'type':'image','image':str(image_path(root,row.get('image',row.get('file_path'))))},
                                                     {'type':'text','text':CATEGORY_PROMPT}]}])
        texts=[processor.apply_chat_template(m,tokenize=False,add_generation_prompt=True) for m in messages]
        images,_=process_vision_info(messages)
        inputs=processor(text=texts,images=images,padding=True,return_tensors='pt').to(device)
        with torch.inference_mode():output=model.generate(**inputs,max_new_tokens=48,do_sample=False,use_cache=True)
        responses=processor.batch_decode(output[:,inputs['input_ids'].shape[1]:],skip_special_tokens=True)
        for row,response in zip(batch,responses):
            valid=True
            try:
                names=extract_json(response)
                if not isinstance(names,list) or not names or any(not isinstance(n,str) for n in names):raise ValueError('Invalid categories')
            except (ValueError,TypeError):valid=False;names=[]
            matched=sorted(c for c,aliases in ALIASES.items() if any(any(a in name for a in aliases) for name in names))
            predictions.append({'id':row['id'],'truth':row['label'],'raw_response':response,'categories':names,'matched_classes':matched,'valid_json':valid})
        if start%200==0:print(json.dumps({'inference_images':min(start+batch_size,len(rows)),'total':len(rows)}),flush=True)
    perclass={}
    for c in LABELS:
        tp=sum(r['truth']==c and c in r['matched_classes'] for r in predictions)
        fp=sum(r['truth']!=c and c in r['matched_classes'] for r in predictions)
        fn=sum(r['truth']==c and c not in r['matched_classes'] for r in predictions)
        perclass[c]={'tp':tp,'fp':fp,'fn':fn,'precision':tp/(tp+fp) if tp+fp else 0.,
                     'recall':tp/(tp+fn) if tp+fn else 0.,'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.}
    return {'images':len(rows),'prompt':CATEGORY_PROMPT,'aliases':ALIASES,'per_class':perclass,
            'macro_f1':float(np.mean([r['f1'] for r in perclass.values()])),
            'primary_five_macro_f1':float(np.mean([perclass[c]['f1'] for c in PRIMARY])),
            'valid_json_rate':sum(r['valid_json'] for r in predictions)/len(rows),'predictions':predictions}


def run(args):
    config=load_config(args.config);device=resolve_device(config['device']);torch.set_num_threads(4)
    torch.manual_seed(args.seed);rng=random.Random(args.seed)
    out=Path(args.output)
    if out.exists() and any(out.iterdir()):raise ValueError('Trial output must be empty')
    out.mkdir(parents=True)
    model=Qwen2_5_VLForConditionalGeneration.from_pretrained(config['vl_model'],dtype=torch.bfloat16,
                attn_implementation='eager' if device.startswith('npu') else 'sdpa').to(device)
    processor=AutoProcessor.from_pretrained(config['vl_model'],max_pixels=config['max_pixels'])
    val=read_jsonl(args.val_catalog)
    if args.evaluate_only:
        if args.adapter:
            from peft import PeftModel
            model=PeftModel.from_pretrained(model,args.adapter)
        started=time.monotonic();result=predict(model,processor,val,args.image_root,device,args.eval_batch_size)
        result.update(seconds=time.monotonic()-started,adapter=args.adapter,config=config,
                      catalog_sha256=hashlib.sha256(Path(args.val_catalog).read_bytes()).hexdigest())
        (out/'evaluation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');return
    train=read_jsonl(args.train);model,trainable=configure_training(model,args.mode,args.rank,args.rank*2)
    model.config.use_cache=False;model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    collator=SFTCollator(processor,args.image_root)
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=args.learning_rate,weight_decay=.01)
    report={'status':'running','implementation_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'args':vars(args),'train_images':len(train),'validation_images':len(val),
            'train_manifest_sha256':hashlib.sha256(Path(args.train).read_bytes()).hexdigest(),
            'val_manifest_sha256':hashlib.sha256(Path(args.val_catalog).read_bytes()).hexdigest(),
            'trainable_parameters':sum(trainable.values()),'trainable_names':trainable,
            'selection_metric':'primary_five_macro_f1','checkpoints':[],'steps':[]}
    def save():(out/'trial.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    started=time.monotonic();save();batches=[]
    for epoch in range(math.ceil(max(args.epochs))):
        order=list(train);rng.shuffle(order)
        batches.extend(order[i:i+args.batch_size] for i in range(0,len(order),args.batch_size))
    remaining=math.ceil(max(args.epochs)*len(train));limited=[]
    for batch in batches:
        if remaining<=0:break
        limited.append(batch[:remaining]);remaining-=len(limited[-1])
    batches=limited
    targets=[math.ceil(e*len(train)) for e in args.epochs];check_index=0;seen=0;running=0.
    warmup=max(1,int(math.ceil(len(batches)/args.grad_accum)*.05))
    model.train();optimizer.zero_grad(set_to_none=True);step_started=time.monotonic()
    for micro,rows in enumerate(batches):
        batch={k:v.to(device) for k,v in collator(rows).items()}
        with torch.autocast(device_type=device.split(':')[0],dtype=torch.bfloat16):loss=model(**batch).loss
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite Qwen loss')
        divisor=min(args.grad_accum,len(batches)-(micro//args.grad_accum)*args.grad_accum)
        (loss/divisor).backward();running+=loss.detach().float().item()/divisor;seen+=len(rows)
        if (micro+1)%args.grad_accum==0 or micro+1==len(batches):
            step=len(report['steps'])+1;norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.)
            if not torch.isfinite(norm):raise RuntimeError('Nonfinite gradient norm')
            for group in optimizer.param_groups:group['lr']=args.learning_rate*min(1.,step/warmup)
            optimizer.step();optimizer.zero_grad(set_to_none=True)
            entry={'step':step,'loss':running,'gradient_norm':norm.item(),'epoch':seen/len(train),'examples_seen':seen,
                   'seconds':time.monotonic()-step_started,'learning_rate':optimizer.param_groups[0]['lr']}
            report['steps'].append(entry);print(json.dumps(entry),flush=True);running=0.;save();step_started=time.monotonic()
            if check_index<len(targets) and seen>=targets[check_index]:
                epoch=args.epochs[check_index];target=out/f'epoch-{epoch:g}'
                model.save_pretrained(target);processor.save_pretrained(target)
                model.gradient_checkpointing_disable()
                result=predict(model,processor,val,args.image_root,device,args.eval_batch_size)
                (out/f'validation-{epoch:g}.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
                digest=hashlib.sha256((target/'adapter_model.safetensors').read_bytes()).hexdigest()
                checkpoint={'epoch_requested':epoch,'actual_epoch':seen/len(train),'step':step,'path':str(target),
                            'selection_score':result['primary_five_macro_f1'],'macro_f1':result['macro_f1'],
                            'valid_json_rate':result['valid_json_rate'],'weight_sha256':digest}
                report['checkpoints'].append(checkpoint);check_index+=1;save();print(json.dumps({'checkpoint':checkpoint}),flush=True)
                model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});model.train();step_started=time.monotonic()
            if args.max_steps and step>=args.max_steps:break
    report['status']='smoke-complete' if args.max_steps else 'complete';report['seconds']=time.monotonic()-started;report['training_step_seconds']=sum(s['seconds'] for s in report['steps'])
    report['peak_allocated_gib']=torch.npu.max_memory_allocated()/1024**3 if hasattr(torch,'npu') else (torch.cuda.max_memory_allocated()/1024**3 if torch.cuda.is_available() else None)
    if report['checkpoints']:report['best']=max(report['checkpoints'],key=lambda r:r['selection_score'])
    save()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--train')
    p.add_argument('--val-catalog',required=True);p.add_argument('--image-root',required=True);p.add_argument('--output',required=True)
    p.add_argument('--learning-rate',type=float,default=1e-5);p.add_argument('--epochs',nargs='+',type=float,default=[.5,1,2])
    p.add_argument('--batch-size',type=int,default=4);p.add_argument('--eval-batch-size',type=int,default=8);p.add_argument('--grad-accum',type=int,default=4)
    p.add_argument('--rank',type=int,default=64);p.add_argument('--mode',default='llm-lora',choices=['llm-lora','mlp-llm']);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--max-steps',type=int);p.add_argument('--evaluate-only',action='store_true');p.add_argument('--adapter')
    run(p.parse_args())
