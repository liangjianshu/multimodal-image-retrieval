"""Supervised contrastive multimodal fine-tuning; no Qwen dependency in the loss."""
import hashlib
import json
import math
import random
from contextlib import nullcontext
from pathlib import Path

from .data import image_path, read_jsonl
from .gemma import GemmaEncoder, query_payload


def validate_triplets(rows, image_root):
    from PIL import Image
    seen=set(); hashes=set()
    for row in rows:
        if not isinstance(row.get('id'),str) or row['id'] in seen: raise ValueError('Triplets require unique string IDs')
        seen.add(row['id'])
        if not (row.get('text') and isinstance(row['text'],str)) and not row.get('image'):
            raise ValueError('Triplet needs text and/or a reference image')
        digests={}
        for key in ('positive','negative','image'):
            if key not in row and key=='image': continue
            if not isinstance(row.get(key),str): raise ValueError('Positive/negative image paths are required')
            p=image_path(image_root,row[key])
            with Image.open(p) as im: im.verify()
            digests[key]=hashlib.sha256(p.read_bytes()).hexdigest(); hashes.add(digests[key])
        if digests['positive']==digests['negative']: raise ValueError('Positive and negative are identical')
        if digests.get('image')==digests['positive']: raise ValueError('Reference must differ from positive image')
    if not rows: raise ValueError('Empty triplet dataset')
    return hashes


def payloads(row, root):
    return [query_payload(row.get('text'),image_path(root,row['image']) if row.get('image') else None),
            {'image':str(image_path(root,row['positive']))}, {'image':str(image_path(root,row['negative']))}]


def pairwise_evaluate(encoder, rows, root):
    import numpy as np
    # Validation uses fixed explicit positive/negative pairs, not full-gallery recall.
    wins=[]; margins=[]
    for row in rows:
        vectors=encoder._encode(payloads(row,root))
        pos,neg=float(vectors[0]@vectors[1]),float(vectors[0]@vectors[2])
        wins.append(pos>neg);margins.append(pos-neg)
    return {'triplets':len(rows),'positive_over_negative_accuracy':float(np.mean(wins)),
            'mean_cosine_margin':float(np.mean(margins)), 'scope':'explicit-pair-ranking-not-full-gallery-recall'}


def run_embedding_training(args, config):
    import torch
    from sentence_transformers.losses import MultipleNegativesRankingLoss
    from sentence_transformers.util import batch_to_device
    train,val=read_jsonl(args.train),read_jsonl(args.val)
    train_hashes=validate_triplets(train,args.image_root);val_hashes=validate_triplets(val,args.image_root)
    if train_hashes & val_hashes: raise ValueError('Train/validation images overlap by content hash')
    if not math.isfinite(args.epochs) or not math.isfinite(args.learning_rate) or args.epochs<=0 or args.grad_accum<1 or args.learning_rate<=0 or (args.max_steps is not None and args.max_steps<1): raise ValueError('Invalid training parameters')
    random.seed(args.seed);torch.manual_seed(args.seed);torch.set_num_threads(4)
    encoder=GemmaEncoder(config); model=encoder.model
    output=Path(args.output)
    if output.exists() and any(output.iterdir()): raise ValueError('Use a new output directory; checkpoints are never overwritten')
    output.mkdir(parents=True,exist_ok=True)
    def save_report(report): (output/'training_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    before=pairwise_evaluate(encoder,val,args.image_root)
    report={'status':'training','base_encoder':encoder.fingerprint,'seed':args.seed,'epochs_requested':args.epochs,
            'learning_rate':args.learning_rate,'grad_accum':args.grad_accum,'micro_batch_size':1,
            'train_triplets':len(train),'val_triplets':len(val),'train_unique_images':len(train_hashes),
            'val_unique_images':len(val_hashes),'image_hash_overlap':0,'loss':'MultipleNegativesRankingLoss',
            'supervision':'user-supplied query/positive/explicit-negative triplets','before':before,
            'train_manifest_sha256':hashlib.sha256(Path(args.train).read_bytes()).hexdigest(),
            'val_manifest_sha256':hashlib.sha256(Path(args.val).read_bytes()).hexdigest(),'steps':[]}
    save_report(report);print('Baseline validation',before,flush=True)
    # Full fine-tuning of the loaded text+vision model. No frozen image vectors.
    # Keep master parameters and Adam moments in FP32; low learning rates can
    # otherwise round away when directly updating BF16 parameters.
    model.float()
    for parameter in model.parameters(): parameter.requires_grad_(True)
    report['trainable_parameters']=sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.learning_rate)
    device_type=encoder.device.split(':')[0]
    use_amp=device_type=='npu' or (device_type=='cuda' and torch.cuda.is_bf16_supported())
    report['precision']='fp32-master-bf16-autocast' if use_amp else 'fp32'
    loss_fn=MultipleNegativesRankingLoss(model)
    micro_steps=math.ceil(len(train)*args.epochs)
    if args.max_steps: micro_steps=min(micro_steps,args.max_steps*args.grad_accum)
    order=list(range(len(train)));random.shuffle(order)
    model.train();optimizer.zero_grad(set_to_none=True);running=0.
    for micro in range(micro_steps):
        if micro and micro%len(train)==0: random.shuffle(order)
        row=train[order[micro%len(train)]]
        features=[batch_to_device(model.tokenize([p],processing_kwargs=encoder.processing),encoder.device)
                  for p in payloads(row,args.image_root)]
        with torch.autocast(device_type=device_type,dtype=torch.bfloat16) if use_amp else nullcontext():
            loss=loss_fn(features,None)
        if not torch.isfinite(loss): raise RuntimeError('Nonfinite training loss')
        group_start=(micro//args.grad_accum)*args.grad_accum
        divisor=min(args.grad_accum,micro_steps-group_start)
        (loss/divisor).backward();running+=float(loss.detach().cpu())/divisor
        if (micro+1)%args.grad_accum==0 or micro+1==micro_steps:
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            if not torch.isfinite(norm): raise RuntimeError('Nonfinite gradient norm')
            optimizer.step();optimizer.zero_grad(set_to_none=True)
            entry={'step':len(report['steps'])+1,'loss':running,'micro_steps':micro+1,
                   'gradient_norm':float(norm.detach().cpu())}
            report['steps'].append(entry);running=0.;save_report(report);print(entry,flush=True)
    # Match the base model's inference precision for before/after evaluation.
    if use_amp: model.bfloat16()
    model.eval()
    report['after']=pairwise_evaluate(encoder,val,args.image_root)
    model.save_pretrained(str(output/'final'))
    digest=hashlib.sha256()
    with (output/'final/model.safetensors').open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024),b''): digest.update(block)
    report['checkpoint_sha256']=digest.hexdigest()
    report['status']='complete'; report['actual_epochs']=micro_steps/len(train)
    save_report(report)
    inference={**config,'backend':'qwen-embedding','embedding_model':'local/embeddinggemma2-finetuned',
               'embedding_revision':report['checkpoint_sha256'],'embedding_weights_sha256':report['checkpoint_sha256'],
               'embedding_model_path':str((output/'final').resolve())}
    (output/'inference-config.json').write_text(json.dumps(inference,ensure_ascii=False,indent=2)+'\n')
    return report
