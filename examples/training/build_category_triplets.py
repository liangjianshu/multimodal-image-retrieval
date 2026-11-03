"""Recover category supervision from ImageNet paths, never from legacy answers.

This is a category-retrieval pilot, not attribute or instance-level supervision.
COCO examples are intentionally excluded: their multi-object annotations need a
separate relevance policy before they can safely supply negative images.
"""
import argparse
import collections
import hashlib
import json
import random
import re
from pathlib import Path

from PIL import Image, ImageOps

LABELS = dict(zip(
    'chair cup bicycle skateboard oven couch bowl motorcycle snowboard microwave cat donut keyboard orange pizza'.split(),
    '椅子 杯子 自行车 滑板 烤箱 沙发 碗 摩托车 滑雪板 微波炉 猫 甜甜圈 键盘 橙子 披萨'.split()))
NEIGHBORS = {'chair':'couch', 'cup':'bowl', 'bicycle':'motorcycle',
             'skateboard':'snowboard', 'oven':'microwave'}
NEIGHBORS.update({v:k for k,v in list(NEIGHBORS.items())})


def pixel_hash(path):
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert('RGB')
        return hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()


def build(manifest, root, output, seed=42):
    output=Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use an empty output directory')
    records=json.loads(Path(manifest).read_text())
    unique={}; conflicts=set(); selected=0
    for record in records:
        if not record['id'].startswith('imagenet21k_'): continue
        match=re.search(r'<\|vision_start\|>(.*?)<\|vision_end\|>',record['conversations'][0]['value'])
        if not match: raise ValueError('Missing image path')
        relative=match[1].removeprefix('./')
        label=Path(relative).stem.rsplit('_',1)[0]
        if label not in LABELS: raise ValueError('Unknown source category')
        digest=pixel_hash(Path(root)/relative);selected+=1
        if digest in unique and unique[digest]['label']!=label: conflicts.add(digest)
        unique.setdefault(digest, {'id':digest[:20], 'image':relative, 'label':label,
                                  'label_cn':LABELS[label], 'pixel_sha256':digest})
    for digest in conflicts: unique.pop(digest)
    rng=random.Random(seed);by_label=collections.defaultdict(list)
    for row in sorted(unique.values(),key=lambda r:r['id']): by_label[row['label']].append(row)
    partitions={name:[] for name in ['train','val','test']}
    for label,rows in sorted(by_label.items()):
        if len(rows)<10: raise ValueError(f'Insufficient unique images for {label}')
        rng.shuffle(rows);n_hold=max(2,int(len(rows)*.15))
        partitions['val'].extend(rows[:n_hold]);partitions['test'].extend(rows[n_hold:2*n_hold])
        partitions['train'].extend(rows[2*n_hold:])
    output.mkdir(parents=True,exist_ok=True)
    def write(name,rows):
        (output/name).write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in rows))
    report={'seed':seed,'source_records':selected,'unique_images':len(unique),
            'excluded_conflicting_pixel_hashes':len(conflicts),
            'label_source':'ImageNet category in source filename; legacy assistant answers ignored',
            'scope':'category-level only; no attribute, instance, or joint-query claims',
            'split':'70/15/15 approximately, stratified after decoded-pixel deduplication',
            'qwen_comparison_limit':'Old Qwen may have trained on these images. Only pretrained-vs-finetuned embedding comparison is valid here.',
            'splits':{}}
    all_hashes=[]
    for name,rows in partitions.items():
        groups=collections.defaultdict(list)
        for row in rows: groups[row['label']].append(row)
        triples=[]
        for row in rows:
            label=row['label'];other_labels=sorted(set(groups)-{label})
            negative_label=NEIGHBORS.get(label) if rng.random()<.75 else None
            if negative_label not in groups: negative_label=rng.choice(other_labels)
            negative=rng.choice(groups[negative_label])
            ref=rng.choice([r for r in groups[label] if r['id']!=row['id']])
            common={'positive':row['image'],'negative':negative['image'],'label':label,
                    'negative_label':negative_label}
            triples.append({'id':row['id']+'-text','text':f'寻找{row["label_cn"]}的图片',**common})
            triples.append({'id':row['id']+'-image','image':ref['image'],**common})
        rng.shuffle(triples)
        write(name+'-catalog.jsonl',rows);write(name+'.jsonl',triples)
        report['splits'][name]={'images':len(rows),'triplets':len(triples),
                               'classes':dict(collections.Counter(r['label'] for r in rows))}
        all_hashes.append({r['pixel_sha256'] for r in rows})
    assert all(not a&b for i,a in enumerate(all_hashes) for b in all_hashes[i+1:])
    report['decoded_pixel_overlap_between_splits']=0
    report['source_manifest_sha256']=hashlib.sha256(Path(manifest).read_bytes()).hexdigest()
    (output/'data-report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--manifest',required=True);parser.add_argument('--image-root',required=True)
    parser.add_argument('--output',required=True);parser.add_argument('--seed',type=int,default=42)
    args=parser.parse_args();build(args.manifest,args.image_root,args.output,args.seed)
