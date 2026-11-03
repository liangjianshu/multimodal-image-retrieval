"""Build expanded, audited training data around an immutable historical test.

True synset labels supervise both models. No legacy generated assistant labels
are used. Exact IDs, decoded pixels and conservative dHash neighbours of test
images are removed before stratified train/validation allocation.
"""
import argparse
import collections
import hashlib
import io
import json
from pathlib import Path
import random
import tarfile

from PIL import Image, ImageOps

LABELS=dict(zip('chair cup bicycle skateboard oven couch bowl motorcycle snowboard microwave cat donut keyboard orange pizza'.split(),
               '椅子 杯子 自行车 滑板 烤箱 沙发 碗 摩托车 滑雪板 微波炉 猫 甜甜圈 键盘 橙子 披萨'.split()))
SYNSETS=dict(zip('n03001627 n03147509 n02834778 n04225987 n03862676 n04256520 n02881193 n03790512 n04251791 n03761084 n02121620 n07639069 n03614007 n07747607 n07873807'.split(),LABELS))
NEIGHBOURS=dict(zip(list(LABELS)[:5],list(LABELS)[5:10]));NEIGHBOURS.update({v:k for k,v in list(NEIGHBOURS.items())})
CATEGORY_PROMPT='请识别图片中主要物体的类别。只返回一个 JSON 数组，包含最具体的中文类别名称，不要输出其他内容。'


def metadata(raw):
    with Image.open(io.BytesIO(raw)) as image:
        image=ImageOps.exif_transpose(image).convert('RGB')
        pixels=hashlib.sha256(str(image.size).encode()+image.tobytes()).hexdigest()
        small=image.convert('L').resize((9,8),Image.Resampling.LANCZOS)
        values=list(small.getdata());bits=0
        for y in range(8):
            for x in range(8):bits=(bits<<1)|int(values[y*9+x]>values[y*9+x+1])
        return {'pixel_sha256':pixels,'dhash':f'{bits:016x}','width':image.width,'height':image.height}


class BKTree:
    def __init__(self):self.root=None
    def add(self,value,index):
        if self.root is None:self.root=[value,[index],{}];return
        node=self.root
        while True:
            distance=(value^node[0]).bit_count()
            if distance==0:node[1].append(index);return
            if distance not in node[2]:node[2][distance]=[value,[index],{}];return
            node=node[2][distance]
    def find(self,value,radius=2):
        pending=[self.root] if self.root is not None else [];found=[]
        while pending:
            node=pending.pop();distance=(value^node[0]).bit_count()
            if distance<=radius:found.extend(node[1])
            pending.extend(child for edge,child in node[2].items() if distance-radius<=edge<=distance+radius)
        return found


def write_jsonl(path,rows):path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))


def build(args):
    root=Path(args.output).resolve();root.mkdir(parents=True,exist_ok=True)
    image_root=Path(args.image_root).resolve()
    testroot=Path(args.test_root)
    historical=[json.loads(line) for line in (testroot/'catalog.jsonl').read_text().splitlines()]
    if len(historical)!=2000:raise ValueError('Restore all original 2000 test images before constructing the study')
    test=[];test_ids=set();test_pixels=set();testtree=BKTree()
    for i,row in enumerate(historical):
        path=testroot/row['file_path'];meta=metadata(path.read_bytes())
        identity=Path(row['file_name']).stem;test_ids.add(identity);test_pixels.add(meta['pixel_sha256']);testtree.add(int(meta['dhash'],16),i)
        test.append({**row,**meta,'id':identity,'historical_id':row['id'],'image':str(path.resolve().relative_to(image_root)),'label_cn':row['categories'][0]})
    poolpath=root/'source-pool.jsonl'
    if poolpath.exists():pool=[json.loads(line) for line in poolpath.read_text().splitlines()]
    else:
        pool=[];invalid=[]
        for archive in sorted(Path(args.archives).glob('*.tar')):
            if archive.stem not in SYNSETS:continue
            label=SYNSETS[archive.stem]
            with tarfile.open(archive) as tar:
                for member in tar:
                    if not member.isfile():continue
                    name=Path(member.name).name;identity=Path(name).stem
                    if not identity.startswith(archive.stem+'_'):continue
                    raw=tar.extractfile(member).read()
                    try:meta=metadata(raw)
                    except Exception as error:invalid.append({'id':identity,'error':str(error)});continue
                    path=root/'pool'/archive.stem/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
                    pool.append({'id':identity,'image':str(path.resolve().relative_to(image_root)),'label':label,'label_cn':LABELS[label],
                                 'source':str(archive),**meta})
            print(json.dumps({'extracted':archive.name,'total':len(pool)}),flush=True)
        public=Path(args.public_pool)
        for row in map(json.loads,(public/'source-manifest.jsonl').read_text().splitlines()):
            if row['wnid'] not in SYNSETS:continue
            path=public/row['file_path'];label=SYNSETS[row['wnid']]
            pool.append({'id':row['id'],'image':str(path.resolve().relative_to(image_root)),'label':label,'label_cn':LABELS[label],
                         'source':row['source'],'source_revision':row['source_revision'],**metadata(path.read_bytes())})
        write_jsonl(poolpath,pool);(root/'invalid-source-images.json').write_text(json.dumps(invalid,indent=2))
    excluded=[];candidates=[];seen_ids=set()
    for row in sorted(pool,key=lambda r:r['id']):
        reason=None
        if row['id'] in seen_ids:reason='duplicate-source-id'
        elif row['id'] in test_ids:reason='historical-test-id'
        elif row['pixel_sha256'] in test_pixels:reason='historical-test-identical-pixels'
        elif testtree.find(int(row['dhash'],16)):reason='conservative-test-dhash-distance-at-most-2'
        seen_ids.add(row['id'])
        if reason:excluded.append({'id':row['id'],'reason':reason})
        else:candidates.append(row)
    # Cluster exact and near duplicates before splitting. Conflicting class groups
    # are excluded entirely; one representative remains in other groups.
    parent=list(range(len(candidates)));tree=BKTree();pixels={}
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    for i,row in enumerate(candidates):
        matches=tree.find(int(row['dhash'],16))
        if row['pixel_sha256'] in pixels:matches.append(pixels[row['pixel_sha256']])
        for j in matches:parent[find(i)]=find(j)
        tree.add(int(row['dhash'],16),i);pixels[row['pixel_sha256']]=i
    groups=collections.defaultdict(list)
    for i,row in enumerate(candidates):groups[find(i)].append(row)
    retained=[]
    for group in groups.values():
        group.sort(key=lambda r:r['id'])
        if len({r['label'] for r in group})>1:
            excluded.extend({'id':r['id'],'reason':'conflicting-label-near-duplicate-cluster'} for r in group)
        else:
            retained.append(group[0]);excluded.extend({'id':r['id'],'reason':'duplicate-cluster'} for r in group[1:])
    byclass=collections.defaultdict(list)
    for row in retained:byclass[row['label']].append(row)
    rng=random.Random(args.seed);train=[];val=[];unused=[]
    for label in LABELS:
        rows=sorted(byclass[label],key=lambda r:r['id']);rng.shuffle(rows)
        nval=20 if label in list(LABELS)[:5] else args.val_per_class
        if len(rows)<args.train_per_class+nval:raise ValueError(f'Insufficient clean {label}: {len(rows)}')
        val.extend(rows[:nval]);train.extend(rows[nval:nval+args.train_per_class]);unused.extend(rows[nval+args.train_per_class:])
    report={'seed':args.seed,'image_root':str(image_root),'source_images':len(pool),'clean_pool':len(retained),'exclusions':dict(collections.Counter(r['reason'] for r in excluded)),
            'test_policy':'Original 2000 IDs and normal/fine grouping unchanged; includes both original JPEG and public-copy PNG',
            'near_duplicate_policy':'Conservative 64-bit dHash distance <=2; exclusions are potential duplicates, not proof of identity',
            'label_policy':'True ImageNet synset; never legacy assistant text','qwen_task':'Image-grounded category JSON SFT; not attribute-description supervision',
            'splits':{},'unused_images':len(unused)}
    for name,rows in [('train',train),('val',val),('test',test)]:
        path=root/f'{name}-catalog.jsonl';write_jsonl(path,rows)
        report['splits'][name]={'images':len(rows),'classes':dict(collections.Counter(r['label'] for r in rows)),
                              'manifest_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
        if name=='test':continue
        write_jsonl(root/f'{name}-qwen.jsonl',[{'id':r['id'],'file_path':r['image'],'label':r['label'],
                  'prompt':CATEGORY_PROMPT,'answer':json.dumps([r['label_cn']],ensure_ascii=False)} for r in rows])
        classrows=collections.defaultdict(list)
        for r in rows:classrows[r['label']].append(r)
        triplets=[]
        for row in rows:
            label=row['label'];neglabel=NEIGHBOURS.get(label) if rng.random()<.75 else None
            if neglabel is None:neglabel=rng.choice([c for c in LABELS if c!=label])
            negative=rng.choice(classrows[neglabel]);ref=rng.choice([r for r in classrows[label] if r['id']!=row['id']])
            common={'positive':row['image'],'negative':negative['image'],'label':label,'negative_label':neglabel}
            triplets.append({'id':row['id']+'-text','text':f'寻找{row["label_cn"]}的图片',**common})
            triplets.append({'id':row['id']+'-image','image':ref['image'],**common})
        rng.shuffle(triplets);write_jsonl(root/f'{name}-embedding.jsonl',triplets)
    write_jsonl(root/'excluded.jsonl',excluded);write_jsonl(root/'unused-catalog.jsonl',unused)
    (root/'data-report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--test-root',required=True);p.add_argument('--archives',required=True)
    p.add_argument('--public-pool',required=True);p.add_argument('--output',required=True);p.add_argument('--image-root',required=True)
    p.add_argument('--train-per-class',type=int,default=300);p.add_argument('--val-per-class',type=int,default=90);p.add_argument('--seed',type=int,default=42)
    build(p.parse_args())
