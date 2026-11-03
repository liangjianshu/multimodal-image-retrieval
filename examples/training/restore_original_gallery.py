"""Recover fixed historical image IDs from archives without resampling them."""
import argparse
import collections
import hashlib
import io
import json
from pathlib import Path
import tarfile

from PIL import Image, ImageOps


def image_metadata(raw):
    with Image.open(io.BytesIO(raw)) as image:
        image=ImageOps.exif_transpose(image).convert('RGB')
        pixels=hashlib.sha256(str(image.size).encode()+image.tobytes()).hexdigest()
        return {'width':image.width,'height':image.height,'pixel_sha256':pixels,
                'sha256':hashlib.sha256(raw).hexdigest()}


def restore(manifest, archives, output, public_pool=None):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    original=json.loads(Path(manifest).read_text())['images']
    desired={r['file_name']:r for r in original}
    if len(desired)!=len(original): raise ValueError('Historical manifest contains duplicate filenames')
    recovered={};archive_log=[]
    for archive in sorted(Path(archives).glob('*.tar')):
        archive_rows=[]
        with tarfile.open(archive) as tar:
            for member in tar:
                name=Path(member.name).name
                if not member.isfile() or name not in desired:continue
                row=desired[name];raw=tar.extractfile(member).read();metadata=image_metadata(raw)
                relative=f'images/{row["label"]}/{name}'
                target=output/relative;target.parent.mkdir(parents=True,exist_ok=True)
                if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest()!=metadata['sha256']:
                    raise ValueError(f'Existing recovery differs for {name}')
                target.write_bytes(raw)
                recovered[name]={'id':str(row['id']),'file_path':relative,'file_name':name,
                    'label':row['label'],'categories':[row['label_cn']],'type':row['type'],
                    'recovery_source':archive.name,**metadata}
                archive_rows.append(name)
        archive_log.append({'archive':archive.name,'recovered':len(archive_rows)})
    # Public copy retains original IDs but stores PNG, not the original JPEG bytes.
    if public_pool:
        pool=Path(public_pool)
        source={r['id']:r for r in map(json.loads,(pool/'source-manifest.jsonl').read_text().splitlines())}
        for name,row in desired.items():
            if name in recovered or Path(name).stem not in source:continue
            provenance=source[Path(name).stem];raw=(pool/provenance['file_path']).read_bytes()
            metadata=image_metadata(raw)
            if metadata['pixel_sha256']!=provenance['pixel_sha256']:raise ValueError('Public pool changed')
            relative=f'images/{row["label"]}/{Path(name).stem}.png'
            target=output/relative;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
            recovered[name]={'id':str(row['id']),'file_path':relative,'file_name':name,
                'label':row['label'],'categories':[row['label_cn']],'type':row['type'],
                'recovery_source':provenance['source'],'source_revision':provenance['source_revision'],
                'encoding_note':'Public source PNG; original ImageNet ID preserved, original JPEG byte hash unavailable',**metadata}
    # Also retain separately downloaded/verified original images on repeat runs.
    for name,row in desired.items():
        if name in recovered:continue
        relative=f'images/{row["label"]}/{name}';target=output/relative
        if target.exists():
            recovered[name]={'id':str(row['id']),'file_path':relative,'file_name':name,
                'label':row['label'],'categories':[row['label_cn']],'type':row['type'],
                'recovery_source':'separately-restored',**image_metadata(target.read_bytes())}
    rows=[recovered[r['file_name']] for r in original if r['file_name'] in recovered]
    missing=[r for r in original if r['file_name'] not in recovered]
    (output/'catalog.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    (output/'missing.json').write_text(json.dumps(missing,ensure_ascii=False,indent=2)+'\n')
    report={'requested':len(original),'restored':len(rows),'missing':len(missing),
            'status':'complete' if not missing else 'partial-not-full-historical-test',
            'source_manifest_sha256':hashlib.sha256(Path(manifest).read_bytes()).hexdigest(),
            'classes':dict(collections.Counter(r['label'] for r in rows)),
            'missing_classes':dict(collections.Counter(r['label'] for r in missing)),
            'archives':archive_log,'note':'Identity matched by historical filename. Original manifest had no image hashes; new hashes preserve recovered bytes.'}
    (output/'recovery-report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True)
    p.add_argument('--archives',required=True);p.add_argument('--output',required=True)
    p.add_argument('--public-pool')
    args=p.parse_args();restore(args.manifest,args.archives,args.output,args.public_pool)
