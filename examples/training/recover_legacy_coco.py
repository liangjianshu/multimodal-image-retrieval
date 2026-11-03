"""Recover truncated COCO files for overlap auditing, without altering old data.

The downloaded original must strictly extend the retained byte prefix. This
identifies the same image without claiming the old model saw the repaired file.
"""
import argparse
import concurrent.futures
import hashlib
import io
import json
from pathlib import Path

import requests
from PIL import Image,ImageOps


def recover(args):
    failures=json.loads(Path(args.audit).read_text())['read_failures']
    annotations=json.loads(Path(args.annotations).read_text());images={r['id']:r for r in annotations['images']}
    output=Path(args.output).resolve();output.mkdir(parents=True,exist_ok=True)
    def repair(record):
        if not record['id'].startswith('coco_'):raise ValueError('This recovery script only handles COCO image IDs')
        identity=int(record['id'].rsplit('_',1)[1]);source=images[identity]['coco_url'];original=Path(record['path']).read_bytes()
        for url in dict.fromkeys([source.replace('http://','https://'),source]):
            try:
                with requests.get(url,timeout=(15,60)) as response:
                    response.raise_for_status();raw=response.content
                with Image.open(io.BytesIO(raw)) as image:ImageOps.exif_transpose(image).convert('RGB').load()
                if not raw.startswith(original):raise ValueError('Recovered source does not extend the retained truncated bytes')
                path=output/f'{identity:012d}.jpg';path.write_bytes(raw)
                return record['path'],{'file_path':str(path),'source_url':url,'coco_id':identity,
                    'old_bytes':len(original),'new_bytes':len(raw),'old_sha256':hashlib.sha256(original).hexdigest(),
                    'new_sha256':hashlib.sha256(raw).hexdigest(),'old_prefix_matches':True}
            except Exception as error:print(json.dumps({'id':identity,'url':url,'error':str(error)}),flush=True)
        raise RuntimeError(f'Could not recover exact COCO source {identity}')
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:repairs=dict(pool.map(repair,failures))
    (output/'repairs.json').write_text(json.dumps(repairs,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'recovered':len(repairs),'manifest':str(output/'repairs.json')}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--audit',required=True);p.add_argument('--annotations',required=True)
    p.add_argument('--output',required=True);recover(p.parse_args())
