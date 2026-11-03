import importlib.util
import io
import json
from pathlib import Path
import random
import sys
import tarfile
import types

from PIL import Image

ROOT=Path(__file__).resolve().parents[1]/'examples/training'
def module(name):
    spec=importlib.util.spec_from_file_location(name,ROOT/(name+'.py'));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def picture(format):
    stream=io.BytesIO();Image.new('RGB',(12,9),(15,100,200)).save(stream,format=format);return stream.getvalue()


def test_restore_keeps_original_ids_when_public_source_uses_png(tmp_path):
    recovery=module('restore_original_gallery');archives=tmp_path/'archives';archives.mkdir()
    jpeg=picture('JPEG');png=picture('PNG')
    with tarfile.open(archives/'class.tar','w') as tar:
        info=tarfile.TarInfo('nested/n1_1.JPEG');info.size=len(jpeg);tar.addfile(info,io.BytesIO(jpeg))
    original={'images':[{'id':0,'file_name':'n1_1.JPEG','label':'cup','label_cn':'杯子','type':'normal'},
                        {'id':1,'file_name':'n2_2.JPEG','label':'chair','label_cn':'椅子','type':'fine'}]}
    manifest=tmp_path/'original.json';manifest.write_text(json.dumps(original))
    pool=tmp_path/'public';pool.mkdir();(pool/'n2_2.png').write_bytes(png)
    (pool/'source-manifest.jsonl').write_text(json.dumps({'id':'n2_2','file_path':'n2_2.png','source':'pinned-parquet',
                     'source_revision':'abc',**recovery.image_metadata(png)})+'\n')
    recovery.restore(manifest,archives,tmp_path/'out',pool)
    rows=[json.loads(r) for r in (tmp_path/'out/catalog.jsonl').read_text().splitlines()]
    assert [r['id'] for r in rows]==['0','1']
    assert rows[1]['file_name']=='n2_2.JPEG' and rows[1]['file_path'].endswith('.png')
    assert rows[1]['type']=='fine' and rows[1]['source_revision']=='abc'
    assert json.loads((tmp_path/'out/recovery-report.json').read_text())['restored']==2


def test_bktree_neighbours_match_exhaustive_hamming():
    m=module('build_restored_study');rng=random.Random(9);values=[rng.getrandbits(64) for _ in range(200)]
    values.extend([values[0]^1,values[0]^3,values[0]])
    tree=m.BKTree()
    for i,v in enumerate(values):tree.add(v,i)
    for q in values[:10]:
        assert sorted(tree.find(q))==[i for i,v in enumerate(values) if (q^v).bit_count()<=2]


def test_parallel_range_reader_preserves_seek_boundaries(monkeypatch):
    m=module('recover_hf_imagenet');content=bytes(range(251))*20
    def get(url,headers,timeout):
        a,b=map(int,headers['Range'].removeprefix('bytes=').split('-'))
        return types.SimpleNamespace(status_code=206,headers={'Content-Range':f'bytes {a}-{b}/{len(content)}'},
             content=content[a:b+1],raise_for_status=lambda:None,close=lambda:None)
    monkeypatch.setitem(sys.modules,'requests',types.SimpleNamespace(get=get))
    with m.ParallelRangeFile('https://example.invalid',workers=3,block_size=100) as file:
        file.seek(97);assert file.read(707)==content[97:804]
        file.seek(-15,2);assert file.read()==content[-15:]
        file.seek(3);assert file.read(212)==content[3:215]
