import importlib.util
import json
from pathlib import Path

from PIL import Image


def test_recovered_supervision_ignores_wrong_answers_and_splits_pixels(tmp_path):
    path=Path(__file__).resolve().parents[1]/'examples/training/build_category_triplets.py'
    spec=importlib.util.spec_from_file_location('category_builder',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    images=tmp_path/'images';images.mkdir();records=[]
    for ci,label in enumerate(['cup','bowl']):
        for i in range(12):
            name=f'images/{label}_{i}.png'
            Image.new('RGB',(8,8),(ci*100+i,20,30)).save(tmp_path/name)
            records.append({'id':f'imagenet21k_hardneg_{ci}_{i}','conversations':[
                {'from':'user','value':f'Describe: <|vision_start|>./{name}<|vision_end|>'},
                {'from':'assistant','value':'This is a bicycle.'}]})
    # Same pixels under a different filename must stay in one split.
    duplicate='images/cup_100.png'
    Image.open(tmp_path/'images/cup_0.png').save(tmp_path/duplicate)
    records.append({'id':'imagenet21k_pos_duplicate','conversations':[
        {'from':'user','value':f'<|vision_start|>./{duplicate}<|vision_end|>'}]})
    manifest=tmp_path/'source.json';manifest.write_text(json.dumps(records))
    output=tmp_path/'output';module.build(manifest,tmp_path,output)
    hashes=[]
    for split in ['train','val','test']:
        catalog=[json.loads(line) for line in (output/f'{split}-catalog.jsonl').read_text().splitlines()]
        hashes.append({r['pixel_sha256'] for r in catalog})
        triples=[json.loads(line) for line in (output/f'{split}.jsonl').read_text().splitlines()]
        for row in triples:
            assert row['label']!=row['negative_label']
            assert Path(row['positive']).stem.startswith(row['label']+'_')
            assert Path(row['negative']).stem.startswith(row['negative_label']+'_')
            assert 'bicycle' not in row.get('text','')
    assert sum(map(len,hashes))==24
    assert not hashes[0]&hashes[1] and not hashes[0]&hashes[2] and not hashes[1]&hashes[2]
