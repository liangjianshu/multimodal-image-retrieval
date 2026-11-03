"""Audit historical training images against recovered test images.

Renamed/re-encoded training copies prevent a pure filename audit. Exact decoded
pixels are conclusive; dHash matches are conservative potential overlaps.
"""
import argparse
import collections
import json
from pathlib import Path
import re
from build_restored_study import metadata,BKTree,write_jsonl


def run(args):
    test=[json.loads(r) for r in Path(args.test_catalog).read_text().splitlines()]
    tree=BKTree();pixels=collections.defaultdict(list)
    for row in test:
        tree.add(int(row['dhash'],16),row['id']);pixels[row['pixel_sha256']].append(row['id'])
    rows=json.loads(Path(args.legacy_manifest).read_text());seen=set();matches=[];failures=[]
    for row in rows:
        match=re.search(r'<\|vision_start\|>(.*?)<\|vision_end\|>',row['conversations'][0]['value'])
        if not match:continue
        relative=match[1].removeprefix('./');path=Path(args.legacy_root)/relative
        if str(path) in seen:continue
        seen.add(str(path))
        try:meta=metadata(path.read_bytes())
        except Exception as error:failures.append({'id':row['id'],'path':str(path),'error':str(error)});continue
        exact=pixels.get(meta['pixel_sha256'],[]);near=tree.find(int(meta['dhash'],16))
        if exact or near:matches.append({'legacy_id':row['id'],'path':str(path),'exact_pixel_matches':exact,'potential_dhash_matches':near})
    exact_ids=sorted({i for r in matches for i in r['exact_pixel_matches']});potential_ids=sorted({i for r in matches for i in r['potential_dhash_matches']}|set(exact_ids))
    report={'legacy_records':len(rows),'unique_paths_examined':len(seen),'read_failures':failures,
            'test_images':len(test),'exact_test_overlap_count':len(exact_ids),'potential_test_overlap_count':len(potential_ids),
            'exact_test_ids':exact_ids,'potential_test_ids':potential_ids,'matches':matches,
            'qualification':'Current retained legacy training manifest, not a cryptographically archived checkpoint input. dHash similarity is a potential duplicate, not definitive training membership.'}
    Path(args.output).write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print({k:v for k,v in report.items() if k not in ('matches','exact_test_ids','potential_test_ids','read_failures')},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--test-catalog',required=True);p.add_argument('--legacy-manifest',required=True)
    p.add_argument('--legacy-root',required=True);p.add_argument('--output',required=True)
    run(p.parse_args())
