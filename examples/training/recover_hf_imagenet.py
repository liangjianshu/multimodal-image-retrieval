"""Read only matching ImageNet classes from pinned Parquet row groups.

Keeps the original ImageNet IDs and PNG bytes supplied by the source dataset.
The output is private dataset material, not a repository artifact.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import time
import concurrent.futures
from collections import OrderedDict

from PIL import Image, ImageOps



class ParallelRangeFile(io.RawIOBase):
    """Seekable HTTP reader with bounded parallel range downloads.

    Every response range is checked; a server ignoring Range cannot silently
    substitute the wrong bytes. Only requested Parquet column ranges are read.
    """
    def __init__(self,url,workers=8,block_size=1024*1024):
        import requests
        self.url=url;self.position=0;self.block_size=block_size;self.cache=OrderedDict()
        self.pool=concurrent.futures.ThreadPoolExecutor(max_workers=workers)
        response=requests.get(url,headers={'Range':'bytes=0-0'},timeout=(20,90))
        response.raise_for_status()
        content_range=response.headers.get('Content-Range','')
        if response.status_code!=206 or not content_range.startswith('bytes 0-0/'):
            raise ValueError('Source does not support validated byte ranges')
        self.size=int(content_range.split('/')[-1]);response.close()
    def readable(self):return True
    def seekable(self):return True
    def tell(self):return self.position
    def seek(self,offset,whence=0):
        self.position=offset if whence==0 else self.position+offset if whence==1 else self.size+offset
        if self.position<0:raise ValueError('Negative file offset')
        return self.position
    def fetch(self,index):
        import requests
        start=index*self.block_size;end=min(start+self.block_size,self.size)-1
        for attempt in range(4):
            try:
                response=requests.get(self.url,headers={'Range':f'bytes={start}-{end}'},timeout=(20,90))
                response.raise_for_status();raw=response.content
                if response.status_code!=206 or response.headers.get('Content-Range')!=f'bytes {start}-{end}/{self.size}' or len(raw)!=end-start+1:
                    raise ValueError('Incorrect HTTP byte range')
                return index,raw
            except Exception:
                if attempt==3:raise
                time.sleep(1+attempt)
    def read(self,size=-1):
        end=self.size if size<0 else min(self.position+size,self.size)
        if end<=self.position:return b''
        indices=list(range(self.position//self.block_size,(end-1)//self.block_size+1))
        chunks={i:self.cache[i] for i in indices if i in self.cache}
        for i,raw in self.pool.map(self.fetch,[i for i in indices if i not in chunks]):chunks[i]=raw
        raw=b''.join(chunks[i] for i in indices)
        result=raw[self.position-indices[0]*self.block_size:self.position-indices[0]*self.block_size+end-self.position]
        self.position=end
        for i in indices:self.cache[i]=chunks[i];self.cache.move_to_end(i)
        while len(self.cache)>8:self.cache.popitem(last=False)
        return result
    def close(self):
        self.pool.shutdown(wait=True);super().close()


def recover(shards_path, output, endpoint, classes, parallel_ranges=0):
    from huggingface_hub import HfFileSystem
    import pyarrow.parquet as pq
    plan=json.loads(Path(shards_path).read_text());root=Path(output);root.mkdir(parents=True,exist_ok=True)
    manifest=root/'source-manifest.jsonl'
    rows={r['id']:r for r in map(json.loads,manifest.read_text().splitlines())} if manifest.exists() else {}
    fs=HfFileSystem(endpoint=endpoint)
    wanted=set(classes);shards={s['file']:s for c in wanted for s in plan['classes'][c]}
    for name,shard in sorted(shards.items()):
        print(json.dumps({'shard':shard['shard'],'status':'opening'}),flush=True)
        for attempt in range(4):
            try:
                url=endpoint+'/datasets/gmongaras/Imagenet21K/resolve/'+plan['revision']+'/data/'+name.rsplit('/',1)[1]
                stream=ParallelRangeFile(url,parallel_ranges) if parallel_ranges else fs.open(name,'rb',block_size=4*1024*1024)
                with stream:
                    parquet=pq.ParquetFile(stream)
                    for group in range(parquet.num_row_groups):
                        meta=parquet.metadata.row_group(group)
                        stats=next(meta.column(k).statistics for k in range(meta.num_columns) if meta.column(k).path_in_schema=='id')
                        low,high=stats.min.split('_')[0],stats.max.split('_')[0]
                        if not any(low<=c<=high for c in wanted):continue
                        ids=parquet.read_row_group(group,columns=['id']).column('id').to_pylist()
                        needed={i for i in ids if i.split('_')[0] in wanted and i not in rows}
                        if not needed:continue
                        print(json.dumps({'shard':shard['shard'],'group':group,'status':'fetching','images':len(needed)}),flush=True)
                        records=parquet.read_row_group(group,columns=['id','image']).to_pylist()
                        for record in records:
                            if record['id'] not in needed:continue
                            identity=record['id'];wnid=identity.split('_')[0];raw=record['image']
                            if isinstance(raw,dict):raw=raw['bytes']
                            with Image.open(io.BytesIO(raw)) as image:
                                image=ImageOps.exif_transpose(image).convert('RGB')
                                pixels=hashlib.sha256(str(image.size).encode()+image.tobytes()).hexdigest()
                                width,height=image.size
                            relative=f'{wnid}/{identity}.png';target=root/relative;target.parent.mkdir(exist_ok=True)
                            target.write_bytes(raw)
                            row={'id':identity,'wnid':wnid,'file_path':relative,'sha256':hashlib.sha256(raw).hexdigest(),
                                 'pixel_sha256':pixels,'width':width,'height':height,'source':name,
                                 'source_revision':plan['revision'],'row_group':group}
                            with manifest.open('a') as file:file.write(json.dumps(row,ensure_ascii=False)+'\n')
                            rows[identity]=row
                        print(json.dumps({'shard':shard['shard'],'group':group,'saved':len(needed),'total':len(rows)}),flush=True)
                break
            except Exception as error:
                print(json.dumps({'shard':shard['shard'],'attempt':attempt+1,'error':str(error)[:250]}),flush=True)
                if attempt==3:raise
                time.sleep(2)
    counts={c:sum(r['wnid']==c for r in rows.values()) for c in sorted(wanted)}
    (root/'download-report.json').write_text(json.dumps({'revision':plan['revision'],'counts':counts,'images':len(rows)},indent=2)+'\n')
    print(counts,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--shards',required=True);p.add_argument('--output',required=True)
    p.add_argument('--endpoint',default='https://huggingface.co');p.add_argument('--classes',nargs='+',required=True)
    p.add_argument('--parallel-ranges',type=int,default=0)
    a=p.parse_args();recover(a.shards,a.output,a.endpoint,a.classes,a.parallel_ranges)
