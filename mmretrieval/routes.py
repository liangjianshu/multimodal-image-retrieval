"""Two retrieval routes sharing Qwen intent parsing, with independent indexes."""
import json
import time
from pathlib import Path

import numpy as np

from .data import image_path, read_jsonl, validate_catalog, retrieval_metrics
from .gemma import GemmaEncoder, signature
from .image_index import ImageIndex
from .models import Qwen, Encoders
from .retrieval import Index, fuse_query, normalized, relative_cutoff


class RoutedRetriever:
    def __init__(self, route, index, router, encoder_factory, config):
        if route not in ('qwen-semantic', 'qwen-embedding'):
            raise ValueError('Unknown retrieval route')
        self.route, self.index, self.router = route, index, router
        self.factory, self.config, self.encoder = encoder_factory, config, None

    def search(self, text=None, image=None, *, top_k=5, exclude_ids=(), min_score=None, parsed_intent=None):
        if not (text and text.strip()) and image is None:
            raise ValueError('Provide text or a reference image')
        started = time.perf_counter()
        intent = parsed_intent if parsed_intent is not None else self.router.intent(text or '', image, parse_image_intent=True)
        timing = {'intent_seconds': time.perf_counter()-started, 'encoder_load_seconds': 0., 'encoding_seconds': 0., 'ranking_seconds': 0.}
        k = intent.get('top_k') if top_k is not None and intent.get('top_k') else top_k
        record = {'backend':self.route, 'intent':intent, 'search_text':text, 'top_k':k,
                  'exclude_ids':sorted(set(exclude_ids)), 'candidate_count':0, 'results':[],
                  'retrieval_executed':False, 'timing':timing, 'intent_reused':parsed_intent is not None}
        if not intent['needs_retrieval']:
            return {**record, 'status':'not_retrieval', 'elapsed_seconds':time.perf_counter()-started}
        if self.encoder is None:
            t = time.perf_counter(); self.encoder = self.factory()
            if self.route == 'qwen-embedding': self.index.verify(self.encoder.fingerprint)
            timing['encoder_load_seconds'] = time.perf_counter()-t
        t = time.perf_counter()
        candidates = np.array([i for i,r in enumerate(self.index.rows) if r['id'] not in exclude_ids], dtype=int)
        if self.route == 'qwen-embedding':
            vector = self.encoder.query(text, image)
            timing['encoding_seconds'] = time.perf_counter()-t
            t = time.perf_counter()
            results = self.index.search(vector, k, exclude_ids, min_score)
        else:
            query_text = text if text and text.strip() else intent['description']
            # The historical route represents gallery images as generated text.
            vector = fuse_query(self.encoder.text(query_text), self.encoder.image(image) if image else None,
                                self.config['image_weight'])
            category_vector = normalized(self.encoder.text(intent.get('target_category') or query_text))
            timing['encoding_seconds'] = time.perf_counter()-t
            t = time.perf_counter()
            if len(candidates):
                coarse = self.index.category[candidates] @ category_vector
                candidates = candidates[coarse >= relative_cutoff(float(coarse.max()), self.config['category_ratio'])]
            scores = self.index.description[candidates] @ vector
            order = sorted(range(len(candidates)), key=lambda j:(-float(scores[j]),self.index.rows[candidates[j]]['id']))
            results = [{**self.index.rows[candidates[j]], 'score':float(scores[j])} for j in order
                       if min_score is None or scores[j] >= min_score]
            if k is not None: results = results[:k]
            record['search_text'] = query_text
        timing['ranking_seconds'] = time.perf_counter()-t
        return {**record, 'status':'retrieved', 'retrieval_executed':True, 'results':results,
                'candidate_count':len(candidates), 'candidate_ids':[self.index.rows[i]['id'] for i in candidates],
                'elapsed_seconds':time.perf_counter()-started}


def execute_route(args, config):
    route = config['backend']
    semantic = route == 'qwen-semantic'
    if args.command == 'index':
        rows = validate_catalog(read_jsonl(args.manifest), args.image_root)
        if semantic:
            router = Qwen(config, vision=True)
            for row in rows:
                description = router.describe(image_path(args.image_root,row['file_path']))
                row.update(retrieval_categories=description['categories'], retrieval_description=description['description'],
                           retrieval_source={'model':config['vl_model'],'adapter':config['adapter']})
            encoder = Encoders(config)
            cats = [encoder.text(', '.join(r['retrieval_categories'])) for r in rows]
            descriptions = [encoder.text(r['retrieval_description']) for r in rows]
            images = [encoder.image(image_path(args.image_root,r['file_path'])) for r in rows]
            index = Index(rows,cats,descriptions,images,{'version':1,'backend':route,
                'text_encoder':config['text_encoder'],'image_encoder':config['image_encoder']})
        else:
            encoder = GemmaEncoder(config)
            index = ImageIndex(rows,encoder.images([image_path(args.image_root,r['file_path']) for r in rows]),
                               {**encoder.fingerprint,'version':1})
        index.save(args.output)
        return {'backend':route,'output':args.output,'images':len(rows)}
    if args.mode not in ('auto', 'semantic' if semantic else 'embedding'):
        raise ValueError('Mode does not match retrieval route')
    if semantic:
        from .cli import checked_index
        index = checked_index(args.index,config)
    else:
        index = ImageIndex.load(args.index); index.verify(signature(config))
    pipeline = RoutedRetriever(route,index,Qwen(config,vision=True),
        lambda: Encoders(config) if semantic else GemmaEncoder(config),config)
    def search(q):
        reference = image_path(args.image_root,q['image']) if q.get('image') else None
        excluded = set(q.get('exclude_ids',[]))
        if reference:
            excluded.update(r['id'] for r in index.rows if image_path(args.image_root,r['file_path']) == reference)
        return pipeline.search(q.get('text'),reference,top_k=None if args.all_matches else args.top_k,
                               exclude_ids=excluded,min_score=args.min_score)
    if args.command == 'search':
        return search({'text':args.text,'image':args.image,'exclude_ids':args.exclude_id})
    queries=read_jsonl(args.queries); details=[]; seen=set(); ids={r['id'] for r in index.rows}
    for q in queries:
        if not isinstance(q.get('id'),str) or q['id'] in seen: raise ValueError('Unique query IDs required')
        seen.add(q['id'])
        relevant=q.get('relevant_ids')
        if not isinstance(relevant,list) or not relevant or not set(relevant)<=ids: raise ValueError('Valid relevant_ids required')
        result=search(q)
        if set(result['exclude_ids']) & set(relevant): raise ValueError('Relevant image excluded')
        details.append({'id':q['id'],'intent':result['intent'],**retrieval_metrics([r['id'] for r in result['results']],relevant)})
    if not details: raise ValueError('Empty queries')
    report={'backend':route,'query_count':len(details),'queries':details,
            'macro':{k:float(np.mean([r[k] for r in details])) for k in ('precision','recall','f1')}}
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    Path(args.output).write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    return report
