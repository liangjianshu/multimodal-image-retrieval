"""Run the unchanged 12-image showcase with validation-selected study weights.

Qwen intent routing remains a common frozen upstream model. The new Qwen
category model and the new embedding model are evaluated as retrieval changes.
"""
import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from mmretrieval.cli import load_config
from mmretrieval.data import read_jsonl, image_path, write_jsonl
from mmretrieval.models import Qwen, Encoders, extract_json
from mmretrieval.gemma import GemmaEncoder
from mmretrieval.retrieval import Index
from mmretrieval.image_index import ImageIndex
from mmretrieval.routes import RoutedRetriever
from mmretrieval.intent import ROUTE_INTENT_PROMPT
from examples.showcase.run import audit_adapter, write_json
from examples.training.report_restored_study import collect
from examples.training.build_restored_study import CATEGORY_PROMPT
from mmretrieval.historical_prompts import DESCRIPTION_PROMPT


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def release_device(torch, device):
    gc.collect()
    if device.startswith('npu'):
        torch.npu.empty_cache()
    elif device.startswith('cuda'):
        torch.cuda.empty_cache()


def describe_selected(model, path):
    """Use the actual new training prompt for category prediction."""
    raw_categories = model.generate(CATEGORY_PROMPT, path)
    categories = extract_json(raw_categories)
    if not isinstance(categories, list) or not categories or any(not isinstance(c, str) or not c.strip() for c in categories):
        raise ValueError('Selected Qwen returned invalid categories')
    description = model.generate(DESCRIPTION_PROMPT, path).strip()
    if not description:
        raise ValueError('Selected Qwen returned empty description')
    return {'categories': categories, 'description': description, 'raw_categories': raw_categories}


def run(args):
    import torch
    torch.set_num_threads(4); torch.manual_seed(42)
    study = Path(args.study_root); models = Path(args.model_root); out = Path(args.output)
    # Refuse pilot/partial selections and provenance inconsistencies before loading models.
    formal = collect(study); selected = formal['selection']['selected']
    public = load_config(ROOT / 'configs/route-a-qwen-semantic.json')
    config = {**public, 'device': args.device, 'vl_model': str(models / 'qwen-vl-7b'),
              'text_encoder': str(models / 'taiyi'), 'image_encoder': str(models / 'clip'),
              'embedding_model_path': str(models / 'embeddinggemma-2'), 'adapter': args.intent_adapter}
    for family, filename in [('qwen', 'adapter_model.safetensors'), ('embedding', 'model.safetensors')]:
        if digest(Path(selected[family]['path']) / filename) != selected[family]['weight_sha256']:
            raise ValueError(f'Selected {family} weights changed')
    gallery = ROOT / 'examples/showcase'
    sources = json.loads((gallery / 'sources.json').read_text())
    rows = read_jsonl(gallery / 'catalog.jsonl')
    for source in sources:
        if digest(gallery / source['file_path']) != source['sha256']:
            raise ValueError('Showcase image changed')
    if {r['id'] for r in rows} != {r['id'] for r in sources}:
        raise ValueError('Showcase gallery differs from verified sources')
    queries = json.loads((gallery / 'queries-hybrid.json').read_text())
    provenance = {'selection_sha256': digest(study / 'selection.json'),
        'gallery_sources_sha256': digest(gallery / 'sources.json'),
        'queries_sha256': digest(gallery / 'queries-hybrid.json'),
        'intent_adapter_sha256': digest(Path(args.intent_adapter) / 'adapter_model.safetensors'),
        'intent_prompt_sha256': hashlib.sha256(ROUTE_INTENT_PROMPT.encode()).hexdigest(),
        'category_prompt_sha256': hashlib.sha256(CATEGORY_PROMPT.encode()).hexdigest(),
        'description_prompt_sha256': hashlib.sha256(DESCRIPTION_PROMPT.encode()).hexdigest(),
        'qwen_weight_sha256': selected['qwen']['weight_sha256'],
        'embedding_weight_sha256': selected['embedding']['weight_sha256']}
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'provenance.json').exists() and json.loads((out / 'provenance.json').read_text()) != provenance:
        raise ValueError('Existing showcase belongs to different inputs or selected weights')
    write_json(out / 'provenance.json', provenance)
    if (out / 'shared-intents.json').exists():
        shared = json.loads((out / 'shared-intents.json').read_text())
    else:
        router = Qwen(config, vision=True); intent_audit = audit_adapter(router, args.intent_adapter)
        intent_audit['base_model'] = public['vl_model']
        intents, seconds = {}, {}
        for query in queries:
            start = time.perf_counter()
            intents[query['id']] = router.intent(query.get('text') or '',
                image_path(gallery, query['image']) if query.get('image') else None, parse_image_intent=True)
            seconds[query['id']] = time.perf_counter() - start
            print('intent', query['id'], intents[query['id']], flush=True)
        shared = {'intents': intents, 'seconds': seconds, 'adapter_verification': intent_audit}
        write_json(out / 'shared-intents.json', shared)
        del router; release_device(torch, args.device)
    for query in queries:
        intent = shared['intents'][query['id']]
        if intent['needs_retrieval'] != query.get('expected_retrieval', True):
            raise ValueError(f'Intent regression: {query["id"]}')
        if 'expected_top_k' in query and intent['top_k'] != query['expected_top_k']:
            raise ValueError(f'Quantity regression: {query["id"]}')
    category_config = {**config, 'adapter': selected['qwen']['path']}
    if (out / 'catalog.described.jsonl').exists() and (out / 'category-adapter-verification.json').exists():
        rows = read_jsonl(out / 'catalog.described.jsonl')
        category_audit = json.loads((out / 'category-adapter-verification.json').read_text())
    else:
        model = Qwen(category_config, vision=True)
        category_audit = audit_adapter(model, selected['qwen']['path']); category_audit['base_model'] = public['vl_model']
        for row in rows:
            prediction = describe_selected(model, image_path(gallery, row['file_path']))
            row.update(retrieval_categories=prediction['categories'], retrieval_description=prediction['description'],
                raw_category_response=prediction['raw_categories'],
                retrieval_source={'model': public['vl_model'], 'adapter_sha256': category_audit['sha256']})
            print('description', row['id'], prediction, flush=True)
        write_jsonl(out / 'catalog.described.jsonl', rows)
        write_json(out / 'category-adapter-verification.json', category_audit)
        del model; release_device(torch, args.device)
    if category_audit['sha256'] != selected['qwen']['weight_sha256']:
        raise ValueError('Description adapter differs from selected Qwen')
    description_sha = digest(out / 'catalog.described.jsonl')

    def replay(folder, backend, encoder, index, state, public_config):
        pipeline = RoutedRetriever(backend, index, None, lambda: encoder, config)
        records = []
        for query in queries:
            reference = image_path(gallery, query['image']) if query.get('image') else None
            excluded = [r['id'] for r in rows if reference and image_path(gallery, r['file_path']) == reference]
            result = pipeline.search(query.get('text'), reference, top_k=3, exclude_ids=excluded,
                                     parsed_intent=shared['intents'][query['id']])
            result['timing']['cached_intent_original_seconds'] = shared['seconds'][query['id']]
            records.append({**query, **result, 'image': query.get('image'),
                            'mode': 'semantic' if backend == 'qwen-semantic' else 'embedding'})
            print(folder, query['id'], [(r['id'], round(r['score'], 4)) for r in result['results']], flush=True)
        report = {'created_at': datetime.now(timezone.utc).isoformat(), 'backend': backend,
            'gallery_size': len(rows), 'config': {**public_config, 'backend': backend}, 'training_state': state,
            'adapter_verification': shared['adapter_verification'], 'intent_adapter_verification': shared['adapter_verification'],
            'category_adapter_verification': category_audit,
            'encoder': getattr(encoder, 'fingerprint', {'text_encoder': public['text_encoder'], 'image_encoder': public['image_encoder']}),
            'shared_intents': True, 'intent_prompt_sha256': provenance['intent_prompt_sha256'],
            'description_cache_sha256': description_sha, 'study_provenance': provenance,
            'notice': 'Validation-selected retrieval models; common frozen Qwen intent model. Real inference on 12 CC0 images, not a benchmark. No manual reranking.',
            'queries': records}
        index.save(out / folder / 'index'); write_json(out / folder / 'results.json', report)
    encoder = Encoders(config)
    index = Index(rows, [encoder.text(', '.join(r['retrieval_categories'])) for r in rows],
        [encoder.text(r['retrieval_description']) for r in rows],
        [encoder.image(image_path(gallery, r['file_path'])) for r in rows],
        {'version': 1, 'text_encoder': public['text_encoder'], 'image_encoder': public['image_encoder']})
    replay('route-a', 'qwen-semantic', encoder, index, 'study-finetuned',
           {**public, 'adapter': 'models/study-qwen-best'})
    del encoder, index; release_device(torch, args.device)
    for folder, cfg, state in [('route-b-baseline', config, 'pretrained-baseline'),
            ('route-b', {**load_config(Path(selected['embedding']['path']) / 'inference-config.json'), 'device': args.device}, 'study-finetuned')]:
        encoder = GemmaEncoder(cfg)
        index = ImageIndex(rows, encoder.images([image_path(gallery, r['file_path']) for r in rows]), {**encoder.fingerprint, 'version': 1})
        pub = {**public, **{k: cfg[k] for k in ('embedding_model', 'embedding_revision', 'embedding_weights_sha256')}}
        replay(folder, 'qwen-embedding', encoder, index, state, pub)
        del encoder, index; release_device(torch, args.device)
    write_json(out / 'showcase-complete.json', provenance)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('study-root', 'model-root', 'intent-adapter', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--device', default='npu:0')
    run(parser.parse_args())
