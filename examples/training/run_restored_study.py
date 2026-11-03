"""Resumable single-device experiment orchestration with frozen validation selection."""
import argparse
import csv
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def write(path,data):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')


def run(args):
    import fcntl
    study=Path(args.study_root).resolve();repo=Path(__file__).resolve().parents[2]
    runtime=Path(args.runtime_root).resolve();legacy=Path(args.legacy_root).resolve();data=study/'data';output=data/'study-v1'
    study.mkdir(parents=True,exist_ok=True);statefile=study/'study-status.json'
    lock=(study/'driver.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    code={str(p.relative_to(repo)):hashlib.sha256(p.read_bytes()).hexdigest() for folder in ['mmretrieval','examples/training'] for p in (repo/folder).glob('*.py')}
    write(study/'implementation-manifest.json',code)
    state={'status':'preparing','started_at':datetime.now(timezone.utc).isoformat(),'commands':[],'args':vars(args)}
    def command(name,argv):
        logfile=study/'logs'/f'{name}.log';logfile.parent.mkdir(exist_ok=True)
        entry={'name':name,'argv':list(map(str,argv)),'started_at':datetime.now(timezone.utc).isoformat(),'log':str(logfile)}
        state['current']=entry;state['status']='running';write(statefile,state);print(json.dumps(entry),flush=True)
        started=time.monotonic()
        with logfile.open('a') as log:result=subprocess.run(entry['argv'],cwd=repo,stdout=log,stderr=subprocess.STDOUT)
        entry.update(returncode=result.returncode,seconds=time.monotonic()-started);state['commands'].append(entry);write(statefile,state)
        if result.returncode:state['status']='failed';write(statefile,state);raise RuntimeError(f'{name} failed; inspect {logfile}')
    python=sys.executable;script=lambda name:str(repo/'examples/training'/name)
    while not (data/'hf-class-pool/download-report.json').exists():
        state['status']='waiting-for-exact-image-recovery';write(statefile,state)
        if not args.wait_for_download:raise RuntimeError('Image recovery is still running')
        time.sleep(30)
    if not (output/'data-report.json').exists():
        command('restore',[python,script('restore_original_gallery.py'),'--manifest',legacy/'project_one/imagenet_json.json',
             '--archives',legacy/'project_three/dataset/_temp','--public-pool',data/'hf-class-pool','--output',data/'original-eval'])
        recovery=json.loads((data/'original-eval/recovery-report.json').read_text())
        if recovery['restored']!=2000:raise RuntimeError('Exact historical gallery is incomplete')
        command('build-data',[python,script('build_restored_study.py'),'--test-root',data/'original-eval',
             '--archives',legacy/'project_three/dataset/_temp','--public-pool',data/'hf-class-pool',
             '--output',output,'--image-root',data,'--train-per-class','300','--val-per-class','90'])
    if not (study/'legacy-overlap.json').exists():
        command('audit-old-training',[python,script('audit_legacy_overlap.py'),'--test-catalog',output/'test-catalog.jsonl',
             '--legacy-manifest',legacy/'project_three/dataset/finetune_dataset.json','--legacy-root',legacy/'project_three/dataset',
             '--output',study/'legacy-overlap.json'])
    embedding=json.loads((runtime/'gemma-runtime.json').read_text());qwen=json.loads((runtime/'hybrid-runtime.json').read_text())
    qwen.update(adapter=None,text_encoder=str(runtime/'models/taiyi'),image_encoder=str(runtime/'models/clip'))
    write(study/'embedding-config.json',embedding);write(study/'qwen-config.json',qwen)
    trial_records=[]
    def trial(kind,lr,temperature=.05,mode='llm-lora'):
        name=f'{kind}-lr{lr:g}'+(f'-t{temperature:g}' if kind=='embedding' else '-'+mode)
        target=study/'trials'/name;index=0
        while target.exists():
            if (target/'trial.json').exists():
                old=json.loads((target/'trial.json').read_text())
                if old.get('status')=='complete':trial_records.append(old);return old
            index+=1;target=study/'trials'/f'{name}-attempt{index+1}'
        argv=[python,script('run_'+kind+'_trial.py'),'--config',study/f'{kind}-config.json',
              '--train',output/('train-embedding.jsonl' if kind=='embedding' else 'train-qwen.jsonl'),
              '--val-catalog',output/'val-catalog.jsonl','--image-root',data,'--output',target,
              '--learning-rate',str(lr),'--epochs','.5','1','2','--batch-size',str(args.embedding_batch if kind=='embedding' else args.qwen_batch),
              '--grad-accum','4']
        if kind=='embedding':argv+=['--temperature',str(temperature)]
        else:argv+=['--mode',mode]
        command(name,argv);result=json.loads((target/'trial.json').read_text());trial_records.append(result)
        write(study/'trials-summary.json',trial_records);return result
    # All model and epoch choices are frozen before the first test evaluation.
    for lr in (1e-6,2e-6,5e-6):trial('embedding',lr)
    best_embedding=max(trial_records,key=lambda r:r['best']['selection_score'])
    trial('embedding',best_embedding['args']['learning_rate'],temperature=.1)
    for lr in (5e-6,1e-5,2e-5):trial('qwen',lr)
    qwentrials=[r for r in trial_records if 'mode' in r['args']]
    best_qwen=max(qwentrials,key=lambda r:r['best']['selection_score'])
    trial('qwen',best_qwen['args']['learning_rate'],mode='mlp-llm')
    embeddings=[r for r in trial_records if 'temperature' in r['args']];qwentrials=[r for r in trial_records if 'mode' in r['args']]
    selected={kind:max(records,key=lambda r:r['best']['selection_score'])['best'] for kind,records in [('embedding',embeddings),('qwen',qwentrials)]}
    selection={'frozen_before_test':True,'selected':selected,'selection_time':datetime.now(timezone.utc).isoformat(),
               'data_report_sha256':hashlib.sha256((output/'data-report.json').read_bytes()).hexdigest()}
    if (study/'selection.json').exists():
        old=json.loads((study/'selection.json').read_text())
        if old['selected']!=selected:raise RuntimeError('Selection changed after test started')
    else:write(study/'selection.json',selection)
    write(study/'trials-summary.json',trial_records)
    table=[]
    for record in trial_records:
        for checkpoint in record['checkpoints']:
            table.append({'model':'qwen' if 'mode' in record['args'] else 'embedding',
                          'learning_rate':record['args']['learning_rate'],'temperature':record['args'].get('temperature',''),
                          'mode':record['args'].get('mode','full-text-and-vision'),'epoch':checkpoint['actual_epoch'],
                          'validation_score':checkpoint['selection_score'],'checkpoint':checkpoint['path'],
                          'trial_seconds':record['seconds']})
    with (study/'hyperparameters.csv').open('w') as file:
        writer=csv.DictWriter(file,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    variants=[('embedding-base','embedding',study/'embedding-config.json',None),
              ('embedding-best','embedding',Path(selected['embedding']['path'])/'inference-config.json',None),
              ('qwen-base','qwen',study/'qwen-config.json',None),
              ('qwen-old','qwen',study/'qwen-config.json',legacy/'project_three/output_lora_llm'),
              ('qwen-best','qwen',study/'qwen-config.json',selected['qwen']['path'])]
    for name,backend,config,adapter in variants:
        argv=[python,script('evaluate_study.py'),'--backend',backend,'--config',config,'--val-catalog',output/'val-catalog.jsonl',
              '--test-catalog',output/'test-catalog.jsonl','--image-root',data,'--output',study/'evaluation'/name,
              '--overlap-audit',study/'legacy-overlap.json']
        if adapter:argv+=['--adapter',adapter]
        command('evaluate-'+name,argv)
    state['status']='experiments-complete-pending-report-and-demo';state['completed_at']=datetime.now(timezone.utc).isoformat();write(statefile,state)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--study-root',required=True);p.add_argument('--runtime-root',required=True)
    p.add_argument('--legacy-root',required=True);p.add_argument('--wait-for-download',action='store_true')
    p.add_argument('--embedding-batch',type=int,default=8);p.add_argument('--qwen-batch',type=int,default=4)
    run(p.parse_args())
