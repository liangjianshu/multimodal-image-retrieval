"""Fixed-label retrieval metrics and validation-only threshold calibration."""
import numpy as np

PRIMARY=('chair','cup','bicycle','skateboard','oven')


def ranking_metrics(scores,relevant,eligible=None):
    scores=np.asarray(scores);relevant=np.asarray(relevant,dtype=bool)
    ids=np.arange(len(scores)) if eligible is None else np.asarray(eligible)
    count=int(relevant[ids].sum())
    ids=ids[np.isfinite(scores[ids])]
    order=ids[np.argsort(-scores[ids],kind='stable')];hits=relevant[order]
    precision=np.cumsum(hits)/np.arange(1,len(hits)+1)
    result={'average_precision':float((precision*hits).sum()/count) if count else 0.,'relevant':count}
    for k in (1,5,10,20):
        selected=hits[:k];result[f'precision_at_{k}']=float(selected.sum()/k)
        result[f'recall_at_{k}']=float(selected.sum()/count) if count else 0.
    return result


def threshold_metrics(scores,relevant,threshold):
    predicted=(np.asarray(scores)>=threshold)&np.isfinite(scores);truth=np.asarray(relevant,dtype=bool)
    tp=int((predicted&truth).sum());fp=int((predicted&~truth).sum());fn=int((~predicted&truth).sum())
    return {'threshold':float(threshold),'tp':tp,'fp':fp,'fn':fn,
            'precision':tp/(tp+fp) if tp+fp else 0.,'recall':tp/(tp+fn) if tp+fn else 0.,
            'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.}


def best_threshold(scores,relevant):
    scores=np.asarray(scores);relevant=np.asarray(relevant,dtype=bool)
    ids=np.flatnonzero(np.isfinite(scores))
    if not len(ids):return threshold_metrics(scores,relevant,1.)
    order=ids[np.argsort(-scores[ids],kind='stable')];s=scores[order];y=relevant[order]
    endpoints=np.r_[np.flatnonzero(s[:-1]!=s[1:]),len(s)-1]
    tp=np.cumsum(y)[endpoints];selected=endpoints+1;denominator=selected+relevant.sum()
    f1=np.divide(2*tp,denominator,out=np.zeros_like(tp,dtype=float),where=denominator!=0)
    best=int(np.argmax(f1));return threshold_metrics(scores,relevant,float(s[endpoints[best]]))


def summarise(values):
    keys=[k for k in values[0] if k.startswith(('average_precision','precision_at_','recall_at_'))]
    return {k:float(np.mean([r[k] for r in values])) for k in keys}


def evaluate_vectors(rows,vectors,text_queries,text_vectors,thresholds=None,
                     candidate_categories=None,image_query_vectors=None):
    labels=np.array([r['label'] for r in rows]);per_query=[];calibration={}
    for query,vector in zip(text_queries,text_vectors):
        scores=vectors@vector;truth=labels==query['label']
        if candidate_categories is not None:
            scores=np.where([query['label'] in cats for cats in candidate_categories],scores,-np.inf)
        result={**query,'mode':'text',**ranking_metrics(scores,truth)}
        if thresholds is None:
            calibrated=best_threshold(scores,truth);calibration[query['id']]=calibrated['threshold']
            result['validation_best_f1']=calibrated
        else:result['calibrated_f1']=threshold_metrics(scores,truth,thresholds[query['id']])
        per_query.append(result)
    image_scores=vectors@(vectors if image_query_vectors is None else image_query_vectors).T
    for i in range(len(rows)):
        image_scores[i,i]=-np.inf
        if candidate_categories is not None:
            query_classes=set(candidate_categories[i])
            image_scores[:,i]=np.where([bool(query_classes.intersection(cats)) for cats in candidate_categories],image_scores[:,i],-np.inf)
    if thresholds is None:
        primary_indices=[i for i,r in enumerate(rows) if r['label'] in PRIMARY]
        values=[];truths=[]
        for i in primary_indices:
            eligible=np.delete(np.arange(len(rows)),i)
            values.append(image_scores[eligible,i]);truths.append(labels[eligible]==rows[i]['label'])
        image_threshold=best_threshold(np.concatenate(values),np.concatenate(truths))['threshold']
        calibration['__image_global__']=image_threshold
    else:image_threshold=thresholds['__image_global__']
    for i,row in enumerate(rows):
        scores=image_scores[:,i];eligible=np.delete(np.arange(len(rows)),i)
        per_query.append({'id':row['id'],'label':row['label'],'mode':'image','type':row.get('type'),
                          **ranking_metrics(scores,labels==row['label'],eligible),
                          'calibrated_f1':threshold_metrics(scores[eligible],labels[eligible]==row['label'],image_threshold)})
    summary={}
    for mode in ('text','image'):
        values=[r for r in per_query if r['mode']==mode]
        classes=sorted(set(r['label'] for r in values))
        perclass={c:summarise([r for r in values if r['label']==c]) for c in classes}
        primary=[perclass[c] for c in PRIMARY if c in perclass]
        summary[mode]={'queries':len(values),'micro':summarise(values),'macro':summarise(list(perclass.values())),
                       'primary_five_macro':summarise(primary),'per_class':perclass}
        if mode=='image':
            summary[mode]['primary_five_calibrated_f1']=float(np.mean([r['calibrated_f1']['f1'] for r in values if r['label'] in PRIMARY]))
            summary[mode]['calibrated_global_threshold']=image_threshold
            summary[mode]['by_query_size']={kind:summarise([r for r in values if r.get('type')==kind])
                for kind in ('normal','fine') if any(r.get('type')==kind for r in values)}
        elif thresholds is not None:
            summary[mode]['primary_five_calibrated_f1']=float(np.mean([r['calibrated_f1']['f1'] for r in values if r['label'] in PRIMARY]))
    return {'gallery_size':len(rows),'metrics':summary,'per_query':per_query,'validation_thresholds':calibration}


def class_contrastive_loss(query,images,query_labels,image_labels,temperature=.05):
    """Multi-positive InfoNCE: same-class candidates never become false negatives.

This is class-level relevance, not instance/attribute discrimination.
"""
    import torch
    if temperature<=0:raise ValueError('temperature must be positive')
    logits=torch.nn.functional.normalize(query.float(),dim=-1)@torch.nn.functional.normalize(images.float(),dim=-1).T/temperature
    positives=torch.tensor([[a==b for b in image_labels] for a in query_labels],device=logits.device,dtype=torch.bool)
    if not positives.any(dim=1).all():raise ValueError('Every query needs at least one positive')
    if positives.all(dim=1).any():raise ValueError('Every query needs at least one negative')
    return (torch.logsumexp(logits,dim=1)-torch.logsumexp(logits.masked_fill(~positives,float('-inf')),dim=1)).mean()
