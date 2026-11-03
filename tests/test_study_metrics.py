import numpy as np
import pytest
from mmretrieval.study_metrics import best_threshold,ranking_metrics,threshold_metrics,class_contrastive_loss

def test_calibration_does_not_split_equal_scores():
    result=best_threshold([.9,.8,.8,.1],[1,1,0,0])
    assert result['threshold']==.8
    assert result['tp']==2 and result['fp']==1
    assert result['f1']==.8

def test_image_self_exclusion_changes_denominator():
    result=ranking_metrics([1,.9,.2],[1,1,0],eligible=[1,2])
    assert result['relevant']==1
    assert result['average_precision']==1
    assert result['precision_at_5']==.2

def test_validation_threshold_applied_without_reoptimising():
    threshold=best_threshold([.9,.7,.2],[1,1,0])['threshold']
    test=threshold_metrics([.8,.6,.5],[0,1,1],threshold)
    assert test['f1']==0

def test_contrastive_loss_treats_same_class_as_positive():
    torch=pytest.importorskip('torch')
    query=torch.tensor([[1.,0.]],requires_grad=True)
    images=torch.tensor([[1.,0.],[.8,.2],[0.,1.]],requires_grad=True)
    loss=class_contrastive_loss(query,images,['cup'],['cup','cup','chair'],.1)
    wrong=class_contrastive_loss(query,images,['cup'],['cup','other','chair'],.1)
    assert loss<wrong
    loss.backward()
    assert torch.isfinite(query.grad).all() and torch.isfinite(images.grad).all()

def test_no_negative_batch_is_rejected():
    torch=pytest.importorskip('torch')
    with pytest.raises(ValueError,match='negative'):
        class_contrastive_loss(torch.ones(1,2),torch.ones(2,2),['a'],['a','a'])

def test_coarse_filter_misses_remain_in_recall_denominator():
    result=ranking_metrics([.9,float('-inf'),.8],[1,1,0])
    assert result['average_precision']==.5
    assert result['recall_at_5']==.5
    calibrated=best_threshold([.9,float('-inf'),.8],[1,1,0])
    assert calibrated['f1']==pytest.approx(2/3)

def test_empty_coarse_filter_returns_zero():
    assert ranking_metrics([float('-inf')],[1])['average_precision']==0
    assert best_threshold([float('-inf')],[1])['f1']==0

def test_image_f1_uses_validation_threshold_and_excludes_self():
    from mmretrieval.study_metrics import evaluate_vectors
    rows=[{'id':'a','label':'cup'},{'id':'b','label':'cup'},{'id':'c','label':'chair'},{'id':'d','label':'chair'}]
    vectors=np.array([[1.,0.],[.99,.01],[0.,1.],[.01,.99]])
    vectors=vectors/np.linalg.norm(vectors,axis=1,keepdims=True)
    queries=[{'id':'cup','label':'cup','text':'杯子'},{'id':'chair','label':'chair','text':'椅子'}]
    val=evaluate_vectors(rows,vectors,queries,np.eye(2))
    threshold=val['validation_thresholds']['__image_global__']
    assert threshold>.99
    assert val['metrics']['image']['primary_five_calibrated_f1']==1.
    test=evaluate_vectors(rows,np.array([[1.,0.],[0.,1.],[1.,0.],[0.,1.]]),queries,np.eye(2),val['validation_thresholds'])
    assert test['metrics']['image']['calibrated_global_threshold']==threshold
    assert test['metrics']['image']['primary_five_calibrated_f1']==0.
