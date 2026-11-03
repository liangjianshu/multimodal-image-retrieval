from unittest.mock import Mock
import numpy as np
import pytest
from mmretrieval.routes import RoutedRetriever
from mmretrieval.image_index import ImageIndex
from mmretrieval.retrieval import Index
from mmretrieval.cli import load_config


def decision(needed=True):
    return dict(needs_retrieval=needed,target_category='杯子',description='花纹杯',constraints=[],top_k=2)


def test_b_uses_intent_but_does_not_filter_categories_or_replace_inputs():
    rows=[{'id':'a','retrieval_categories':['椅子']},{'id':'b','retrieval_categories':['杯子']},{'id':'c'}]
    index=ImageIndex(rows,[[1.,0.],[.8,.6],[0.,1.]],{'dimension':2})
    model,encoder=Mock(),Mock(); model.intent.return_value=decision()
    encoder.query.return_value=np.array([1.,0.]); encoder.fingerprint={'dimension':2}
    route=RoutedRetriever('qwen-embedding',index,model,lambda:encoder,load_config(None))
    r=route.search('找两张红杯子，不要蓝色','ref.jpg')
    assert [x['id'] for x in r['results']]==['a','b']
    assert r['candidate_count']==3
    encoder.query.assert_called_once_with('找两张红杯子，不要蓝色','ref.jpg')
    model.intent.assert_called_once_with('找两张红杯子，不要蓝色','ref.jpg',parse_image_intent=True)


@pytest.mark.parametrize('backend',['qwen-semantic','qwen-embedding'])
def test_both_routes_stop_nonretrieval_before_loading_encoder(backend):
    router=Mock();router.intent.return_value=decision(False)
    factory=Mock(side_effect=AssertionError('No retrieval for chat'))
    pipeline=RoutedRetriever(backend,Mock(),router,factory,load_config(None))
    r=pipeline.search('只描述这张图片','ref.jpg')
    assert r['status']=='not_retrieval' and r['results']==[]
    factory.assert_not_called()


def test_a_ranks_description_after_category_prefilter_not_image_similarity():
    rows=[{'id':'cup1'},{'id':'cup2'},{'id':'chair'}]
    index=Index(rows,[[1.,0.],[1.,0.],[0.,1.]],[[.8,.6],[1.,0.],[1.,0.]],[[1.,0.],[0.,1.],[1.,0.]],{})
    router,encoder=Mock(),Mock();router.intent.return_value=decision()
    encoder.text.return_value=np.array([1.,0.])
    r=RoutedRetriever('qwen-semantic',index,router,lambda:encoder,load_config(None)).search('杯子')
    assert [x['id'] for x in r['results']]==['cup2','cup1']
    assert r['candidate_count']==2


def test_default_is_semantic_route():
    assert load_config(None)['backend']=='qwen-semantic'


def test_triplet_validation_detects_renamed_identical_images(tmp_path):
    from PIL import Image
    from mmretrieval.embedding_training import validate_triplets
    Image.new('RGB',(16,16),'red').save(tmp_path/'a.png')
    (tmp_path/'b.png').write_bytes((tmp_path/'a.png').read_bytes())
    with pytest.raises(ValueError,match='identical'):
        validate_triplets([dict(id='x',text='red',positive='a.png',negative='b.png')],tmp_path)
