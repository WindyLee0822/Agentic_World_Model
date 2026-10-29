import pytest
from tts.env.appworld_ours_model_driven import AppWorldOursModelDrivenPredictor as Experimental
from tts.env.appworld_ours_wm import AppWorldOursWMPredictor as Original

@pytest.mark.parametrize('mode,visible',[('write_with_rejection',True),('clean_read',False)])
def test_only_static_contract_is_added_to_prompt(monkeypatch,mode,visible):
    p=Experimental(None)
    context={'mode':mode,'api_documentation':{'description':'endpoint-contract-marker'},
             'success_schema':{'payment_request_id':1},'failure_schema':{'message':'string'},
             'args':{'access_token':'private-argument-marker'},
             'writes':[{'result':'private-write-marker'}]}
    seen=[]
    def parent(self,prompt,final_instr,**kwargs):
        seen.append((prompt,kwargs))
        return 'model-generated-output'
    monkeypatch.setattr(Original,'_predict',parent)
    assert p._predict('Predict','PREDICT',python_context=context)=='model-generated-output'
    prompt,kwargs=seen[0]
    assert ('endpoint-contract-marker' in prompt) is visible
    assert ('payment_request_id' in prompt) is visible
    assert 'private-argument-marker' not in prompt
    assert 'private-write-marker' not in prompt
    assert kwargs['python_context'] is context
