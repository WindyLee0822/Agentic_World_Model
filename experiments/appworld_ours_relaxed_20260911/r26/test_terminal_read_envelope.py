import json
from types import SimpleNamespace
import pytest
from tts.env.appworld_ours_model_driven import AppWorldOursModelDrivenPredictor as Experimental
from tts.env.appworld_ours_wm import AppWorldOursWMPredictor as Original
from tts.env.appworld_wm import AppWorldWMQuery

QUERY={'app':'venmo','api':'show_sent_payment_requests','args':{}}

def test_terminal_query_preserves_inspection_and_default_behavior():
    output='Inspected current writes\n'+json.dumps({'queries':[QUERY],'facts':{'count':1}})
    queries,facts=Experimental._parse_python_read_payload(output)
    assert queries==[QUERY]
    assert facts=={'generated_stdout':'Inspected current writes','terminal_facts':{'count':1}}
    assert Original._parse_python_read_payload(output)==([],{'generated_stdout':output})

@pytest.mark.parametrize('output',[
    'prefix\n'+json.dumps({'queries':[QUERY]})+'\nsuffix',
    'prefix '+json.dumps({'queries':[QUERY]}),
])
def test_no_embedded_or_nonterminal_requests(output):
    assert Experimental._parse_python_read_payload(output)[0]==[]

def test_malformed_terminal_request_requires_repair():
    assert Experimental._parse_python_read_payload('prefix\n{"queries":"invalid"}') is None

@pytest.mark.parametrize('api,allowed',[('show_sent_payment_requests',True),('create_payment_request',False)])
def test_generated_mixed_output_uses_existing_read_gate(monkeypatch,api,allowed):
    monkeypatch.setenv('TTS_APPWORLD_WM_PYTHON_READ_TURNS','2')
    p=Experimental(None);p.start_rollout()
    native_calls=[]
    def native(app,api,**kwargs):
        native_calls.append((app,api))
        return {'count':7}
    gate=AppWorldWMQuery(native,SimpleNamespace(
        allows_grounding=lambda app,api:api=='show_sent_payment_requests',
        hydrate_read_args=lambda app,api,args:args))
    programs=iter([
        'PYTHON_READ:\nprint("Inspecting current writes")\nprint_json('+repr({'queries':[dict(QUERY,api=api)]})+')',
        'PYTHON_FINAL:\nprint_json({"observed":data["reads"][0]["result"]})',
    ])
    monkeypatch.setattr(p,'_complete',lambda *a,**k:next(programs))
    result=json.loads(p._python_read_loop('Predict','PREDICT: <JSON>',context={},
        read_fn=lambda app,api,args:gate.read_raw(app+'.'+api,args),max_tokens=2048,
        temperature=0,budgeted=True,python_required=True,output_kind='json'))
    if allowed:
        assert native_calls==[('venmo',api)]
        assert result['observed']=={'count':7}
    else:
        assert native_calls==[]
        assert 'write API' in result['observed']


def test_empty_terminal_queries_still_preserve_generated_inspection():
    assert Experimental._parse_python_read_payload('inspected\n{"queries":[]}') == (
        [], {'generated_stdout':'inspected','terminal_facts':None})
