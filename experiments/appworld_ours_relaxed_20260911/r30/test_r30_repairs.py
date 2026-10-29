import json
import pytest
from tts.llm.python_tool import PurePythonTool
from tts.env.appworld_ours_model_driven import AppWorldOursModelDrivenPredictor as Experimental
from tts.env.appworld_ours_wm import AppWorldOursWMPredictor as Original

@pytest.fixture
def tool():
    t = PurePythonTool()
    yield t
    t.close()

@pytest.mark.parametrize('code,expected', [
    ('print_json([true, false, null])', [True, False, None]),
    ('print_json(json.loads("[1,2]"))', [1,2]),
    ('try:\n json.loads("invalid")\nexcept json.JSONDecodeError:\n print_json("handled")', 'handled'),
    ('from json import JSONDecodeError\ntry:\n json.loads("invalid")\nexcept JSONDecodeError:\n print_json("handled")', 'handled'),
    ('print_json(type(data).__name__)', 'dict'),
    ('print_json(1855358211218584)', 1855358211218584),
    ('x=[1,2]; del x[0]; assert x == [2]; print_json(x)', [2]),
])
def test_compatibility_is_per_call(tool, code, expected):
    assert tool.run(code, {}).error
    r = tool.run(code, {}, appworld_compat=True)
    assert r.error is None, r.error
    assert json.loads(r.output) == expected
    assert tool.run(code, {}).error  # No shared-worker leakage into default mode.

@pytest.mark.parametrize('code', [
    'import os\nprint(os.environ)',
    'print(data.__class__)',
    'print(type(data).__mro__)',
    'print(open("/etc/passwd").read())',
    'import socket\nprint(socket.socket())',
    'print(eval("1+1"))',
])
def test_compatibility_keeps_host_access_blocked(tool, code):
    assert tool.run(code, {}, appworld_compat=True).error


def test_copy_cost_does_not_consume_generated_line_budget(tool):
    code = 'from copy import deepcopy\nx=deepcopy(data)\nx["items"][0]["x"]=99\nprint_json([x["items"][0]["x"],data["items"][0]["x"]])'
    context = {'items':[{'x':i} for i in range(500)]}
    assert tool.run(code, context, max_steps=100).error
    r = tool.run(code, context, max_steps=100, appworld_compat=True)
    assert r.error is None, r.error
    assert json.loads(r.output)==[99,0]
    assert context['items'][0]['x']==0
    assert tool.run('while True:\n pass', {}, max_steps=100, appworld_compat=True).error


def test_failed_assert_does_not_produce_a_verdict(tool):
    r=tool.run('assert False\nprint("VERDICT: SUCCESS")', {}, appworld_compat=True)
    assert r.error and 'AssertionError' in r.error
    assert not r.output


def test_runtime_mode_only_enabled_for_experimental_predictor():
    assert Experimental(None)._python_tool_limits()['appworld_compat'] is True
    assert 'appworld_compat' not in Original(None)._python_tool_limits()

@pytest.mark.parametrize('recipient,private,success', [
    ('right@example.test',True,True),
    ('wrong@example.test',True,False),
    ('right@example.test',False,False),
])
def test_state_then_verdict_uses_actual_candidate_without_gold_leak(monkeypatch,recipient,private,success):
    import tts.env.appworld_ours_model_driven as m
    monkeypatch.setattr(m,'_api_doc',lambda *a:{'description':'Create the requested record'})
    p=Experimental(None);p.start_rollout();p.program_only=True
    p.write_history=[('venmo','create_payment_request',{'user_email':recipient,'private':private},{'message':'success'})]
    prompts=[]
    class Chat:
        def complete(self,messages,**kwargs):
            prompts.append(str(messages))
            if len(prompts)==1:
                return '''PYTHON_FINAL:
assert 'gold' not in data and 'checklist' not in data
args=data['writes'][0]['args']
print_json({'changes':{'added':1},'final_state':{'email':args['user_email'],'privacy_type':'privately' if args['private'] else 'publicly'},'unresolved':[]})'''
            return '''PYTHON_FINAL:
actual=data['predicted_state']['final_state']
expected=data['gold']['sections']['expected']
print_json({'actual':actual,'expected':expected})
print('VERDICT: SUCCESS' if actual==expected else 'VERDICT: FAIL')'''
    p._chat=Chat()
    checklist='GOLD VALUES (expected): '+json.dumps({'email':'right@example.test','privacy_type':'privately'})
    assert p._reference_verdict('Create the payment request',checklist,0) is success
    assert len(prompts)==2
    assert 'GOLD VALUES (expected)' not in prompts[0]
    assert 'GOLD VALUES (expected)' in prompts[1]
    assert p.n_python_calls==2
