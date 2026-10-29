import argparse
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from tts_qwen.env.appworld_qwen_defaults import configured, repair_profile, _REQUIRED, _DEFAULTS


def runner():
    spec = importlib.util.spec_from_file_location('appworld_default_runner', Path(__file__).resolve().parents[1]/'experiments/run_qwen.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def args(**values):
    return SimpleNamespace(**dict(dict(agent='llm_wm', llm_agent='Qwen3.5-35B-A3B',
        appworld_qwen_repair='v3', appworld_wm_tool_mode='python', predict_all_actions=True,
        success_score='simulator_success', wm_temperature=0.6), **values))


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    for key in set(_REQUIRED)|set(_DEFAULTS):
        monkeypatch.delenv(key, raising=False)


@pytest.mark.parametrize('method,options,expected',[
    ('ours', {}, 'v3'), ('b2', {}, 'legacy'), ('b1', {}, 'legacy'),
    ('agentworld', {}, 'legacy'), ('ours', {'agent':'llm'}, 'legacy'),
    ('ours', {'llm_agent':'gpt-5.4'}, 'legacy'),
    ('ours', {'appworld_qwen_repair':'legacy'}, 'legacy'),
    ('ours', {'appworld_wm_tool_mode':'text'}, 'legacy'),
    ('ours', {'predict_all_actions':False,'_predict_all_actions_explicit':True}, 'legacy'),
])
def test_profile(method, options, expected):
    assert repair_profile(args(**options),method)==expected


def test_context_restores_environment_and_respects_explicit_options(monkeypatch):
    monkeypatch.setenv('TTS_APPWORLD_WM_HARD_TIMEOUT_S','2400')
    a=args(_wm_temperature_explicit=True,wm_temperature=.2)
    before=dict(os.environ)
    with pytest.raises(RuntimeError):
        with configured(a,'ours') as effective:
            assert all(os.environ[k]==v for k,v in _REQUIRED.items())
            assert effective.wm_temperature==.2
            assert effective.success_score=='simulator_confidence_guarded'
            assert os.environ['TTS_APPWORLD_VERIFIER_REFERENCE']=='0'
            assert os.environ['TTS_APPWORLD_WM_HARD_TIMEOUT_S']=='2400'
            raise RuntimeError('test')
    assert dict(os.environ)==before
    assert a.success_score=='simulator_success'


def test_mixed_cells_keep_profiles_and_caches_separate(monkeypatch):
    run=runner();seen=[]
    def fake(payload):
        env,spec,cells,a,usable=payload
        seen.append((cells,a,dict(os.environ)))
        return []
    monkeypatch.setattr(run,'_run_task_cells_unrepaired',fake)
    cells=[('b2',1,None,None),('ours',1,None,None),('b2',2,None,None),('ours',2,None,None)]
    before=dict(os.environ)
    run._run_task_cells(('appworld',None,cells,args(),False))
    assert len(seen)==2
    assert [c[0] for c in seen[0][0]]==['b2','b2']
    assert seen[0][2]==before
    assert seen[0][1].success_score=='simulator_success'
    assert [c[0] for c in seen[1][0]]==['ours','ours']
    assert seen[1][1]._task_max_n==2
    assert seen[1][2]['TTS_APPWORLD_OURS_QWEN_COMPLETE_CHILDREN']=='1'
    assert dict(os.environ)==before


def test_historical_validated_records_resume_and_legacy_is_rejected():
    import json
    run=runner();p=argparse.ArgumentParser()
    file=Path(__file__).resolve().parents[1]/'experiments/appworld-evidence-repair-20260914/closure_validation12/records.jsonl'
    if not file.exists():pytest.skip('local historical evaluation artifact')
    records=[json.loads(line) for line in file.read_text().splitlines()]
    with configured(args(),'ours') as effective:
        run._validate_appworld_resume(records,effective,p,{'ours'})
        legacy=[{**r,'appworld_qwen_evidence_guard':False,'appworld_qwen_complete_children':False} for r in records]
        with pytest.raises(SystemExit):run._validate_appworld_resume(legacy,effective,p,{'ours'})
    with pytest.raises(SystemExit):run._validate_appworld_resume(records,args(appworld_qwen_repair='legacy'),p,{'ours'})


def test_local_world_model_endpoint_is_inherited_and_overrides_are_kept(monkeypatch):
    monkeypatch.setenv('OPENAI_BASE_URL','http://localhost:9001/v1')
    with configured(args(local=True),'ours') as effective:
        assert effective.wm_base_url=='http://localhost:9001/v1'
        assert effective.wm_model=='Qwen3.5-35B-A3B'
    with configured(args(wm_base_url='http://explicit/v1',wm_model='explicit'),'ours') as effective:
        assert effective.wm_base_url=='http://explicit/v1'
        assert effective.wm_model=='explicit'
