"""Default selection, mixed-method isolation and resume compatibility for OfficeBench V2."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

def runner():
    path=Path(__file__).resolve().parents[1]/'experiments/run_qwen.py'
    spec=importlib.util.spec_from_file_location('run_repair_test',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

def args(**overrides):
    return SimpleNamespace(**dict(dict(agent='llm_wm',officebench_wm_repair='v2',
        predict_all_actions=False,_predict_all_actions_explicit=False,
        officebench_wm_tool_mode='python'),**overrides))

@pytest.mark.parametrize('method,options,expected',[
    ('ours',{},'v2'),('b2',{},'legacy'),('agentworld',{},'legacy'),
    ('ours',{'agent':'llm'},'legacy'),('ours',{'officebench_wm_repair':'legacy'},'legacy'),
    ('ours',{'_predict_all_actions_explicit':True},'legacy'),
    ('ours',{'_predict_all_actions_explicit':True,'predict_all_actions':True},'v2'),
    ('ours',{'officebench_wm_tool_mode':'text'},'legacy'),
])
def test_effective_selection(method,options,expected):
    assert runner()._officebench_repair_profile(args(**options),method)==expected

def test_mixed_methods_and_multiple_n_restore_adapter(monkeypatch):
    run=runner()
    from tts_qwen.env import officebench_lowlevel as low,officebench_wm_predict as wm
    original=(low._DRIVER_PATH,wm.OfficeBenchWMPredictor._predict)
    observed=[];drivers=[]
    def cells(payload):
        env,spec,group,options,usable=payload
        repaired=wm.OfficeBenchWMPredictor._predict is not original[1]
        observed.append((group,options.predict_all_actions,repaired))
        if repaired:
            assert Path(low._DRIVER_PATH).exists()
            drivers.append(Path(low._DRIVER_PATH))
        return []
    monkeypatch.setattr(run,'_run_task_cells_unrepaired',cells)
    selected=args()
    run._run_task_cells(('officebench',None,[('b2',1,None,None),('ours',1,None,None),('ours',2,None,None),('b2',2,None,None)],selected,False))
    assert [(len(g),all_reads,repaired) for g,all_reads,repaired in observed]==[(2,False,False),(2,True,True)]
    run._run_task_cells(('officebench',None,[('b2',1,None,None)],selected,False))
    assert observed[-1][1:]==(False,False)
    assert not selected.predict_all_actions
    assert (low._DRIVER_PATH,wm.OfficeBenchWMPredictor._predict)==original
    assert all(not p.exists() for p in drivers)

def test_other_environments_bypass_adapter(monkeypatch):
    run=runner();payload=('appworld',None,[('ours',1,None,None)],args(),False)
    seen=[]
    monkeypatch.setattr(run,'_run_task_cells_unrepaired',lambda p:seen.append(p) or [])
    run._run_task_cells(payload)
    assert seen==[payload] and seen[0] is payload

def test_restore_on_failure(monkeypatch):
    run=runner()
    from tts_qwen.env import officebench_lowlevel as low,officebench_wm_predict as wm
    original=(low._DRIVER_PATH,wm.OfficeBenchWMPredictor._predict)
    def fail(payload):raise RuntimeError('test failure')
    monkeypatch.setattr(run,'_run_task_cells_unrepaired',fail)
    with pytest.raises(RuntimeError):run._run_task_cells(('officebench',None,[('ours',1,None,None)],args(),False))
    assert (low._DRIVER_PATH,wm.OfficeBenchWMPredictor._predict)==original

def test_resume_rejects_legacy_and_accepts_v2():
    run=runner();legacy={'method':'ours','predict_all_actions':False}
    repaired={'method':'ours','predict_all_actions':True,'officebench_wm_repair':'v2'}
    b2={'method':'b2','predict_all_actions':False,'officebench_wm_repair':'legacy'}
    assert run._officebench_resume_mismatches([legacy],args())==[legacy]
    assert not run._officebench_resume_mismatches([repaired,b2],args())
    assert not run._officebench_resume_mismatches([legacy],args(officebench_wm_repair='legacy'))
    assert not run._officebench_resume_mismatches([{'method':'ours','predict_all_actions':True}],args(),historical_profile='v2')
    assert run._officebench_resume_mismatches([repaired],args(officebench_wm_repair='legacy'))
    assert not run._officebench_resume_mismatches([legacy,b2],args(methods='b2'))

def test_promoted_reader_preserves_full_input_and_default_reader():
    from unittest.mock import patch
    from tts_qwen.env import officebench_lowlevel as low,officebench_repair_v2 as repair
    with repair.install():
        spec=importlib.util.spec_from_file_location('promoted_driver_test',low._DRIVER_PATH)
        driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver)
        with patch.object(driver,'_ensure_path'),patch.object(driver,'_read_one_file',return_value='x'*2100+'TAIL'):
            request={'files':[{'path':'/testbed/data/a'}],'extras':False}
            assert len(driver.handle_read_state(request)['state']['/testbed/data/a'])==2000
            assert driver.handle_read_state({**request,'wm_full_read':True})['state']['/testbed/data/a'].endswith('TAIL')
