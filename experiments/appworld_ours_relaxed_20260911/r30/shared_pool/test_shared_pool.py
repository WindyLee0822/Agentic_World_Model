import pytest
from tts.harness import _wm_agent_rollouts
from tts.config import Config
from tts.trajectory import Trajectory,RewardInfo
from tts.budget import Budget
from tts.cache import ReadCache

class Env:
    task_id='fixed'
    def __init__(self,name='appworld'):
        self.env_name=name
        self.calls=[]
    def run_agent_rollout_wm(self,config,method,seed,budget,cache):
        self.calls.append(seed)
        if seed==config.seed+1:
            raise RuntimeError('exhausted candidate')
        return Trajectory(final_message=str(seed)),RewardInfo(reward=float(seed)),None

@pytest.mark.parametrize('enabled,name,method,expected',[(True,'appworld','ours',4),(False,'appworld','ours',5),(True,'officebench','ours',5),(True,'appworld','b1',5)])
def test_shared_prefix(monkeypatch,enabled,name,method,expected):
    monkeypatch.setenv('TTS_APPWORLD_FIXED_CANDIDATE_POOL','1' if enabled else '0')
    cfg=Config();cfg.simulator.predict_all_actions=True;cfg.solver.rollout_retries=0
    env=Env(name);cache={};pools=[]
    for n in [4,2,1]:
        pools.append(_wm_agent_rollouts(env,cfg,method,n,Budget(),ReadCache(),cache))
    assert len(env.calls)==expected
    for small in pools[1:]:
        for i,item in enumerate(small):
            if enabled and name=='appworld' and method=='ours' or i!=1:
                assert item.trajectory is pools[0][i].trajectory
                assert item.predicted is pools[0][i].predicted
    assert pools[0][1].trajectory.done is False
