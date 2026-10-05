import json, os, time
from pathlib import Path
from tts.config import Config
from tts.harness import run_decision
from tts.env.real_appworld import AppWorldTaskEnv

def main():
    os.environ.update(AZURE_CLIENT_ID=os.environ['DEFAULT_IDENTITY_CLIENT_ID'], APPWORLD_ROOT='/tmp/appworld-strict-data', AZURE_OPENAI_ENDPOINT='https://ai-m365researchefficientaieus2212654625773.cognitiveservices.azure.com/', AZURE_OPENAI_API_VERSION='2024-12-01-preview', AZURE_OPENAI_DEPLOYMENT='DeepSeek-V4-Pro', AZURE_AUTH_MODE='default', TTS_APPWORLD_PREDICT_ALL_PROGRAM_ONLY='1', TTS_APPWORLD_PREGROUND_CLEAN_READS='0', TTS_APPWORLD_DETERMINISTIC_GROUNDED_CLEAN_READS='0', TTS_APPWORLD_AGENT_REFERENCE='0', TTS_APPWORLD_INSTRUCTION_PREFLIGHTS='0')
    cfg=Config(method='ours',env='appworld',N=2)
    cfg.solver.agent='llm_wm'
    cfg.solver.llm_agent='DeepSeek-V4-Pro'
    cfg.solver.temperature=1
    cfg.solver.max_steps=40
    started=time.monotonic()
    env=AppWorldTaskEnv(task_id='024c982_1')
    outcome=run_decision(env,cfg)
    result=outcome.record.to_dict()
    result['candidate_setup_prediction']=[r.predicted.metadata if r.predicted else None for r in outcome.rollouts]
    result['candidate_grounded_success']=[r.grounded.success if r.grounded else None for r in outcome.rollouts]
    result['elapsed_seconds']=time.monotonic()-started
    Path('strict-n2-result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)
    assert result['n_commits']==1 and result['simulator_writes']==0
    assert result['candidate_grounded_success']==[None,None]
    assert all(m and m['setup_predictions_enabled'] and m['setup_predictions']>0 for m in result['candidate_setup_prediction'])

if __name__ == '__main__': main()
