"""Offline scorer audit; no native actions, selection, or commit is performed."""
import copy
import json
import os
from pathlib import Path

from tts.env.officebench_wm import OfficeBenchVerifier
from tts.env.officebench_wm_predict import OfficeBenchWMPredictor, write_targets
from tts.llm.chat_client import ChatClient

ROOT = Path(__file__).parent
os.environ.update(TTS_OFFICEBENCH_WM_TOOL_MODE='python',
                  TTS_OFFICEBENCH_WM_PYTHON='1',
                  TTS_OFFICEBENCH_VERIFIER_PERCHECK='1',
                  TTS_OFFICEBENCH_VERIFIER_SAMPLES='5')
config = next(t for t in json.loads((ROOT / 'tasks.json').read_text())
              if t['task_id'] == '3-12/0')
traces = []
for line in (ROOT / 'analysis/run.log').read_text().splitlines():
    if '[officebench-commit-trace] ' in line:
        trace = json.loads(line.split('[officebench-commit-trace] ', 1)[1])
        if trace['task_id'] == config['task_id']:
            traces.append(trace)
# The three N configurations committed the same eight observed actions.
assert len(traces) == 24 and traces[:8] == traces[8:16] == traces[16:]
actions = [t['action'] for t in traces[:8]]
observations = [t['predicted'] for t in traces[:8]]

class EmptyMailbox:
    # This fixture has no initial emails; only successful predicted sends are replayed.
    def list(self, path):
        return '[]'

predictor = OfficeBenchWMPredictor(None)
predictor.bind_query_interface(EmptyMailbox())
for action, obs in zip(actions, observations):
    if action['app'] == 'email' and obs.startswith('Successfully'):
        for target in write_targets(action):
            predictor.applied_writes.setdefault(target, []).append((action, obs))
state = {f'emails/{name}': predictor._mailbox_state(f'emails/{name}')
         for name in ('Bob', 'Tom')}
# The native check tests only PDF existence. This is a logical decoded projection,
# not a claim that this script reproduced PDF bytes/layout.
state['data/power_failure.pdf'] = next(a['contents'] for a in actions
                                      if a['action'] == 'write_to_file')
assert all(state.values())
client = ChatClient(model='gpt-5.4', backend='azure',
    azure_endpoint='https://ai-m365researchefficientaieus2212654625773.cognitiveservices.azure.com/',
    azure_api_version='2024-12-01-preview', azure_auth_mode='gcr',
    max_retries=1, max_rate_limit_retries=1)

class PrivateChat:
    def __init__(self):
        self.calls = 0
    def chat(self, system, prompt, **kwargs):
        for private in ('Power outage detected.', 'Power Failure Notification',
                        'SCORER_PRIVATE_SENTINEL_9382'):
            assert private not in system + prompt
        self.calls += 1
        response = client.chat(system, prompt, **kwargs)
        with (ROOT / 'verifier_followup_programs.jsonl').open('a') as stream:
            stream.write(json.dumps({'case': case, 'call': self.calls,
                                     'response': response}) + '\n')
        return response

results = []
for case in ('recovered_success', 'wrong_recipient_content', 'unknown_recipient_state'):
    copied = copy.deepcopy(state)
    copied['unused_private_note'] = 'SCORER_PRIVATE_SENTINEL_9382'
    if case == 'wrong_recipient_content':
        copied['emails/Tom'] = 'Unrelated meeting invitation'
    elif case == 'unknown_recipient_state':
        copied.pop('emails/Tom')
    chat = PrivateChat()
    verifier = OfficeBenchVerifier(chat, sample_seed=20260922)
    score = verifier.predict_success(config['task'], actions, observations,
        config=config, final_state=copied, final_answer='None')
    result = dict(case=case, confidence=score,
                  breakdown=verifier.last_percheck_breakdown,
                  precheck=verifier.last_precheck_failure,
                  python_calls=verifier.n_python_calls,
                  python_failures=verifier.n_python_failures,
                  chat_calls=chat.calls, private_prompt_checks_passed=True)
    results.append(result)
    print(json.dumps(result), flush=True)
    (ROOT / 'verifier_followup.json').write_text(json.dumps({
        'kind': 'offline scorer audit of frozen selected trajectory plus negative controls',
        'native_actions_executed': 0, 'results': results}, indent=2))
assert results[0]['confidence'] >= 0.8
assert all(r['confidence'] < 0.5 for r in results[1:])
