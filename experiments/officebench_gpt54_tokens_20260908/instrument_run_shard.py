"""Add per-call usage logging to an isolated AML code snapshot, without changing prompts."""
from pathlib import Path

hook = '''
import inspect as _inspect
import json as _json
import os as _os
import threading as _threading
_token_log_lock = _threading.Lock()
def _log_token_call(messages, response):
    frame = _inspect.currentframe().f_back
    identity = {}
    contexts = []
    component = 'other'
    while frame is not None:
        name = frame.f_code.co_name
        filename = frame.f_code.co_filename
        local = frame.f_locals
        if name == '_run_task_cells':
            spec = local.get('spec')
            identity = {'subset': spec.subset, 'spec_index': spec.index,
                        'method': local.get('method'), 'n_rollouts': local.get('N')}
        if name == 'run_officebench_agent':
            component = 'actor' if component == 'other' else component
        if 'officebench_wm' in filename:
            component = 'wm'
            if name == 'run_python_prediction_loop':
                contexts.append(local.get('context', {}))
        frame = frame.f_back
    usage = getattr(response, 'usage', None)
    row = dict(identity, component=component, messages=messages, contexts=contexts,
               output=response.choices[0].message.content or '',
               usage=usage.model_dump() if usage is not None else None)
    path = _os.environ.get('TTS_TOKEN_LOG')
    if path:
        path = path.removesuffix('.jsonl') + '.' + str(_os.getpid()) + '.jsonl'
        with _token_log_lock:
            with open(path, 'a') as f:
                f.write(_json.dumps(row, default=str) + '\\n')
'''
p = Path('src/tts/llm/chat_client.py')
s = p.read_text()
anchor = '                usage = getattr(resp, "usage", None)'
assert s.count(anchor) == 1
s = s.replace(anchor, '                _log_token_call(messages, resp)\n' + anchor)
s += hook
p.write_text(s)
