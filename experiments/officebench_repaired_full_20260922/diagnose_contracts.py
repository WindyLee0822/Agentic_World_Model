"""Reproduce wiring/logic defects with copied synthetic inputs; no native actions."""
import ast
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).parent
SOURCE = Path('/tmp/officebench-repaired-full-n124-20260922')


def isolated_function(path, name, namespace, method=False):
    tree = ast.parse(path.read_text())
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
    if method:
        # Supply only the path normalizer imported inside this method; retain its actual logic.
        node.body = [n for n in node.body if not isinstance(n, ast.ImportFrom)]
    module = ast.Module(body=[node], type_ignores=[])
    exec(compile(module, str(path), 'exec'), namespace)
    return namespace[name]


ns = dict(Dict=Dict, List=List, Optional=Optional, _int_env=lambda name, default: default)
wm = SOURCE / 'src/tts/env/officebench_wm.py'
state_files = isolated_function(wm, 'state_files_from_config', ns)
atomic = isolated_function(wm, '_atomic_requirements', ns)
email = {'evaluation': [{'function': 'evaluate_contain', 'args': {
    'doc_type': 'email', 'username': 'PRIVATE_USER', 'keywords': ['notice']}}]}
docx = {'evaluation': [{'function': 'evaluate_file_exist', 'args': {'file': 'data/result.docx'}}]}


def norm(path):
    return path.removeprefix('/testbed/').removeprefix('./')


class Predictor:
    applied_writes = {'data/result.docx': [({'app': 'word', 'action': 'write_to_file'}, 'Success')]}
    global_dirty = False
    def was_written(self, target):
        return target in self.applied_writes
    def is_deleted(self, target):
        return False
    def predict_file_state(self, path, initial):
        return 'Valid document contents'
    def predicted_write_log(self):
        return 'Synthetic predicted write'


predict_state = isolated_function(SOURCE / 'src/tts/env/real_officebench.py', '_predicted_state',
                                 dict(os=os, _norm_path=norm), method=True)
docx_state = predict_state(None, None, Predictor(), state_files(docx), initial_state={})
negative = {'evaluation': [{'function': 'evaluate_not_contain',
                           'args': {'file': 'data/result.txt', 'keywords': ['alpha', 'beta']}}]}
content = 'alpha is present; the other keyword is absent'
native_negative = not all(k in content for k in negative['evaluation'][0]['args']['keywords'])
split_negative = all(not all(k in content for k in c['args']['keywords']) for c in atomic(negative))
report = {
    'source': str(SOURCE), 'native_actions_executed': 0,
    'email_specs': state_files(email),
    'email_specs_should_reference': 'emails/PRIVATE_USER',
    'docx_existence_specs': state_files(docx),
    'docx_existence_predicted_state': docx_state,
    'negation': {'native_result': native_negative, 'split_result': split_negative,
                 'atomic_checks': atomic(negative)},
}
assert state_files(email) == [{'path': './calendar/PRIVATE_USER.ics', 'doc_type': 'ics'}]
assert docx_state['data/result.docx'].startswith('[incompatible file format:')
assert native_negative and not split_negative
(ROOT / 'analysis/contract_reproductions.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
