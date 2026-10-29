"""Validated OfficeBench V2 adapter; install within a single worker's Ours cells.

Promoted from experiments/officebench-evidence-repair-v2-20260913/repair.py.
Prediction behavior is preserved; generated drivers live in a temporary directory.
"""
import ast
import hashlib
import inspect
import json
from pathlib import Path
from unittest.mock import patch
from . import _officebench_repair_base as baseline

OBSERVATION_KINDS = {'clean_read', 'dirty_read', 'write_observation'}


def evidence_error(code):
    """Reject direct replacement of the supplied input (not a full provenance proof)."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None  # The native validator reports syntax errors.
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id == 'data' and isinstance(node.ctx, (ast.Store, ast.Del)):
            return 'Do not replace supplied data with examples. Read the existing data variables.'
        if isinstance(node, ast.arg) and node.arg == 'data':
            return 'Do not shadow the supplied data variable with a function argument.'
        if isinstance(node, ast.Subscript) and isinstance(node.ctx, (ast.Store, ast.Del)):
            base = node.value
            while isinstance(base, (ast.Subscript, ast.Attribute)):
                base = base.value
            if isinstance(base, ast.Name) and base.id == 'data':
                return 'Do not overwrite copied input data. Build transformed output in a separate variable.'
    return None


def describe(context):
    """Give every operand a separate bounded preview, rather than truncating a combined blob."""
    result = {}
    for key in ('action', 'target', 'pre_write_content', 'initial_content', 'writes', 'history', 'initial_inventory'):
        value = context.get(key)
        if value is not None:
            text = json.dumps(value, ensure_ascii=False, default=str)
            limit = 4000 if key in ('writes', 'history') else 2500
            result[key] = {'variable': f'data[{key!r}]', 'chars': len(text), 'preview': text[:limit], 'preview_truncated': len(text) > limit}
    result['reads'] = []
    for index, record in enumerate(context.get('reads') or []):
        value = record.get('result', '')
        result['reads'].append({'variable': f'data["reads"][{index}]["result"]',
            'op': record.get('op'), 'path': record.get('path'), 'chars': len(str(value)),
            'preview': str(value)[:2500], 'preview_truncated': len(str(value)) > 2500})
    return json.dumps(result, ensure_ascii=False)


def install(profile='wmfix2', trace_dir=None):
    if profile in ('wmfix', 'control'):
        return baseline.install(profile, trace_dir)
    assert profile == 'wmfix2'
    from tts_qwen.env import officebench_lowlevel as low, officebench_wm_predict as wm, officebench_wm_python as py
    from tts_qwen.llm.python_tool import PythonToolResult
    stack = baseline.install('wmfix', trace_dir)

    def log(kind, payload):
        if trace_dir:
            directory = Path(trace_dir)
            directory.mkdir(parents=True, exist_ok=True)
            with (directory / 'evidence_trace.jsonl').open('a') as f:
                f.write(json.dumps({'kind': kind, **payload}, default=str) + '\n')

    # Only this process copies the augmented driver into newly created containers.
    # Native verifier reads retain their 2,000-character behavior. Full reads are WM-only.
    driver_source = Path(low._DRIVER_PATH).read_text()
    needle = 'out[path] = _read_one_file(abspath, spec.get("doc_type", "txt"))[:2000]'
    assert driver_source.count(needle) == 1
    driver_source = driver_source.replace(needle,
        'out[path] = _read_one_file(abspath, spec.get("doc_type", "txt"))' +
        '\n        if not req.get("wm_full_read", False):\n            out[path] = out[path][:2000]')
    import tempfile
    driver_dir = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix='officebench-v2-')))
    driver_path = driver_dir / ('driver_' + hashlib.sha256(driver_source.encode()).hexdigest()[:12] + '.py')
    if driver_path.exists():
        assert driver_path.read_text() == driver_source
    else:
        # Generated once before concurrent workers start, or atomically via identical contents.
        import os, tempfile
        fd, temporary = tempfile.mkstemp(dir=driver_dir, suffix='.tmp')
        with os.fdopen(fd, 'w') as f:
            f.write(driver_source)
        os.replace(temporary, driver_path)
    stack.enter_context(patch.object(low, '_DRIVER_PATH', str(driver_path)))
    original_pristine = wm.OfficeBenchWMQuery._read_pristine
    def pristine(self, safe_path):
        if self.allow_action_execution or not getattr(self, '_v2_full_read', False):
            return original_pristine(self, safe_path)
        try:
            state = self._session._driver('read_state', {'files': [{'path': safe_path, 'doc_type': wm._doc_type(safe_path)}], 'extras': False, 'wm_full_read': True}).get('state', {})
            result = str(state.get(safe_path, '[file does not exist]'))
            log('input_read', {'path': safe_path, 'chars': len(result), 'would_truncate': len(result) > 2000})
            return result
        except Exception as error:
            return f'ERROR: {type(error).__name__}: {error}'
    stack.enter_context(patch.object(wm.OfficeBenchWMQuery, '_read_pristine', pristine))

    original_run = py._run_python
    def run(code, context, stats, **kwargs):
        if context.get('kind') not in OBSERVATION_KINDS:
            return original_run(code, context, stats, **kwargs)
        error = evidence_error(code)
        if error:
            stats.python_calls += 1
            stats.python_failures += 1
            result = PythonToolResult(error='evidence contract: ' + error)
        else:
            result = original_run(code, context, stats, **kwargs)
        log('program', {'action': context.get('action'), 'context_kind': context.get('kind'),
            'code': code, 'error': result.error, 'output': result.output,
            'evidence': describe(context)})
        return result
    stack.enter_context(patch.object(py, '_run_python', run))

    # Clone the previously tested loop; its protocol includes public app semantics.
    baseline_loop = wm.run_python_prediction_loop
    namespace = dict(baseline_loop.__globals__)
    original_protocol = namespace['_python_protocol']
    def protocol(context, **kwargs):
        text = original_protocol(context, **kwargs)
        if context.get('kind') not in OBSERVATION_KINDS:
            return text
        return text + '\nEVIDENCE CONTRACT:\n' + describe(context) + '''
The supplied data variable is real copied input. Never define example data or replace data.
Compute from full variables, not preview literals. Previews can be truncated; variables are not.
READ provides pristine extracted file text, not the final app observation. For dirty reads, apply
data['writes'] in order. A missing pristine destination can have been created by a prior write:
read the source operands of those transitions before predicting its current contents.
Do not summarize or invent document contents. If the input is ICS, parse only BEGIN:VEVENT blocks,
unfold continuation lines, decode escaped text, and reproduce the public app's time/None formatting.
The real driver returns stdout.split('OBSERVATION:')[-1].strip(); match that visible text.
Available Python excludes os, pathlib, open, icalendar, email, delete statements and exit().
Use supported string/list/dict operations, re, json, datetime, and print_json; short executable code only.
'''
    original_resolve = py._resolve_final
    def resolve(response, **kwargs):
        context = kwargs['context']
        if context.get('kind') in OBSERVATION_KINDS:
            kwargs['prompt'] += '\nCURRENT COPIED INPUTS FOR FINAL AND REPAIR:\n' + describe(context)
        return original_resolve(response, **kwargs)
    source = inspect.getsource(py.run_python_prediction_loop)
    needle = '"variable": f"data[\'reads\'][{index}][\'result\']",'
    assert source.count(needle) == 1
    source = source.replace(needle, needle + '\n                "preview": str(record.get("result", ""))[:2500],\n                "result_chars": len(str(record.get("result", ""))),')
    source = source.replace('f"{model_prompt}\\n\\nYou are out of read turns.',
        'f"{model_prompt}\\nCURRENT COPIED INPUTS:\\n{describe(current_context())}\\n\\nYou are out of read turns.')
    namespace.update(_python_protocol=protocol, _run_python=run, _resolve_final=resolve, describe=describe)
    exec(compile(source, '<officebench_evidence_v2>', 'exec'), namespace)
    enhanced_loop = namespace['run_python_prediction_loop']
    def loop(*args, **kwargs):
        selected = enhanced_loop if kwargs['context'].get('kind') in OBSERVATION_KINDS else baseline_loop
        return selected(*args, **kwargs)
    stack.enter_context(patch.object(wm, 'run_python_prediction_loop', loop))
    original_predict = wm.OfficeBenchWMPredictor._predict
    def predict(self, *args, **kwargs):
        if self._query is not None and (kwargs.get('python_context') or {}).get('kind') in OBSERVATION_KINDS:
            with patch.object(self._query, '_v2_full_read', True, create=True):
                return original_predict(self, *args, **kwargs)
        return original_predict(self, *args, **kwargs)
    stack.enter_context(patch.object(wm.OfficeBenchWMPredictor, '_predict', predict))

    # Normalize at the same layer as the real driver, observation predictions only.
    for name in ('predict_read', 'predict_write'):
        original = getattr(wm.OfficeBenchWMPredictor, name)
        def normalized(self, action, *args, _original=original, **kwargs):
            result = _original(self, action, *args, **kwargs)
            if 'World model failed to predict safely' not in result:
                result = result.split('OBSERVATION:')[-1].strip()
            return result
        stack.enter_context(patch.object(wm.OfficeBenchWMPredictor, name, normalized))
    return stack
