"""Differential checks against OfficeBench app function bodies on disposable files."""
import ast
import os
from pathlib import Path

import pytest

from tts.env.officebench_semantics import cli_literal
from tts.env.officebench_wm_predict import symbolic_file_state


def native(relative, namespace):
    root = os.environ.get('OFFICEBENCH_ROOT')
    if not root:
        pytest.skip('OFFICEBENCH_ROOT is needed for native differential checks')
    path = Path(root) / 'apps' / relative
    tree = ast.parse(path.read_text())
    tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    exec(compile(tree, str(path), 'exec'), namespace)
    return namespace


@pytest.mark.parametrize('value', ['1', '1.0', 'True', 'False', '0', '2.5', '"1"', 'hello', '"a,b"', 'None'])
def test_excel_state_matches_native_cli_and_workbook(tmp_path, value):
    openpyxl = pytest.importorskip('openpyxl')
    parser = pytest.importorskip('fire.parser')
    assert cli_literal(value) == parser.DefaultParseValue(value)
    path = str(tmp_path / 'book.xlsx')
    wb = openpyxl.Workbook()
    wb.active.cell(2, 2, 9)
    wb.save(path)
    setter = native('excel_app/excel_set_cell.py', {'os': os, 'openpyxl': openpyxl})
    reader = native('excel_app/excel_read_file.py', {'os': os, 'openpyxl': openpyxl})
    initial = reader['excel_read_file'](path)
    assert setter['excel_set_cell'](path, parser.DefaultParseValue(value), 2, 2)
    predicted = symbolic_file_state('book.xlsx', initial, [({'app': 'excel', 'action': 'set_cell', 'text': value, 'row_idx': 2, 'column_idx': 2}, 'ok')])
    assert predicted == reader['excel_read_file'](path)


def test_word_append_and_quote_stripping_match_native(tmp_path):
    Document = pytest.importorskip('docx').Document
    parser = pytest.importorskip('fire.parser')
    import shlex
    path = str(tmp_path / 'doc.docx')
    doc = Document(); doc.add_paragraph('Existing text'); doc.save(path)
    app = native('word_app/word_write_to_file.py', {'os': os, 'Document': Document})
    action = {'app': 'word', 'action': 'write_to_file', 'file_path': path, 'contents': 'Bob\'s "new" paragraph'}
    argv = shlex.split(app['construct_action']('/', action))
    parsed = parser.DefaultParseValue(argv[argv.index('--contents') + 1])
    assert app['word_write_to_file'](path, parsed)
    assert symbolic_file_state(path, 'Existing text', [(action, 'ok')]) == '\n'.join(p.text for p in Document(path).paragraphs)


def test_word_txt_is_not_plain_text(tmp_path):
    Document = pytest.importorskip('docx').Document
    path = str(tmp_path / 'out.txt')
    app = native('word_app/word_create_new_file.py', {'Document': Document})
    assert app['word_create_new_file'](path)
    assert Path(path).read_bytes().startswith(b'PK')


def test_large_rendered_evidence_and_regex_match_methods():
    from tts.env.officebench_wm_python import _run_python, PythonLoopResult
    text = 'x' * 25000
    result = _run_python("import re\nm = re.search('x+', data['text'])\nprint_json({'output': data['text'][m.start():m.end()]})", {'text': text}, PythonLoopResult())
    assert not result.error
    import json
    assert json.loads(result.output)['output'] == text


def test_verifier_percheck_uses_all_configured_votes(monkeypatch):
    from tts.env.officebench_wm import OfficeBenchVerifier
    monkeypatch.setenv('TTS_OFFICEBENCH_VERIFIER_PERCHECK', '1')
    monkeypatch.setenv('TTS_OFFICEBENCH_VERIFIER_SAMPLES', '5')
    monkeypatch.setenv('TTS_OFFICEBENCH_WM_PYTHON', '0')
    class Chat:
        answers = iter(['STATUS: MET', 'SUCCESS', 'FAILURE', 'SUCCESS', 'FAILURE', 'SUCCESS'])
        count = 0
        def chat(self, *args, **kwargs):
            self.count += 1
            return next(self.answers)
    chat = Chat(); verifier = OfficeBenchVerifier(chat)
    score = verifier.predict_success('create output', [], [], config={'evaluation': [{'function': 'evaluate_file_exist', 'args': {'file': 'out.txt'}}]}, final_state={'out.txt': 'ok'})
    assert score == 0.6
    assert chat.count == 6
    assert verifier.last_percheck_breakdown['atomic_score'] == 1


def test_inventory_accepts_file_and_read_directory_is_recoverable(tmp_path, monkeypatch):
    from tts.env import officebench_driver as driver
    monkeypatch.setattr(driver, '_TESTBED', str(tmp_path))
    file = tmp_path / 'data.txt'; file.write_text('hello')
    result = driver.handle_list_state({'path': str(file)})
    assert result['missing'] is False
    assert result['entries'] == [{'name': 'data.txt', 'path': str(file), 'type': 'file'}]
    assert driver._read_one_file(str(tmp_path), 'txt').startswith('ERROR: operand is a directory')


def test_query_envelope_is_repaired_not_exposed_as_observation():
    from tts.env.officebench_wm_python import run_python_prediction_loop
    prompts = []
    replies = iter([
        'PYTHON_FINAL:\nprint_json({"queries": [{"op": "read", "path": "/testbed/a.txt"}]})',
        'PYTHON_FINAL:\nprint_json({"text": "corrected observation"})',
    ])
    def complete(prompt, budget):
        prompts.append(prompt)
        return next(replies)
    result = run_python_prediction_loop(complete, None, 'predict', 'PREDICT: text', context={}, force_always=True)
    assert result.valid and result.output == 'corrected observation'
    assert result.python_failures == 1
    assert 'not a valid text result' in prompts[1]


@pytest.mark.parametrize('action,predicted,actual', [
    ({'app': 'calendar', 'action': 'create_event'}, 'OBSERVATION: Successfully create a new event to Bob\'s calendar.', 'Successfully create a new event to Bob\'s calendar.'),
    ({'app': 'excel', 'action': 'read_file'}, '(1, 1): Name\t\n', 'The following is the table from the excel file:\n(1, 1): Name'),
    ({'app': 'shell', 'action': 'command', 'command': 'cp a b'}, '', 'Successfully executed command: cp a b. The output was [].'),
])
def test_recorded_driver_observation_boundaries(action, predicted, actual):
    from tts.env.officebench_semantics import normalize_agent_observation
    assert normalize_agent_observation(action, predicted) == actual


@pytest.mark.parametrize('app,extension', [('word', 'docx'), ('excel', 'xlsx')])
def test_creation_requires_existing_parent_not_existing_output(tmp_path, app, extension):
    from tts.env.officebench_wm_predict import action_inputs
    Document = pytest.importorskip('docx').Document
    openpyxl = pytest.importorskip('openpyxl')
    data = tmp_path / 'data'; data.mkdir()
    destination = data / ('new.' + extension)
    action = {'app': app, 'action': 'create_new_file', 'file_path': '/testbed/data/' + destination.name}
    assert not destination.exists()
    dependencies = action_inputs(action)
    assert dependencies
    assert all((tmp_path / dependency).is_dir() for dependency in dependencies)
    module = native(f'{app}_app/{app}_create_new_file.py', {'os': os, 'Document': Document, 'openpyxl': openpyxl})
    assert module[f'{app}_create_new_file'](str(destination))
    assert destination.is_file()


@pytest.mark.parametrize('app,extension', [('word', 'docx'), ('excel', 'xlsx')])
@pytest.mark.parametrize('parent_exists', [True, False])
def test_grounded_creation_matches_native_without_executing_candidate(tmp_path, app, extension, parent_exists):
    from tts.env.officebench_wm_predict import OfficeBenchWMPredictor, OfficeBenchWMQuery
    Document = pytest.importorskip('docx').Document
    openpyxl = pytest.importorskip('openpyxl')
    if parent_exists:
        (tmp_path/'data').mkdir()
    destination = tmp_path/'data'/('new.'+extension)
    class Session:
        def apps_catalogue(self): return {}
        def list_state(self, path):
            assert path == '/testbed/data'
            return {'entries': [], 'missing': not parent_exists}
        def exec_action(self, *args, **kwargs):
            raise AssertionError('candidate execution is forbidden')
    class NoChat:
        def chat(self, *args, **kwargs):
            raise AssertionError('known creation preconditions should not be guessed')
    predictor = OfficeBenchWMPredictor(NoChat()); predictor.program_only = True
    predictor.bind_query_interface(OfficeBenchWMQuery(Session(), lambda: '/', allow_action_execution=False))
    action = {'app': app, 'action': 'create_new_file', 'file_path': '/testbed/data/'+destination.name}
    predictions = [predictor.predict_write(action), predictor.predict_write(action)]
    assert not destination.exists()  # both candidates were predictions over copied evidence
    module = native(f'{app}_app/{app}_create_new_file.py', {'os': os, 'Document': Document, 'openpyxl': openpyxl})
    actual = [module['main'](str(destination)).split('OBSERVATION:')[-1].strip().replace(str(tmp_path), '/testbed') for _ in range(2)]
    assert predictions == actual


def test_inventory_schema_distinguishes_empty_directory_missing_and_file():
    from tts.env.officebench_semantics import inventory_schema
    assert inventory_schema('/testbed/data', '[]')['root_exists'] is True
    assert inventory_schema('/testbed/data', '[]')['root_type'] == 'directory'
    assert inventory_schema('/testbed/data', '[directory does not exist]')['root_exists'] is False
    assert inventory_schema('/testbed/data', 'ERROR: timeout')['root_exists'] is None
    assert inventory_schema('/testbed/data/a.txt', '[{"path":"/testbed/data/a.txt","type":"file"}]')['root_type'] == 'file'


def test_contract_repair_keeps_environment_content_out_of_prompts():
    from tts.env.officebench_semantics import action_contract
    from tts.env.officebench_wm_python import run_python_prediction_loop
    action = {'app': 'excel', 'action': 'set_cell', 'file_path': '/testbed/data/a.xlsx'}
    context = {'action': action, 'observation_action': action, 'targets': ['data/a.xlsx'],
               'enforce_observation_contract': True,
               'current_state': {'data/a.xlsx': {'status': 'exact', 'exists': True,
                                                 'content': 'PRIVATE_WORKBOOK_SENTINEL'}}}
    replies = iter([
        'PYTHON_FINAL:\nprint("Successfully edited the cell.")',
        'PYTHON_FINAL:\nprint("The file " + data["action"]["file_path"] + " does not exist. Failed to write to the file.")',
        'PYTHON_FINAL:\nprint("Successfully write text to " + data["action"]["file_path"])',
    ])
    prompts = []
    def complete(prompt, budget):
        prompts.append(prompt)
        return next(replies)
    result = run_python_prediction_loop(complete, None, action_contract(action), 'PREDICT: text',
                                        context=context, force_always=True)
    assert result.valid and result.python_failures == 2
    assert result.output == 'Successfully write text to /testbed/data/a.xlsx'
    assert all('PRIVATE_WORKBOOK_SENTINEL' not in p for p in prompts)
    assert 'exact current-state existence' in prompts[-1]


def test_calendar_contract_repairs_raw_timestamp_without_exposing_event_data():
    from tts.env.officebench_semantics import action_contract
    from tts.env.officebench_wm_python import run_python_prediction_loop
    action = {'app': 'calendar', 'action': 'list_events', 'username': 'Alice'}
    context = {'action': action, 'enforce_observation_contract': True,
               'reads': [{'result': '20140101T141000Z'}]}
    code = '''PYTHON_FINAL:
import datetime
value = datetime.datetime.strptime(data['reads'][0]['result'], '%Y%m%dT%H%M%SZ').strftime('%Y-%m-%d %H:%M:%S')
print('Successfully list events for ' + data['action']['username'] + ':\\nStart Time: ' + value)
'''
    replies = iter([
        "PYTHON_FINAL:\nprint('Successfully list events for ' + data['action']['username'] + ':\\nStart Time: ' + data['reads'][0]['result'])",
        code,
    ])
    prompts = []
    def complete(prompt, budget):
        prompts.append(prompt); return next(replies)
    result = run_python_prediction_loop(complete, None, action_contract(action), 'PREDICT: text',
                                        context=context, force_always=True)
    assert result.valid and result.python_failures == 1
    assert result.output.endswith('Start Time: 2014-01-01 14:10:00')
    assert all('20140101T141000Z' not in p for p in prompts)
    import datetime
    module = native('calendar_app/calendar_list_events.py', {})
    from types import SimpleNamespace
    assert result.output.split('Start Time: ')[1] == module['format_time'](
        SimpleNamespace(dt=datetime.datetime(2014, 1, 1, 14, 10)))


def test_current_state_creation_then_edit_uses_overlay_not_pristine_absence():
    from tts.env.officebench_wm_predict import OfficeBenchWMPredictor
    class Query:
        def read_raw(self, path): return '[file does not exist]'
    predictor = OfficeBenchWMPredictor(None)
    predictor._query = Query()
    target = 'data/new.xlsx'
    actions = [
        {'app': 'excel', 'action': 'create_new_file'},
        {'app': 'excel', 'action': 'set_cell', 'row_idx': 1, 'column_idx': 1, 'text': 'PRIVATE_CELL'},
    ]
    for action in actions:
        action['file_path'] = '/testbed/' + target
        predictor.applied_writes.setdefault(target, []).append((action, 'ok'))
        predictor.transition_log.record(action, 'ok', [target], applied=True)
    record = predictor._current_state_data(target)
    assert record['status'] == 'exact' and record['exists'] is True
    assert '(1, 1): PRIVATE_CELL' in record['content']
    predictor.global_dirty = True
    record = predictor._current_state_data(target)
    assert record['status'] == 'unknown' and record['exists'] is None


def test_related_transitions_keep_order_without_duplicate_multifile_actions():
    from tts.env.officebench_wm_predict import OfficeBenchWMPredictor
    predictor = OfficeBenchWMPredictor(None)
    for number, paths, applied in [(1, ['data/a', 'data/b'], True),
                                    (2, ['data/b'], False), (3, ['data/a'], True)]:
        predictor.transition_log.record({'sequence': number}, 'outcome', paths, applied=applied)
    assert [a['sequence'] for a, _ in predictor._related_writes('data')] == [1, 3]
    assert [entry['applied'] for entry in predictor._write_history_data()] == [True, False, True]


def test_clean_read_contract_rejects_dropped_content_without_revealing_it():
    from tts.env.officebench_semantics import observation_contract_error
    content = '(1, 1): PRIVATE_SOURCE_CELL\t\n'
    context = {'enforce_observation_contract': True, 'kind': 'clean_read',
               'action': {'app': 'excel', 'action': 'read_file', 'file_path': '/testbed/a.xlsx'},
               'reads': [{'status': 'ok', 'result': content}]}
    error = observation_contract_error(context, 'The following is the table from the excel file:\n(1, 1): invented')
    assert error and 'PRIVATE_SOURCE_CELL' not in error
    assert observation_contract_error(context, 'The following is the table from the excel file:\n' + content) is None
    context['reads'][0]['result'] = 'literal OBSERVATION: retained tail'
    assert observation_contract_error(context, 'retained tail') is None


def test_conversion_current_input_and_native_output_name():
    from tts.env.officebench_wm_predict import write_targets, OfficeBenchWMPredictor
    from tts.env.officebench_semantics import observation_contract_error, action_contract
    action = {'app': 'word', 'action': 'convert_to_pdf', 'word_file_path': '/testbed/data/source.docx', 'pdf_file_path': '/testbed/data/requested.pdf'}
    assert write_targets(action) == ['data/source.pdf']
    context = {'enforce_observation_contract': True, 'action': action, 'inputs': ['data/source.docx'], 'current_state': {'data/source.docx': {'status': 'exact', 'exists': True}}}
    assert observation_contract_error(context, 'The word file /testbed/data/source.docx does not exist. Failed to convert the file to pdf.')
    assert observation_contract_error(context, 'Successfully convert /testbed/data/source.docx to /testbed/data/requested.pdf') is None
    assert 'INPUTS' in action_contract(action)


def test_email_overlay_both_mailboxes_overwrite_and_no_native_actions():
    import json
    from tts.env.officebench_wm_predict import OfficeBenchWMPredictor, write_targets
    class Query:
        def list(self, path): return '[]'
        def read_raw(self, path): raise AssertionError('empty pristine mailbox has no messages')
    predictor = OfficeBenchWMPredictor(None)
    predictor._query = Query()
    for content in ['obsolete', 'replacement body longer than twenty characters']:
        action = {'app': 'email', 'action': 'send_email', 'sender': 'Alice@example.org', 'recipient': 'Tom@example.org', 'subject': 'same', 'content': content}
        for target in write_targets(action):
            predictor.applied_writes.setdefault(target, []).append((action, 'Successfully sent email to Tom@example.org.'))
    for target in ['emails/Alice', 'emails/Tom']:
        full = predictor._mailbox_state(target)
        assert full.count('Email ID: same.eml') == 1
        assert 'obsolete' not in full and 'replacement body longer' in full
        assert 'From: Alice@example.com' in full and 'To: Tom@example.com' in full
        short = predictor._mailbox_state(target, 20)
        assert 'Content: replacement body lon...' in short


def test_ocr_contract_rejects_string_transcription():
    from tts.env.officebench_semantics import observation_contract_error
    context = {'enforce_observation_contract': True, 'action': {'app': 'ocr', 'action': 'recognize_file', 'file_path': '/testbed/private.png'}, 'reads': [{'status': 'ok', 'result': 'Private 7329\nSecond line\n'}]}
    assert observation_contract_error(context, 'The text from /testbed/private.png is:\nPrivate 7329\nSecond line') is None
    assert observation_contract_error(context, 'The text from /testbed/private.png is:\nPrivate 7328\nSecond line')


def test_zero_confidence_selection_requires_audit_improvement(monkeypatch):
    from tts.env.real_officebench import OfficeBenchTaskEnv
    from tts.trajectory import RolloutResult, Trajectory, RewardInfo
    from tts.config import Config, SolverConfig
    monkeypatch.setenv('TTS_OFFICEBENCH_COMPARATIVE_RERANK', '1')
    def forbidden(**kwargs): raise AssertionError('no free-form comparison for an all-zero pool')
    monkeypatch.setattr('tts.llm.make_chat_client', forbidden)
    env = OfficeBenchTaskEnv.__new__(OfficeBenchTaskEnv)
    env._instruction = 'produce PDF and send emails'
    env._config = {}
    rollouts = [RolloutResult(i, Trajectory(steps=[]), RewardInfo(reward=0.0)) for i in range(4)]
    for r in rollouts:
        r.predicted.metadata['verifier_percheck'] = {'met': 1, 'unmet': 2, 'unknown': 0, 'total': 3}
    rollouts[0].predicted.metadata['comparative_rerank'] = {'pool_size': 2, 'best': 1}
    env.refine_rollout_scores(Config(seed=7, solver=SolverConfig(llm_agent='gpt-5.4')), rollouts)
    assert rollouts[0].predicted.metadata['selection_override']['best'] == 1
    # A fresh nested pool may improve partial completion, without claiming success.
    for r in rollouts: r.predicted.metadata.pop('comparative_rerank', None)
    rollouts[3].predicted.metadata['verifier_percheck']['met'] = 2
    rollouts[3].predicted.metadata['verifier_percheck']['unmet'] = 1
    env.refine_rollout_scores(Config(seed=7, solver=SolverConfig(llm_agent='gpt-5.4')), rollouts)
    assert rollouts[0].predicted.metadata['selection_override']['best'] == 3
    assert all(r.predicted.reward == 0 for r in rollouts)


def test_conversion_snapshot_preserves_source_at_conversion_time():
    from tts.env.officebench_wm_predict import OfficeBenchWMPredictor
    class Query:
        def workdir(self): return '/testbed'
        def list(self, path): return '[]'
        def read_raw(self, path): return '[file does not exist]'
    class Predictor(OfficeBenchWMPredictor):
        contexts = []
        def _predict(self, prompt, final, **kwargs):
            data = kwargs['python_context']; self.contexts.append(data)
            action = data.get('action', {})
            if data['kind'] == 'final_file_state':
                return data['conversion_source']['state']['content']
            if action.get('action') == 'convert_to_pdf':
                return 'Successfully convert /testbed/data/a.docx to /testbed/data/a.pdf'
            return 'Successfully write contents to /testbed/data/a.docx'
    p = Predictor(None); p.program_only = True; p._query = Query()
    p.predict_write({'app': 'word', 'action': 'create_new_file', 'file_path': '/testbed/data/a.docx'})
    p.predict_write({'app': 'word', 'action': 'write_to_file', 'file_path': '/testbed/data/a.docx', 'contents': 'original private payload'})
    p.predict_write({'app': 'word', 'action': 'convert_to_pdf', 'word_file_path': '/testbed/data/a.docx', 'pdf_file_path': '/testbed/data/a.pdf'})
    p.predict_write({'app': 'word', 'action': 'write_to_file', 'file_path': '/testbed/data/a.docx', 'contents': 'later source edit'})
    assert p.predict_file_state('data/a.pdf', '[file does not exist]') == 'original private payload'
    assert 'later source edit' in p._current_state_data('data/a.docx')['content']
    assert p._current_state_data('data/a.pdf')['conversion_source']['state']['content'] == 'original private payload'
    p.predict_write({'app': 'word', 'action': 'create_new_file', 'file_path': '/testbed/data/a.pdf'})
    assert 'data/a.pdf' not in p.conversion_sources


def test_conversion_reconstruction_rejects_stale_absence_but_allows_uncertainty():
    from tts.env.officebench_semantics import observation_contract_error
    ctx = {'kind': 'final_file_state', 'enforce_observation_contract': True,
           'conversion_source': {'state': {'status': 'exact', 'exists': True, 'content': 'private'}}}
    assert observation_contract_error(ctx, '[file does not exist]')
    assert observation_contract_error(ctx, '[unknown predicted content]') is None
    assert observation_contract_error(ctx, 'private') is None


def test_email_comma_recipient_matches_native_fire_failure():
    from email.message import EmailMessage
    from types import SimpleNamespace
    import shlex
    from tts.env.officebench_semantics import email_cli_data, observation_contract_error
    parser = pytest.importorskip('fire.parser')
    app = native('email_app/email_send_email.py', {'os': SimpleNamespace(makedirs=lambda *a, **k: None), 'EmailMessage': EmailMessage})
    action = {'app': 'email', 'action': 'send_email', 'sender': 'Alice', 'recipient': 'Bob, Tom', 'subject': 'notice', 'content': 'body'}
    argv = shlex.split(app['construct_action']('/', dict(action)))
    recipient = parser.DefaultParseValue(argv[argv.index('--recipient') + 1])
    assert recipient == ('Bob', 'Tom') == email_cli_data(action)['recipient']
    expected = app['main']('Alice', recipient, 'notice', 'body').split('OBSERVATION:')[-1].strip()
    assert expected == "Failed to send email to ('Bob', 'Tom')."
    ctx = {'action': action, 'enforce_observation_contract': True}
    assert observation_contract_error(ctx, expected) is None
    assert observation_contract_error(ctx, 'Successfully sent email to Bob, Tom.')
    action['recipient'] = ['Bob', 'Tom']
    assert email_cli_data(action)['multiple_recipients'] is True


def test_inventory_find_preserves_native_order(tmp_path, monkeypatch):
    import subprocess
    from tts.env import officebench_driver as driver
    from tts.env.officebench_wm_predict import _emulate_inventory_pipeline
    (tmp_path / 'z.txt').write_text('z')
    (tmp_path / 'middle').mkdir()
    (tmp_path / 'middle' / 'b.txt').write_text('b')
    (tmp_path / 'a.txt').write_text('a')
    monkeypatch.setattr(driver, '_TESTBED', str(tmp_path))
    entries = driver.handle_list_state({'path': str(tmp_path), 'recursive': True})['entries']
    for item in entries:
        item['path'] = item['path'].replace(str(tmp_path), '/testbed', 1)
    predicted = _emulate_inventory_pipeline([['find', '/testbed', '-type', 'f']], entries)
    actual = subprocess.check_output(['find', str(tmp_path), '-type', 'f'], text=True).replace(str(tmp_path), '/testbed').strip()
    assert predicted == actual
