import json
from email.message import EmailMessage

import pytest

from tts.env.officebench_checks import check_predicted
from tts.env.officebench_wm import OfficeBenchVerifier, state_files_from_config, _atomic_requirements
from tts.env.officebench_wm_predict import OfficeBenchWMPredictor
from tts.env.real_officebench import OfficeBenchTaskEnv


def check(fn, **args):
    return {'function': fn, 'args': args}


def test_native_predicates_and_uncertainty():
    state = {'./data/s.xlsx': '(1, 1): Bob\t(1, 2): 2100000\t\n',
             'data/a.txt': 'alpha', 'data/b.txt': 'beta'}
    assert check_predicted(check('evaluate_excel_cell_value', file='data/s.xlsx',
        matches=[{'row': 1, 'col': 2, 'value': '2100000'}]), state)[0] == 'MET'
    assert check_predicted(check('evaluate_not_contain', file='data/a.txt',
        keywords=['alpha', 'beta']), state)[0] == 'MET'
    assert len(_atomic_requirements({'evaluation': [check('evaluate_not_contain',
        file='data/a.txt', keywords=['alpha', 'beta'])]})) == 1
    assert check_predicted(check('evaluate_diff_contain_text', input_file='data/a.txt',
        output_file='data/b.txt', keywords=['-alpha', '+beta']), state)[0] == 'MET'
    assert check_predicted(check('evaluate_file_exist', file='data/missing'), state)[0] == 'UNKNOWN'
    assert check_predicted(check('evaluate_file_exist', file='data/a.txt'),
        {'data/a.txt': '[unknown predicted content]'})[0] == 'UNKNOWN'
    assert check_predicted(check('evaluate_contain', file='data/a.txt', keywords=['2100000']),
        {'data/a.txt': '2,100,000'})[0] == 'MET'
    assert check_predicted(check('evaluate_contain', file='data/a.docx', doc_type='xlsx',
        keywords=['alpha']), {'data/a.docx': 'alpha'})[0] == 'UNMET'


def test_exact_match_does_not_guess_lost_cell_types():
    c = check('evaluate_exact_match', result_file='data/a.xlsx', expected_file='ref.xlsx', doc_type='xlsx')
    assert check_predicted(c, {'data/a.xlsx': '(1, 1): [Empty Cell]\t\n'},
                          {'ref.xlsx': '[Empty Sheet]'})[0] == 'MET'
    assert check_predicted(c, {'data/a.xlsx': '(1, 1): 1\t\n'},
                          {'ref.xlsx': '(1, 1): 1\t\n'})[0] == 'UNKNOWN'


class Query:
    def __init__(self, files=None):
        self.files = files or {}
    def list(self, path, **kwargs):
        if path.startswith('/testbed/emails/'):
            return json.dumps([{'name': k.rsplit('/', 1)[-1], 'path': k, 'type': 'file'}
                               for k in self.files if k.startswith(path + '/')])
        if path in ('/testbed', '/testbed/data'):
            return json.dumps([{'name': k.rsplit('/', 1)[-1], 'path': k, 'type': 'file'}
                               for k in self.files if k.rsplit('/', 1)[0] == path])
        return '[directory does not exist]'
    def read_raw(self, path):
        return self.files.get(path, '[file does not exist]')


class NoChat:
    def chat(self, *args, **kwargs):
        raise AssertionError('pure predicted-state checks must not disclose content to a model')


def test_email_docx_evidence_handoff_and_verdict(monkeypatch):
    monkeypatch.setenv('TTS_OFFICEBENCH_VERIFIER_TYPED', '1')
    monkeypatch.setenv('TTS_OFFICEBENCH_VERIFIER_PERCHECK', '1')
    p = OfficeBenchWMPredictor(NoChat()); p.program_only = True
    p.bind_query_interface(Query())
    action = {'app': 'email', 'action': 'send_email', 'sender': 'Alice', 'recipient': 'Bob',
              'subject': 'Notice', 'content': 'PRIVATE_SUCCESS_CONTENT'}
    p.applied_writes['emails/Bob'] = [(action, 'Successfully sent email to Bob.')]
    p.applied_writes['data/a.docx'] = [({'app': 'word', 'action': 'create_new_file',
                                     'file_path': '/testbed/data/a.docx'}, 'Successfully created')]
    cfg = {'evaluation': [check('evaluate_file_exist', file='data/a.docx'),
           check('evaluate_contain', doc_type='email', username='Bob', keywords=['PRIVATE_SUCCESS_CONTENT'])]}
    specs = state_files_from_config(cfg)
    assert {'path': 'emails/Bob', 'doc_type': 'email'} in specs
    assert {'path': 'data/a.docx', 'doc_type': 'docx'} in specs
    env = OfficeBenchTaskEnv.__new__(OfficeBenchTaskEnv)
    state = env._predicted_state(None, p, specs, initial_state={})
    v = OfficeBenchVerifier(NoChat())
    assert v.predict_success('make doc and notify Bob', [], [], config=cfg, final_state=state) == 1
    assert v.n_python_failures == 0
    assert all(c['status'] == 'MET' for c in v.last_percheck_breakdown['checks'])
    state.pop('emails/Bob')
    assert v.predict_success('', [], [], config=cfg, final_state=state) == 0
    assert v.last_percheck_breakdown['unknown'] == 1


def test_single_email_read_keeps_native_body_and_suffix():
    msg = EmailMessage()
    msg['From'], msg['To'], msg['Subject'] = 'a@example.com', 'b@example.com', 'Notice'
    msg.set_content('private ' + 'long content ' * 40)
    p = OfficeBenchWMPredictor(NoChat()); p.program_only = True
    p.bind_query_interface(Query({'/testbed/emails/Bob/a.eml': msg.as_string()}))
    obs = p.predict_read({'app': 'email', 'action': 'read_email', 'username': 'Bob', 'email_id': 'a'}, None, dirty=False)
    assert obs == ('Successfully read email a for Bob:\nFrom: a@example.com\nTo: b@example.com\n'
                   'Subject: Notice\nContent: ' + msg.get_content() + '...').strip()
    assert p.predict_read({'app': 'email', 'action': 'read_email', 'username': 'Bob',
                           'email_id': 'missing'}, None, dirty=False) == 'Failed to read email missing for Bob.'


def test_shell_move_snapshot_and_mkdir_creation_are_pure():
    files = {'/testbed/data/a.docx': 'PRIVATE_DOCUMENT'}
    p = OfficeBenchWMPredictor(NoChat()); p.program_only = True
    p.bind_query_interface(Query(files))
    move = {'app': 'shell', 'action': 'command', 'command': 'mv /testbed/data/a.docx /testbed/data/b.docx'}
    assert p.predict_write(move) == 'Successfully executed command: ' + move['command'] + '. The output was [].'
    assert p.predict_file_state('data/b.docx', '[file does not exist]') == 'PRIVATE_DOCUMENT'
    assert p.predict_file_state('data/a.docx', 'PRIVATE_DOCUMENT') == '[file does not exist]'
    p.predict_write({'app': 'shell', 'action': 'command', 'command': 'mkdir -p /testbed/data/reports'})
    assert p.predict_write({'app': 'word', 'action': 'create_new_file',
        'file_path': '/testbed/data/reports/report.docx'}) == 'Successfully create new file /testbed/data/reports/report.docx'
    assert files == {'/testbed/data/a.docx': 'PRIVATE_DOCUMENT'}


def test_zero_pool_cannot_discard_incumbent_for_unknown_audit(monkeypatch):
    from tts.trajectory import RolloutResult, Trajectory, RewardInfo
    from tts.config import Config, SolverConfig
    monkeypatch.setenv('TTS_OFFICEBENCH_COMPARATIVE_RERANK', '1')
    env = OfficeBenchTaskEnv.__new__(OfficeBenchTaskEnv)
    env._instruction, env._config = '', {}
    rs = [RolloutResult(i, Trajectory(steps=[], done=True), RewardInfo(reward=0.0)) for i in range(2)]
    rs[0].predicted.metadata['hard_invalid'] = True
    rs[1].predicted.metadata['verifier_percheck'] = {'met': 0, 'unmet': 1, 'unknown': 2, 'total': 3}
    env.refine_rollout_scores(Config(solver=SolverConfig(llm_agent='gpt-5.4')), rs)
    assert rs[0].predicted.metadata['selection_override']['best'] == 0
