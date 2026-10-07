import copy
import json
from types import SimpleNamespace

from tts_qwen.llm.python_tool import PurePythonTool
from tts_qwen.env.appworld_qwen_simulation import QwenAppWorldSession, _READ_PROGRAM_CACHE


def test_unknown_table_is_not_empty_but_legacy_is_unchanged():
    tool = PurePythonTool()
    data = {'state': {'apps': {'phone': {'Alarm': [{'id': 1, 'user_id': 9}], 'Contact': []}}}}
    original = copy.deepcopy(data)
    try:
        code = "print_json(data['state']['apps']['phone'].get('Alarms', []))"
        assert tool.run(code, data).output == '[]'
        guarded = tool.run(code, data, evidence_schema_guard=True)
        assert 'Unknown evidence key' in guarded.error
        assert 'Alarm' in guarded.error
        assert tool.run("print_json(data['state']['apps']['phone'].get('Contact', []))", data,
                        evidence_schema_guard=True).output == '[]'
        wrong_field = tool.run("print_json(data['state']['apps']['phone']['Alarm'][0].get('user', {}))",
                               data, evidence_schema_guard=True)
        assert 'Unknown evidence key' in wrong_field.error
        assert data == original
    finally:
        tool.close()


def test_optional_arguments_and_locally_added_fields_are_allowed():
    tool = PurePythonTool()
    data = {'arguments': {}, 'state': {'apps': {'phone': {'Alarm': [{'id': 1}, {'id': 2, 'label': None}]}}}}
    code = """row = data['state']['apps']['phone']['Alarm'][0]
row['snooze_minutes'] = 20
print_json([row.get('snooze_minutes'), row.get('label'), data['arguments'].get('page_index', 0)])"""
    try:
        assert json.loads(tool.run(code, data, evidence_schema_guard=True).output) == [20, None, 0]
    finally:
        tool.close()


def test_session_repairs_silent_wrong_table_and_supplies_json(monkeypatch):
    monkeypatch.setenv('TTS_APPWORLD_OURS_QWEN_EVIDENCE_GUARD', '1')
    monkeypatch.setenv('TTS_APPWORLD_OURS_QWEN_COMPUTED_READS', '1')
    monkeypatch.setattr('tts_qwen.env.appworld_wm._api_doc', lambda *args: {'method': 'GET'})
    _READ_PROGRAM_CACHE.clear()
    session = QwenAppWorldSession(model='qwen', base_url='http://unused/v1', initial_state={
        'apps': {'phone': {'Alarm': [{'id': 27, 'snooze_minutes': 10}]}}})
    programs = iter([
        "print(json.dumps(data['state']['apps']['phone'].get('Alarms', [])))",
        "print(json.dumps(data['state']['apps']['phone']['Alarm']))",
    ])
    calls = []
    def create(**kwargs):
        calls.append(copy.deepcopy(kwargs))
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=next(programs)))], usage=None)
    session.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    result = json.loads(session.predict_tool('phone.show_alarms', {}))
    assert result == [{'id': 27, 'snooze_minutes': 10}]
    assert len(calls) == 2
    assert 'Unknown evidence key' in calls[1]['messages'][-1]['content']
    assert len(session.computation_errors) == 1


def test_example_schema_does_not_force_nullable_evidence_to_empty_list():
    import pytest
    from tts_qwen.env.appworld_qwen_simulation import validate_api_prediction
    doc = {'response_schemas': {'success': [{'alarm_id': 1, 'repeat_days': ['monday']}]}}
    response = [{'alarm_id': 308, 'repeat_days': None}]
    validate_api_prediction(response, doc, allow_null_fields=True)
    with pytest.raises(ValueError):
        validate_api_prediction(response, doc)
    with pytest.raises(ValueError):
        validate_api_prediction(None, doc, allow_null_fields=True)
    with pytest.raises(ValueError):
        validate_api_prediction([{'alarm_id': 308}], doc, allow_null_fields=True)


def test_complete_children_restores_owned_order_items_without_other_orders(monkeypatch):
    from tts_qwen.env.appworld_qwen_simulation import complete_fixed_snapshot
    monkeypatch.setattr('tts_qwen.env.appworld_agentworld._record_to_dict', lambda row: dict(row))
    tables = {
        'User': [{'id': 1, 'email': 'owner@test'}, {'id': 2, 'email': 'other@test'}],
        'Order': [{'id': 10, 'user_id': 1}, {'id': 20, 'user_id': 2}],
        'OrderItem': [{'id': 101, 'order_id': 10}, {'id': 102, 'order_id': 10},
                      {'id': 201, 'order_id': 20}],
    }
    models = SimpleNamespace(SQLModel=SimpleNamespace(model_names=lambda: list(tables)), **{
        name: SimpleNamespace(all=lambda rows=rows: rows) for name, rows in tables.items()})
    world = SimpleNamespace(models=SimpleNamespace(amazon=models))
    initial = {'supervisor': {'email': 'owner@test'}, 'apps': {'amazon': {
        'User': [tables['User'][0]], 'Order': [tables['Order'][0]],
        'OrderItem': [tables['OrderItem'][0]],
    }}}
    legacy = copy.deepcopy(initial)
    monkeypatch.delenv('TTS_APPWORLD_OURS_QWEN_COMPLETE_CHILDREN', raising=False)
    complete_fixed_snapshot(world, legacy, {}, 'Find an item in my order')
    assert [r['id'] for r in legacy['apps']['amazon']['OrderItem']] == [101]
    fixed = copy.deepcopy(initial)
    monkeypatch.setenv('TTS_APPWORLD_OURS_QWEN_COMPLETE_CHILDREN', '1')
    complete_fixed_snapshot(world, fixed, {}, 'Find an item in my order')
    assert {r['id'] for r in fixed['apps']['amazon']['OrderItem']} == {101, 102}
    assert len(tables['OrderItem']) == 3


def test_complete_children_follows_nested_containment(monkeypatch):
    from tts_qwen.env.appworld_qwen_simulation import complete_fixed_snapshot
    monkeypatch.setenv('TTS_APPWORLD_OURS_QWEN_COMPLETE_CHILDREN', '1')
    monkeypatch.setattr('tts_qwen.env.appworld_agentworld._record_to_dict', lambda row: dict(row))
    tables = {
        'User': [{'id': 1, 'email': 'owner@test'}, {'id': 2, 'email': 'other@test'}],
        'Project': [{'id': 10, 'user_id': 1}, {'id': 20, 'user_id': 2}],
        'Task': [{'id': 30, 'project_id': 10}, {'id': 40, 'project_id': 20}],
        'TaskComment': [{'id': 50, 'task_id': 30, 'user_id': 2}],
        'Attachment': [{'id': 60, 'task_comment_id': 50}],
    }
    models = SimpleNamespace(SQLModel=SimpleNamespace(model_names=lambda: list(tables)), **{
        name: SimpleNamespace(all=lambda rows=rows: rows) for name, rows in tables.items()})
    packet = {'supervisor': {'email': 'owner@test'}, 'apps': {'todoist': {'User': [tables['User'][0]]}}}
    complete_fixed_snapshot(SimpleNamespace(models=SimpleNamespace(todoist=models)), packet, {}, 'Read my project')
    assert [r['id'] for r in packet['apps']['todoist']['Task']] == [30]
    assert [r['id'] for r in packet['apps']['todoist']['Attachment']] == [60]
