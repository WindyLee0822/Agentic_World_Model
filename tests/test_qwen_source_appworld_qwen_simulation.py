import copy
import json
from types import SimpleNamespace

from tts_qwen.env.appworld_qwen_simulation import NonThinkingChat, QwenAppWorldSession, complete_fixed_snapshot
from tts_qwen.llm.agentworld import AgentWorldSession


def test_nonthinking_adapter_is_request_local():
    calls = []
    client = SimpleNamespace(complete=lambda *args, **kwargs: calls.append(kwargs) or "code")
    extra = {"chat_template_kwargs": {"other": True}}
    assert NonThinkingChat(client).complete([], extra_body=extra) == "code"
    assert calls[0]["extra_body"]["chat_template_kwargs"] == {
        "other": True, "enable_thinking": False,
    }
    assert extra == {"chat_template_kwargs": {"other": True}}


def test_fixed_snapshot_is_bounded_balanced_and_never_mutated():
    state = {
        "supervisor": {"email": "owner@example.test"},
        "apps": {
            "amazon": {
                "Product": [{"id": i, "name": "x" * 400} for i in range(100)],
                "User": [{"id": 1001, "email": "owner@example.test"}],
                "CartEntry": [{"id": 1, "product_id": 99}],
            },
            "spotify": {"Song": [{"id": 7}]},
        },
    }
    original = copy.deepcopy(state)
    session = QwenAppWorldSession(initial_state=state, model="qwen", base_url="http://unused/v1")
    raw = session._snapshot("amazon", {"product_id": 99}, limit=2000)
    packet = json.loads(raw)
    assert len(raw) <= 2000
    assert set(packet["apps"]) == {"amazon"}
    assert packet["apps"]["amazon"]["Product"][0]["id"] == 99
    assert packet["apps"]["amazon"]["User"]
    assert packet["apps"]["amazon"]["CartEntry"]
    assert state == original


def test_thinking_override_does_not_change_default_agentworld_requests(monkeypatch):
    monkeypatch.setattr("tts_qwen.env.appworld_wm._api_doc", lambda *args: {})
    for session_cls, extra in ((AgentWorldSession, {}), (QwenAppWorldSession, {"initial_state": {}})):
        calls = []
        session = session_cls(model="qwen", base_url="http://unused/v1", system_prompt="test", **extra)
        def create(**kwargs):
            calls.append(kwargs)
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{"ok":true}'))], usage=None)
        session.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        assert json.loads(session.predict_tool("amazon.login", {"username": "test"}))["ok"]
        body = calls[0]["extra_body"]
        if session_cls is AgentWorldSession:
            assert "chat_template_kwargs" not in body
        else:
            assert body["chat_template_kwargs"] == {"enable_thinking": False}


def test_initial_closure_includes_contacts_and_friend_identities(monkeypatch):
    monkeypatch.setattr("tts_qwen.env.appworld_agentworld._record_to_dict", lambda row: dict(row))
    users = [{"id": 1, "email": "owner@test"}, {"id": 2, "email": "friend@test"}]

    def models(**tables):
        return SimpleNamespace(
            SQLModel=SimpleNamespace(model_names=lambda: list(tables)),
            **{name: SimpleNamespace(all=lambda rows=rows: rows) for name, rows in tables.items()},
        )

    world = SimpleNamespace(models=SimpleNamespace(
        phone=models(User=users, Contact=[{"id": 3, "user_id": 1, "email": "friend@test"},
                                         {"id": 4, "user_id": 2, "email": "outsider@test"}],
                     GlobalTextMessage=[{"id": 7, "sender_id": 1, "receiver_id": 2}]),
        venmo=models(User=users, Friendship=[{"id": 9, "user_id": 1, "friend_id": 2}]),
    ))
    packet = {"supervisor": {"email": "owner@test"}, "apps": {
        "phone": {"User": [users[0]]}, "venmo": {"User": [users[0]]},
    }}
    complete_fixed_snapshot(world, packet, {}, "Reset friends on venmo to match my phone friends")
    assert [row["id"] for row in packet["apps"]["phone"]["Contact"]] == [3]
    assert "GlobalTextMessage" not in packet["apps"]["phone"]
    assert {row["id"] for row in packet["apps"]["venmo"]["User"]} == {1, 2}


def test_computed_read_uses_copied_snapshot_and_preserves_whitespace(monkeypatch):
    monkeypatch.setenv("TTS_APPWORLD_OURS_QWEN_COMPUTED_READS", "1")
    monkeypatch.setattr("tts_qwen.env.appworld_wm._api_doc", lambda *args: {"method": "GET"})
    state = {"apps": {"file_system": {"File": [{"path": "~/x", "content": "one\ntwo\n"}]}}}
    original = copy.deepcopy(state)
    session = QwenAppWorldSession(initial_state=state, model="qwen", base_url="http://unused/v1")
    code = '''```python
import json
row = data['state']['apps']['file_system']['File'][0]
result = {'content': row['content']}
row['content'] = 'local mutation only'
print(json.dumps(result))
```'''
    def create(**kwargs):
        assert kwargs["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=code))], usage=None)
    session.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    result = json.loads(session.predict_tool("file_system.show_file", {"file_path": "~/x"}))
    assert result == {"content": "one\ntwo\n"}
    assert state == original
    assert session.n_computations == 1
    assert session.computation_errors == []


def test_computation_rejects_replacement_data_and_repairs(monkeypatch):
    monkeypatch.setenv("TTS_APPWORLD_OURS_QWEN_COMPUTED_READS", "1")
    monkeypatch.setattr("tts_qwen.env.appworld_wm._api_doc", lambda *args: {"method": "GET"})
    state = {"supervisor": {"email": "unique@test"}, "apps": {"phone": {"Contact": [{"id": 17}]}}}
    session = QwenAppWorldSession(initial_state=state, model="qwen", base_url="http://unused/v1")
    programs = iter([
        'import json\ndata = {"invented": True}\nprint(json.dumps(data))',
        'import json\nprint(json.dumps(data["state"]["apps"]["phone"]["Contact"]))',
    ])
    calls = []
    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=next(programs)))], usage=None)
    session.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    assert json.loads(session.predict_tool("phone.search_contacts", {})) == [{"id": 17}]
    assert len(calls) == 2
    assert "do not assign to data" in session.computation_errors[0]
    # Identical evidence can reuse the learned program, but still computes on
    # copied inputs and never needs an environment call.
    assert json.loads(session.predict_tool("phone.search_contacts", {})) == [{"id": 17}]
    assert len(calls) == 2
    assert session.computation_trace[-1]["cache_hit"]


def test_computed_verification_receives_all_apps_and_simulated_mutations(monkeypatch):
    monkeypatch.setenv("TTS_APPWORLD_OURS_QWEN_COMPUTED_VERIFIER", "1")
    state = {"apps": {"phone": {"Contact": [{"email": "a@test"}]}, "venmo": {"Friendship": []}}}
    session = QwenAppWorldSession(initial_state=state, model="qwen", base_url="http://unused/v1")
    session.transitions = [{"name": "venmo.add_friend", "arguments": {"user_email": "a@test"},
                            "observation": {"message": "success"}}]
    code = '''import json
expected = {row['email'] for row in data['state']['apps']['phone']['Contact']}
added = {row['arguments']['user_email'] for row in data['transitions']}
ok = expected == added
print(json.dumps({'success': ok, 'success_probability': float(ok), 'reason': 'set comparison'}))'''
    session.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kwargs:
        SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=code))], usage=None))))
    result = json.loads(session.predict_tool("__agentworld__.verify_task", {"instruction": "sync friends"}))
    assert result["success_probability"] == 1
    assert state["apps"]["venmo"]["Friendship"] == []


def test_repair_wrong_collection_element_shape(monkeypatch):
    monkeypatch.setenv("TTS_APPWORLD_OURS_QWEN_COMPUTED_READS", "1")
    monkeypatch.setattr("tts_qwen.env.appworld_wm._api_doc", lambda *args: {
        "method": "GET", "response_schemas": {"success": ["string"], "failure": {"message": "string"}},
    })
    state = {"apps": {"file_system": {"File": [{"path": "/home/test/x"}]}}}
    programs = iter([
        "import json\nprint(json.dumps(data['state']['apps']['file_system']['File']))",
        "import json\nprint(json.dumps([r['path'] for r in data['state']['apps']['file_system']['File']]))",
    ])
    session = QwenAppWorldSession(initial_state=state, model="qwen", base_url="http://unused/v1")
    session.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kwargs:
        SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=next(programs)))], usage=None))))
    assert json.loads(session.predict_tool("file_system.show_directory", {})) == ["/home/test/x"]
    assert len(session.computation_errors) == 1
    assert "response[0] must be str" in session.computation_errors[0]


def test_direct_prediction_repairs_schema_without_recording_invalid_write(monkeypatch):
    monkeypatch.setattr("tts_qwen.env.appworld_wm._api_doc", lambda *args: {
        "method": "POST", "response_schemas": {"success": {"playlist_id": 1}},
    })
    responses = iter(['{"id": 99}', '{"playlist_id": 99}'])
    calls = []
    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=next(responses)))], usage=None)
    session = QwenAppWorldSession(initial_state={}, model="qwen", base_url="http://unused/v1")
    session.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    result = json.loads(session.predict_tool("spotify.create_playlist", {"title": "test"}))
    assert result == {"playlist_id": 99}
    assert len(session.transitions) == 1
    assert session.transitions[0]["observation"] == result
    assert len(calls) == 2
    assert "missing documented fields" in str(calls[-1]["messages"])


def test_initial_snapshot_includes_catalog_and_only_owners_funding(monkeypatch):
    monkeypatch.setattr("tts_qwen.env.appworld_agentworld._record_to_dict", lambda row: dict(row))
    class Row(dict):
        def __getattr__(self, name):
            return self.get(name)

    def models(**tables):
        return SimpleNamespace(
            SQLModel=SimpleNamespace(model_names=lambda: list(tables)),
            **{name: SimpleNamespace(all=lambda rows=rows: rows) for name, rows in tables.items()},
        )

    world = SimpleNamespace(models=SimpleNamespace(
        admin=models(MainUser=[Row(id=7, email="owner@test"), Row(id=8, email="other@test")],
                     PaymentCard=[Row(id=1, main_user_id=7, card_number="111", balance=500),
                                  Row(id=2, main_user_id=8, card_number="222", balance=900)]),
        supervisor=models(Supervisor=[Row(id=1, email="owner@test")], Address=[], PaymentCard=[],
                          AccountPassword=[Row(id=1, account_name="amazon", password="secret")]),
        amazon=models(User=[Row(id=1, email="owner@test")],
                      Product=[Row(id=1, name="first"), Row(id=2, name="searchable")], Seller=[],
                      Order=[Row(id=100, user_id=2), Row(id=1, user_id=1)]),
    ))
    packet = {"supervisor": {"email": "owner@test"}, "apps": {"amazon": {}}}
    complete_fixed_snapshot(world, packet, {}, "Find and order a product")
    assert {row["id"] for row in packet["apps"]["amazon"]["Product"]} == {1, 2}
    assert [row["card_number"] for row in packet["apps"]["amazon"]["FundingCard"]] == ["111"]
    assert packet["apps"]["amazon"]["IdSequence"] == [{"table": "Order", "next_id": 101}]
    assert packet["apps"]["supervisor"]["AccountPassword"][0]["password"] == "secret"


def test_computation_resolves_app_specific_user_ids(monkeypatch):
    monkeypatch.setenv("TTS_APPWORLD_OURS_QWEN_COMPUTED_VERIFIER", "1")
    state = {"supervisor": {"email": "OWNER@test"}, "apps": {
        "supervisor": {"Supervisor": [{"id": 1, "email": "owner@test"}]},
        "venmo": {"User": [{"id": 1, "email": "other@test"}, {"id": 23, "email": "owner@test"}]},
        "phone": {"User": [{"id": 99, "email": "owner@test"}]},
    }}
    original = copy.deepcopy(state)
    code = '''import json
users = data['task_users']
ok = users['venmo']['id'] == 23 and users['phone']['id'] == 99
print(json.dumps({'success': ok, 'success_probability': float(ok)}))'''
    session = QwenAppWorldSession(initial_state=state, model="qwen", base_url="http://unused/v1")
    def create(**kwargs):
        assert "IDs are app-specific" in str(kwargs["messages"])
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=code))], usage=None)
    session.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    result = json.loads(session.predict_tool("__agentworld__.verify_task", {"instruction": "check fixture"}))
    assert result["success"]
    assert state == original


def test_computed_read_sees_app_ledger_and_applies_write_arguments(monkeypatch):
    monkeypatch.setenv("TTS_APPWORLD_OURS_QWEN_COMPUTED_READS", "1")
    monkeypatch.setattr("tts_qwen.env.appworld_wm._api_doc", lambda *args: {"method": "GET"})
    state = {"apps": {"simple_note": {"Note": [{"id": 17, "content": "old"}]}}}
    session = QwenAppWorldSession(initial_state=state, model="qwen", base_url="http://unused/v1")
    session.transitions = [
        {"name": "simple_note.update_note", "arguments": {"note_id": 17, "content": "new\n"},
         "observation": {"message": "Note updated successfully."}},
        {"name": "phone.delete_contact", "arguments": {"contact_id": 17},
         "observation": {"message": "Contact deleted successfully."}},
    ]
    code = '''import json
state = data['state']
rows = state['apps']['simple_note']['Note']
for t in data['transitions']:
    if t['app'] == 'simple_note' and t['api'] == 'update_note':
        for row in rows:
            if row['id'] == t['args']['note_id']:
                row['content'] = t['arguments']['content']
print(json.dumps(rows[0]))'''
    def create(**kwargs):
        prompt = str(kwargs["messages"])
        assert 'simple_note.update_note' in prompt
        assert 'Note updated successfully.' in prompt
        assert 'phone.delete_contact' not in prompt
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=code))], usage=None)
    session.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    assert json.loads(session.predict_tool("simple_note.show_note", {"note_id": 17}))["content"] == "new\n"
    assert state['apps']['simple_note']['Note'][0]['content'] == 'old'


def test_rejected_prediction_is_removed_from_history(monkeypatch):
    monkeypatch.setattr("tts_qwen.env.appworld_wm._api_doc", lambda *args: {
        "method": "POST", "response_schemas": {"success": {"playlist_id": 1}},
    })
    session = QwenAppWorldSession(initial_state={}, model="qwen", base_url="http://unused/v1")
    calls = []
    def create(**kwargs):
        calls.append(copy.deepcopy(kwargs))
        content = '{"wrong_id": 123}' if len(calls) == 1 else '{"playlist_id": 456}'
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))], usage=None)
    session.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    args = {"title": "before"}
    session.predict_tool("spotify.create_playlist", args)
    args["title"] = "after"
    assert session.transitions[0]["arguments"]["title"] == "before"
    assert not any('wrong_id' in m['content'] for m in calls[1]['messages'] if m['role'] == 'assistant')
    assert not any('wrong_id' in m['content'] for m in session.messages)
    assert len(session.transitions) == 1


def test_direct_prediction_keeps_ledger_after_history_eviction(monkeypatch):
    monkeypatch.setattr("tts_qwen.env.appworld_wm._api_doc", lambda *args: {"method": "GET"})
    session = QwenAppWorldSession(initial_state={}, model="qwen", base_url="http://unused/v1", max_history_chars=100)
    session.transitions = [{"name": "spotify.create_playlist", "arguments": {"title": "early playlist"},
                            "observation": {"playlist_id": 123}}]
    session.messages.extend([{'role': 'user', 'content': 'later read ' * 100},
                             {'role': 'assistant', 'content': '[]'}])
    def create(**kwargs):
        assert 'early playlist' in kwargs['messages'][0]['content']
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='[]'))], usage=None)
    session.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    assert json.loads(session.predict_tool('spotify.show_playlist_library', {})) == []


def test_dead_worker_falls_back_once_and_recovers_between_solver_steps(monkeypatch):
    from tts_qwen.llm.python_tool import PythonToolResult
    monkeypatch.setenv("TTS_APPWORLD_OURS_QWEN_COMPUTED_READS", "1")
    monkeypatch.setattr("tts_qwen.env.appworld_wm._api_doc", lambda *args: {"method": "GET"})
    events = []
    worker = SimpleNamespace(
        run=lambda *a, **kw: events.append('run') or PythonToolResult(error='worker unavailable', fatal=True),
        prepare=lambda: events.append('prepare') or True,
    )
    monkeypatch.setattr('tts_qwen.llm.python_tool.shared_python_tool', lambda: worker)
    responses = iter(['import json\nprint(json.dumps([]))', '[]'])
    session = QwenAppWorldSession(initial_state={}, model="qwen", base_url="http://unused/v1")
    session.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kwargs:
        SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=next(responses)))], usage=None))))
    assert json.loads(session.predict_tool('phone.search_contacts', {})) == []
    assert events == ['run']
    assert len(session.computation_errors) == 1
    NonThinkingChat(SimpleNamespace(complete=lambda *a, **kw: events.append('solver'))).complete([])
    assert events == ['run', 'prepare', 'solver']


def test_scoped_solver_thinking_has_bounded_code_fallback(monkeypatch):
    monkeypatch.setenv('TTS_APPWORLD_OURS_QWEN_SOLVER_THINKING', '1')
    calls = []
    responses = iter(['', '```python\nprint(1)\n```'])
    def complete(messages, **kwargs):
        calls.append((copy.deepcopy(messages), copy.deepcopy(kwargs)))
        return next(responses)
    messages = [{'role': 'user', 'content': 'task'}]
    original = copy.deepcopy(messages)
    raw = NonThinkingChat(SimpleNamespace(complete=complete)).complete(messages, max_tokens=2048)
    assert 'print(1)' in raw
    assert calls[0][1]['extra_body']['chat_template_kwargs']['enable_thinking'] is True
    assert calls[0][1]['max_tokens'] == 8192
    assert calls[1][1]['extra_body']['chat_template_kwargs']['enable_thinking'] is False
    assert calls[1][1]['max_tokens'] == 2048
    assert messages == original


def test_scoped_solver_thinking_keeps_executable_first_response(monkeypatch):
    monkeypatch.setenv('TTS_APPWORLD_OURS_QWEN_SOLVER_THINKING', '1')
    calls = []
    code = '```python\nprint(2)\n```'
    def complete(messages, **kwargs):
        calls.append(kwargs)
        return code
    assert NonThinkingChat(SimpleNamespace(complete=complete)).complete([]) == code
    assert len(calls) == 1


def test_queue_index_contract_repairs_computed_observation(monkeypatch):
    monkeypatch.setenv('TTS_APPWORLD_OURS_QWEN_COMPUTED_READS', '1')
    monkeypatch.setattr('tts_qwen.env.appworld_wm._api_doc', lambda *args: {
        'method': 'GET', 'response_schemas': {'success': [{'position': 1}]},
    })
    programs = iter([
        'import json\nprint(json.dumps([{"position": i + 1} for i in range(2)]))',
        'import json\nprint(json.dumps([{"position": i} for i in range(2)]))',
    ])
    calls = []
    def create(**kwargs):
        calls.append(copy.deepcopy(kwargs))
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=next(programs)))], usage=None)
    session = QwenAppWorldSession(initial_state={}, model='qwen', base_url='http://unused/v1')
    session.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    assert json.loads(session.predict_tool('spotify.show_song_queue', {})) == [{'position': 0}, {'position': 1}]
    assert 'zero-based' in str(calls[1]['messages'])
    assert 'zero-based' in session.computation_trace[0]['error']


def test_queue_index_contract_does_not_change_other_endpoints():
    import pytest
    from tts_qwen.env.appworld_qwen_simulation import validate_api_prediction
    doc = {'response_schemas': {'success': [{'position': 1}]}}
    validate_api_prediction([{'position': 1}], doc, 'other.show_items')
    validate_api_prediction([], doc, 'spotify.show_song_queue')
    with pytest.raises(ValueError, match='zero-based'):
        validate_api_prediction([{'position': 1}], doc, 'spotify.show_song_queue')
