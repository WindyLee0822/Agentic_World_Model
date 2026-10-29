"""Regressions for the isolated Ours predict-all execution path."""

from types import SimpleNamespace

import pytest

from tts.env import appworld_wm
from tts.env.appworld_ours_wm import AppWorldOursWMPredictor
from tts.env.appworld_wm import AppWorldWMPredictor, WMHardTimeout, install_wm_requester


class ScriptChat:
    def __init__(self, *responses):
        self.responses = iter(responses)
        self.calls = 0

    def complete(self, *args, **kwargs):
        self.calls += 1
        return next(self.responses)


def test_predicted_read_write_dirty_read_and_verdict_never_execute_writes(monkeypatch):
    monkeypatch.setenv("TTS_APPWORLD_PREDICT_ALL_PROGRAM_ONLY", "1")
    monkeypatch.setenv("TTS_APPWORLD_PREGROUND_CLEAN_READS", "0")
    monkeypatch.setenv("TTS_APPWORLD_DETERMINISTIC_GROUNDED_CLEAN_READS", "0")
    methods = {"show_items": "GET", "delete_item": "DELETE"}
    monkeypatch.setattr(appworld_wm, "_api_doc", lambda app, api: {
        "method": methods.get(api, "POST"), "response_schemas": {"success": {"message": "string"}}})
    real_calls = []

    def request(app, api, **kwargs):
        real_calls.append((app, api))
        assert api == "show_items", "candidate write reached the real environment"
        return [{"id": 1}, {"id": 2}]

    read = 'PYTHON_READ:\nprint_json({"queries":[{"app":"example","api":"show_items","args":{}}]})'
    chat = ScriptChat(
        read, "PYTHON_FINAL:\nprint_json(data['reads'][-1]['result'])",
        "PYTHON_FINAL:\nprint_json({'message':'Deleted.'})",
        read,
        "PYTHON_FINAL:\ndeleted = {w['args']['id'] for w in data['writes']}\n"
        "print_json([r for r in data['reads'][-1]['result'] if r['id'] not in deleted])",
        "PYTHON_FINAL:\nprint_json({'verdict':'SUCCESS'})",
    )
    predictor = AppWorldOursWMPredictor(chat)
    world = SimpleNamespace(requester=SimpleNamespace(request=request))
    install_wm_requester(world, predictor, predict_all_actions=True)
    predictor.start_rollout()
    assert world.requester.request("example", "show_items") == [{"id": 1}, {"id": 2}]
    assert world.requester.request("example", "delete_item", id=1) == {"message": "Deleted."}
    assert world.requester.request("example", "show_items") == [{"id": 2}]
    assert predictor.predict_success("Delete item 1", [], [], reference_checklist="Item 1 deleted") == 1.0
    assert predictor.n_predicted_reads == 2
    assert predictor.n_writes == 1
    assert predictor.n_python_calls == 6
    assert real_calls == [("example", "show_items"), ("example", "show_items")]
    assert chat.calls == 6


@pytest.mark.parametrize("operation", ["read", "write"])
def test_exhaustion_cannot_fabricate_observations(operation):
    predictor = AppWorldOursWMPredictor(ScriptChat(), max_llm_calls=1)
    predictor.n_llm_calls = 1
    with pytest.raises(WMHardTimeout, match="unpredicted observation"):
        if operation == "read":
            predictor.predict_read("example", "show_items", {}, [{"id": 1}], {}, dirty=True)
        else:
            predictor.predicted_write("example", "delete_item", {"id": 1}, {"message": "string"})
    assert predictor.write_history == []
    assert predictor.history == []
    assert predictor.hard_timed_out


def test_setup_time_is_not_charged_and_baseline_defaults_are_unchanged(monkeypatch):
    for key in ("TTS_APPWORLD_WM_TIME_BUDGET_S", "TTS_APPWORLD_WM_HARD_TIMEOUT_S",
                "TTS_APPWORLD_OURS_WM_TIME_BUDGET_S", "TTS_APPWORLD_OURS_WM_HARD_TIMEOUT_S"):
        monkeypatch.delenv(key, raising=False)
    baseline = AppWorldWMPredictor(ScriptChat())
    ours = AppWorldOursWMPredictor(ScriptChat())
    monkeypatch.setattr("tts.env.appworld_ours_wm._wall_clock", lambda: 10000.0)
    ours.start_rollout()
    assert (baseline.time_budget_s, baseline.hard_timeout_s) == (240.0, 600.0)
    assert (ours._deadline, ours._hard_deadline) == (11800.0, 12400.0)


def test_explicit_budgets_are_respected(monkeypatch):
    monkeypatch.setenv("TTS_APPWORLD_WM_TIME_BUDGET_S", "100")
    monkeypatch.setenv("TTS_APPWORLD_OURS_WM_TIME_BUDGET_S", "200")
    assert AppWorldOursWMPredictor(ScriptChat()).time_budget_s == 200
    assert AppWorldOursWMPredictor(ScriptChat(), time_budget_s=300).time_budget_s == 300
    assert AppWorldWMPredictor(ScriptChat()).time_budget_s == 100


def test_program_budgets_allow_final_print_without_changing_baseline(monkeypatch):
    monkeypatch.delenv("TTS_APPWORLD_OURS_WM_PROGRAM_TOKENS", raising=False)
    monkeypatch.delenv("TTS_APPWORLD_OURS_WM_VERIFIER_TOKENS", raising=False)
    caps = []

    class Chat:
        def complete(self, messages, **kwargs):
            caps.append(kwargs["max_tokens"])
            return "PYTHON_FINAL:\nprint_json({'verdict': 'SUCCESS'})"

    ours = AppWorldOursWMPredictor(Chat())
    ours._complete("write", max_tokens=800)
    ours._complete("verdict", max_tokens=700, budgeted=False)
    ours._complete("large", max_tokens=8192)
    AppWorldWMPredictor(Chat())._complete("baseline", max_tokens=800)
    assert caps == [2048, 4096, 8192, 800]


@pytest.mark.parametrize("feedback", [False, True])
def test_agent_history_records_executed_code_only_when_enabled(monkeypatch, feedback):
    from tts.env import appworld_react_agent

    monkeypatch.setattr(appworld_react_agent, "build_initial_messages", lambda *a, **kw: [])
    messages_seen = []
    response = "```python\nx = 1\n```\n```python\ny = 2\n```"

    class Chat:
        def complete(self, messages, **kwargs):
            messages_seen.append([dict(m) for m in messages])
            return response

    executed = []
    world = SimpleNamespace(execute=lambda code: executed.append(code) or "ok",
                            task_completed=lambda: False)
    appworld_react_agent.run_react_code_agent(
        world, Chat(), max_steps=2, executed_code_feedback=feedback)
    assert executed == ["x = 1", "x = 1"]
    prior_assistant = next(m['content'] for m in messages_seen[1] if m['role'] == 'assistant')
    assert ("y = 2" not in prior_assistant) == feedback


def test_simulator_receives_documented_read_signatures_without_write_tools(monkeypatch):
    from tts.env import appworld_ours_wm

    monkeypatch.setattr(appworld_ours_wm, "_api_docs", lambda app: {
        "show_profile": {"method": "GET", "description": "Show public profile.",
                         "parameters": [{"name": "email", "required": True}]},
        "delete_user": {"method": "DELETE", "parameters": []},
    })
    captured = []
    monkeypatch.setattr(AppWorldWMPredictor, "_predict",
                        lambda self, prompt, final_instr, **kwargs: captured.append(prompt) or "result")
    result = AppWorldOursWMPredictor(ScriptChat())._predict(
        "Simulate the call", "PREDICT", python_context={"app": "venmo"})
    assert result == "result"
    assert "venmo.show_profile(email [required])" in captured[0]
    assert "delete_user" not in captured[0]
    assert "plain-text ERROR results are not JSON" in captured[0]


def test_computed_fact_names_survive_read_rounds_only_for_ours():
    values = {"account": {"verified": True}, "python_read_error_1": "overwrite"}
    ours_facts = {"python_read_error_1": "original"}
    baseline_facts = {}
    AppWorldOursWMPredictor(ScriptChat())._record_python_facts(ours_facts, values, 2)
    AppWorldWMPredictor(ScriptChat())._record_python_facts(baseline_facts, values, 2)
    assert ours_facts["account"] == {"verified": True}
    assert ours_facts["python_read_error_1"] == "original"
    assert baseline_facts == {"python_computed_facts_2": values}


def test_generated_fact_can_drive_the_next_simulator_verdict():
    predictor = AppWorldOursWMPredictor(ScriptChat(
        "PYTHON_READ:\nprint_json({'queries': [], 'facts': {'account': {'verified': True}}})",
        "PYTHON_FINAL:\nprint_json({'verdict': 'SUCCESS' if data['facts']['account']['verified'] else 'FAIL'})",
    ))
    predictor.bind_query_interface(SimpleNamespace(
        read_raw=lambda *args: pytest.fail("computed facts do not need a database query")))
    predictor.program_only = True
    verdict = predictor._predict(
        "Judge the account", "VERDICT", python_context={},
        max_tokens=700, python_output_kind="verdict", budgeted=False)
    assert verdict == "VERDICT: SUCCESS"
    assert predictor.n_python_calls == 2
    assert predictor.n_python_failures == 0


def test_optional_diagnostics_capture_swallowed_transport_error_without_solver_output(monkeypatch, tmp_path, capsys):
    import json

    class FailingChat:
        def complete(self, *args, **kwargs):
            raise RuntimeError("diagnostic transport failure")

    monkeypatch.setenv("TTS_APPWORLD_OURS_DIAGNOSTICS_DIR", str(tmp_path))
    predictor = AppWorldOursWMPredictor(FailingChat())
    predictor.set_diagnostic_context("task-1", 3)
    assert predictor._complete("prediction", max_tokens=800) == ""
    predictor._chat.stream.close()
    record = json.loads(next(tmp_path.glob("wm-*.jsonl")).read_text())
    assert record["task_id"] == "task-1" and record["seed"] == 3
    assert record["error"] == "diagnostic transport failure"
    assert "RuntimeError" in record["traceback"]
    assert capsys.readouterr().out == ""


def test_large_native_page_is_preserved_by_ours_and_baseline_limit_is_unchanged(monkeypatch):
    for key in ("TTS_APPWORLD_WM_PYTHON_MAX_OUTPUT_CHARS",
                "TTS_APPWORLD_OURS_WM_PYTHON_MAX_OUTPUT_CHARS"):
        monkeypatch.delenv(key, raising=False)
    page = [{"order_id": i, "products": [{"description": "x" * 1800}]} for i in range(20)]
    code = "print_json(data['page'])"
    ours = AppWorldOursWMPredictor(ScriptChat())
    result = ours._run_python_prediction(code, {"page": page})
    import json
    assert result.error is None
    assert json.loads(result.output) == page
    baseline_result = AppWorldWMPredictor(ScriptChat())._run_python_prediction(code, {"page": page})
    assert "20000" in baseline_result.error


def test_get_download_is_predicted_and_never_available_as_grounding(monkeypatch):
    monkeypatch.setattr(appworld_wm, "_api_doc", lambda app, api: {
        "method": "GET", "response_schemas": {"success": {"message": "string", "file_path": "string"}}})
    predictor = AppWorldOursWMPredictor(ScriptChat(
        "PYTHON_FINAL:\nprint_json({'outcome': 'success', 'response': {'message': 'Downloaded.', 'file_path': '~/downloads/receipt.txt'}})"))
    world = SimpleNamespace(requester=SimpleNamespace(
        request=lambda *a, **kw: pytest.fail("download mutated the real environment")))
    install_wm_requester(world, predictor, predict_all_actions=True)
    result = world.requester.request("amazon", "download_order_receipt", order_id=1)
    assert result["file_path"] == "~/downloads/receipt.txt"
    assert predictor.dirty_apps == {"amazon", "file_system"}
    assert "ERROR" in predictor.query_interface.read_raw("amazon.download_order_receipt", {"order_id": 1})
    baseline = AppWorldWMPredictor(ScriptChat())
    assert not baseline.is_write_api("amazon", "download_order_receipt")
    assert baseline.allows_grounding("amazon", "download_order_receipt")


def test_write_preconditions_receive_applied_transitions(monkeypatch):
    monkeypatch.setenv("TTS_APPWORLD_PREDICT_ALL_PROGRAM_ONLY", "1")
    monkeypatch.setattr(appworld_wm, "_api_doc", lambda app, api: {
        "method": "POST", "response_schemas": {"success": {"message": "string"}}})
    chat = ScriptChat(
        "PYTHON_READ:\nprint_json({'queries': [], 'facts': {'added': any("
        "w['name'] == 'amazon.add_product_to_cart' and w['args']['product_id'] == 17 "
        "for w in data['writes'])}})",
        "PYTHON_FINAL:\n"
        "print_json({'outcome':'success','status_code':200,'response':"
        "{'message':'Ordered.' if data['facts']['added'] else 'Empty cart.'}})",
    )
    predictor = AppWorldOursWMPredictor(chat)
    predictor.bind_query_interface(appworld_wm.AppWorldWMQuery(
        lambda *a, **kw: pytest.fail('no real write may execute'), predictor))
    args = {"product_id": 17, "quantity": 1}
    predictor._record_applied_write("amazon", "add_product_to_cart", args,
                                    {"message": "Added."}, ("add", 17))
    result, failure = predictor._predict_with_rejection(
        "amazon", "place_order", {}, {"message": "string"}, False)
    assert result == {"message": "Ordered."}
    assert failure is None
    assert predictor.n_python_calls == 2


def test_venmo_funding_uses_generated_prediction_only_for_ours(monkeypatch):
    monkeypatch.setattr(appworld_wm, "_api_doc", lambda app, api: {
        "method": "POST", "response_schemas": {"success": {"message": "string", "bank_transfer_id": "integer"}}})
    schema = {"message": "string", "bank_transfer_id": "integer"}
    baseline = AppWorldWMPredictor(ScriptChat())
    monkeypatch.setattr(baseline, "_venmo_balance", lambda: 100.0)
    result, failure = baseline._predict_grounded_venmo_funding(
        "withdraw_from_venmo_balance", {"amount": 10}, schema)
    assert result["bank_transfer_id"] == 1_000_001
    assert failure is None
    chat = ScriptChat("PYTHON_FINAL:\nprint_json({'outcome':'success', 'status_code':200, "
                      "'response':{'message':'Money withdrawn from Venmo balance.', 'bank_transfer_id':42}})")
    ours = AppWorldOursWMPredictor(chat)
    ours.program_only = True
    monkeypatch.setattr(ours, "_venmo_balance", lambda: 100.0)
    result = ours.predict_write("venmo", "withdraw_from_venmo_balance", {"amount": 10}, schema)
    assert result["bank_transfer_id"] == 42
    assert ours.n_python_calls == 1
    assert chat.calls == 1


@pytest.mark.parametrize('mode,response', [
    ('clean_read', 'null'),
    ('dirty_read', 'not JSON'),
    ('write_response', 'null'),
    ('write_with_rejection', '{"outcome":"success","response":null}'),
])
def test_unusable_predictions_cannot_reach_inherited_synthetic_fallback(monkeypatch, mode, response):
    monkeypatch.setattr(AppWorldWMPredictor, '_predict', lambda *a, **kw: response)
    predictor = AppWorldOursWMPredictor(ScriptChat())
    with pytest.raises(RuntimeError, match='Simulator program did not produce'):
        predictor._predict('Predict', 'JSON', python_context={'mode':mode})


def test_write_validation_reads_nested_predicted_state_without_real_writes(monkeypatch):
    methods = {'show_items':'GET', 'place_order':'POST'}
    monkeypatch.setattr(appworld_wm, '_api_doc', lambda app, api: {
        'method': methods.get(api, 'POST'), 'response_schemas': {'success': {'message':'string'}}})
    read = "PYTHON_READ:\nprint_json({'queries':[{'app':'example','api':'show_items','args':{}}]})"
    chat = ScriptChat(
        read,  # Enclosing write requests logical state.
        read,  # Nested dirty-read predictor requests pristine evidence.
        "PYTHON_FINAL:\nprint_json(json_value(data['reads'][0]['result']) + "
        "[w['args']['id'] for w in data['writes'] if w['api']=='add_item'])",
        "PYTHON_FINAL:\nprint_json({'outcome':'success','response': "
        "{'message':'Ordered.' if json_value(data['reads'][0]['result']) == [17] else 'Wrong state'}})",
    )
    calls = []
    def native_read(app, api, **args):
        calls.append((app, api))
        assert api == 'show_items'
        return []
    predictor = AppWorldOursWMPredictor(chat)
    predictor.program_only = True
    predictor.bind_query_interface(appworld_wm.AppWorldWMQuery(native_read, predictor))
    predictor._record_applied_write('example','add_item',{'id':17},{'message':'Added.'},('add',17))
    response, failure = predictor._predict_with_rejection('example','place_order',{}, {'message':'string'},False)
    assert response == {'message':'Ordered.'}
    assert failure is None
    assert calls == [('example','show_items')]
    assert predictor.n_predicted_reads == 1
    assert predictor.n_python_calls == 4
    cached = predictor._read_write_precondition('example', 'show_items', {})
    cached.append(999)
    assert predictor._read_write_precondition('example', 'show_items', {}) == [17]
    assert calls == [('example','show_items')]
    assert predictor._logical_read_key('example', 'show_items', {'access_token':'a'}) != (
        predictor._logical_read_key('example', 'show_items', {'access_token':'b'}))
    predictor._record_applied_write('example','add_item',{'id':18},{'message':'Added.'},('add',18))
    chat.responses = iter([read,
        "PYTHON_FINAL:\nprint_json([w['args']['id'] for w in data['writes'] if w['api']=='add_item'])"])
    assert predictor._read_write_precondition('example', 'show_items', {}) == [17,18]
    assert calls == [('example','show_items'), ('example','show_items')]


def test_generated_inspection_stdout_is_facts_but_invalid_queries_are_not_executed():
    parse = AppWorldOursWMPredictor._parse_python_read_payload
    assert parse('{"cart_keys":["cart_items"]}') == ([], {'cart_keys':['cart_items']})
    stdout = 'Cart keys: ["cart_items"]\nCard: {"payment_card_id": 17}'
    assert parse(stdout) == ([], {'generated_stdout':stdout})
    assert parse('{"queries":[{"app":"amazon"}]}') is None


def test_debug_stdout_does_not_execute_embedded_query_envelopes():
    text = 'Inspecting a query example:\n{"queries":[{"app":"amazon","api":"show_cart","args":{}}]}'
    assert AppWorldOursWMPredictor._parse_python_read_payload(text) == ([], {'generated_stdout':text})


def test_forbidden_type_inspection_gets_actionable_repair_without_relaxing_guard():
    predictor = AppWorldOursWMPredictor(ScriptChat())
    bad = predictor._run_python_prediction("print_json(type(data).__name__)", {})
    assert bad.error and 'str(type(value))' in bad.error
    assert not bad.output
    repaired = predictor._run_python_prediction("print_json(str(type(data)))", {})
    assert repaired.error is None
    assert 'dict' in repaired.output


def test_private_balance_read_does_not_relax_solver_or_allow_private_writes(monkeypatch):
    from tts.env import appworld_ours_wm
    doc = lambda app, api: {'method':'GET' if api=='show_card_balance' else 'POST',
                           'parameters':[{'name':'card_number'}]}
    monkeypatch.setattr(appworld_wm, '_api_doc', doc)
    monkeypatch.setattr(appworld_ours_wm, '_api_doc', doc)
    class Requester:
        raise_on_extra_parameters = True
        def request(self, app, api, **args):
            if self.raise_on_extra_parameters:
                raise ValueError('private API unavailable to solver')
            return {'balance':123}
    requester = Requester()
    predictor = AppWorldOursWMPredictor(ScriptChat())
    query = appworld_wm.AppWorldWMQuery(requester.request, predictor)
    predictor.bind_query_interface(query)
    assert query.read_value('admin.show_card_balance', {'card_number':123}) == {'balance':123}
    assert requester.raise_on_extra_parameters is True
    with pytest.raises(ValueError, match='private API unavailable'):
        requester.request('admin','show_card_balance',card_number=123)
    with pytest.raises(ValueError, match='not an allowed read'):
        query.read_value('admin.credit_to_card', {'card_number':123})
    with pytest.raises(ValueError, match='Unexpected parameters'):
        query.read_value('admin.show_card_balance', {'unexpected':123})


def test_verifier_program_failure_retains_candidate_with_zero_confidence(monkeypatch):
    calls = []
    def fail(*args, **kwargs):
        calls.append(1)
        raise RuntimeError('program-only AppWorld prediction did not produce a valid Python result')
    monkeypatch.setattr(AppWorldWMPredictor, 'predict_success', fail)
    p = AppWorldOursWMPredictor(ScriptChat())
    p.write_history.append(('example','add_item',{'id':17},{'message':'Added.'}))
    before = list(p.write_history)
    assert p.predict_success('task', [], []) == 0.0
    assert p.write_history == before
    assert calls == [1]
    assert p.verifier_error.startswith('program-only')
    with pytest.raises(RuntimeError):
        AppWorldWMPredictor(ScriptChat()).predict_success('task', [], [])
    def unrelated(*args, **kwargs):
        raise RuntimeError('unrelated bug')
    monkeypatch.setattr(AppWorldWMPredictor, 'predict_success', unrelated)
    with pytest.raises(RuntimeError, match='unrelated bug'):
        p.predict_success('task', [], [])


def test_verifier_inspects_fresh_reads_before_executing_final_program(monkeypatch):
    monkeypatch.setattr(appworld_wm, '_api_doc', lambda *args: {'method':'GET'})
    read = "PYTHON_READ:\nprint_json({'queries':[{'app':'example','api':'show_items','args':{}}]})"
    inspect = "PYTHON_READ:\nprint_json({'queries':[], 'facts':{'items':json_value(data['reads'][0]['result'])}})"
    chat = ScriptChat(read,
        "PYTHON_FINAL:\nprint_json({'verdict':'SUCCESS'})",  # skipped until inspection
        inspect,
        "PYTHON_FINAL:\nprint_json({'verdict':'SUCCESS' if data['facts']['items']==[17] else 'FAIL'})")
    p = AppWorldOursWMPredictor(chat)
    p.program_only = True
    p.bind_query_interface(appworld_wm.AppWorldWMQuery(lambda *a, **kw:[17],p))
    result=p._predict('Verify','VERDICT',python_context={'gold':{}},python_output_kind='verdict',budgeted=False,max_tokens=700)
    assert result=='VERDICT: SUCCESS'
    assert chat.calls==4
    assert p.n_python_calls==3  # The ungrounded final program was never executed.
    assert AppWorldWMPredictor(ScriptChat())._requires_read_inspection({},'verdict') is False
