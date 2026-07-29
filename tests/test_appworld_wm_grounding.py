"""AppWorld world-model request routing: read-grounded prefix before the first write.

These tests exercise :func:`tts.env.appworld_wm.install_wm_requester` with a fake ``world`` and a
fake chat client (no AppWorld package, no API access).  They pin the routing contract:

* every documented read executes against the *real* requester — exactly, no prediction;
* a write (``POST``/``PUT``/``PATCH``/``DELETE``) is predicted (the real requester is never called);
* task completion is predictor-local and unknown API metadata is blocked fail-closed;
* a read of an app a prior write dirtied is predicted (post-write state);
* **invariant** — until the first write, no read is ever predicted (the read-only prefix is fully
  read-grounded), even for an app that gets written later.
"""

from __future__ import annotations

import json

import pytest

from tts.env import appworld_wm
from tts.env.appworld_wm import (
    AppWorldWMPredictor,
    AppWorldWMQuery,
    WMPredictedHTTPError,
    install_wm_requester,
    run_agentic_read_loop,
)
from tts.env.real_appworld import _verifier_read_only_request


class _FakeChat:
    """A chat client whose ``complete`` returns a canned JSON write/read response."""

    def __init__(self, payload):
        self._payload = payload
        self.calls = 0

    def complete(self, messages, **kwargs):
        self.calls += 1
        return json.dumps(self._payload)


class _FakeRequester:
    """Records every real request it receives (i.e. every call routed as a read)."""

    def __init__(self):
        self.calls = []

    def request(self, app, api, client=None, raise_on_failure=None, show=False, track=True, **data):
        self.calls.append((app, api, dict(data)))
        if (app, api) == ("supervisor", "show_active_task"):
            return {"instruction": "task", "status": "pending", "answer": None}
        return {"real": True, "app": app, "api": api}


class _FakeWorld:
    def __init__(self):
        self.requester = _FakeRequester()


@pytest.fixture
def methods(monkeypatch):
    """Map each (app, api) to an HTTP method for the routing logic."""
    table = {
        ("spotify", "show_song"): "GET",
        ("spotify", "ping"): "HEAD",          # non-mutating, non-GET -> must be a read
        ("spotify", "like_song"): "POST",     # write
        ("venmo", "show_balance"): "GET",
        ("supervisor", "complete_task"): "POST",
        ("supervisor", "show_active_task"): "GET",
        ("spotify", "login"): "POST",
        ("spotify", "logout"): "POST",
        ("admin", "credit_to_card"): "POST",
    }
    monkeypatch.setattr(appworld_wm, "_api_method",
                        lambda app, api: table.get((app, api), "GET"))
    monkeypatch.setattr(appworld_wm, "_api_doc",
                        lambda app, api: (
                            {"method": table[(app, api)]}
                            if (app, api) in table else {}
                        ))
    monkeypatch.setattr(appworld_wm, "_success_schema", lambda app, api: {"message": "ok"})
    return table


def test_reads_before_first_write_are_real(methods):
    world = _FakeWorld()
    predictor = AppWorldWMPredictor(_FakeChat({"message": "Liked."}))
    install_wm_requester(world, predictor)

    # A GET and a HEAD before any write must both hit the real requester (read-grounded prefix).
    r1 = world.requester.request("spotify", "show_song", song_id=1)
    r2 = world.requester.request("spotify", "ping")
    assert r1["real"] is True and r2["real"] is True
    assert [(a, b) for a, b, _ in world.requester.calls] == [
        ("spotify", "show_song"), ("spotify", "ping")]
    assert predictor.first_write_seen is False
    assert predictor.n_writes == 0


def test_write_is_predicted_not_executed(methods):
    world = _FakeWorld()
    chat = _FakeChat({"message": "Liked."})
    predictor = AppWorldWMPredictor(chat)
    install_wm_requester(world, predictor)

    out = world.requester.request("spotify", "like_song", song_id=1)
    assert out == {"message": "Liked."}            # predicted payload
    assert chat.calls == 1                          # the LLM predicted it
    assert world.requester.calls == []              # the real requester was NOT called
    assert predictor.first_write_seen is True
    assert "spotify" in predictor.dirty_apps


def test_supervisor_completion_is_local_logical_state(methods):
    world = _FakeWorld()
    chat = _FakeChat({"message": "unused"})
    predictor = AppWorldWMPredictor(chat)
    install_wm_requester(world, predictor)

    result = world.requester.request(
        "supervisor", "complete_task", answer="42", status="success")

    assert result == {"message": "Marked the active task complete."}
    assert world.requester.calls == []
    assert chat.calls == 0
    assert predictor.simulated_task_completion == {"answer": "42", "status": "success"}
    assert predictor.write_history[-1][0:2] == ("supervisor", "complete_task")

    active = world.requester.request("supervisor", "show_active_task", track=False)
    assert active["status"] == "success"
    assert active["answer"] == "42"
    assert [(app, api) for app, api, _ in world.requester.calls] == [
        ("supervisor", "show_active_task")]


def test_logout_and_admin_writes_are_never_executed(methods):
    world = _FakeWorld()
    chat = _FakeChat({"message": "simulated"})
    predictor = AppWorldWMPredictor(chat)
    install_wm_requester(world, predictor)

    login = world.requester.request("spotify", "login", username="u", password="p")
    logout = world.requester.request("spotify", "logout", access_token="token")
    credit = world.requester.request(
        "admin", "credit_to_card", card_id=1, amount=10)

    assert login["real"] is True
    assert logout == {"message": "simulated"}
    assert credit == {"message": "simulated"}
    assert [(app, api) for app, api, _ in world.requester.calls] == [
        ("spotify", "login")]


def test_unknown_api_metadata_is_blocked_fail_closed(methods):
    world = _FakeWorld()
    predictor = AppWorldWMPredictor(_FakeChat({"message": "unused"}))
    install_wm_requester(world, predictor)

    with pytest.raises(Exception, match="blocked to preserve read-only"):
        world.requester.request("spotify", "undocumented_endpoint")
    body = world.requester.request(
        "spotify", "undocumented_endpoint", raise_on_failure=False)

    assert "blocked to preserve read-only" in body["message"]
    assert world.requester.calls == []


def test_dirty_read_predicted_clean_read_real(methods):
    world = _FakeWorld()
    chat = _FakeChat({"message": "post-write"})
    predictor = AppWorldWMPredictor(chat)
    install_wm_requester(world, predictor)

    world.requester.request("spotify", "like_song", song_id=1)   # writes -> spotify dirty
    real_calls_after_write = len(world.requester.calls)

    # read of the DIRTY app -> predicted (real requester still consulted for pre-write baseline)
    dirty = world.requester.request("spotify", "show_song", song_id=1)
    assert dirty == {"message": "post-write"}

    # read of an UNTOUCHED app -> exact real read
    clean = world.requester.request("venmo", "show_balance")
    assert clean["real"] is True
    assert ("venmo", "show_balance") in [(a, b) for a, b, _ in world.requester.calls]


def test_invariant_no_prediction_before_first_write(methods):
    """Reads to an app that is written *later* are still real while no write has happened yet."""
    world = _FakeWorld()
    chat = _FakeChat({"message": "x"})
    predictor = AppWorldWMPredictor(chat)
    install_wm_requester(world, predictor)

    # Read spotify BEFORE writing it -> must be real (not predicted), chat untouched.
    pre = world.requester.request("spotify", "show_song", song_id=1)
    assert pre["real"] is True
    assert chat.calls == 0
    assert predictor.first_write_seen is False


class _RejectChat:
    """Returns a fixed SUCCESS payload for the first-time stable-success prompt, and a FAILURE wrapper
    for the rejection-aware prompt (which asks for an ``"outcome"`` field)."""

    def __init__(self, success_payload, failure_body):
        self.success_payload = success_payload
        self.failure_body = failure_body
        self.calls = 0

    def complete(self, messages, **kwargs):
        self.calls += 1
        content = messages[0]["content"]
        if '"outcome"' in content:  # rejection-aware prompt
            return json.dumps({"outcome": "failure", "status_code": 422, "response": self.failure_body})
        return json.dumps(self.success_payload)


def test_repeat_write_predicts_rejection_and_raises(methods):
    """A repeat of the same state-changing write to the same target is predicted as a real rejection."""
    world = _FakeWorld()
    chat = _RejectChat({"message": "Liked."}, {"message": "You have already liked this song."})
    predictor = AppWorldWMPredictor(chat)
    install_wm_requester(world, predictor)

    first = world.requester.request("spotify", "like_song", song_id=7)
    assert first == {"message": "Liked."}
    assert predictor.applied_writes  # recorded as applied

    # the SAME write again -> the WM raises the real-format HTTP error (the agent must react to it).
    with pytest.raises(Exception) as exc:
        world.requester.request("spotify", "like_song", song_id=7)
    assert "Response status code is 422" in str(exc.value)
    assert "already liked" in str(exc.value).lower()
    # a rejected write does NOT mutate state: still a single applied write, n_writes counts the attempt.
    assert len(predictor.applied_writes) == 1


def test_rejected_write_returns_body_when_raise_on_failure_false(methods):
    world = _FakeWorld()
    chat = _RejectChat({"message": "Liked."}, {"message": "You have already liked this song."})
    predictor = AppWorldWMPredictor(chat)
    install_wm_requester(world, predictor)

    world.requester.request("spotify", "like_song", song_id=7)
    # raise_on_failure=False -> the real requester returns the failure body instead of raising.
    out = world.requester.request("spotify", "like_song", song_id=7, raise_on_failure=False)
    assert out == {"message": "You have already liked this song."}


def test_distinct_writes_each_succeed(methods):
    """Distinct targets are not duplicates -> each uses the success path (burst-dedup preserved)."""
    world = _FakeWorld()
    chat = _RejectChat({"message": "Liked."}, {"message": "already liked"})
    predictor = AppWorldWMPredictor(chat)
    install_wm_requester(world, predictor)

    a = world.requester.request("spotify", "like_song", song_id=1)
    b = world.requester.request("spotify", "like_song", song_id=2)
    assert a == {"message": "Liked."} and b == {"message": "Liked."}
    assert len(predictor.applied_writes) == 2


def test_query_interface_lists_read_apis_and_blocks_writes(methods, monkeypatch):
    world = _FakeWorld()
    predictor = AppWorldWMPredictor(_FakeChat({"message": "Liked."}))
    install_wm_requester(world, predictor)
    query = predictor.query_interface
    assert isinstance(query, AppWorldWMQuery)

    monkeypatch.setattr(appworld_wm, "_api_docs", lambda app: {
        "show_song": {"method": "GET", "description": "Show one song."},
        "like_song": {"method": "POST", "description": "Like one song."},
    })
    listing = query.list("spotify")
    assert "spotify.show_song" in listing
    assert "spotify.like_song" not in listing

    before = len(world.requester.calls)
    blocked = query.call("spotify.like_song", {"song_id": 1})
    assert blocked.startswith("ERROR:")
    assert len(world.requester.calls) == before

    # Agent-routing passthrough exceptions are NOT subagent read permissions.
    assert query.call("supervisor.complete_task", {}).startswith("ERROR:")
    assert query.call("spotify.login", {}).startswith("ERROR:")
    assert query.call("gmail.send_email", {}).startswith("ERROR:")
    assert len(world.requester.calls) == before


def test_verifier_request_requires_documented_read_only_api(methods):
    calls = []

    def request(app, api, **kwargs):
        calls.append((app, api, kwargs))
        return {"ok": True}

    assert _verifier_read_only_request(
        request, "spotify", "show_song", song_id=1) == {"ok": True}
    blocked_write = _verifier_read_only_request(
        request, "spotify", "like_song", song_id=1)
    blocked_unknown = _verifier_read_only_request(
        request, "spotify", "undocumented_endpoint")
    blocked_transport = _verifier_read_only_request(
        request, "spotify", "show_song", track=False)

    assert "not a documented read-only API" in blocked_write["error"]
    assert "not a documented read-only API" in blocked_unknown["error"]
    assert "Transport-only" in blocked_transport["error"]
    assert [(app, api) for app, api, _ in calls] == [("spotify", "show_song")]


def test_read_credentials_are_hydrated_without_exposing_token(methods, monkeypatch):
    seen = []

    def request(app, api, **kwargs):
        seen.append((app, api, kwargs))
        return {"song_id": 7}

    monkeypatch.setattr(appworld_wm, "_api_doc", lambda app, api: {
        "method": "GET",
        "parameters": [{"name": "access_token", "required": True}],
    })
    predictor = AppWorldWMPredictor(_FakeChat({"message": "unused"}))
    predictor.note_read(
        "spotify", "login", {"username": "u"}, {"access_token": "real-secret-token"})
    query = AppWorldWMQuery(request, predictor)

    output = query.read(
        "spotify.show_song", {"song_id": 7, "access_token": "<redacted>"})

    assert seen[0][2]["access_token"] == "real-secret-token"
    assert "real-secret-token" not in output


def test_dirty_query_returns_pristine_state_plus_trace(methods):
    world = _FakeWorld()
    predictor = AppWorldWMPredictor(_FakeChat({"message": "Liked."}))
    install_wm_requester(world, predictor)

    world.requester.request("spotify", "like_song", song_id=1)
    logical = predictor.query_interface.read("spotify.show_song", {"song_id": 1})

    assert '"pre_write_result"' in logical
    assert '"predicted_transitions"' in logical
    assert "APPLIED" in logical
    assert "like_song" in logical


class _AgenticScriptChat:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0
        self.prompts = []

    def complete(self, messages, **kwargs):
        self.prompts.append(messages[-1]["content"])
        response = self.responses[min(self.calls, len(self.responses) - 1)]
        self.calls += 1
        return response


def test_write_prediction_uses_read_then_predict_subagent(methods, monkeypatch):
    world = _FakeWorld()
    chat = _AgenticScriptChat([
        'READ spotify.show_song {"song_id": 7}',
        'PREDICT: {"message": "Liked."}',
    ])
    predictor = AppWorldWMPredictor(chat)
    # The textual READ/LIST/CALL/TRACE protocol is now a legacy opt-in.
    monkeypatch.setenv("TTS_APPWORLD_WM_TOOL_MODE", "text")
    install_wm_requester(world, predictor)

    result = world.requester.request("spotify", "like_song", song_id=7)

    assert result == {"message": "Liked."}
    assert chat.calls == 2
    assert predictor.query_interface.reads == 1
    assert ("spotify", "show_song") in [(app, api) for app, api, _ in world.requester.calls]
    assert "Facts gathered so far" in chat.prompts[-1]


def test_uniform_loop_supports_list_call_and_trace(methods, monkeypatch):
    world = _FakeWorld()
    predictor = AppWorldWMPredictor(_FakeChat({"message": "ok"}))
    install_wm_requester(world, predictor)
    query = predictor.query_interface
    monkeypatch.setattr(appworld_wm, "_api_docs", lambda app: {
        "show_song": {"method": "GET", "description": "Show one song."},
    })
    responses = iter([
        "LIST spotify\nCALL venmo.show_balance {}\nTRACE spotify",
        'PREDICT: {"message": "done"}',
    ])

    terminal = run_agentic_read_loop(
        lambda prompt, budget: next(responses),
        query,
        "Predict a result.",
        "PREDICT: <JSON>",
    )

    assert terminal == ("predict", '{"message": "done"}')
    assert query.lists == 1
    assert query.calls == 1
    assert query.traces == 1
