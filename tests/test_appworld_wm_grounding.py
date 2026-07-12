"""AppWorld world-model request routing: read-grounded prefix before the first write.

These tests exercise :func:`tts.env.appworld_wm.install_wm_requester` with a fake ``world`` and a
fake chat client (no AppWorld package, no API access).  They pin the routing contract:

* every read (any non-mutating method, plus passthrough apps/auth) executes against the *real*
  requester — exactly, no prediction;
* a write (``POST``/``PUT``/``PATCH``/``DELETE``) is predicted (the real requester is never called);
* a read of an app a prior write dirtied is predicted (post-write state);
* **invariant** — until the first write, no read is ever predicted (the read-only prefix is fully
  read-grounded), even for an app that gets written later.
"""

from __future__ import annotations

import json

import pytest

from tts.env import appworld_wm
from tts.env.appworld_wm import AppWorldWMPredictor, install_wm_requester, WMPredictedHTTPError


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
    }
    monkeypatch.setattr(appworld_wm, "_api_method",
                        lambda app, api: table.get((app, api), "GET"))
    monkeypatch.setattr(appworld_wm, "_success_schema", lambda app, api: {"message": "ok"})
    monkeypatch.setattr(appworld_wm, "_api_doc", lambda app, api: {})
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
