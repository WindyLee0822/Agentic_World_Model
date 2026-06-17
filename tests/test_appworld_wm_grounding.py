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
from tts.env.appworld_wm import AppWorldWMPredictor, install_wm_requester


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
