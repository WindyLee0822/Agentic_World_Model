"""AppWorld world-model request routing: read-grounded prefix before the first write.

These tests exercise :func:`tts_qwen.env.appworld_wm.install_wm_requester` with a fake ``world`` and a
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

from tts_qwen.env import appworld_wm
from tts_qwen.env import appworld_agentworld
from tts_qwen.env.appworld_wm import (
    AppWorldWMPredictor,
    AppWorldWMQuery,
    WMPredictedHTTPError,
    install_wm_requester,
    run_agentic_read_loop,
)
from tts_qwen.env.real_appworld import (
    _install_simulated_token_replay_aliases,
    _verifier_read_only_request,
)
from tts_qwen.llm.agentworld import AgentWorldSession


class _FakeChat:
    """A chat client whose ``complete`` returns a canned JSON write/read response."""

    def __init__(self, payload):
        self._payload = payload
        self.calls = 0

    def complete(self, messages, **kwargs):
        self.calls += 1
        return f"PYTHON_FINAL:\nprint_json({self._payload!r})"


class _FakeRequester:
    """Records every real request it receives (i.e. every call routed as a read)."""

    def __init__(self):
        self.calls = []

    def request(self, app, api, client=None, raise_on_failure=None, show=False, track=True, **data):
        self.calls.append((app, api, dict(data)))
        if (app, api) == ("supervisor", "show_active_task"):
            return {"instruction": "task", "status": "pending", "answer": None}
        return {"real": True, "app": app, "api": api}


def test_selected_replay_rebinds_only_ephemeral_simulated_access_token():
    class ReplayRequester:
        def __init__(self):
            self.calls = []

        def request(self, app, api, client=None, raise_on_failure=None,
                    show=False, track=True, **data):
            self.calls.append((app, api, dict(data)))
            if api == "login":
                return {"access_token": "real-opaque-token"}
            return {"ok": True}

    world = type("ReplayWorld", (), {})()
    world.requester = ReplayRequester()
    _install_simulated_token_replay_aliases(world)

    world.requester.request(
        _app_name="amazon", _api_name="login", username="u", password="p"
    )
    world.requester.request(
        "amazon", "show_cart", access_token="simulated-amazon-token"
    )
    world.requester.request(
        "spotify", "show_playlists", access_token="simulated-spotify-token"
    )

    assert world.requester.calls[1][2]["access_token"] == "real-opaque-token"
    assert world.requester.calls[2][2]["access_token"] == "simulated-spotify-token"


class _FakeWorld:
    def __init__(self):
        self.requester = _FakeRequester()


class _FakeRecord:
    def __init__(self, **values):
        self.values = values

    def to_dict(self, keep_computed=False):
        return dict(self.values)


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
    install_wm_requester(world, predictor, predict_all_actions=False)

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


def test_exact_shadow_actions_execute_only_on_disposable_world(methods):
    world = _FakeWorld()
    chat = _FakeChat({"message": "must not be used"})
    predictor = AppWorldWMPredictor(chat)
    install_wm_requester(world, predictor, exact_shadow_actions=True)

    written = world.requester.request("spotify", "like_song", song_id=1)
    read = world.requester.request("spotify", "show_song", song_id=1)

    assert written == {"real": True, "app": "spotify", "api": "like_song"}
    assert read == {"real": True, "app": "spotify", "api": "show_song"}
    assert chat.calls == 0
    assert [(app, api) for app, api, _ in world.requester.calls] == [
        ("spotify", "like_song"), ("spotify", "show_song")
    ]
    assert predictor.n_writes == 1
    assert predictor.first_write_seen is True


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


def test_strict_simulation_never_calls_real_requester(methods, monkeypatch):
    world = _FakeWorld()
    predictor = AppWorldWMPredictor(_FakeChat({"message": "simulated"}))
    monkeypatch.setattr(appworld_wm, "_available_app_names", lambda world: ("spotify",))
    monkeypatch.setattr(appworld_wm, "_public_api_docs", lambda app: {
        "show_song": {"app_name": app, "api_name": "show_song", "description": "Show one song."}
    })
    predictor.predict_read = lambda app, api, args, baseline, schema, dirty=True: {
        "simulated": True,
        "app": app,
        "api": api,
    }

    install_wm_requester(
        world,
        predictor,
        predict_all_actions=True,
        simulate_all_actions=True,
    )

    passwords = world.requester.request("supervisor", "show_account_passwords")
    login = world.requester.request("spotify", "login", username="u", password="p")
    read = world.requester.request("spotify", "show_song", song_id=1)
    docs = world.requester.request(
        "api_docs", "show_api_descriptions", app_name="spotify"
    )

    assert passwords == [
        {"account_name": "spotify", "password": "simulated-password"}
    ]
    assert login == {"access_token": "simulated-spotify-token"}
    assert read == {"simulated": True, "app": "spotify", "api": "show_song"}
    assert docs == [{"name": "show_song", "description": "Show one song."}]
    assert world.requester.calls == []
    assert predictor.query_interface is None


def test_strict_persistent_simulator_routes_reads_and_writes_without_real_calls(
    methods, monkeypatch
):
    class FakeSimulator:
        def __init__(self):
            self.calls = []

        def predict_api(
            self, app, api, arguments, *, reference_context=None, max_tokens=None
        ):
            self.calls.append(
                (app, api, dict(arguments), reference_context, max_tokens)
            )
            return {"predicted": f"{app}.{api}"}

    world = _FakeWorld()
    predictor = AppWorldWMPredictor(_FakeChat({"unused": True}))
    simulator = FakeSimulator()
    predictor.bind_persistent_simulator(simulator)
    monkeypatch.setattr(appworld_wm, "_available_app_names", lambda world: ("spotify",))

    install_wm_requester(
        world,
        predictor,
        predict_all_actions=True,
        simulate_all_actions=True,
    )

    read = world.requester.request("spotify", "show_song", song_id=7)
    write = world.requester.request("spotify", "like_song", song_id=7)

    assert read == {"predicted": "spotify.show_song"}
    assert write == {"predicted": "spotify.like_song"}
    assert [(app, api) for app, api, _, _, _ in simulator.calls] == [
        ("spotify", "show_song"),
        ("spotify", "like_song"),
    ]
    assert [call[-1] for call in simulator.calls] == [3072, 2048]
    assert world.requester.calls == []
    assert predictor.query_interface is None
    assert predictor.n_predicted_reads == 1
    assert predictor.n_writes == 1


@pytest.mark.parametrize("paginated", [False, True])
def test_strict_simulator_only_paginates_documented_endpoints(monkeypatch, paginated):
    from types import SimpleNamespace

    calls = []
    parameters = ([{"name": "page_index"}, {"name": "page_limit"}]
                  if paginated else [])
    monkeypatch.setattr(appworld_wm, "_api_doc", lambda app, api: {
        "method": "GET", "parameters": parameters,
        "response_schemas": {"success": []},
    })
    predictor = AppWorldWMPredictor(_FakeChat({}))
    predictor.bind_persistent_simulator(SimpleNamespace(
        predict_api=lambda *args, **kwargs: calls.append(kwargs) or [],
    ))
    world = _FakeWorld()
    install_wm_requester(world, predictor, simulate_all_actions=True)
    assert world.requester.request("spotify", "show_song_queue") == []
    prompt = calls[0]["reference_context"]
    assert ("This API has no page_index/page_limit pagination" in prompt) is not paginated
    assert world.requester.calls == []


def test_strict_invalid_arguments_cannot_become_successful_mutations(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(appworld_wm, "_api_doc", lambda app, api: {
        "method": "POST", "parameters": [{"name": "source_file_path", "required": True}],
    })
    predictor = AppWorldWMPredictor(_FakeChat({}))
    calls = []
    predictor.bind_persistent_simulator(SimpleNamespace(
        predict_api=lambda *args, **kwargs: calls.append(args) or {"message": "success"},
    ))
    world = _FakeWorld()
    install_wm_requester(world, predictor, simulate_all_actions=True)
    with pytest.raises(Exception, match="Missing required parameters: source_file_path"):
        world.requester.request("file_system", "move_file", source_path="~/a")
    assert calls == []
    assert predictor.write_history == []
    assert world.requester.calls == []


def test_strict_persistent_simulator_prompts_stateful_collection_invariants(
    monkeypatch,
):
    class FakeSimulator:
        def __init__(self):
            self.calls = []

        def predict_api(
            self, app, api, arguments, *, reference_context=None, max_tokens=None
        ):
            self.calls.append((app, api, reference_context))
            return [] if api in {
                "show_transactions",
                "show_received_payment_requests",
                "show_account_passwords",
            } else {
                "message": "ok"
            }

    docs = {
        ("venmo", "show_transactions"): {
            "method": "GET",
            "parameters": [
                {"name": "page_index", "required": False},
                {"name": "page_limit", "required": False},
            ],
        },
        ("venmo", "show_received_payment_requests"): {
            "method": "GET",
            "parameters": [{"name": "page_index", "required": False}],
        },
        ("venmo", "approve_payment_request"): {
            "method": "POST",
            "parameters": [
                {"name": "payment_request_id", "required": True},
                {"name": "payment_card_id", "required": False},
            ],
        },
        ("spotify", "remove_song_from_playlist"): {
            "method": "POST",
            "parameters": [
                {"name": "playlist_id", "required": True},
                {"name": "song_id", "required": True},
            ],
        },
        ("spotify", "add_song_to_playlist"): {
            "method": "POST",
            "parameters": [
                {"name": "playlist_id", "required": True},
                {"name": "song_id", "required": True},
            ],
        },
        ("amazon", "move_product_from_wish_list_to_cart"): {
            "method": "POST",
            "parameters": [
                {"name": "product_id", "required": True},
                {"name": "quantity", "required": False},
            ],
        },
        ("amazon", "place_order"): {
            "method": "POST",
            "parameters": [
                {"name": "payment_card_id", "required": True},
                {"name": "address_id", "required": True},
            ],
        },
        ("spotify", "login"): {
            "method": "POST",
            "parameters": [
                {"name": "username", "required": True},
                {"name": "password", "required": True},
            ],
        },
        ("file_system", "show_file"): {
            "method": "GET",
            "parameters": [
                {"name": "file_path", "required": True},
                {"name": "access_token", "required": True},
            ],
        },
        ("supervisor", "show_account_passwords"): {
            "method": "GET",
            "parameters": [],
        },
    }
    monkeypatch.setattr(appworld_wm, "_api_doc", lambda app, api: docs.get((app, api), {}))
    monkeypatch.setattr(
        appworld_wm,
        "_api_method",
        lambda app, api: docs.get((app, api), {}).get("method", "GET"),
    )
    monkeypatch.setattr(appworld_wm, "_success_schema", lambda app, api: {})
    monkeypatch.setattr(
        appworld_wm,
        "_available_app_names",
        lambda world: ("amazon", "spotify", "venmo", "file_system"),
    )

    world = _FakeWorld()
    predictor = AppWorldWMPredictor(_FakeChat({"unused": True}))
    predictor.predict_fixed_file_read = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("persistent file read bypassed the learned simulator")
    )
    predictor.configure_initial_state({
        "supervisor": {"email": "task-user@example.com"},
        "apps": {
            "venmo": {
                "User": [
                    {"id": 7, "email": "task-user@example.com"},
                    {"id": 8, "email": "sender@example.com"},
                ],
                "PaymentRequest": [
                    {"id": 101, "sender_id": 8, "receiver_id": 7},
                    {"id": 102, "sender_id": 7, "receiver_id": 8},
                ],
            }
        },
    })
    simulator = FakeSimulator()
    predictor.bind_persistent_simulator(simulator)
    install_wm_requester(
        world, predictor, predict_all_actions=True, simulate_all_actions=True
    )

    world.requester.request(
        "venmo", "show_transactions", page_index=1, page_limit=20
    )
    world.requester.request(
        "venmo", "show_received_payment_requests", page_index=0
    )
    world.requester.request(
        "venmo", "approve_payment_request", payment_request_id=101, payment_card_id=5
    )
    world.requester.request(
        "spotify", "remove_song_from_playlist", playlist_id=8, song_id=22
    )
    world.requester.request(
        "spotify", "add_song_to_playlist", playlist_id=99, song_id=22
    )
    world.requester.request(
        "amazon", "move_product_from_wish_list_to_cart", product_id=9, quantity=2
    )
    world.requester.request(
        "amazon", "place_order", payment_card_id=5, address_id=6
    )
    world.requester.request("spotify", "login", username="u", password="wrong")
    world.requester.request(
        "file_system", "show_file", file_path="~/x", access_token="virtual"
    )
    world.requester.request("supervisor", "show_account_passwords")

    contexts = {(app, api): context for app, api, context in simulator.calls}
    assert "stable slice" in contexts[("venmo", "show_transactions")]
    assert "Never repeat a row" in contexts[("venmo", "show_transactions")]
    assert "documented page limit" in contexts[("venmo", "show_transactions")]
    assert "receiver_id equals that task-user ID" in contexts[
        ("venmo", "show_received_payment_requests")
    ]
    assert '"id":101' in contexts[("venmo", "show_received_payment_requests")]
    assert '"id":102' not in contexts[("venmo", "show_received_payment_requests")]
    assert "subtract amounts from all earlier successful approvals" in contexts[
        ("venmo", "approve_payment_request")
    ]
    assert "scoped to the explicit target collection" in contexts[
        ("spotify", "remove_song_from_playlist")
    ]
    assert "true move, not a copy" in contexts[
        ("amazon", "move_product_from_wish_list_to_cart")
    ]
    assert "clear_cart does not restore wishlist membership" in contexts[
        ("amazon", "move_product_from_wish_list_to_cart")
    ]
    assert "never replaces or silently drops pre-existing" in contexts[
        ("amazon", "move_product_from_wish_list_to_cart")
    ]
    assert "compute the payable total" in contexts[("amazon", "place_order")]
    assert "insufficient-balance failure" in contexts[("amazon", "place_order")]
    assert "successfully created playlist starts empty" in contexts[
        ("spotify", "add_song_to_playlist")
    ]
    assert "account_name is the app identifier" in contexts[
        ("supervisor", "show_account_passwords")
    ]
    assert "Wrong credentials" in contexts[("spotify", "login")]
    assert ("file_system", "show_file") in contexts


def test_agentworld_verifier_uses_exact_source_target_set_semantics():
    class FakeSession:
        max_tokens = 8192

        def __init__(self):
            self.reference_context = None

        def predict_tool(
            self, name, arguments, *, reference_context=None, max_tokens=None
        ):
            self.reference_context = reference_context
            return '{"success_probability":0.5,"success":true,"reason":"test"}'

    session = FakeSession()
    simulator = appworld_agentworld.AppWorldAgentWorldSimulator(
        session, strict_task_audit=True
    )
    simulator.verify_task("Move matching songs into an archive playlist.")

    assert "exact intersection" in session.reference_context
    assert "multiple source containers" in session.reference_context
    assert "already-approved, denied" in session.reference_context
    assert "only initially pending" in session.reference_context
    assert "debit that card's running balance" in session.reference_context
    assert "Amazon orders" in session.reference_context
    assert "uncaught API exception terminates that code block" in session.reference_context
    assert "does not prove it was removed" in session.reference_context
    assert "compact fixed task evidence" in session.reference_context
    assert "oversized hallucinated single-page response" in session.reference_context
    assert "wishlist-to-cart operations are moves" in session.reference_context
    assert "cart is subsequently cleared" in session.reference_context


def test_strict_agentworld_verifier_reconciles_explicit_failure_probability():
    class FakeSession:
        max_tokens = 8192

        def predict_tool(self, name, arguments, **kwargs):
            return '{"success_probability":0.8,"success":false,"reason":"extra item"}'

    strict = appworld_agentworld.AppWorldAgentWorldSimulator(
        FakeSession(), strict_task_audit=True
    )
    baseline = appworld_agentworld.AppWorldAgentWorldSimulator(FakeSession())

    assert strict.verify_task("Buy only the listed items.")[0] == 0.49
    # The standalone AgentWorld baseline retains its pre-existing interpretation.
    assert baseline.verify_task("Buy only the listed items.")[0] == 0.8


def test_strict_persistent_simulator_raises_predicted_duplicate_as_http_error():
    class FakeSession:
        def predict_tool(self, *_args, **_kwargs):
            return '{"message":"Song already exists in playlist"}'

    strict = appworld_agentworld.AppWorldAgentWorldSimulator(
        FakeSession(), strict_task_audit=True
    )

    with pytest.raises(appworld_agentworld.AgentWorldHTTPError):
        strict.predict_api(
            "spotify",
            "add_song_to_playlist",
            {"playlist_id": 10, "song_id": 20},
        )


def test_standalone_agentworld_duplicate_parsing_is_unchanged(monkeypatch):
    class FakeSession:
        def predict_tool(self, *_args, **_kwargs):
            return '{"message":"Song already exists in playlist"}'

    monkeypatch.setattr(appworld_agentworld, "_public_api_docs", lambda _app: {})
    baseline = appworld_agentworld.AppWorldAgentWorldSimulator(FakeSession())

    assert baseline.predict_api(
        "spotify",
        "add_song_to_playlist",
        {"playlist_id": 10, "song_id": 20},
    ) == {"message": "Song already exists in playlist"}


def test_variant_family_closure_uses_fixed_rows_without_candidate_execution():
    products = [
        _FakeRecord(id=10, name="Acme Trail Shoe", seller_id=3,
                    relative_size="small", color="blue", has_variations=True),
        _FakeRecord(id=11, name="Acme Trail Shoe", seller_id=3,
                    relative_size="medium", color="blue", has_variations=True),
        _FakeRecord(id=12, name="Acme Trail Shoe", seller_id=3,
                    relative_size="large", color="red", has_variations=True),
        _FakeRecord(id=99, name="Other Shoe", seller_id=4,
                    relative_size="large", color="red", has_variations=True),
    ]

    class ProductModel:
        @staticmethod
        def all():
            return products

    class Models:
        shop = type("ShopModels", (), {"Product": ProductModel})()

    world = type("World", (), {"models": Models()})()
    app_state = {
        "Order": [{"id": 7, "created_at": "2026-01-03"}],
        "OrderItem": [{"id": 8, "order_id": 7, "product_id": 11}],
        "Product": [products[1].to_dict(), products[3].to_dict()],
    }

    added = appworld_agentworld.expand_task_relevant_product_variants(
        world,
        "shop",
        app_state,
        "Return the Acme Trail Shoe and buy the next available size and color.",
    )

    assert added == 2
    assert [row["id"] for row in app_state["Product"]] == [11, 99]
    assert [row["id"] for row in app_state["Product"]].count(11) == 1
    assert app_state["ProductVariantFamily"] == [{
        "name": "Acme Trail Shoe",
        "seller_id": 3,
        "variants": [
            {"product_id": 10, "relative_size": "small", "color": "blue",
             "inventory_quantity": None},
            {"product_id": 11, "relative_size": "medium", "color": "blue",
             "inventory_quantity": None},
            {"product_id": 12, "relative_size": "large", "color": "red",
             "inventory_quantity": None},
        ],
    }]


def test_file_to_spotify_closure_adds_only_exact_frozen_title_matches():
    songs = [
        _FakeRecord(id=1, title="Already Linked", duration=100),
        _FakeRecord(id=2, title="Missing From Playlists", duration=200),
        _FakeRecord(id=3, title="Unrelated Song", duration=300),
    ]

    class SongModel:
        @staticmethod
        def all():
            return songs

    world = type(
        "World",
        (),
        {"models": type("Models", (), {
            "spotify": type("Spotify", (), {"Song": SongModel})()
        })()},
    )()
    state = {
        "apps": {
            "file_system": {
                "File": [{"content": "- Missing From Playlists by An Artist"}]
            },
            "spotify": {"Song": [songs[0].to_dict()]},
        }
    }

    added = appworld_agentworld.expand_file_referenced_spotify_songs(world, state)

    assert added == 1
    assert [row["id"] for row in state["apps"]["spotify"]["Song"]] == [1, 2]
    assert state["apps"]["spotify"]["Song"][1]["_retrieval_provenance"] == (
        "file_reference_catalog_match_only_not_playlist_membership"
    )


def test_variant_family_closure_joins_latest_matching_category_order():
    products = [
        _FakeRecord(id=10, name="Acme Trail Shoe", product_type="shoes", seller_id=3,
                    relative_size="small", color="blue", has_variations=True),
        _FakeRecord(id=11, name="Acme Trail Shoe", product_type="shoes", seller_id=3,
                    relative_size="medium", color="red", has_variations=True),
        _FakeRecord(id=99, name="Unrelated Camera", product_type="camera", seller_id=4,
                    relative_size="medium", color="black", has_variations=False),
    ]

    class ProductModel:
        @staticmethod
        def all():
            return products

    class Models:
        shop = type("ShopModels", (), {"Product": ProductModel})()

    world = type("World", (), {"models": Models()})()
    app_state = {
        "Order": [
            {"id": 7, "created_at": "2026-01-03"},
            {"id": 8, "created_at": "2026-02-03"},
        ],
        "OrderItem": [
            {"id": 70, "order_id": 7, "product_id": 10},
            {"id": 80, "order_id": 8, "product_id": 99},
        ],
        "Product": [products[0].to_dict(), products[2].to_dict()],
    }

    added = appworld_agentworld.expand_task_relevant_product_variants(
        world,
        "shop",
        app_state,
        "The last shoes I bought are too small; buy the next larger size in red.",
    )

    assert added == 1
    assert [row["id"] for row in app_state["Product"]] == [10, 99]
    assert [row["product_id"] for row in
            app_state["ProductVariantFamily"][0]["variants"]] == [10, 11]


def test_agentworld_unwraps_list_envelope_without_environment_read(monkeypatch):
    monkeypatch.setattr(
        appworld_agentworld,
        "_public_api_docs",
        lambda app: {"show_transactions": {"response_schemas": {"success": []}}},
    )
    result = appworld_agentworld._normalize_agentworld_result_shape(
        "venmo",
        "show_transactions",
        {"status": "success", "transactions": [{"transaction_id": 3}]},
        "unused",
    )
    assert result == [{"transaction_id": 3}]


def test_agentworld_history_bound_keeps_complete_recent_pairs():
    session = AgentWorldSession.__new__(AgentWorldSession)
    session.max_history_chars = 12
    session.messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "old-user"},
        {"role": "assistant", "content": "old-answer"},
        {"role": "user", "content": "new"},
        {"role": "assistant", "content": "answer"},
    ]
    assert session._bounded_messages() == [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "new"},
        {"role": "assistant", "content": "answer"},
    ]


def test_agentworld_context_overflow_retries_until_lower_bound_converges():
    class Completions:
        def __init__(self):
            self.budgets = []

        def create(self, **kwargs):
            budget = kwargs["max_tokens"]
            self.budgets.append(budget)
            if budget > 4096:
                prompt_lower_bound = 32768 - budget + 1
                raise RuntimeError(
                    "This model's maximum context length is 32768 tokens. "
                    f"However, you requested {budget} output tokens and your "
                    f"prompt contains at least {prompt_lower_bound} input tokens."
                )
            message = type("Message", (), {"content": '{"ok":true}'})()
            choice = type("Choice", (), {"message": message})()
            return type(
                "Response",
                (),
                {"choices": [choice], "usage": type("Usage", (), {"total_tokens": 7})()},
            )()

    completions = Completions()
    session = AgentWorldSession.__new__(AgentWorldSession)
    session.model = "Qwen3.5-35B-A3B"
    session.temperature = 0.0
    session.top_p = 0.95
    session.top_k = 20
    session.max_tokens = 8192
    session.max_history_chars = 8000
    session.context_overflow_retries = 7
    session.messages = [{"role": "system", "content": "fixed state"}]
    session.client = type(
        "Client", (), {"chat": type("Chat", (), {"completions": completions})()}
    )()
    session.total_tokens = 0
    session._turn = 0

    result = session.predict_tool("amazon.show_orders", {}, max_tokens=8192)

    assert result == '{"ok":true}'
    assert completions.budgets[0] == 8192
    assert completions.budgets[-1] <= 4096
    assert all(
        later < earlier
        for earlier, later in zip(completions.budgets, completions.budgets[1:])
    )
    assert session.total_tokens == 7


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
    install_wm_requester(world, predictor, predict_all_actions=False)

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
    install_wm_requester(world, predictor, predict_all_actions=False)

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
        content = messages[-1]["content"]
        if '"outcome"' in content:  # rejection-aware prompt
            payload = {
                "outcome": "failure",
                "status_code": 422,
                "response": self.failure_body,
            }
            return f"PYTHON_FINAL:\nprint_json({payload!r})"
        return f"PYTHON_FINAL:\nprint_json({self.success_payload!r})"


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


def test_fixed_initial_state_is_exposed_to_fully_simulated_reads(methods):
    chat = _AgenticScriptChat(["{}"])
    predictor = AppWorldWMPredictor(chat)
    predictor.python_disabled = True
    predictor.configure_initial_state({"apps": {"spotify": {"Song": [{"id": 7}]}}})

    predictor.predict_read(
        "spotify", "show_song", {"song_id": 7}, None, {"song_id": 1}, dirty=False
    )

    messages = chat.prompts
    assert "data['fixed_initial_state']" in messages[0]


def test_write_prediction_uses_read_then_predict_subagent(methods, monkeypatch):
    world = _FakeWorld()
    chat = _AgenticScriptChat([
        (
            "PYTHON_READ:\n"
            'print_json({"queries":[{"app":"spotify","api":"show_song",'
            '"args":{"song_id":7}}]})'
        ),
        'PYTHON_FINAL:\nprint_json({"message": "Liked."})',
    ])
    predictor = AppWorldWMPredictor(chat)
    # The textual READ/LIST/CALL/TRACE protocol is now a legacy opt-in.
    monkeypatch.setenv("TTS_APPWORLD_WM_TOOL_MODE", "text")
    monkeypatch.setenv("TTS_APPWORLD_PREDICT_ALL_PROGRAM_ONLY", "1")
    install_wm_requester(world, predictor)

    result = world.requester.request("spotify", "like_song", song_id=7)

    assert result == {"message": "Liked."}
    assert chat.calls == 2
    assert predictor.query_interface.reads == 1
    assert ("spotify", "show_song") in [(app, api) for app, api, _ in world.requester.calls]
    assert "data['reads'][0]['result']" in chat.prompts[-1]
    assert '"real": true' not in chat.prompts[-1].lower()


def test_predict_all_clean_read_uses_pregrounded_prompted_world_model(
        methods, monkeypatch):
    world = _FakeWorld()
    chat = _AgenticScriptChat([
        "PYTHON_FINAL:\n"
        "value = data['grounded_clean_read']\n"
        "print(value)",
    ])
    predictor = AppWorldWMPredictor(chat)
    monkeypatch.setenv("TTS_APPWORLD_PREGROUND_CLEAN_READS", "1")
    monkeypatch.setenv("TTS_APPWORLD_DETERMINISTIC_GROUNDED_CLEAN_READS", "0")
    monkeypatch.setenv("TTS_APPWORLD_WM_TOOL_MODE", "text")
    original_predict_read = predictor.predict_read

    def predict_read(*args, **kwargs):
        assert args[3] is None
        assert world.requester.calls == []
        return original_predict_read(*args, **kwargs)

    predictor.predict_read = predict_read
    install_wm_requester(world, predictor, predict_all_actions=True)

    result = world.requester.request("spotify", "show_song", song_id=7)

    assert result == {"app": "spotify", "api": "show_song", "real": True}
    assert chat.calls == 1
    assert predictor.n_predicted_reads == 1
    assert predictor.n_dirty_reads == 0
    assert len(world.requester.calls) == 1
    assert predictor.query_interface.reads == 1
    assert predictor.history[-1] == (
        "spotify",
        "show_song",
        {"song_id": 7},
        {"app": "spotify", "api": "show_song", "real": True},
    )
    assert "EXACT GROUNDED CLEAN-READ PROTOCOL" in chat.prompts[0]


def test_predict_all_can_use_non_forced_program_predictions(methods, monkeypatch):
    world = _FakeWorld()
    chat = _FakeChat({"message": "Liked."})
    predictor = AppWorldWMPredictor(chat)
    monkeypatch.setenv("TTS_APPWORLD_PREDICT_ALL_PROGRAM_ONLY", "0")

    install_wm_requester(world, predictor, predict_all_actions=True)

    assert predictor.program_only is False
    assert world.requester.request("spotify", "like_song", song_id=1) == {
        "message": "Liked."
    }
    assert chat.calls == 1


def test_predict_all_deterministic_grounded_read_still_uses_simulator(
        methods, monkeypatch):
    world = _FakeWorld()
    chat = _FakeChat({"message": "must not be used"})
    predictor = AppWorldWMPredictor(chat)
    monkeypatch.setenv("TTS_APPWORLD_PREGROUND_CLEAN_READS", "1")
    monkeypatch.setenv("TTS_APPWORLD_DETERMINISTIC_GROUNDED_CLEAN_READS", "1")
    original_predict_read = predictor.predict_read

    def predict_read(*args, **kwargs):
        assert args[3] is None
        assert world.requester.calls == []
        return original_predict_read(*args, **kwargs)

    predictor.predict_read = predict_read

    install_wm_requester(world, predictor, predict_all_actions=True)
    result = world.requester.request("spotify", "show_song", song_id=7)

    assert result == {"app": "spotify", "api": "show_song", "real": True}
    assert chat.calls == 0
    assert predictor.n_predicted_reads == 1
    assert predictor.n_dirty_reads == 0
    assert predictor.query_interface.reads == 1
    assert predictor.history[-1] == (
        "spotify",
        "show_song",
        {"song_id": 7},
        {"app": "spotify", "api": "show_song", "real": True},
    )


def test_predict_all_clean_read_python_protocol_copies_host_json():
    prompt = AppWorldWMPredictor._python_protocol(
        {
            "grounded_clean_read_required": True,
            "grounded_clean_read": {"song_id": 7},
        },
        required=False,
        context_chars=100,
        allow_reads=True,
    )

    assert "EXACT GROUNDED CLEAN-READ PROTOCOL" in prompt
    assert "Return a PYTHON_FINAL program" in prompt
    assert "Only generated Python may process" in prompt


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
