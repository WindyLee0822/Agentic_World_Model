"""The REAL τ²-bench environments as trajectory-level :class:`~tts.abcs.TaskEnvironment`\\ s.

A *trajectory* is a sequence of real τ² tool-calls.  Execution and scoring cohere to the
**official** τ²-bench implementation for **every** domain (airline, retail, telecom, mock,
banking_knowledge):

* the real agent (B0) is the official τ² LLM customer-service agent conversing with the official
  user simulator over the real environment, scored by the official
  ``evaluate_simulation(..., EvaluationType.ALL)`` — so a single rollout reproduces the official
  τ²-bench number for that domain;
* stepping uses the official ``environment.get_response(ToolCall(...))`` (the tool execution that
  mutates the domain's Pydantic ``DB``);
* the read-grounded world model (B1/Ours) predicts observations and candidate rewards **without
  touching the real env**. Clean read tools execute exactly; writes and dirty reads are predicted
  from copied task/action/history data plus host-mediated, read-only evidence.

The recursive world-model subagent uses restricted Python by default. ``PYTHON_READ`` programs emit
structured record reads, id listings, or read-only tool calls; the host validates and executes them,
then copies their results into the next turn. ``PYTHON_FINAL`` programs compute the final observation
or success verdict. Generated code never receives the tau2 environment, database, toolkit, files,
network, credentials, or write tools. The former textual ``READ``/``LIST``/``CALL`` protocol remains
available only as an explicit ablation via ``TTS_TAU2_WM_TOOL_MODE=text``.

Requires the real ``tau2`` package (Python ≥3.12 — the ``dw`` conda env).
"""

from __future__ import annotations

import copy
import json
import os
import re
import time
from dataclasses import replace
from datetime import datetime
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

from ..abcs import EntityStore, RestrictedView, ShadowEvaluator, TaskEnvironment, WorldModel
from ..config import BudgetConfig, SimulatorConfig
from ..trajectory import Observation, RewardInfo, Step, Trajectory
from ..types import Action
from ..worldmodel import make_world_model
from .tau2_wm_python import (
    prepare_tau2_python_tool,
    run_tau2_python_prediction_loop,
    tau2_wm_tool_mode,
)

#: airline tools that mutate the backing DB — kept for backward-compatibility (the generic path
#: derives per-domain mutating tools from the toolkit via :func:`tau2_mutating_tools`).
MUTATING_TOOLS = {
    "book_reservation",
    "cancel_reservation",
    "update_reservation_flights",
    "update_reservation_passengers",
    "update_reservation_baggages",
    "send_certificate",
}


def real_tau2_available() -> bool:
    try:
        import tau2  # noqa: F401
    except Exception:
        return False
    return True


def real_tau2_subset_size(domain: str) -> int:
    from .datasets import tau2_subset_size

    if real_tau2_available():
        try:
            from tau2.registry import registry

            return len(list(registry.get_tasks_loader(domain)()))
        except Exception:
            pass
    return tau2_subset_size(domain)


def _env_constructor(domain: str):
    """The official ``get_environment`` factory for ``domain`` (accepts ``db=`` for replay/fork)."""
    from tau2.registry import registry

    return registry.get_env_constructor(domain)


#: Banking's ``get_environment`` builds a knowledge-base retrieval pipeline.  The backing DATABASE
#: (``TransactionalDB``) the world model reads is identical for every retrieval variant — only the
#: ``KB_search`` tool differs.  We therefore use the official retrieval (real KB) when an embedding
#: backend is configured (``OPENAI_API_KEY``), and otherwise fall back to ``bm25_grep`` so the env
#: still constructs offline: the WM keeps faithful read access to the real DB and predicts with its
#: own LLM; only ``KB_search`` is a BM25 approximation in that offline case.  Override explicitly
#: with ``TTS_BANKING_RETRIEVAL_VARIANT``.
_EMBEDDER_AZURE_PATCHED = False


def _align_tau2_embedder_azure() -> bool:
    """Route tau2's OpenAI KB embedder through the project's Azure OpenAI deployment (AAD).

    The official ``banking_knowledge`` retrieval (the default ``alltools`` variant) embeds KB queries
    with OpenAI ``text-embedding-3-large`` via the raw ``openai.OpenAI`` client, which needs a direct
    ``OPENAI_API_KEY``.  On the project's Azure-only setup that key is absent, so we rebind
    :class:`tau2.knowledge.embedders.openai_embedder.OpenAIEmbedder` to use an ``AzureOpenAI`` client
    authenticated with the *same* AAD token used for chat.  The embedding model/deployment
    (``text-embedding-3-large``) is identical, so retrieval is byte-for-byte the official method — the
    world model and the real-env baselines both call it.  Returns ``True`` when the Azure-backed
    embedder is active (a direct ``OPENAI_API_KEY`` short-circuits this and is used as-is).
    """
    global _EMBEDDER_AZURE_PATCHED
    from ..llm.local import local_llm_configured

    if local_llm_configured():
        # Self-hosted chat server: don't route KB embeddings through Azure (banking falls back to
        # the offline BM25 approximation in local mode — see ``_banking_env_kwargs``).
        return False
    if os.environ.get("OPENAI_API_KEY"):
        return False  # a real OpenAI key is present — use the official embedder unchanged
    if _EMBEDDER_AZURE_PATCHED:
        return True
    from ..llm.azure import azure_api_version, azure_endpoint, fetch_aad_token

    endpoint = azure_endpoint()
    if not endpoint:
        return False  # truly offline — caller falls back to the BM25 approximation
    try:
        from openai import AzureOpenAI
        from tau2.knowledge.embedders import openai_embedder as _oe
    except Exception:
        return False
    api_version = azure_api_version()

    def _azure_init(self, model: str = "text-embedding-3-large", api_key=None):  # noqa: ARG001
        self.model = model  # for Azure, `model` is the deployment name (same string here)
        self.client = AzureOpenAI(
            azure_endpoint=endpoint,
            api_version=api_version,
            azure_ad_token_provider=lambda: fetch_aad_token(),  # refreshed per request
        )

    def _azure_embed(self, texts):
        """Embed in small batches with backoff — the Azure S0 tier rate-limits the one-shot index of
        the full KB (~700 docs).  Results are cached on disk by tau2 after the first success."""
        import time as _time

        import numpy as _np

        if not texts:
            raise ValueError("No text to embed.")
        out = []
        batch = int(os.environ.get("TTS_AZURE_EMBED_BATCH", "16"))
        for start in range(0, len(texts), batch):
            chunk = texts[start:start + batch]
            for attempt in range(6):
                try:
                    resp = self.client.embeddings.create(input=chunk, model=self.model)
                    out.extend(item.embedding for item in resp.data)
                    break
                except Exception as err:  # noqa: BLE001 - retry only on rate limits
                    if "429" not in str(err) and "rate limit" not in str(err).lower():
                        raise
                    if attempt == 5:
                        raise
                    _time.sleep(min(60, 5 * (attempt + 1)))
        return _np.array(out)

    _oe.OpenAIEmbedder.__init__ = _azure_init  # all embedder registries share this class object
    _oe.OpenAIEmbedder.embed = _azure_embed
    _EMBEDDER_AZURE_PATCHED = True
    return True


def _banking_env_kwargs() -> dict:
    override = os.environ.get("TTS_BANKING_RETRIEVAL_VARIANT")
    if override:
        return {"retrieval_variant": override}
    # The official banking retrieval (default `alltools` variant) embeds KB queries with OpenAI
    # text-embedding-3-large.  Use it verbatim when we have a direct OpenAI key OR can route that exact
    # embedding model through Azure; only drop to the offline BM25 approximation when neither exists.
    # A self-hosted chat server exposes no embeddings, so local mode uses the BM25 approximation.
    from ..llm.local import local_llm_configured

    if not local_llm_configured() and (os.environ.get("OPENAI_API_KEY") or _align_tau2_embedder_azure()):
        return {}  # official KB retrieval — identical to the real evaluation
    return {"retrieval_variant": "bm25_grep"}  # offline: real DB + BM25 KB approximation


_ENV_KWARGS_FNS = {"banking_knowledge": _banking_env_kwargs}


def _make_environment(domain: str, db=None):
    ctor = _env_constructor(domain)
    kwargs = _ENV_KWARGS_FNS[domain]() if domain in _ENV_KWARGS_FNS else {}
    if db is not None:
        kwargs["db"] = db
    return ctor(**kwargs)


def _align_tau2_judge_models(model: str) -> None:
    """Point tau2's LLM judges (NL-assertions / env-interface) at ``model``.

    The official ``EvaluationType.ALL`` reward runs an NL-assertion judge whose default
    (``tau2.config.DEFAULT_LLM_NL_ASSERTIONS``) is a bare ``gpt-4.1-...`` name that litellm routes to
    **OpenAI** — failing with "Missing OPENAI_API_KEY" on Azure-only setups.  We rebind it (and the
    env-interface model) to the same Azure model string so every tau2 LLM call uses Azure.  Only
    applied for ``azure/*`` models (or when a local server is configured — the default judge string
    ``gpt-4.1`` would otherwise hit OpenAI / the local server as a non-existent model), so plain
    OpenAI setups are untouched.
    """
    from ..llm.local import local_llm_configured

    if not (model.startswith("azure/") or local_llm_configured()):
        return
    import importlib

    targets = [
        ("tau2.config", ("DEFAULT_LLM_NL_ASSERTIONS", "DEFAULT_LLM_ENV_INTERFACE",
                         "DEFAULT_LLM_AGENT", "DEFAULT_LLM_USER")),
        ("tau2.evaluator.evaluator_nl_assertions", ("DEFAULT_LLM_NL_ASSERTIONS",)),
    ]
    for mod_name, names in targets:
        try:
            mod = importlib.import_module(mod_name)
        except Exception:
            continue
        for name in names:
            if hasattr(mod, name):
                try:
                    setattr(mod, name, model)
                except Exception:
                    pass


@lru_cache(maxsize=None)
def tau2_mutating_tools(domain: str) -> frozenset:
    """The DB-mutating (``ToolType.WRITE``) agent tools of ``domain`` (derived from the toolkit)."""
    try:
        from tau2.environment.toolkit import ToolType

        env = _make_environment(domain)
        tk = env.tools
        return frozenset(
            name for name in tk.get_tools()
            if tk.tool_type(name) == ToolType.WRITE
        )
    except Exception:
        return frozenset(MUTATING_TOOLS) if domain == "airline" else frozenset()


@lru_cache(maxsize=None)
def tau2_agent_tools(domain: str) -> frozenset:
    """All (assistant-side) agent tool names of ``domain`` (derived from the toolkit)."""
    try:
        return frozenset(_make_environment(domain).tools.get_tools().keys())
    except Exception:
        return frozenset()


def tau2_task_world_model_eligible(domain: str, task) -> bool:
    """True iff ``task`` has a DB-mutating agent action (so the world-model prediction is
    discriminative).  The agentic world model reads records, lists ids, and calls read-only tools,
    so it no longer requires single-control: dual-control telecom and the retrieval-only
    banking_knowledge tasks are eligible too (used to pick the world-model task subset)."""
    acts = (task.evaluation_criteria.actions if task and task.evaluation_criteria else None) or []
    if not acts:
        return False
    mutating = tau2_mutating_tools(domain)
    return any(a.name in mutating for a in acts)


# --------------------------------------------------------------------------- #
# DB introspection — buckets (dict- or list-keyed entity collections)
# --------------------------------------------------------------------------- #


def _singular(name: str) -> str:
    return name[:-1] if name.endswith("s") else name


def _attr(obj, name):
    """Get a field from a pydantic model or a plain dict row."""
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def _list_id_field(bucket: str, element) -> Optional[str]:
    """The primary-id attribute of a list-bucket element (``lines`` → ``line_id``, etc.)."""
    fields = (getattr(type(element), "model_fields", None)
              or (element.keys() if isinstance(element, dict) else {}) or {})
    fields = list(fields)
    cand = f"{_singular(bucket)}_id"
    if cand in fields:
        return cand
    if "id" in fields:
        return "id"
    for f in fields:  # last resort: first ``*_id`` field
        if f.endswith("_id"):
            return f
    return None


class _DBModel:
    """Introspected view of a domain ``DB``: its entity *buckets* and per-bucket id sets.

    A *bucket* is a top-level DB field holding entities, either a ``dict[id -> model]`` (airline,
    retail, mock) or a ``list[model]`` keyed by an id attribute (telecom).  This drives generic,
    domain-agnostic privileged reads and partial-DB reconstruction for the read-grounded world
    model.
    """

    def __init__(self, db) -> None:
        self._db = db
        self.db_class = type(db)
        # bucket -> ("dict", None) | ("list", id_field)
        self.buckets: Dict[str, Tuple[str, Optional[str]]] = {}
        self.id_sets: Dict[str, set] = {}
        for name in getattr(self.db_class, "model_fields", {}):
            val = getattr(db, name, None)
            table = getattr(val, "data", None)  # DatabaseTable-style container (e.g. banking)
            if isinstance(val, dict) and val and _is_model(next(iter(val.values()))):
                self.buckets[name] = ("dict", None)
                self.id_sets[name] = set(map(str, val.keys()))
            elif isinstance(val, list) and val and _is_model(val[0]):
                idf = _list_id_field(name, val[0])
                if idf is None:
                    continue
                self.buckets[name] = ("list", idf)
                self.id_sets[name] = {
                    str(getattr(el, idf)) for el in val if getattr(el, idf, None) is not None
                }
            elif isinstance(table, dict) and table:
                self.buckets[name] = ("table", None)
                self.id_sets[name] = set(map(str, table.keys()))
            elif isinstance(table, list) and table:
                idf = _list_id_field(name, table[0])
                if idf is None:
                    continue
                self.buckets[name] = ("table-list", idf)
                self.id_sets[name] = {
                    str(_attr(el, idf)) for el in table if _attr(el, idf) is not None
                }

    def has_buckets(self) -> bool:
        return bool(self.buckets)

    def read(self, bucket: str, entity_id: str):
        kind = self.buckets.get(bucket)
        if kind is None:
            return None
        if kind[0] == "dict":
            d = getattr(self._db, bucket)
            return d.get(entity_id) if entity_id in d else d.get(str(entity_id))
        if kind[0] == "table":
            d = getattr(self._db, bucket).data
            return d.get(entity_id) if entity_id in d else d.get(str(entity_id))
        if kind[0] == "table-list":
            idf = kind[1]
            for el in getattr(self._db, bucket).data:
                if str(_attr(el, idf)) == str(entity_id):
                    return el
            return None
        idf = kind[1]  # list bucket
        for el in getattr(self._db, bucket):
            if str(getattr(el, idf, None)) == str(entity_id):
                return el
        return None

    def matches(self, value) -> List[Tuple[str, str]]:
        """``(bucket, id)`` refs for any bucket whose id-set contains ``value`` (a scalar)."""
        if not isinstance(value, (str, int)):
            return []
        v = str(value)
        return [(bucket, v) for bucket, ids in self.id_sets.items() if v in ids]


def _is_model(obj) -> bool:
    return hasattr(type(obj), "model_fields") or hasattr(type(obj), "model_dump")


def _scan_scalars(obj, _depth: int = 0):
    """Yield scalar (str/int) leaves of a pydantic model / dict / list (bounded recursion)."""
    if _depth > 6:
        return
    if isinstance(obj, (str, int)) and not isinstance(obj, bool):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _scan_scalars(v, _depth + 1)
    elif isinstance(obj, (list, tuple, set)):
        for v in obj:
            yield from _scan_scalars(v, _depth + 1)
    elif _is_model(obj):
        try:
            data = obj.model_dump()
        except Exception:
            return
        yield from _scan_scalars(data, _depth + 1)


# --------------------------------------------------------------------------- #
# Privileged, read-only entity store (any domain)
# --------------------------------------------------------------------------- #


class Tau2EntityStore(EntityStore):
    """Read-only, deep-copying reads of a real τ² domain ``DB`` (privileged).

    Works for dict-keyed buckets (airline/retail/mock) and list-keyed buckets (telecom) alike via
    :class:`_DBModel` introspection.  Reads are deep-copied so the world model can never mutate the
    real environment.
    """

    def __init__(self, db, model: Optional[_DBModel] = None) -> None:
        self._model = model or _DBModel(db)

    def read(self, entity_type: str, entity_id: str) -> Optional[Dict]:
        entity = self._model.read(entity_type, entity_id)
        if entity is None:
            return None
        return {"model": copy.deepcopy(entity)}  # deep copy: cannot mutate real env

    def list_ids(self, entity_type: str) -> List[str]:
        return sorted(self._model.id_sets.get(entity_type, set()))


# --------------------------------------------------------------------------- #
# LLM world-model predictor — predicts observations + success from read facts +
# history, WITHOUT ever executing a tool or constructing an environment.
# --------------------------------------------------------------------------- #


def _action_str(a: Action) -> str:
    import json

    return f"{a.name}({json.dumps(a.args or {}, sort_keys=True, default=str)})"


def _action_data(a: Action) -> Dict:
    return {"name": a.name, "args": dict(a.args or {})}


def _conversation_text(messages: Optional[List[Any]], final_message: str = "") -> str:
    """Compact τ² conversation text for the reference-grounded trajectory verifier."""
    lines: List[str] = []
    for message in messages or []:
        role = str(getattr(message, "role", "") or type(message).__name__)
        content = getattr(message, "content", None)
        if content:
            lines.append(f"{role}: {str(content).strip()}")
        for call in getattr(message, "tool_calls", None) or []:
            lines.append(
                f"{role} tool: {getattr(call, 'name', '?')}"
                f"({json.dumps(dict(getattr(call, 'arguments', {}) or {}), sort_keys=True)})"
            )
    if final_message and not any(final_message in line for line in lines):
        lines.append(f"assistant final: {final_message.strip()}")
    return ("\n".join(lines) or "(no conversation text)")[:20_000]


def _assistant_text(messages: Optional[List[Any]], final_message: str = "") -> str:
    """Assistant-authored natural language only, excluding user and tool-result contents."""
    lines = [
        str(getattr(message, "content", "")).strip()
        for message in messages or []
        if str(getattr(message, "role", "")).casefold() == "assistant"
        and getattr(message, "content", None)
    ]
    if final_message and final_message.strip() not in lines:
        lines.append(final_message.strip())
    return "\n".join(lines)


def _parse_verdict(text: str) -> bool:
    """Parse a SUCCESS/FAIL verdict from the model's reply (last mention wins)."""
    up = (text or "").upper()
    return up.rfind("SUCCESS") > up.rfind("FAIL")


def _model_to_text(model, limit: int = 4000) -> str:
    import json

    if model is None:
        return "<not found>"
    if isinstance(model, dict):
        return json.dumps(model, default=str)[:limit]
    try:
        return json.dumps(model.model_dump(), default=str)[:limit]
    except Exception:
        return str(model)[:limit]


def _row_text(row) -> str:
    if row and isinstance(row, dict) and "model" in row:
        return _model_to_text(row["model"])
    return "<not found>"


def _parse_agentic(text: str):
    """Parse the subagent's turn into (reads, lists, calls, terminal). Terminal stops the loop."""
    import json

    up = (text or "").upper()
    pi = up.find("PREDICT:")
    if pi != -1:
        return [], [], [], ("predict", text[pi + len("PREDICT:"):].strip())
    vi = up.find("VERDICT:")
    if vi != -1:
        return [], [], [], ("verdict", "SUCCESS" if up.rfind("SUCCESS") > up.rfind("FAIL") else "FAIL")
    reads, lists, calls = [], [], []
    for line in (text or "").splitlines():
        s = line.strip()
        u = s.upper()
        if u.startswith("READ "):
            p = s.split()
            if len(p) >= 3:
                reads.append((p[1].strip(',"\''), p[2].strip(',"\'')))
        elif u.startswith("LIST "):
            p = s.split()
            if len(p) >= 2:
                lists.append(p[1].strip(',"\''))
        elif u.startswith("CALL "):
            rest = s[5:].strip()
            name = rest.split()[0] if rest else ""
            args = {}
            br = rest.find("{")
            if br != -1:
                try:
                    args = json.loads(rest[br:])
                except Exception:
                    args = {}
            if name:
                calls.append((name, args))
    return reads, lists, calls, None


class _LLMWorldModelPredictor:
    """Predicts the τ² environment's responses with an LLM — the *world-simulator subagent*.

    In its agentic mode the subagent **proactively queries** the backing DB (read-only, multi-turn)
    to gather what it needs, then decides when to emit the predicted observation (or success
    verdict).  It **never executes a tool and never constructs an environment** (real or forked) and
    can only READ — so it is a genuine predictive model that respects the non-forkable premise.
    """

    #: max read turns the subagent may take before being forced to predict.
    MAX_TURNS = 6

    def __init__(self, domain: str, model: Optional[str] = None) -> None:
        self._domain = domain
        self._model_name = model
        self._client = None
        self.n_python_calls = 0
        self.n_python_failures = 0
        self.n_python_forced = 0
        self.n_python_mode_misses = 0

    def _chat(self):
        if self._client is None:
            from ..llm import make_chat_client

            try:
                from ..llm.azure import azure_deployment

                default = azure_deployment()
            except Exception:
                default = None
            self._client = make_chat_client(model=self._model_name or default)
        return self._client

    def _complete(self, prompt: str, max_tokens: int) -> str:
        return self._chat().complete(
            [{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=max_tokens,
        )

    def _record_python_result(self, result) -> None:
        self.n_python_calls += result.python_calls
        self.n_python_failures += result.python_failures
        self.n_python_forced += result.python_forced
        self.n_python_mode_misses += result.python_mode_misses

    def python_metadata(self) -> Dict[str, int]:
        return {
            "python_calls": self.n_python_calls,
            "python_failures": self.n_python_failures,
            "python_forced": self.n_python_forced,
            "python_mode_misses": self.n_python_mode_misses,
        }

    # -- one-shot prediction over a pre-read shadow (used by the flat/B1 path) -------------- #
    def predict_success(self, facts: str, candidate_actions: List[Action],
                        gold_actions: List[Action], tool_docs: str = "") -> bool:
        gold = "\n".join(_action_str(a) for a in gold_actions) or "(none)"
        cand = "\n".join(_action_str(a) for a in candidate_actions) or "(none)"
        ref = f"\nTool reference (behaviour of each tool):\n{tool_docs}\n" if tool_docs else ""
        header = (
            f"You are a world model of the τ²-bench {self._domain} database. Decide whether the "
            f"CANDIDATE tool-call sequence leaves the database in the SAME final state as the GOLD "
            f"(known-correct) sequence, starting from the initial records below. Reason about each "
            f"sequence's net effect on the records.\n{ref}\n"
            f"Initial records (read-grounded):\n{facts}\n\n"
            f"GOLD actions:\n{gold}\n\nCANDIDATE actions:\n{cand}"
        )
        final = "On the LAST line answer exactly 'VERDICT: SUCCESS' or 'VERDICT: FAIL'."
        if tau2_wm_tool_mode() == "python":
            result = run_tau2_python_prediction_loop(
                self._complete,
                None,
                header,
                final,
                context={
                    "domain": self._domain,
                    "initial_records": facts,
                    "gold_actions": [_action_data(action) for action in gold_actions],
                    "candidate_actions": [
                        _action_data(action) for action in candidate_actions
                    ],
                    "tool_docs": tool_docs,
                },
                max_turns=self.MAX_TURNS,
                max_tokens=500,
                output_kind="verdict",
            )
            self._record_python_result(result)
            return result.valid and result.output == "SUCCESS"
        return _parse_verdict(self._complete(f"{header}\n\n{final}", 500))

    def predict_observation(self, facts: str, history: List[tuple], action: Action,
                            tool_doc: str = "") -> Tuple[str, bool]:
        hist = "\n".join(f"{_action_str(a)} -> {o}" for a, o in history) or "(none)"
        ref = f"\nTool reference (what {action.name} does and RETURNS):\n{tool_doc}\n" if tool_doc else ""
        
        # Add deterministic ID/timestamp generation rules for create actions
        id_rules = ""
        if self._domain == "airline":
            if action.name == "book_reservation":
                id_rules = ("\nIMPORTANT: New reservation IDs are assigned in this EXACT sequence: "
                           "HATHAT (1st reservation), HATHAU (2nd), HATHAV (3rd). Check the known "
                           "records and history to determine which IDs are already used. "
                           "Timestamps are ALWAYS '2024-05-15T15:00:00'.\n")
            elif "certificate" in action.name.lower() or "payment" in action.name.lower():
                id_rules = ("\nIMPORTANT: New payment certificate IDs follow the sequence: "
                           "certificate_3221322, certificate_3221323, certificate_3221324. "
                           "Check which are already used.\n")
        
        header = (
            f"You are a world model of the τ²-bench {self._domain} database. Predict the EXACT result "
            f"the tool call would return, given the known records and the history of prior calls "
            f"(which may have changed state).\n{ref}{id_rules}\n"
            f"Known records (read-grounded):\n{facts}\n\nHistory of prior calls:\n{hist}\n\n"
            f"Tool call: {_action_str(action)}"
        )
        final = (
            f"Return ONLY the tool's result, in the SAME JSON schema as the matching record above "
            f"(e.g. a tool documented to 'return the updated reservation' must return the full "
            f"reservation JSON with the updated fields — not a summary sentence). Begin with "
            f"'ERROR:' only if the call would genuinely fail. Validate every referenced entity and "
            f"payment identifier against the known records; never invent an identifier or assume a "
            f"write succeeded when an argument is invalid."
        )
        if tau2_wm_tool_mode() == "python":
            result = run_tau2_python_prediction_loop(
                self._complete,
                None,
                header,
                final,
                context={
                    "domain": self._domain,
                    "initial_records": facts,
                    "history": [
                        {"action": _action_data(prior), "observation": observation}
                        for prior, observation in history
                    ],
                    "action": _action_data(action),
                    "tool_doc": tool_doc,
                    "deterministic_rules": id_rules,
                },
                max_turns=self.MAX_TURNS,
                max_tokens=1500,
                output_kind="text",
            )
            self._record_python_result(result)
            text = result.output
        else:
            text = (self._complete(f"{header}\n\n{final}", 1500) or "").strip()
        return text, text.upper().startswith("ERROR")

    # -- agentic, multi-turn DB-querying prediction (used by Ours) --------------------------- #
    def _read_loop(self, q, entity_types: List[str], header: str, final_instr: str):
        """Run the legacy textual read loop; ``q`` exposes query/list_ids/call."""
        known: Dict[str, str] = {}
        read_tools = getattr(q, "read_tools", []) or []
        call_line = (f"  CALL <tool> {{json args}}   (run a read-only tool: {', '.join(read_tools)})\n"
                     if read_tools else "")
        proto = (
            f"You may QUERY the database (read-only) before answering. Available record types: "
            f"{', '.join(entity_types) or '(none)'}.\n"
            f"Each turn output one or more query lines:\n"
            f"  READ <type> <id>           (fetch a record by id)\n"
            f"  LIST <type>                (list ids of a type)\n"
            f"{call_line}"
            f"OR, when you have enough information, the final line:\n  {final_instr}\n"
            f"Query only what you need; you have at most {self.MAX_TURNS} read turns."
        )
        for turn in range(self.MAX_TURNS):
            facts = "\n".join(f"{k}: {v}" for k, v in known.items()) or "(nothing read yet)"
            prompt = f"{header}\n\n{proto}\n\nFacts gathered so far:\n{facts}\n\nYour move:"
            resp = self._chat().complete([{"role": "user", "content": prompt}],
                                         temperature=0.0, max_tokens=1500)
            reads, lists, calls, term = _parse_agentic(resp)
            if term is not None:
                return term
            acted = False
            for lt in lists[:3]:
                known[f"ids[{lt}]"] = ", ".join(q.list_ids(lt)[:60])
                acted = True
            if reads:
                rows = q.query([(t, i) for t, i in reads], f"turn{turn}")
                for (t, i), row in rows.items():
                    known[f"{t} {i}"] = _row_text(row)
                acted = True
            for name, args in calls[:4]:
                known[f"call {name}({args})"] = str(q.call(name, args))[:2000]
                acted = True
            if not acted:  # no parseable command -> treat the reply as the answer
                return ("predict", (resp or "").strip())
        # turn budget exhausted -> force a final answer from what was gathered
        facts = "\n".join(f"{k}: {v}" for k, v in known.items()) or "(nothing read)"
        prompt = f"{header}\n\nFacts gathered:\n{facts}\n\nYou are out of read turns. {final_instr}"
        resp = self._chat().complete([{"role": "user", "content": prompt}],
                                     temperature=0.0, max_tokens=1500)
        _, _, _, term = _parse_agentic(resp)
        return term if term is not None else ("predict", (resp or "").strip())

    def predict_observation_agentic(
        self,
        domain,
        action,
        history,
        q,
        entity_types,
        tool_doc="",
        seed_facts=None,
    ) -> Tuple[str, bool]:
        hist = "\n".join(f"{_action_str(a)} -> {o}" for a, o in history) or "(none)"
        ref = f"Tool reference (what {action.name} does and RETURNS):\n{tool_doc}\n\n" if tool_doc else ""
        grounded = json.dumps(
            seed_facts or {}, ensure_ascii=True, sort_keys=True, default=str)
        
        # Add deterministic ID/timestamp rules
        id_rules = ""
        if domain == "airline":
            if action.name == "book_reservation":
                id_rules = ("\nIMPORTANT: New reservation IDs follow EXACT sequence: HATHAT, HATHAU, "
                           "HATHAV. Timestamps are always '2024-05-15T15:00:00'. Check known records "
                           "to determine which IDs are already used.\n")
            elif "certificate" in action.name.lower() or "payment" in action.name.lower():
                id_rules = ("\nIMPORTANT: Payment certificate IDs follow sequence: certificate_3221322, "
                           "certificate_3221323, certificate_3221324.\n")
            elif action.name == "update_reservation_flights":
                id_rules = (
                    "\nIMPORTANT UPDATE SEMANTICS: Use the automatically grounded reservation and "
                    "flight-date records below. For every requested segment, take the requested "
                    "cabin's exact price from flight.dates[date].prices, multiply old and new segment "
                    "totals by the passenger count, and append the signed difference "
                    "(new_total - old_total) to payment_history under the supplied payment_id. A "
                    "downgrade therefore appends a negative refund. Preserve every other reservation "
                    "field and output the complete updated reservation. Use PYTHON_FINAL for this "
                    "arithmetic and record transformation.\n"
                )
        
        header = (
            f"You are a world model of the τ²-bench {domain} database, predicting the EXACT result of "
            f"a tool call.\n{ref}{id_rules}Automatically grounded records for this call:\n"
            f"{grounded}\n\nHistory of prior calls (may have changed state):\n{hist}\n\n"
            f"Tool call to predict: {_action_str(action)}"
        )
        final = ("PREDICT: <the tool's result. If the tool returns a record (e.g. the updated "
                 "order/reservation), output the COMPLETE record as one JSON object in the SAME "
                 "schema as the records above — every field, with the changed fields updated — not a "
                 "summary. Validate every referenced entity and payment identifier with PYTHON_READ "
                 "when needed; never invent an identifier or assume a write succeeded when an "
                 "argument is invalid. Begin with 'ERROR:' only if the call genuinely fails>")
        if tau2_wm_tool_mode() == "python":
            result = run_tau2_python_prediction_loop(
                self._complete,
                q,
                header,
                final,
                context={
                    "domain": domain,
                    "action": _action_data(action),
                    "history": [
                        {"action": _action_data(prior), "observation": observation}
                        for prior, observation in history
                    ],
                    "tool_doc": tool_doc,
                    "deterministic_rules": id_rules,
                    "seed_facts": seed_facts or {},
                    "entity_types": list(entity_types),
                    "read_tools": list(getattr(q, "read_tools", []) or []),
                },
                max_turns=self.MAX_TURNS,
                max_tokens=1500,
                output_kind="text",
                force_long=(
                    domain == "airline"
                    and action.name == "update_reservation_flights"
                ),
            )
            self._record_python_result(result)
            text = result.output
            return text, text.upper().startswith("ERROR")
        term = self._read_loop(q, entity_types, header, final)
        text = (term[1] if term and term[0] == "predict" else "").strip()
        return text, text.upper().startswith("ERROR")

    def predict_success_agentic(
        self,
        domain,
        candidate_actions,
        gold_actions,
        q,
        entity_types,
        tool_docs="",
        task_text="",
        evaluation_context=None,
        candidate_conversation="",
    ) -> bool:
        gold = "\n".join(_action_str(a) for a in gold_actions) or "(none)"
        cand = "\n".join(_action_str(a) for a in candidate_actions) or "(none)"
        ref = f"Tool reference:\n{tool_docs}\n\n" if tool_docs else ""
        criteria = dict(evaluation_context or {})
        policy = _wm_verifier_policy_guidance(domain)
        header = (
            f"You are a reference-grounded verifier for a τ²-bench {domain} trajectory. Decide "
            f"whether the CANDIDATE fully satisfies the task: its tool calls must leave the database "
            f"in the SAME final state as the GOLD sequence, and its conversation must satisfy every "
            f"required communication and natural-language assertion. Query initial records when "
            f"needed. Missing required communication is failure.\n\n{ref}"
            f"WORLD-MODEL VERIFICATION GUIDANCE:\n{policy}\n\n"
            f"TASK:\n{task_text or '(not provided)'}\n\n"
            f"COMMUNICATION / NL CRITERIA:\n"
            f"{json.dumps(criteria, ensure_ascii=True, sort_keys=True)}\n\n"
            f"GOLD actions:\n{gold}\n\nCANDIDATE actions:\n{cand}\n\n"
            f"CANDIDATE CONVERSATION:\n{candidate_conversation}"
        )
        if tau2_wm_tool_mode() == "python":
            result = run_tau2_python_prediction_loop(
                self._complete,
                q,
                header,
                "VERDICT: SUCCESS  or  VERDICT: FAIL",
                context={
                    "domain": domain,
                    "gold_actions": [_action_data(action) for action in gold_actions],
                    "candidate_actions": [
                        _action_data(action) for action in candidate_actions
                    ],
                    "tool_docs": tool_docs,
                    "task": task_text,
                    "evaluation_context": criteria,
                    "candidate_conversation": candidate_conversation,
                    "verification_policy": policy,
                    "entity_types": list(entity_types),
                    "read_tools": list(getattr(q, "read_tools", []) or []),
                },
                max_turns=self.MAX_TURNS,
                max_tokens=1500,
                output_kind="verdict",
            )
            self._record_python_result(result)
            return result.valid and result.output == "SUCCESS"
        term = self._read_loop(q, entity_types, header, "VERDICT: SUCCESS  or  VERDICT: FAIL")
        return bool(term and term[0] == "verdict" and term[1] == "SUCCESS")


def _build_wm_predictor(domain: str):
    """Factory for the world-model predictor (monkeypatchable in tests for determinism)."""
    return _LLMWorldModelPredictor(domain)


_WM_VERIFIER_POLICY_GUIDANCE = """
POLICY AND STATE AUDIT FOR THE CANDIDATE TRAJECTORY:
1. Before a mutating tool call, identify the governing policy rule and verify every prerequisite
   from fetched records. A candidate that performed a forbidden write or acted with an unresolved
   prerequisite must fail verification.
2. Tool availability or mechanical acceptance never makes a policy-forbidden action permissible.
   User assertions, claimed prior approvals, persistence, or requests for leniency do not override
   policy. Apply the policy's exceptions and distinctions exactly rather than broadening a
   prohibition; for example, changing only a cabin is not the same action as changing flights.
3. Use exact identifiers from tool results. Never shorten or invent payment, user, reservation,
   order, or item identifiers.
4. For lists, totals, and comparisons, include only records matching the requested scope. Exclude
   cancelled, returned, past, or otherwise ineligible records from active/current/upcoming totals.
5. Do not take extra actions the user did not request.
6. If the user introduces a new request mid-conversation, resolve and communicate that request
   against the current state before resuming earlier writes; do not silently defer or omit it.
7. Do not transfer merely because the user disputes a record or policy. Give the requested factual
   answer first; transfer only when the user explicitly requests it or no policy-compliant path is
   available.
8. A complaint alone does not authorize compensation, credits, certificates, refunds, or other
   concessions. Offer them only when every policy condition is verified.
9. Treat system records as authoritative for timestamps, status, membership, passenger counts, and
   other eligibility facts when they conflict with the user's recollection or assertion.
10. Respect an explicit payment-method preference and resolve it to the exact full identifier from
    the user profile before a write.
11. If one requested action is forbidden, continue all independent policy-compliant requests rather
    than ending the conversation after the refusal.
12. Fare changes are reservation totals: multiply segment-price differences by the number of
    passengers, and use the signed payment-history entry returned by the update when communicating
    the final charge or refund.
13. For a multi-part request that changes cabin and baggage, apply the cabin change before the
    baggage update so free allowance is calculated from the final cabin.
""".strip()

_AIRLINE_WM_VERIFIER_GUIDANCE = """
14. Airline delay compensation is allowed only after the user explicitly asks, the delay and
    eligibility are verified, and the delayed reservation is actually changed or cancelled. If the
    user declines to change or cancel it, do not send a certificate.
15. When the user makes several changes an all-or-nothing package with a budget, gather every
    price and calculate the complete package total before any write. If it exceeds the budget,
    perform none of the package's writes; do not apply a partial subset.
16. For "all upcoming" or equivalent requests, enumerate the complete authoritative set, process
    every independently eligible record, and do not stop after only the first few.
17. In airline baggage writes, total_baggages is the requested total while nonfree_baggages is only
    the portion above the membership-and-cabin allowance. Never set both fields to the same count
    unless every requested bag is chargeable.
18. Track every successful write by its full target identifier. Never repeat a successful
    cancellation or update; continue with the next remaining requested record instead.
""".strip()


def _wm_verifier_policy_guidance(domain: str) -> str:
    """Policy guidance for the WM verifier only; never inject it into the benchmark agent."""
    if domain == "airline":
        return f"{_WM_VERIFIER_POLICY_GUIDANCE}\n{_AIRLINE_WM_VERIFIER_GUIDANCE}"
    return _WM_VERIFIER_POLICY_GUIDANCE


def _add_wm_agent_policy_reminder(agent, domain: str = "") -> None:
    """Deprecated no-op retained for compatibility.

    The WM path must use the exact official agent prompt. Policy guidance belongs to the
    world-model verifier, not to candidate generation.
    """
    return None


def _add_airline_cancel_preflight(agent) -> None:
    """Regenerate cancellation calls that authoritative records prove policy-ineligible."""
    if getattr(agent, "_tts_airline_cancel_preflight", False):
        return
    generate_next_message = getattr(agent, "generate_next_message", None)
    if not callable(generate_next_message):
        return
    now = datetime.fromisoformat("2024-05-15T15:00:00")

    def reservation_records(state) -> Dict[str, Dict[str, Any]]:
        records: Dict[str, Dict[str, Any]] = {}
        for message in getattr(state, "messages", None) or []:
            content = getattr(message, "content", None)
            if not isinstance(content, str):
                continue
            try:
                payload = json.loads(content)
            except Exception:
                continue
            if isinstance(payload, dict) and payload.get("reservation_id"):
                records[str(payload["reservation_id"])] = payload
        return records

    def cancellation_block(state, assistant_message) -> str:
        calls = [
            call for call in getattr(assistant_message, "tool_calls", None) or []
            if getattr(call, "name", "") == "cancel_reservation"
        ]
        if not calls:
            return ""
        records = reservation_records(state)
        conversation = _conversation_text(
            getattr(state, "messages", None)).casefold()
        covered_insurance_reason = bool(re.search(
            r"\b(sick|ill|illness|medical|weather|storm|snow|hurricane)\b",
            conversation,
        ))
        for call in calls:
            reservation_id = str(
                dict(getattr(call, "arguments", {}) or {}).get("reservation_id", ""))
            reservation = records.get(reservation_id)
            if reservation is None:
                return (
                    f"read reservation {reservation_id} before cancellation so policy eligibility "
                    "is verified from authoritative records"
                )
            flight_dates = [
                str(flight.get("date", ""))
                for flight in reservation.get("flights", [])
                if isinstance(flight, dict)
            ]
            if any(date and date < now.date().isoformat() for date in flight_dates):
                return f"reservation {reservation_id} contains a flight segment that has already flown"
            created_at = reservation.get("created_at")
            within_24_hours = False
            if isinstance(created_at, str):
                try:
                    age = now - datetime.fromisoformat(created_at)
                    within_24_hours = 0 <= age.total_seconds() <= 24 * 60 * 60
                except ValueError:
                    pass
            if (
                within_24_hours
                or reservation.get("cabin") == "business"
                or (
                    reservation.get("insurance") == "yes"
                    and covered_insurance_reason
                )
            ):
                continue
            return (
                f"reservation {reservation_id} is outside the 24-hour window and has neither "
                "business cabin nor travel insurance; user claims cannot override this policy"
            )
        return ""

    def guarded_generate(message, state):
        feedback = ""
        for attempt in range(3):
            original_messages = list(getattr(state, "messages", []) or [])
            reminder = None
            if feedback:
                from tau2.data_model.message import SystemMessage

                reminder = SystemMessage(
                    role="system",
                    content=(
                        f"Cancellation policy preflight blocked the proposed call: {feedback}. "
                        "Use authoritative records, then respond or choose a policy-compliant action. "
                        "Do not repeat the blocked cancellation."
                    ),
                )
                state.system_messages.insert(0, reminder)
            try:
                assistant_message, next_state = generate_next_message(message, state)
            finally:
                if reminder is not None and reminder in state.system_messages:
                    state.system_messages.remove(reminder)
            feedback = cancellation_block(next_state, assistant_message)
            if not feedback:
                return assistant_message, next_state
            if attempt < 2:
                next_state.messages = original_messages
                state = next_state
                continue

            from tau2.data_model.message import AssistantMessage

            safe_message = AssistantMessage(
                role="assistant",
                content=(
                    "I’m sorry, but I can’t cancel this reservation because it does not meet the "
                    "airline’s cancellation policy. I can help with another policy-compliant option."
                ),
                tool_calls=None,
            )
            next_state.messages[-1] = safe_message
            return safe_message, next_state
        raise AssertionError("unreachable cancellation preflight loop")

    agent.generate_next_message = guarded_generate
    agent._tts_airline_cancel_preflight = True


def _add_airline_baggage_preflight(agent) -> None:
    """Regenerate writes whose paid-baggage count contradicts the official allowance table."""
    if getattr(agent, "_tts_airline_baggage_preflight", False):
        return
    generate_next_message = getattr(agent, "generate_next_message", None)
    if not callable(generate_next_message):
        return
    free_per_passenger = {
        "regular": {"basic_economy": 0, "economy": 1, "business": 2},
        "silver": {"basic_economy": 1, "economy": 2, "business": 3},
        "gold": {"basic_economy": 2, "economy": 3, "business": 4},
    }

    def records(state) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
        reservations: Dict[str, Dict[str, Any]] = {}
        users: Dict[str, Dict[str, Any]] = {}
        for message in getattr(state, "messages", None) or []:
            content = getattr(message, "content", None)
            if not isinstance(content, str):
                continue
            try:
                payload = json.loads(content)
            except Exception:
                continue
            if not isinstance(payload, dict):
                continue
            if payload.get("reservation_id"):
                reservations[str(payload["reservation_id"])] = payload
            if payload.get("user_id") and payload.get("membership"):
                users[str(payload["user_id"])] = payload
        return reservations, users

    def baggage_block(
        state, assistant_message
    ) -> Tuple[str, Optional[Any], Optional[int]]:
        reservations, users = records(state)
        for call in getattr(assistant_message, "tool_calls", None) or []:
            name = getattr(call, "name", "")
            if name not in {"book_reservation", "update_reservation_baggages"}:
                continue
            args = dict(getattr(call, "arguments", {}) or {})
            total_baggages = int(args.get("total_baggages", 0) or 0)
            actual_nonfree = int(args.get("nonfree_baggages", 0) or 0)
            if name == "book_reservation":
                user_id = str(args.get("user_id", ""))
                user = users.get(user_id)
                cabin = str(args.get("cabin", ""))
                passenger_count = len(args.get("passengers") or [])
            else:
                reservation_id = str(args.get("reservation_id", ""))
                reservation = reservations.get(reservation_id)
                if reservation is None:
                    return (
                        f"read reservation {reservation_id} before calculating baggage allowance",
                        None,
                        None,
                    )
                user_id = str(reservation.get("user_id", ""))
                user = users.get(user_id)
                cabin = str(reservation.get("cabin", ""))
                passenger_count = len(reservation.get("passengers") or [])
            if user is None:
                return f"read user {user_id} before calculating baggage allowance", None, None
            membership = str(user.get("membership", "regular"))
            allowance = free_per_passenger.get(
                membership, free_per_passenger["regular"]).get(cabin)
            if allowance is None:
                return f"verify cabin {cabin!r} before calculating baggage allowance", None, None
            expected_nonfree = max(0, total_baggages - allowance * passenger_count)
            if actual_nonfree != expected_nonfree:
                return (
                    (
                        f"{membership} membership with {cabin} cabin and "
                        f"{passenger_count} passenger(s) allows "
                        f"{allowance * passenger_count} free checked bag(s); "
                        f"nonfree_baggages must be {expected_nonfree}, not {actual_nonfree}"
                    ),
                    call,
                    expected_nonfree,
                )
        return "", None, None

    def guarded_generate(message, state):
        feedback = ""
        for attempt in range(3):
            original_messages = list(getattr(state, "messages", []) or [])
            reminder = None
            if feedback:
                from tau2.data_model.message import SystemMessage

                reminder = SystemMessage(
                    role="system",
                    content=(
                        f"Baggage policy preflight blocked the proposed call: {feedback}. "
                        "Regenerate the write with the exact official free-baggage allowance."
                    ),
                )
                state.system_messages.insert(0, reminder)
            try:
                assistant_message, next_state = generate_next_message(message, state)
            finally:
                if reminder is not None and reminder in state.system_messages:
                    state.system_messages.remove(reminder)
            feedback, incorrect_call, expected_nonfree = baggage_block(
                next_state, assistant_message)
            if not feedback:
                return assistant_message, next_state
            if incorrect_call is not None:
                corrected_args = dict(getattr(incorrect_call, "arguments", {}) or {})
                corrected_args["nonfree_baggages"] = expected_nonfree
                incorrect_call.arguments = corrected_args
                return assistant_message, next_state
            if attempt < 2:
                next_state.messages = original_messages
                state = next_state
                continue
            return assistant_message, next_state
        raise AssertionError("unreachable baggage preflight loop")

    agent.generate_next_message = guarded_generate
    agent._tts_airline_baggage_preflight = True


def _add_airline_certificate_preflight(agent) -> None:
    """Reject delayed-flight certificates until a requested change or cancellation occurred."""
    if getattr(agent, "_tts_airline_certificate_preflight", False):
        return
    generate_next_message = getattr(agent, "generate_next_message", None)
    if not callable(generate_next_message):
        return

    def certificate_block(state, assistant_message) -> str:
        sends_certificate = any(
            getattr(call, "name", "") == "send_certificate"
            for call in getattr(assistant_message, "tool_calls", None) or []
        )
        if not sends_certificate:
            return ""
        conversation = _conversation_text(
            getattr(state, "messages", None)).casefold()
        if not re.search(r"\bdelay(?:ed)?\b", conversation):
            return ""
        completed_change = any(
            getattr(call, "name", "") in {
                "cancel_reservation",
                "update_reservation_flights",
            }
            for message in getattr(state, "messages", None) or []
            for call in getattr(message, "tool_calls", None) or []
        )
        if completed_change:
            return ""
        return (
            "a delayed-flight certificate requires the delayed reservation to be changed or "
            "cancelled first; the user has not completed either action"
        )

    def guarded_generate(message, state):
        feedback = ""
        for attempt in range(3):
            original_messages = list(getattr(state, "messages", []) or [])
            reminder = None
            if feedback:
                from tau2.data_model.message import SystemMessage

                reminder = SystemMessage(
                    role="system",
                    content=(
                        f"Compensation policy preflight blocked the proposed call: {feedback}. "
                        "Do not send a certificate; continue with a policy-compliant response."
                    ),
                )
                state.system_messages.insert(0, reminder)
            try:
                assistant_message, next_state = generate_next_message(message, state)
            finally:
                if reminder is not None and reminder in state.system_messages:
                    state.system_messages.remove(reminder)
            feedback = certificate_block(next_state, assistant_message)
            if not feedback:
                return assistant_message, next_state
            if attempt < 2:
                next_state.messages = original_messages
                state = next_state
                continue

            from tau2.data_model.message import AssistantMessage

            safe_message = AssistantMessage(
                role="assistant",
                content=(
                    "I can’t provide a delayed-flight certificate unless the affected reservation "
                    "is changed or cancelled under the compensation policy."
                ),
                tool_calls=None,
            )
            next_state.messages[-1] = safe_message
            return safe_message, next_state
        raise AssertionError("unreachable certificate preflight loop")

    agent.generate_next_message = guarded_generate
    agent._tts_airline_certificate_preflight = True


class _Q:
    """Read-only query interface handed to the agentic predictor: fetch records by id, list ids,
    or invoke a **read-only** tool (e.g. ``KB_search``/``search_direct_flight``) to retrieve info.

    All paths are read-only and budgeted/logged via the world model's reader; the tool runner only
    accepts ``ToolType.READ`` tools, so the subagent can never mutate anything."""

    def __init__(self, reader, read_tool_runner, read_tools: List[str]) -> None:
        self._reader = reader
        self._run = read_tool_runner
        self.read_tools = list(read_tools or [])

    def query(self, refs, label="q"):
        return self._reader.query(refs, label)

    def list_ids(self, entity_type):
        return self._reader.list_ids(entity_type)

    def call(self, name, args):
        if self._run is None or name not in self.read_tools:
            return f"ERROR: tool {name!r} is not an available read-only tool"
        self._reader.probe(f"call:{name}")
        try:
            return self._run(name, args or {})
        except Exception as exc:  # surface tool errors as observations
            return f"ERROR: {type(exc).__name__}: {exc}"


# --------------------------------------------------------------------------- #
# Generic shadow evaluator (read-graph + LLM-predicted reward, no execution)
# --------------------------------------------------------------------------- #


class Tau2ShadowEvaluator(ShadowEvaluator):
    """Drives the read-grounded world model for every tau2 domain with a readable entity DB.

    The world model **reads** (privileged, read-only) the entities a trajectory touches, then a
    predictor (the *world-simulator subagent*) predicts each tool's observation and the
    trajectory's success from those **read-grounded facts + the action history** — it never
    executes a tool or builds an environment, so it cannot write the real env (or any copy) and
    honours the non-forkable premise.  Domain semantics are discovered by introspection:

    * **seed_entities** — entities named (by id) anywhere in the candidate's and gold's args;
    * **child_references** — entities a fetched row transitively references (drives the recursive
      reservation→user+flights closure that grows the read-grounded facts);
    * **reward_from_shadow** — ask the predictor whether the candidate reaches the same final state
      as the gold reference, given the read facts.  Reading more (recursive vs flat) gives the
      predictor more grounding, so it predicts better.
    """

    def __init__(self, domain: str, gold_actions: List[Action], model: _DBModel,
                 predictor=None, tool_docs: Optional[Dict[str, str]] = None,
                 read_tools: Optional[List[str]] = None, read_tool_runner=None,
                 task_text: str = "",
                 evaluation_context: Optional[Dict[str, Any]] = None,
                 gold_action_specs: Optional[List[Any]] = None) -> None:
        self._domain = domain
        self._gold = list(gold_actions)
        self._gold_specs = (
            list(gold_action_specs) if gold_action_specs is not None else list(gold_actions))
        self._model = model
        self._mutating = set(tau2_mutating_tools(domain))
        self._predictor = predictor
        self._tool_docs = tool_docs or {}
        self._read_tools = list(read_tools or [])
        self._read_tool_runner = read_tool_runner  # read-only tool executor (KB_search/search/...)
        self._task_text = task_text
        self._evaluation_context = dict(evaluation_context or {})
        #: Ours drives a multi-turn, LLM-led read loop (the agentic world-model subagent).
        self.agentic = True
        #: running (action, predicted-observation) history used to condition step predictions.
        self._history: List[tuple] = []

    def _query_iface(self, reader):
        return _Q(reader, self._read_tool_runner, self._read_tools)

    def _action_refs(self, action: Action) -> List[Tuple[str, str]]:
        refs: List[Tuple[str, str]] = []
        for value in _scan_scalars(action.args or {}):
            refs.extend(self._model.matches(value))
        return refs

    def seed_entities(self, candidate_actions: List[Action]) -> List[tuple]:
        out: List[tuple] = []
        for a in list(candidate_actions) + self._gold:
            out.extend(self._action_refs(a))
        return out

    def child_references(self, entity_type: str, entity_id: str, row: Dict) -> List[tuple]:
        if not row:
            return []
        model = row.get("model")
        if model is None:
            return []
        refs: List[tuple] = []
        for value in _scan_scalars(model):
            for ref in self._model.matches(value):
                if ref != (entity_type, str(entity_id)):
                    refs.append(ref)
        return refs

    def _required(self, candidate_actions: List[Action]) -> List[tuple]:
        req: List[tuple] = []
        for a in list(candidate_actions) + self._gold:
            if a.name in self._mutating:
                req.extend(self._action_refs(a))
        return list(dict.fromkeys(req))  # de-dup, preserve order

    def _predictor_or_default(self):
        if self._predictor is None:
            self._predictor = _build_wm_predictor(self._domain)
        return self._predictor

    @staticmethod
    def _python_metadata(predictor) -> Dict[str, int]:
        get_metadata = getattr(predictor, "python_metadata", None)
        if not callable(get_metadata):
            return {}
        metadata = get_metadata()
        return dict(metadata) if isinstance(metadata, dict) else {}

    def _facts_text(self, shadow: Dict[tuple, Dict]) -> str:
        """Serialise the read-grounded shadow rows into compact text for the predictor."""
        lines: List[str] = []
        for (btype, bid), row in shadow.items():
            if not row or row.get("model") is None:
                lines.append(f"{btype} {bid}: <absent / not found>")
                continue
            lines.append(f"{btype} {bid}: {_model_to_text(row['model'])}")
        return "\n".join(lines) or "(no records read)"

    def observe(self, shadow: Dict[tuple, Dict], action: Action):
        """Predict one tool call's observation from read-grounded facts + history (no execution)."""
        facts = self._facts_text(shadow)
        content, is_error = self._predictor_or_default().predict_observation(
            facts, self._history, action, tool_doc=self._tool_docs.get(action.name, "")
        )
        self._history.append((action, content))
        return (content, is_error)

    def _seed_action_facts(self, reader, action: Action) -> Dict[str, Any]:
        """Read and compact the records directly referenced by a predicted write."""
        refs = list(dict.fromkeys(self._action_refs(action)))
        if not refs:
            return {}
        rows = reader.query(refs[:20], "action-grounding")
        requested_dates = {
            str(value)
            for value in _scan_scalars(action.args or {})
            if isinstance(value, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", value)
        }
        facts: Dict[str, Any] = {}
        for ref in refs[:20]:
            row = rows.get(ref)
            model = row.get("model") if isinstance(row, dict) else None
            if model is None:
                facts[f"{ref[0]} {ref[1]}"] = None
                continue
            try:
                value = model.model_dump()
            except Exception:
                value = copy.deepcopy(model) if isinstance(model, dict) else str(model)
            if ref[0] == "flights" and isinstance(value, dict) and requested_dates:
                dates = value.get("dates")
                if isinstance(dates, dict):
                    value["dates"] = {
                        date: dates.get(date)
                        for date in sorted(requested_dates)
                        if date in dates
                    }
            facts[f"{ref[0]} {ref[1]}"] = value
        return facts

    def _tool_docs_text(self, actions: List[Action]) -> str:
        used = list(dict.fromkeys(a.name for a in actions))
        lines = [f"- {n}: {self._tool_docs[n]}" for n in used if self._tool_docs.get(n)]
        return "\n".join(lines)

    def reward_from_shadow(self, shadow: Dict[tuple, Dict], candidate_actions: List[Action]) -> RewardInfo:
        """Predict success: does the candidate reach the same final state as gold, given the facts?

        Pure prediction over read-grounded facts — no tool is executed and no environment (real or
        forked) is constructed.  (One-shot; used by the flat/B1 path.)"""
        facts = self._facts_text(shadow)
        docs = self._tool_docs_text(list(candidate_actions) + self._gold)
        predictor = self._predictor_or_default()
        ok = predictor.predict_success(
            facts, list(candidate_actions), self._gold, tool_docs=docs
        )
        return RewardInfo(reward=1.0 if ok else 0.0, success=bool(ok), grounded=False,
                          reason="world-model prediction (read-grounded, no execution)",
                          metadata=self._python_metadata(predictor))

    # -- agentic, multi-turn DB-querying prediction (Ours) ---------------------------------- #
    def observe_agentic(self, reader, action: Action, force_predict: bool = False):
        """Predict the observation for one action.

        If the action is itself a **read-only / retrieval tool** (e.g. ``KB_search``,
        ``search_direct_flight``, ``get_*``), the world model returns the EXACT result the real env
        would — by executing that read-only tool the same way the official evaluation does (no
        mutation).  For a **write** action it predicts via the agentic read loop (no execution).

        ``force_predict=True`` skips the exact-read shortcut and always predicts — used for a read of
        an entity that a prior write in this rollout already changed (a *dirty* read), so the
        observation reflects the predicted post-write state rather than the unmutated real state."""
        if (not force_predict and self._read_tool_runner is not None
                and action.name in set(self._read_tools)):
            reader.probe(f"exec-read:{action.name}")
            try:
                result = str(self._read_tool_runner(action.name, dict(action.args or {})))
            except Exception as exc:
                result = f"ERROR: {type(exc).__name__}: {exc}"
            self._history.append((action, result))
            return (result, result.strip().upper().startswith("ERROR"))
        seed_facts = self._seed_action_facts(reader, action)
        content, is_error = self._predictor_or_default().predict_observation_agentic(
            self._domain, action, self._history, self._query_iface(reader),
            list(self._model.buckets), tool_doc=self._tool_docs.get(action.name, ""),
            seed_facts=seed_facts,
        )
        self._history.append((action, content))
        return (content, is_error)

    def note(self, action: Action, content: str) -> None:
        """Record an exactly-served observation (e.g. a read executed outside ``observe_agentic``)
        in the running history, so subsequent write predictions condition on the full context."""
        self._history.append((action, content))

    @staticmethod
    def _trajectory_tool_actions(trajectory: Trajectory) -> List[Action]:
        """Return assistant and user tool calls when the official message log is available."""
        actions: List[Action] = []
        for message in trajectory.sim_messages or []:
            for tool_call in getattr(message, "tool_calls", None) or []:
                name = getattr(tool_call, "name", None)
                if name:
                    actions.append(Action(str(name), dict(
                        getattr(tool_call, "arguments", None) or {})))
        return actions or list(trajectory.actions)

    @staticmethod
    def _matches_reference_action(candidate: Action, reference: Any) -> bool:
        """Match one action using τ²'s official ``compare_args`` semantics."""
        if candidate.name != getattr(reference, "name", None):
            return False
        reference_args = getattr(reference, "arguments", None)
        if reference_args is None:
            reference_args = getattr(reference, "args", None)
        reference_args = dict(reference_args or {})
        compare_args = getattr(reference, "compare_args", None)
        keys = list(candidate.args) if compare_args is None else list(compare_args)
        candidate_args = {key: candidate.args[key] for key in keys if key in candidate.args}
        expected_args = {key: reference_args[key] for key in keys if key in reference_args}
        return candidate_args == expected_args

    @classmethod
    def _match_reference_writes(
        cls,
        candidate_writes: List[Action],
        gold_writes: List[Any],
    ) -> Tuple[int, List[Any], List[Action]]:
        """One-to-one match candidate writes to gold writes.

        One-to-one matching makes duplicate, extra, and wrong-argument writes visible to the
        world-model verifier instead of letting them hide behind complete gold-action recall.
        """
        unmatched_gold = list(range(len(gold_writes)))
        unexpected: List[Action] = []
        matched = 0
        for candidate in candidate_writes:
            match = next(
                (
                    index for index in unmatched_gold
                    if cls._matches_reference_action(candidate, gold_writes[index])
                ),
                None,
            )
            if match is None:
                unexpected.append(candidate)
                continue
            unmatched_gold.remove(match)
            matched += 1
        missing = [gold_writes[index] for index in unmatched_gold]
        return matched, missing, unexpected

    def _reference_assessment(self, trajectory: Trajectory) -> Tuple[Optional[str], float]:
        """Return a deterministic reference failure and graded structural similarity."""
        candidate_writes = [
            action
            for action in self._trajectory_tool_actions(trajectory)
            if action.name in self._mutating
        ]
        gold_writes = [
            action for action in self._gold_specs
            if getattr(action, "name", None) in self._mutating
        ]
        matched_writes, missing_writes, unexpected_writes = self._match_reference_writes(
            candidate_writes,
            gold_writes,
        )
        duplicate_writes = [
            candidate for candidate in unexpected_writes
            if any(self._matches_reference_action(candidate, gold) for gold in gold_writes)
        ]
        writes_without_reference = unexpected_writes if not gold_writes else []
        if duplicate_writes or writes_without_reference:
            write_score = matched_writes / max(len(gold_writes), len(candidate_writes))
        else:
            write_score = matched_writes / len(gold_writes) if gold_writes else 1.0

        conversation = _assistant_text(
            trajectory.sim_messages, trajectory.final_message)
        compact_conversation = re.sub(r"[^a-z0-9]", "", conversation.casefold())
        required_values = self._evaluation_context.get("communicate_info", [])
        communicated = 0
        for required in self._evaluation_context.get("communicate_info", []):
            compact_required = re.sub(r"[^a-z0-9]", "", str(required).casefold())
            if compact_required and compact_required in compact_conversation:
                communicated += 1
        communication_score = (
            communicated / len(required_values) if required_values else 1.0)
        reference_score = 0.75 * write_score + 0.25 * communication_score

        if missing_writes:
            return (
                "candidate is missing or mismatches required reference mutating actions",
                reference_score,
            )
        invalid_extras = duplicate_writes or writes_without_reference
        if invalid_extras:
            unexpected = ", ".join(action.name for action in invalid_extras[:3])
            return (
                f"candidate contains duplicate or unreferenced mutating actions: {unexpected}",
                reference_score,
            )
        if communication_score < 1.0:
            missing = [
                value for value in required_values
                if re.sub(r"[^a-z0-9]", "", str(value).casefold())
                not in compact_conversation
            ]
            return f"missing required communicated value: {missing[0]}", reference_score
        return None, reference_score

    def reward_agentic(self, reader, trajectory: Trajectory) -> RewardInfo:
        """The subagent proactively queries the DB (via ``reader``) then predicts success."""
        mismatch, reference_score = self._reference_assessment(trajectory)
        if mismatch is not None:
            predictor = self._predictor_or_default()
            return RewardInfo(
                reward=reference_score,
                success=False,
                grounded=False,
                reason=f"reference-grounded verifier: {mismatch}",
                metadata={
                    **self._python_metadata(predictor),
                    "hard_invalid": True,
                    "reference_score": reference_score,
                },
            )
        candidate_actions = self._trajectory_tool_actions(trajectory)
        docs = self._tool_docs_text(list(candidate_actions) + self._gold)
        predictor = self._predictor_or_default()
        ok = predictor.predict_success_agentic(
            self._domain, list(candidate_actions), self._gold, self._query_iface(reader),
            list(self._model.buckets), tool_docs=docs, task_text=self._task_text,
            evaluation_context=self._evaluation_context,
            candidate_conversation=_conversation_text(
                trajectory.sim_messages, trajectory.final_message),
        )
        reward = 1.0 if ok else 0.9 * reference_score
        return RewardInfo(reward=reward, success=bool(ok), grounded=False,
                          reason="world-model prediction (agentic read loop, no execution)",
                          metadata={
                              **self._python_metadata(predictor),
                              "reference_score": reference_score,
                          })


#: Backward-compatible alias (the evaluator is now domain-generic).
AirlineShadowEvaluator = Tau2ShadowEvaluator


# --------------------------------------------------------------------------- #
# Restricted view (solver-facing)
# --------------------------------------------------------------------------- #


class _Tau2RestrictedView(RestrictedView):
    def __init__(self, env: "Tau2TaskEnv") -> None:
        self._env = env

    @property
    def task_id(self) -> str:
        return self._env.task_id

    def instruction(self) -> str:
        return self._env._instruction

    def tool_schema(self) -> List[str]:
        return list(self._env._tool_names)

    def tool_docs(self) -> Dict[str, str]:
        """Public tool behaviour docs (names + docstrings) for an LLM solver — not privileged."""
        return self._env._tool_docstrings()


# --------------------------------------------------------------------------- #
# The task environment
# --------------------------------------------------------------------------- #


class Tau2TaskEnv(TaskEnvironment):
    """One real τ² task (any domain), cohering to the official τ²-bench execution + evaluation."""

    def __init__(self, domain: str = "airline", task=None, subset: Optional[str] = None,
                 _env=None) -> None:
        if not real_tau2_available():
            raise RuntimeError("real tau2 not available (needs Python>=3.12 + tau2 install)")
        self.domain = domain
        self.subset = subset or domain
        self._env = _env if _env is not None else _make_environment(domain)
        self._task = task
        self._commits = 0
        gold = (task.evaluation_criteria.actions if task and task.evaluation_criteria else None) or []
        self._gold_raw = list(gold)
        self._gold = [Action(a.name, dict(a.arguments)) for a in gold]
        self._instruction = self._derive_instruction(task)
        self._tool_names = self._discover_tool_names()
        self._db_model: Optional[_DBModel] = None

    # -- introspection helpers ------------------------------------------ #
    def _discover_tool_names(self) -> List[str]:
        try:
            return sorted(self._env.tools.get_tools().keys())
        except Exception:
            return sorted(MUTATING_TOOLS)

    def _db(self):
        try:
            return self._env.tools.db
        except Exception:
            return None

    def _model(self) -> Optional[_DBModel]:
        if self._db_model is None:
            db = self._db()
            if db is not None:
                self._db_model = _DBModel(db)
        return self._db_model

    def _world_model_eligible(self) -> bool:
        """True iff the agentic world model can ground on this task's backing DB.

        The agentic LLM subagent reads records by id, lists ids, and calls **read-only** tools
        (e.g. ``KB_search``/``search_*``) to gather what it needs, then predicts — so it applies to
        any domain with an introspectable, readable entity DB (airline, retail, mock, telecom,
        telecom-workflow, banking_knowledge).  A domain whose env cannot be constructed or exposes no
        readable buckets falls back to the oracle reward."""
        model = self._model()
        return model is not None and model.has_buckets()

    @staticmethod
    def _derive_instruction(task) -> str:
        if task is None:
            return "Complete the customer request."
        # Prefer the task's STRUCTURED user instructions (the request + the info the user holds).
        # In the real benchmark the agent extracts these by conversing with the user simulator; the
        # llm_wm solver has no user simulator, so we surface them directly as the task spec it plans
        # against.  (Other modes ignore this: the scripted stub doesn't read the instruction, and the
        # official llm agent runs its own orchestrator — so this only feeds the llm_wm solver.)
        us = getattr(task, "user_scenario", None)
        ins = getattr(us, "instructions", None) if us is not None else None
        if ins is not None:
            parts: List[str] = []
            for attr, label in (("reason_for_call", "Request"),
                                ("known_info", "Known info"),
                                ("task_instructions", "Preferences")):
                val = getattr(ins, attr, None)
                if val:
                    parts.append(f"{label}: {str(val).strip()}")
            if parts:
                return "\n".join(parts)[:2000]
        for attr in ("description", "user_scenario", "ticket"):
            val = getattr(task, attr, None)
            if val is None:
                continue
            text = getattr(val, "instructions", None) or getattr(val, "text", None) or str(val)
            if text:
                return str(text)[:2000]
        return f"τ² task {getattr(task, 'id', '?')}"

    # -- identity / metadata -------------------------------------------- #
    @property
    def env_name(self) -> str:
        return f"tau2_{self.domain}"

    @property
    def mutating_tools(self):
        return set(tau2_mutating_tools(self.domain))

    @property
    def task_id(self) -> str:
        return str(getattr(self._task, "id", "?"))

    def restricted_view(self) -> RestrictedView:
        return _Tau2RestrictedView(self)

    def entity_store(self) -> EntityStore:
        model = self._model()
        if model is None or not model.has_buckets():
            raise NotImplementedError(f"tau2 domain {self.domain!r} exposes no entity store")
        return Tau2EntityStore(self._db(), model)

    def _tool_docstrings(self) -> Dict[str, str]:
        """Static return/behaviour docs of the tools (from the already-built env; no new env).

        Includes the assistant tools and, for dual-control domains, the USER tools too — so the world
        model can predict the effect of user-side writes (e.g. ``toggle_roaming``) it would otherwise
        have no documentation for."""
        docs: Dict[str, str] = {}
        toolkits = [getattr(self._env, "tools", None), getattr(self._env, "user_tools", None)]
        for tk in toolkits:
            if tk is None:
                continue
            try:
                names = list(tk.get_tools())
            except Exception:
                names = list(self._tool_names)
            for n in names:
                if n in docs:
                    continue
                fn = getattr(tk, n, None)
                doc = (getattr(fn, "__doc__", None) or "").strip()
                if doc:
                    docs[n] = " ".join(doc.split())[:400]
        return docs

    def _read_tool_names(self) -> List[str]:
        """Names of the **read-only** (``ToolType.READ``) agent tools — safe for the WM to call."""
        try:
            from tau2.environment.toolkit import ToolType

            tk = self._env.tools
            return [n for n in tk.get_tools() if tk.tool_type(n) == ToolType.READ]
        except Exception:
            return []

    def _run_read_tool(self, name: str, args: dict) -> str:
        """Execute a read-only tool on the env and return its result text.

        Hard guard: refuses any tool that is not ``ToolType.READ`` so the world model can never
        mutate state.  (Read tools do not change the DB; asserted by the read-only tests.)"""
        from tau2.data_model.message import ToolCall
        from tau2.environment.toolkit import ToolType

        if name not in self._read_tool_names() or self._env.tools.tool_type(name) != ToolType.READ:
            raise PermissionError(f"{name} is not a read-only tool")
        res = self._env.get_response(ToolCall(id="r", name=name, requestor="assistant",
                                              arguments=dict(args or {})))
        return str(getattr(res, "content", res))

    def _user_read_tool_names(self) -> List[str]:
        """Names of the **read-only** (``ToolType.READ``) USER tools (dual-control domains).

        For dual-control domains (telecom) the user simulator also calls tools; its read-only ones
        (e.g. ``check_status_bar``, ``run_speed_test``) are served exactly by the world model, while
        its writes (e.g. ``toggle_roaming``) are WM-predicted — keeping the real env untouched."""
        try:
            from tau2.environment.toolkit import ToolType

            tk = getattr(self._env, "user_tools", None)
            if tk is None:
                return []
            return [n for n in tk.get_tools() if tk.tool_type(n) == ToolType.READ]
        except Exception:
            return []

    def world_model(self, config: SimulatorConfig, budget, cache=None) -> WorldModel:
        if not self._world_model_eligible():
            raise NotImplementedError(
                f"no read-grounded world model for this {self.domain!r} task "
                "(env exposes no readable entity DB — falls back to the oracle reward)"
            )
        model = self._model()
        criteria = getattr(self._task, "evaluation_criteria", None)
        evaluation_context = {
            "communicate_info": list(getattr(criteria, "communicate_info", None) or []),
            "nl_assertions": list(getattr(criteria, "nl_assertions", None) or []),
        }
        return make_world_model(
            Tau2EntityStore(self._db(), model),
            Tau2ShadowEvaluator(self.domain, self._gold, model, tool_docs=self._tool_docstrings(),
                                read_tools=self._read_tool_names(),
                                read_tool_runner=self._run_read_tool,
                                task_text=self._instruction,
                                evaluation_context=evaluation_context,
                                gold_action_specs=self._gold_raw),
            config, budget, BudgetConfig(), cache,
        )

    def fork(self) -> "Tau2TaskEnv":
        db = self._db()
        forked_env = _make_environment(self.domain, db=copy.deepcopy(db))  # B2 baseline ONLY
        env = Tau2TaskEnv(domain=self.domain, task=self._task, subset=self.subset, _env=forked_env)
        env.split = getattr(self, "split", None)
        env.level = getattr(self, "level", None)
        return env

    def gold_trajectory(self) -> Trajectory:
        criteria = getattr(self._task, "evaluation_criteria", None)
        communicate_info = list(getattr(criteria, "communicate_info", None) or [])
        final_message = "Done."
        if communicate_info:
            final_message = "Done. " + " ".join(str(value) for value in communicate_info)
        return Trajectory(steps=[Step(action=a) for a in self._gold],
                          final_message=final_message)

    # -- stepping / evaluation / commit --------------------------------- #
    def step(self, action: Action) -> Observation:
        from tau2.data_model.message import ToolCall

        tc = ToolCall(id=f"c{self._commits}", name=action.name, requestor="assistant",
                      arguments=dict(action.args))
        res = self._env.get_response(tc)  # official execution; mutates the real DB
        content = getattr(res, "content", None)
        return Observation(content=str(content) if content is not None else "", error=False, raw=res)

    def _official_reward(self, actions: List[Action]) -> RewardInfo:
        """Official τ² evaluation of a tool-call sequence (replays on a scratch env)."""
        from tau2.data_model.message import AssistantMessage, ToolCall, ToolMessage
        from tau2.data_model.simulation import SimulationRun, TerminationReason
        from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation

        scratch = _make_environment(self.domain)
        messages = []
        for i, a in enumerate(actions):
            tc = ToolCall(id=f"c{i}", name=a.name, requestor="assistant", arguments=dict(a.args))
            res = scratch.get_response(tc)
            messages.append(AssistantMessage(role="assistant", content=None, tool_calls=[tc]))
            messages.append(res if isinstance(res, ToolMessage) else ToolMessage(id=f"c{i}", role="tool", content=str(res)))
        messages.append(AssistantMessage(role="assistant", content="Done.", tool_calls=None))
        sim = SimulationRun(
            id="s", task_id=self.task_id, messages=messages,
            termination_reason=TerminationReason.USER_STOP,
            start_time="2020-01-01T00:00:00", end_time="2020-01-01T00:00:01", duration=1.0,
        )
        try:
            ri = evaluate_simulation(sim, self._task, EvaluationType.ENV, solo_mode=False, domain=self.domain)
            reward = float(getattr(ri, "reward", 0.0))
        except Exception as exc:  # an env-assertion can raise on an incomplete trajectory -> fail
            return RewardInfo(reward=0.0, success=False, grounded=True,
                              reason=f"official tau2 ENV evaluation error: {type(exc).__name__}")
        return RewardInfo(reward=reward, success=reward >= 0.999, grounded=True,
                          reason=f"official tau2 ENV evaluation ({self.domain})")

    def evaluate(self, trajectory: Trajectory) -> RewardInfo:
        return self._official_reward(trajectory.actions)

    # -- REAL official agent (B0 / sampling) ---------------------------- #
    def run_agent_rollout(self, config, seed: int):
        """Run the **official** τ² agent loop and return ``(Trajectory, RewardInfo)``.

        Drives the official runner — the LLM customer-service agent (with the official
        :data:`tau2.agent.llm_agent.AGENT_INSTRUCTION` / ``SYSTEM_PROMPT``) conversing with the
        official user simulator over the real environment for ``self.domain`` — and scores the
        conversation with the official ``evaluate_simulation`` (``EvaluationType.ALL``).  The
        returned reward is the genuine benchmark reward, so a single rollout (B0) reproduces the
        official τ²-bench number for that domain.  The realised assistant tool-calls are returned
        as the trajectory.
        """
        from tau2.data_model.message import AssistantMessage
        from tau2.evaluator.evaluator import EvaluationType
        from tau2.run import run_single_task

        from ..llm import ensure_litellm_azure_env

        # Retry transient LLM failures (expired AAD bearer token / rate limits / timeouts) with a
        # freshly-refreshed Azure token — long parallel runs can otherwise lose a task mid-conversation.
        sim = None
        last_err = None
        for attempt in range(4):
            default_model = ensure_litellm_azure_env(force_refresh=attempt > 0)  # fresh token on retry
            _align_tau2_judge_models(default_model)  # route the NL-assertion judge to Azure too
            _align_tau2_embedder_azure()  # route banking KB embeddings (text-embedding-3-large) to Azure
            run_cfg = self._build_run_cfg(config, seed, default_model)
            try:
                sim = run_single_task(run_cfg, self._task, seed=seed, evaluation_type=EvaluationType.ALL)
                break
            except Exception as err:  # noqa: BLE001
                msg = str(err).lower()
                transient = any(k in msg for k in (
                    "authentication", "invalid subscription key", "401", "rate", "429",
                    "timeout", "timed out", "connection", "503", "502", "overloaded"))
                if attempt < 3 and transient:
                    last_err = err
                    time.sleep(2 ** attempt)
                    continue
                raise
        if sim is None:  # pragma: no cover - only if all retries exhausted
            raise RuntimeError(f"tau2 rollout failed after retries: {last_err}")

        steps: List[Step] = []
        final_message = "Done."
        for msg in sim.messages:
            if isinstance(msg, AssistantMessage):
                if msg.content:
                    final_message = str(msg.content)
                for tc in (msg.tool_calls or []):
                    steps.append(Step(action=Action(tc.name, dict(tc.arguments or {}))))
        traj = Trajectory(steps=steps, final_message=final_message)
        reward = float(getattr(sim.reward_info, "reward", 0.0) or 0.0)
        ri = RewardInfo(reward=reward, success=reward >= 0.999, grounded=True,
                        reason="official tau2 simulation (EvaluationType.ALL)")
        return traj, ri

    def _build_run_cfg(self, config, seed: int, default_model: str):
        """The official ``TextRunConfig`` (llm_agent + user_simulator) used for a rollout."""
        from tau2.data_model.simulation import TextRunConfig

        from ..llm.local import litellm_model_for

        solver = config.solver
        llm_args = {"seed": seed}
        if solver.temperature:
            llm_args["temperature"] = solver.temperature
        return TextRunConfig(
            domain=self.domain, agent="llm_agent", user="user_simulator",
            llm_agent=litellm_model_for(solver.llm_agent, default_model),
            llm_args_agent=dict(llm_args),
            llm_user=litellm_model_for(solver.llm_user or solver.llm_agent, default_model),
            llm_args_user=dict(llm_args),
            max_steps=solver.max_steps, seed=seed,
        )

    def run_agent_rollout_wm(self, config, method: str, seed: int, budget, cache=None):
        """Official agent + user simulator interacting with the **world model** (no real writes).

        Builds the official τ² orchestrator with the exact same agent/user configuration, prompts,
        and generation loop as the real-environment path. The only substitution is the
        tool-execution chokepoint — ``environment.get_response`` — which is routed through the
        read-grounded world model: read tools are answered exactly (read-only), write/dual-control
        tools are WM-predicted from read-grounded facts + the running history, and the real backing
        DB is never mutated. Returns ``(Trajectory, predicted RewardInfo, spawn_root)``; the
        trajectory carries the realized official message log (assistant + user tool calls) so the
        selected one can be committed and officially scored — correctly for dual-control domains too.
        """
        if not self._world_model_eligible():
            raise NotImplementedError(
                f"no read-grounded world model for this {self.domain!r} task")

        from tau2.data_model.message import AssistantMessage, ToolMessage
        from tau2.run import build_text_orchestrator

        from ..llm import ensure_litellm_azure_env

        kind = "flat" if method == "b1" else "recursive"
        prepare_tau2_python_tool()
        wm = self.world_model(replace(config.simulator, kind=kind), budget, cache)
        read_assist = set(self._read_tool_names())
        read_user = set(self._user_read_tool_names())
        model = self._model()

        def _entity_refs(action: Action) -> List[Tuple[str, str]]:
            """Entity (type, id) refs named in an action's args (for dirty-read tracking)."""
            refs: List[Tuple[str, str]] = []
            if model is not None:
                for value in _scan_scalars(action.args or {}):
                    refs.extend(model.matches(value))
            return refs

        # Dirty-state tracking: a read of an entity a prior write already changed must be PREDICTED
        # (so it reflects the post-write state), while reads of untouched entities stay exact.
        dirty: set = set()           # entity refs written by entity-keyed writes
        global_writes = [0]          # count of ref-less writes (device/global state, e.g. telecom)

        def _make_wm_get_response(orig_get_response):
            def _wm_get_response(message):
                """Serve one tool call: clean read exact, dirty read + write WM-predicted (no real write).

                Requestor-aware: assistant- and user-side (dual-control) read tools of *untouched*
                entities are executed exactly against the real backing DB (no mutation).  A read of an
                entity a prior write already changed, and every write itself, is predicted by the world
                model (history-conditioned), so reads reflect prior writes and the real env is never
                written."""
                name, requestor = message.name, message.requestor
                action = Action(name, dict(message.arguments or {}))
                refs = _entity_refs(action)
                is_read = name in (read_user if requestor == "user" else read_assist)
                if is_read:
                    read_is_dirty = bool(set(refs) & dirty) if refs else (global_writes[0] > 0)
                    if not read_is_dirty:
                        try:
                            tm = orig_get_response(message)  # clean read -> exact, read-only
                            wm.note(action, str(getattr(tm, "content", "")))
                            return tm
                        except Exception as exc:  # noqa: BLE001
                            return ToolMessage(id=message.id, content=f"Error: {exc}",
                                               requestor=requestor, role="tool", error=True)
                    try:
                        obs = wm.predict_step(action)  # dirty read -> predicted post-write state
                        content, err = str(obs.content), bool(obs.error)
                    except Exception as exc:  # noqa: BLE001
                        content, err = f"Error: {type(exc).__name__}: {exc}", True
                    return ToolMessage(id=message.id, content=content, requestor=requestor,
                                       role="tool", error=err)
                # write -> WM-predicted (no real mutation); mark its entities dirty
                try:
                    obs = wm.step(action)
                    content, err = str(obs.content), bool(obs.error)
                except Exception as exc:  # never let a prediction failure crash the rollout
                    content, err = f"Error: {type(exc).__name__}: {exc}", True
                if refs:
                    dirty.update(refs)
                else:
                    global_writes[0] += 1
                return ToolMessage(id=message.id, content=content, requestor=requestor,
                                   role="tool", error=err)

            return _wm_get_response

        sim = None
        last_err = None
        for attempt in range(4):
            default_model = ensure_litellm_azure_env(force_refresh=attempt > 0)
            _align_tau2_judge_models(default_model)
            _align_tau2_embedder_azure()
            run_cfg = self._build_run_cfg(config, seed, default_model)
            try:
                orch = build_text_orchestrator(run_cfg, self._task, seed=seed)
                # Prompt/loop parity invariant: do not modify orch.agent or orch.user. Only route the
                # environment's tool-execution chokepoint through the world model.
                orch.environment.get_response = _make_wm_get_response(orch.environment.get_response)
                sim = orch.run()
                break
            except Exception as err:  # noqa: BLE001
                msg = str(err).lower()
                transient = any(k in msg for k in (
                    "authentication", "invalid subscription key", "401", "rate", "429",
                    "timeout", "timed out", "connection", "503", "502", "overloaded"))
                if attempt < 3 and transient:
                    last_err = err
                    time.sleep(2 ** attempt)
                    continue
                raise
        if sim is None:  # pragma: no cover - only if all retries exhausted
            raise RuntimeError(f"tau2 WM rollout failed after retries: {last_err}")

        steps: List[Step] = []
        final_message = "Done."
        for msg in sim.messages:
            if isinstance(msg, AssistantMessage):
                if msg.content:
                    final_message = str(msg.content)
                for tc in (msg.tool_calls or []):
                    steps.append(Step(action=Action(tc.name, dict(tc.arguments or {}))))
        agent_traj = Trajectory(steps=steps, final_message=final_message)
        agent_traj.sim_messages = list(sim.messages)
        agent_traj.termination_reason = sim.termination_reason
        # predicted (selection) signal: the WM's own success belief over the realized actions.
        predicted = wm.evaluate(agent_traj)
        return agent_traj, predicted, wm.spawn_root

    def commit_trajectory(self, trajectory: Trajectory) -> RewardInfo:
        """Commit the selected trajectory to the REAL env **once** and score that single run.

        The actions execute on the real env (the single legitimate write) and their genuine tool
        messages are captured; the official ``evaluate_simulation`` then scores that message log.
        This avoids the previously-redundant extra replay on a scratch env (``_official_reward``)
        for the commit path — one real execution, one score.

        When the trajectory carries a realized official message log (``sim_messages`` — produced by
        the world-model rollout, including dual-control *user* tool calls), that faithful log is
        scored directly via ``evaluate_simulation`` (which replays it on a fresh real env), so
        dual-control domains are scored correctly.
        """
        from tau2.data_model.message import AssistantMessage, ToolCall, ToolMessage
        from tau2.data_model.simulation import SimulationRun, TerminationReason
        from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation

        self._commits += 1

        sim_messages = getattr(trajectory, "sim_messages", None)
        if sim_messages:
            # Faithful commit for a world-model rollout.  The recorded tool results in ``sim_messages``
            # are WM *predictions* (e.g. a guessed ``reservation_id``/timestamp), which would make the
            # official env-consistency check RAISE.  So we RE-EXECUTE the realized tool calls (assistant
            # AND user, requestor preserved) on the real env — the single grounded write — substituting
            # the REAL tool results, while keeping the agent/user TEXT messages (so the communicate / NL
            # bases see what the agent actually said).  This is dual-control safe.
            # the recorded results must match what evaluate_simulation re-derives on its OWN fresh env,
            # so we re-execute on a fresh env of the same domain (identical initial state), not on the
            # already-touched ``self._env`` (whose state could diverge for dual-control device tools).
            scratch = _make_environment(self.domain)
            rebuilt = []
            for m in sim_messages:
                if isinstance(m, ToolMessage):
                    continue  # drop the predicted tool result; replaced with the real one below
                rebuilt.append(m)
                for tc in (getattr(m, "tool_calls", None) or []):
                    try:
                        res = scratch.get_response(tc)  # the single grounded execution (requestor preserved)
                    except Exception as exc:  # noqa: BLE001
                        res = ToolMessage(id=tc.id, role="tool", requestor=tc.requestor,
                                          content=f"Error: {exc}", error=True)
                    rebuilt.append(res if isinstance(res, ToolMessage)
                                   else ToolMessage(id=tc.id, role="tool", requestor=tc.requestor,
                                                    content=str(getattr(res, "content", res))))
            term = getattr(trajectory, "termination_reason", None) or TerminationReason.USER_STOP
            sim = SimulationRun(
                id="s", task_id=self.task_id, messages=rebuilt, termination_reason=term,
                start_time="2020-01-01T00:00:00", end_time="2020-01-01T00:00:01", duration=1.0,
            )
            try:
                ri = evaluate_simulation(sim, self._task, EvaluationType.ALL, solo_mode=False,
                                         domain=self.domain)
                reward = float(getattr(ri, "reward", 0.0))
            except Exception as exc:  # noqa: BLE001
                return RewardInfo(reward=0.0, success=False, grounded=True,
                                  reason=f"official tau2 evaluation error: {type(exc).__name__}: {exc}")
            return RewardInfo(reward=reward, success=reward >= 0.999, grounded=True,
                              reason=f"official tau2 evaluation on committed WM trajectory ({self.domain})")

        messages = []
        for i, action in enumerate(trajectory.actions):
            tc = ToolCall(id=f"c{i}", name=action.name, requestor="assistant",
                          arguments=dict(action.args))
            res = self._env.get_response(tc)  # the single real write
            messages.append(AssistantMessage(role="assistant", content=None, tool_calls=[tc]))
            messages.append(res if isinstance(res, ToolMessage)
                            else ToolMessage(id=f"c{i}", role="tool", content=str(getattr(res, "content", res))))
        messages.append(AssistantMessage(role="assistant", content="Done.", tool_calls=None))
        sim = SimulationRun(
            id="s", task_id=self.task_id, messages=messages,
            termination_reason=TerminationReason.USER_STOP,
            start_time="2020-01-01T00:00:00", end_time="2020-01-01T00:00:01", duration=1.0,
        )
        try:
            ri = evaluate_simulation(sim, self._task, EvaluationType.ENV, solo_mode=False, domain=self.domain)
            reward = float(getattr(ri, "reward", 0.0))
        except Exception as exc:  # an env-assertion can raise on an incomplete trajectory -> fail
            return RewardInfo(reward=0.0, success=False, grounded=True,
                              reason=f"official tau2 ENV evaluation error: {type(exc).__name__}")
        return RewardInfo(reward=reward, success=reward >= 0.999, grounded=True,
                          reason=f"official tau2 ENV evaluation on committed trajectory ({self.domain})")

    def state_hash(self) -> str:
        return self._env.get_db_hash()

    def committed_count(self) -> int:
        return self._commits


__all__ = [
    "Tau2TaskEnv",
    "Tau2EntityStore",
    "Tau2ShadowEvaluator",
    "AirlineShadowEvaluator",
    "MUTATING_TOOLS",
    "tau2_mutating_tools",
    "real_tau2_available",
    "real_tau2_subset_size",
]
