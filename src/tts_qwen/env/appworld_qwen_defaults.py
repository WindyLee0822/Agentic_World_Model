"""Method-local defaults for the validated AppWorld Qwen repair.

The runner enters this context inside an AppWorld worker, never around a
mixed-method process pool. No prediction implementation is changed here.
"""
import copy
import os
from contextlib import contextmanager


def repair_profile(args, method):
    model = (getattr(args, "llm_agent", None) or os.environ.get("TTS_LOCAL_MODEL")
             or os.environ.get("AZURE_OPENAI_DEPLOYMENT") or os.environ.get("WM_MODEL")
             or os.environ.get("TTS_AGENT_MODEL") or "")
    wm_model = getattr(args, "wm_model", None)
    if (method != "ours" or getattr(args, "agent", None) != "llm_wm"
            or "qwen" not in model.lower()
            or (wm_model and "qwen" not in wm_model.lower())
            or getattr(args, "appworld_qwen_repair", "legacy") != "v3"
            or getattr(args, "appworld_wm_tool_mode", "python") != "python"
            or (getattr(args, "_predict_all_actions_explicit", False)
                and not getattr(args, "predict_all_actions", True))):
        return "legacy"
    return "v3"


_REQUIRED = {
    "TTS_APPWORLD_OURS_QWEN_SIMULATE_ALL_ACTIONS": "1",
    "TTS_APPWORLD_OURS_QWEN_INITIAL_STATE_CONTEXT": "1",
    "TTS_APPWORLD_OURS_QWEN_PERSISTENT_WM": "1",
    "TTS_APPWORLD_OURS_QWEN_COMPUTED_READS": "1",
    "TTS_APPWORLD_OURS_QWEN_COMPUTED_VERIFIER": "1",
    "TTS_APPWORLD_OURS_QWEN_WRITE_DISPATCH_FIX": "1",
    "TTS_APPWORLD_OURS_QWEN_EVIDENCE_GUARD": "1",
    "TTS_APPWORLD_OURS_QWEN_COMPLETE_CHILDREN": "1",
    "TTS_APPWORLD_OURS_QWEN_EXACT_SHADOW_ACTIONS": "0",
    "TTS_APPWORLD_OURS_QWEN_ORACLE_DIAGNOSTIC": "0",
}
_DEFAULTS = {
    "TTS_APPWORLD_OURS_QWEN_SOLVER_THINKING": "1",
    "TTS_APPWORLD_OURS_QWEN_TRACE_OBSERVATIONS": "1",
    "TTS_APPWORLD_OURS_QWEN_REFERENCE_REPAIR": "0",
    "TTS_APPWORLD_OURS_QWEN_REFERENCE_RERANK": "0",
    "TTS_APPWORLD_INSTRUCTION_PREFLIGHTS": "0",
    "TTS_APPWORLD_AGENT_REFERENCE": "0",
    "TTS_APPWORLD_VERIFIER_REFERENCE": "0",
    "TTS_APPWORLD_VERIFIER_PERCHECK": "0",
    "TTS_APPWORLD_COMPARATIVE_RERANK": "0",
    "TTS_APPWORLD_WM_HARD_TIMEOUT_S": "1800",
}


@contextmanager
def configured(args, method):
    effective = copy.copy(args)
    if repair_profile(args, method) != "v3":
        yield effective
        return
    changes = {**{k: v for k, v in _DEFAULTS.items() if k not in os.environ}, **_REQUIRED}
    previous = {k: os.environ.get(k) for k in changes}
    effective.predict_all_actions = True
    if not getattr(args, "wm_base_url", None):
        from ..llm.local import local_base_url, DEFAULT_LOCAL_BASE_URL
        effective.wm_base_url = (local_base_url() or os.environ.get("OPENAI_BASE_URL")
                                 or (getattr(args, "local_base_url", None) or DEFAULT_LOCAL_BASE_URL
                                     if getattr(args, "local", False) else None))
    if not getattr(args, "wm_model", None):
        effective.wm_model = getattr(args, "llm_agent", None) or os.environ.get("TTS_LOCAL_MODEL")
    if not getattr(args, "wm_api_key", None) and os.environ.get("OPENAI_API_KEY"):
        effective.wm_api_key = os.environ["OPENAI_API_KEY"]
    if not getattr(args, "_success_score_explicit", False) and not (
        getattr(args, "verifier_reference", False) or getattr(args, "verifier_percheck", False)
    ):
        effective.success_score = "simulator_confidence_guarded"
    if not getattr(args, "_wm_temperature_explicit", False):
        effective.wm_temperature = 0.0
    try:
        os.environ.update(changes)
        yield effective
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
