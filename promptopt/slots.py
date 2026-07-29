"""Faithful rendering of verifier *inputs* into template *data slots*.

These functions reproduce the data-formatting the source verifiers do inside their prompt builders
(``officebench_wm.OfficeBenchVerifier._build_prompt`` and
``real_tau2._LLMWorldModelPredictor.predict_success``), so a captured example reproduces the exact
data the baseline prompt would have seen.  Only the *wording* around these slots is optimized.
"""

from __future__ import annotations

from typing import Dict, List, Optional


def officebench_slots(instruction: str, actions: List, observations: List[str], *,
                      config: Optional[Dict] = None, final_state: Optional[Dict] = None,
                      final_answer: Optional[str] = None,
                      gold_files: Optional[Dict] = None) -> Dict[str, str]:
    from tts.env.officebench_wm import (
        _STATE_FILE_CHARS, _GOLD_FILE_CHARS, _TRAJ_OBS_CHARS,
        _requirements_from_config, _is_giveup, _reference_enabled,
    )

    traj_lines = []
    for i, action in enumerate(actions):
        obs = (observations[i] if i < len(observations) else "").replace("\n", "\\n")
        traj_lines.append(f"  step {i}: {action} -> [{obs[:_TRAJ_OBS_CHARS]}]")
    trajectory = "\n".join(traj_lines)

    state = ""
    if final_state:
        state = "\n".join(f"  === {p} ===\n  {str(c)[:_STATE_FILE_CHARS]}" for p, c in final_state.items())

    giveup = ""
    if final_answer is not None and _is_giveup(final_answer):
        giveup = (f"\nNOTE: the agent's final answer looks like a give-up/refusal ({final_answer!r}). "
                  "A task that requires changing files/emails/calendars is NOT accomplished by giving "
                  "up; treat this as FAILURE unless the resulting state above already satisfies the task.")

    requirements = gold = ""
    if config is not None and _reference_enabled():
        reqs = _requirements_from_config(config)
        if reqs:
            requirements = "\n".join(f"  - {r}" for r in reqs)
        if gold_files:
            gold = "\n".join(f"  === gold {label} ===\n  {str(content)[:_GOLD_FILE_CHARS]}"
                             for label, content in gold_files.items())

    return {"instruction": str(instruction), "trajectory": trajectory, "final_state": state,
            "giveup_note": giveup, "requirements": requirements, "gold_files": gold}


def tau2_slots(domain: str, facts: str, candidate_actions: List, gold_actions: List,
               tool_docs: str = "") -> Dict[str, str]:
    from tts.env.real_tau2 import _action_str

    gold = "\n".join(_action_str(a) for a in gold_actions) or "(none)"
    cand = "\n".join(_action_str(a) for a in candidate_actions) or "(none)"
    ref = f"\nTool reference (behaviour of each tool):\n{tool_docs}\n" if tool_docs else ""
    return {"domain": str(domain), "ref": ref, "facts": str(facts), "gold": gold, "cand": cand}


__all__ = ["officebench_slots", "tau2_slots"]
