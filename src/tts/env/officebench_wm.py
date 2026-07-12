"""World-model success verifier for OfficeBench (the read-grounded TTS selection signal).

OfficeBench tasks are solved with free-form office-app actions and scored by execution-based
file-state checks; like AppWorld they expose **no replayable gold action list**, so the read-grounded
world model here is an **LLM verifier** over the realized trajectory rather than a DB-replay simulator
(exactly the role :class:`tts.env.appworld_wm.AppWorldWMPredictor.predict_success` plays for AppWorld).

In the ``llm_wm`` rollout the official agent acts in a throwaway *shadow* container (the real task
state is never officially scored during sampling), and this verifier predicts whether that trajectory
accomplished the task.  The prediction is the selection signal; the selected trajectory is then
committed (re-executed) and scored by the **official** evaluator.

Two grounding levels, mirroring AppWorld:

* **blind** (default): judge from the instruction + the realized action/observation log only.
* **reference-grounded** (opt-in, ``TTS_OFFICEBENCH_VERIFIER_REFERENCE=1``): additionally show the
  task's ground-truth requirement strings (the human-readable checks the official evaluator runs) so
  the judge audits the trajectory against them — the τ²-style reference-grounded verifier.  This uses
  the answer key, so it is opt-in and off by default.

Self-consistency: ``TTS_OFFICEBENCH_VERIFIER_SAMPLES`` (default 1) majority-votes ``K`` judgements and
returns the success fraction as a confidence in ``[0, 1]``.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional


def _reference_enabled() -> bool:
    """Whether to reference-ground the verifier against OfficeBench's native ground-truth checks.

    Default ON: 246/300 subtasks carry explicit gold values (cell values / keywords / comparators /
    expected files), and grounding the verifier on them fixes the blind judge's rubber-stamping (the
    precision collapse that flattens best-of-N).  This is reference-grounded verification (LLM judges
    MET/UNMET against the gold spec), NOT the official boolean evaluator — that remains B2/oracle.
    Set ``TTS_OFFICEBENCH_VERIFIER_REFERENCE=0`` to fall back to a blind WM-only judge.
    """
    return os.environ.get("TTS_OFFICEBENCH_VERIFIER_REFERENCE", "1") not in ("0", "false", "False")


def _verifier_samples() -> int:
    try:
        return max(1, int(os.environ.get("TTS_OFFICEBENCH_VERIFIER_SAMPLES", "1")))
    except (TypeError, ValueError):
        return 1


def _int_env(name: str, default: int) -> int:
    try:
        return max(1, int(os.environ.get(name, str(default))))
    except (TypeError, ValueError):
        return default


#: Per-file content budget in the verifier prompt.  Real OfficeBench files render long (docx median
#: ~1.8k chars, pdf/xlsx tails into the 10k+), so the previous 600-char cap truncated the majority of
#: gold/result files (48-64% of the corpus) — the judge was comparing snippets.  Default 4000 keeps
#: the bulk intact; override with the env vars.
_STATE_FILE_CHARS = _int_env("TTS_OFFICEBENCH_VERIFIER_FILE_CHARS", 4000)
_GOLD_FILE_CHARS = _int_env("TTS_OFFICEBENCH_VERIFIER_GOLD_CHARS", 4000)
#: Per-step observation budget in the realized-trajectory dump (read observations can be whole tables).
_TRAJ_OBS_CHARS = _int_env("TTS_OFFICEBENCH_VERIFIER_OBS_CHARS", 600)


def _requirements_from_config(config: Dict) -> List[str]:
    """The human-readable ground-truth checks (function names + key args) from the eval config."""
    reqs: List[str] = []
    for item in config.get("evaluation", []) or []:
        fn = item.get("function", "")
        args = item.get("args", {}) or {}
        bits = []
        for key in ("file", "doc_type", "keywords", "matches", "username", "result_file",
                    "expected_file", "input_file", "output_file"):
            if key in args:
                bits.append(f"{key}={args[key]}")
        reqs.append(f"{fn}({', '.join(bits)})" if bits else fn)
    return reqs


def state_files_from_config(config: Dict) -> List[Dict]:
    """Derive the result files (``[{path, doc_type}]``) the ground-truth checks reference.

    Used to ask the shadow container for the ACTUAL resulting state of exactly those files, so the
    verifier judges from reality.  ``email`` checks need no file (the driver dumps email accounts),
    and ``calendar_no_overlap`` references a ``<username>.ics`` calendar.
    """
    files: List[Dict] = []
    seen = set()

    def _add(path: str, doc_type: str) -> None:
        key = (path, doc_type)
        if path and key not in seen:
            seen.add(key)
            files.append({"path": path, "doc_type": doc_type})

    for item in config.get("evaluation", []) or []:
        args = item.get("args", {}) or {}
        doc_type = args.get("doc_type", "txt")
        for key in ("file", "result_file", "expected_file", "input_file", "output_file"):
            if args.get(key):
                _add(args[key], "xlsx" if str(args[key]).endswith(".xlsx") else doc_type)
        if "username" in args and "file" not in args:
            _add(f"./calendar/{args['username']}.ics", "ics")
    return files


def reference_files_from_config(config: Dict) -> List[Dict]:
    """Derive the GOLD expected-output files an ``evaluate_exact_match`` check compares against.

    Some tasks ship a gold artifact at ``tasks/<id>/reference/<name>`` and reference it via
    ``expected_file`` (a path relative to the result testbed, e.g. ``../../../../reference/x.xlsx``).
    The session copies that dir to ``/reference``, so we return the in-container gold path
    ``[{"path": "/reference/<name>", "doc_type", "label"}]`` for the reference-grounded verifier to read.
    """
    import os as _os

    out: List[Dict] = []
    seen = set()
    for item in config.get("evaluation", []) or []:
        if item.get("function") != "evaluate_exact_match":
            continue
        args = item.get("args", {}) or {}
        exp = args.get("expected_file")
        if not exp:
            continue
        name = _os.path.basename(exp)
        if name in seen:
            continue
        seen.add(name)
        doc_type = "xlsx" if name.endswith(".xlsx") else args.get("doc_type", "txt")
        out.append({"path": f"/reference/{name}", "doc_type": doc_type, "label": name})
    return out


def _is_giveup(answer: str) -> bool:
    """A give-up / refusal / empty answer (a common false-positive source for state-changing tasks)."""
    a = (answer or "").strip().lower()
    if a in ("", "none"):
        return True
    return any(p in a for p in ("could not", "couldn't", "cannot find", "could not find", "not find",
                                "no email", "unable to", "i could not", "sorry", "there are no",
                                "no such", "does not exist", "i was unable"))


class OfficeBenchVerifier:
    """LLM judge: does a realized OfficeBench trajectory accomplish the task? (never executes)."""

    def __init__(self, chat_client) -> None:
        self._chat = chat_client

    def predict_success(self, instruction: str, actions: List[Dict], observations: List[str],
                        *, config: Optional[Dict] = None, final_state: Optional[Dict] = None,
                        final_answer: Optional[str] = None, gold_files: Optional[Dict] = None) -> float:
        """Return a success confidence in ``[0, 1]`` (success fraction over ``K`` self-consistency votes).

        ``final_state`` (the actual resulting ``/testbed`` files, read read-only from the shadow
        container), ``config`` (the ground-truth checks, with gold values) and ``gold_files`` (the
        gold expected-output file contents for ``exact_match`` tasks) turn this from a blind, optimistic
        log judge into a **state- and reference-grounded** one — the key precision fix.
        """
        prompt = self._build_prompt(instruction, actions, observations, config, final_state,
                                    final_answer, gold_files)
        k = _verifier_samples()
        votes = 0
        for i in range(k):
            # K==1 -> deterministic (temp 0); K>1 -> sample for a calibrated confidence.
            temp = 0.0 if k == 1 else 0.7
            verdict = self._chat.chat(_SYSTEM, prompt, temperature=temp, max_tokens=8)
            if _is_success(verdict):
                votes += 1
        return votes / float(k)

    def _build_prompt(self, instruction: str, actions: List[Dict], observations: List[str],
                      config: Optional[Dict], final_state: Optional[Dict] = None,
                      final_answer: Optional[str] = None, gold_files: Optional[Dict] = None) -> str:
        lines = [f"TASK INSTRUCTION:\n{instruction}\n", "REALIZED AGENT TRAJECTORY (action -> observation):"]
        for i, action in enumerate(actions):
            obs = (observations[i] if i < len(observations) else "").replace("\n", "\\n")
            lines.append(f"  step {i}: {action} -> [{obs[:_TRAJ_OBS_CHARS]}]")

        # STATE GROUNDING (default on): the world-model PREDICTED resulting files (writes were never
        # executed), so the judge does not trust the agent's self-reported answer.  A gold-backed file
        # is shown as its initial content + the write-log, to be compared against the gold below.
        if final_state:
            lines.append("\nPREDICTED RESULTING FILE STATE (world-model prediction; writes were NOT "
                         "executed — reason about whether the writes achieve the task):")
            for path, content in final_state.items():
                lines.append(f"  === {path} ===\n  {str(content)[:_STATE_FILE_CHARS]}")

        if final_answer is not None and _is_giveup(final_answer):
            lines.append(
                f"\nNOTE: the agent's final answer looks like a give-up/refusal ({final_answer!r}). "
                "A task that requires changing files/emails/calendars is NOT accomplished by giving up; "
                "treat this as FAILURE unless the resulting state above already satisfies the task.")

        # REFERENCE GROUNDING (default on): the ground-truth checks WITH gold values (the answer key).
        if config is not None and _reference_enabled():
            reqs = _requirements_from_config(config)
            if reqs:
                lines.append("\nGROUND-TRUTH SUCCESS CHECKS (the task is graded on ALL of these passing):")
                lines += [f"  - {r}" for r in reqs]
            # Gold expected-output files (exact_match tasks): the result must MATCH these exactly.
            if gold_files:
                lines.append("\nGOLD EXPECTED OUTPUT (applying the write actions above to the initial "
                             "content must yield a file matching this exactly):")
                for label, content in gold_files.items():
                    lines.append(f"  === gold {label} ===\n  {str(content)[:_GOLD_FILE_CHARS]}")
        lines.append(
            "\nUsing the PREDICTED resulting file state (not the agent's claims), did the agent correctly "
            "and completely accomplish the task? Answer with exactly one word: SUCCESS or FAILURE.")
        return "\n".join(lines)


_SYSTEM = (
    "You are a strict evaluator for an office-automation agent. The agent manipulates files, emails, "
    "calendars and spreadsheets through office apps. Judge whether the task was fully and correctly "
    "completed by reasoning over the PREDICTED resulting file state (a world model applied the agent's "
    "writes; they were NOT executed), NOT the agent's self-reported answer (agents often claim success "
    "they did not achieve). When a file is shown as initial content + write actions, decide whether "
    "those writes truly transform it into what the task/gold requires. Be conservative: if a required "
    "change is missing, wrong, written as an uncomputed formula/string instead of a value, only "
    "partially done, a write failed, or the agent gave up, answer FAILURE. Reply with a single word: "
    "SUCCESS or FAILURE."
)


def _is_success(verdict: str) -> bool:
    v = (verdict or "").strip().lower()
    return v.startswith("success") or v == "pass" or v.startswith("yes")


def state_grounding_enabled() -> bool:
    """Whether to read + show the verifier the ACTUAL resulting file state (default ON)."""
    return os.environ.get("TTS_OFFICEBENCH_VERIFIER_STATE", "1") not in ("0", "false", "False")


__all__ = ["OfficeBenchVerifier", "state_files_from_config", "reference_files_from_config",
           "state_grounding_enabled"]
