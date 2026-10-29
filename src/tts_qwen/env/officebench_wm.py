"""World-model success verifier for OfficeBench (the read-grounded TTS selection signal).

OfficeBench tasks are solved with free-form office-app actions and scored by execution-based
file-state checks; like AppWorld they expose **no replayable gold action list**, so the read-grounded
world model here is an **LLM verifier** over the realized trajectory rather than a DB-replay simulator
(exactly the role :class:`tts_qwen.env.appworld_wm.AppWorldWMPredictor.predict_success` plays for AppWorld).

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

import json
import os
import re
from typing import Any, Dict, List, Optional

from .officebench_wm_python import (
    officebench_wm_tool_mode,
    run_python_prediction_loop,
)


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


def _percheck_enabled() -> bool:
    return os.environ.get(
        "TTS_OFFICEBENCH_VERIFIER_PERCHECK", "0"
    ) not in ("0", "false", "False")


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default


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

_AUDIT_LENSES = (
    "Completeness: look for any required app, output, or evaluator check the trajectory omitted.",
    "Exactness: verify names, values, paths, formulas, dates, times, and cardinalities against the gold.",
    "Execution: look for malformed, blocked, failed, stale-read, or no-op actions that break the plan.",
    "Final state: compare every predicted result artifact with every ground-truth success check.",
    "Adversarial review: actively try to prove the trajectory fails before accepting it as complete.",
)


def _atomic_requirements(config: Optional[Dict]) -> List[Dict]:
    """Split small keyword/match lists into independently auditable evaluator checks."""
    checks: List[Dict] = []
    split_cap = _int_env("TTS_OFFICEBENCH_VERIFIER_ATOMIC_CAP", 4)
    for item in (config or {}).get("evaluation", []) or []:
        function = item.get("function", "")
        args = dict(item.get("args", {}) or {})
        split_key = None
        for key in ("keywords", "matches"):
            value = args.get(key)
            if isinstance(value, list) and 1 < len(value) <= split_cap:
                split_key = key
                break
        if split_key is None:
            checks.append({"function": function, "args": args})
            continue
        for value in args[split_key]:
            atomic_args = dict(args)
            atomic_args[split_key] = [value]
            checks.append({"function": function, "args": atomic_args})
    return checks


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


def _required_output_specs(config: Optional[Dict]) -> List[tuple[str, bool]]:
    """Output files that must exist, plus whether their checks require substantive content."""
    specs = {}
    for item in (config or {}).get("evaluation", []) or []:
        function = str(item.get("function", "")).lower()
        if any(marker in function for marker in ("not_exist", "notexist", "delete", "remove")):
            continue
        args = item.get("args", {}) or {}
        negative_content = any(
            marker in function for marker in ("not_contain", "notcontain"))
        if negative_content:
            continue
        requires_content = (
            any(marker in function for marker in (
                "contain", "cell", "value", "content", "keyword"))
        )
        for key in ("file", "result_file", "output_file"):
            if args.get(key):
                specs[str(args[key])] = specs.get(str(args[key]), False) or requires_content
    return list(specs.items())


def _state_is_missing(content: object) -> bool:
    text = str(content or "").strip().lower()
    if text.startswith(("[file does not exist]", "[directory does not exist]")):
        return True
    return (
        "initial content" in text
        and "[file does not exist]" in text
        and ("(none)" in text or "(no recorded transitions)" in text)
    )


def _state_is_empty(content: object) -> bool:
    return str(content or "").strip().lower() in (
        "", "[empty sheet]", "[empty document]", "[empty cell]")


def _deterministic_failure(
        config: Optional[Dict],
        actions: List[Dict],
        observations: List[str],
        final_state: Optional[Dict],
) -> Optional[str]:
    """Return a proof of failure that does not require an LLM judgement."""
    from .officebench_wm_predict import (
        _content_arg,
        _looks_like_failure,
        _norm_path,
        write_targets,
    )

    resource_status: Dict[str, str] = {}
    read_targets = set()
    task_text = str((config or {}).get("task", "")).lower()
    requires_preserving_append = (
        "end of" in task_text
        and any(marker in task_text for marker in ("add ", "append ", "insert "))
    )
    for index, action in enumerate(actions):
        if not isinstance(action, dict):
            continue
        observation = observations[index] if index < len(observations) else ""
        if _looks_like_failure(observation):
            continue
        app, name = action.get("app"), action.get("action")
        raw_targets = write_targets(action)
        targets = [
            target for target in raw_targets
            if target not in ("<shell>", "<unknown>", "")
        ]
        if name == "read_file":
            for key in ("file_path", "file", "path"):
                if action.get(key):
                    read_targets.add(_norm_path(str(action[key])))
        if app in ("word", "excel") and name == "create_new_file":
            for target in targets:
                resource_status[target] = (
                    "content" if _content_arg(action) else "empty")
        elif app == "word" and name == "write_to_file":
            for target in targets:
                if requires_preserving_append and target not in read_targets:
                    return (
                        f"append/edit target {target} was overwritten without first reading "
                        "the existing document"
                    )
                if _content_arg(action):
                    resource_status[target] = "content"
        elif app == "excel" and name == "set_cell":
            text = str(action.get("text", ""))
            if "," in text and not (
                len(text) >= 2 and text.startswith('"') and text.endswith('"')
            ):
                return (
                    "excel.set_cell comma-containing text was not wrapped in literal "
                    "double-quote characters and will be rejected by the OfficeBench app"
                )
            for target in targets:
                resource_status[target] = "content"
        elif name in ("convert_to_pdf", "convert_to_image", "convert_to_word"):
            for target in targets:
                resource_status[target] = "derived"
        elif app == "shell":
            command = str(action.get("command", ""))
            if any(target in ("<shell>", "<unknown>") for target in raw_targets):
                resource_status.clear()
            elif re.match(r"^\s*(?:rm|unlink|rmdir)(?:\s|$)", command):
                for target in targets:
                    resource_status[target] = "missing"
            elif re.match(r"^\s*(?::|true)\s*>\s*\S+", command):
                for target in targets:
                    resource_status[target] = "empty"
            else:
                for target in targets:
                    resource_status[target] = "derived"

    if final_state is not None:
        normalized_state = {
            _norm_path(path): content
            for path, content in final_state.items()
            if path != "WRITE LOG (predicted)"
        }
        for path, require_content in _required_output_specs(config):
            target = _norm_path(path)
            content = normalized_state.get(target)
            if content is None:
                continue  # state reconstruction was unavailable; defer unknown evidence to the LLM
            if _state_is_missing(content):
                return f"required output {path} is missing"
            if require_content and (
                    resource_status.get(target) == "empty" or _state_is_empty(content)):
                return f"required output {path} was created but left empty"
    return None


class OfficeBenchVerifier:
    """Agentic LLM judge over a predicted OfficeBench trajectory (never executes writes)."""

    MAX_TURNS = 6

    def __init__(self, chat_client, *, sample_seed: Optional[int] = None) -> None:
        self._chat = chat_client
        self._sample_seed = sample_seed
        self._query = None
        self._flat_system_prompt: Optional[str] = None
        self.last_precheck_failure: Optional[str] = None
        self.last_percheck_breakdown: Optional[Dict[str, Any]] = None
        self.n_python_calls = 0
        self.n_python_failures = 0
        self.n_python_forced = 0
        self.n_python_mode_misses = 0

    def bind_query_interface(self, query) -> None:
        """Allow the verifier to gather additional read-only evidence from the pristine session."""
        self._query = query

    @property
    def uses_fixed_context(self) -> bool:
        return False


    def _system(self, default: str) -> str:
        return self._flat_system_prompt or default

    def _record_python_result(self, result) -> None:
        self.n_python_calls += result.python_calls
        self.n_python_failures += result.python_failures
        self.n_python_forced += result.python_forced
        self.n_python_mode_misses += result.python_mode_misses

    @staticmethod
    def _python_context(
        instruction: str,
        actions: List[Dict],
        observations: List[str],
        config: Optional[Dict],
        final_state: Optional[Dict],
        final_answer: Optional[str],
        gold_files: Optional[Dict],
    ) -> Dict[str, Any]:
        reference_config = config if _reference_enabled() else None
        return {
            "kind": "success_verdict",
            "instruction": instruction,
            "trajectory": [
                {
                    "action": action,
                    "observation": observations[index] if index < len(observations) else "",
                }
                for index, action in enumerate(actions)
            ],
            "requirements": _requirements_from_config(reference_config or {}),
            "config": reference_config or {},
            "final_state": final_state or {},
            "final_answer": final_answer,
            "gold_files": gold_files or {},
        }

    def predict_success(self, instruction: str, actions: List[Dict], observations: List[str],
                        *, config: Optional[Dict] = None, final_state: Optional[Dict] = None,
                        final_answer: Optional[str] = None, gold_files: Optional[Dict] = None) -> float:
        """Return a success confidence in ``[0, 1]`` (success fraction over ``K`` self-consistency votes).

        ``final_state`` (the actual resulting ``/testbed`` files, read read-only from the shadow
        container), ``config`` (the ground-truth checks, with gold values) and ``gold_files`` (the
        gold expected-output file contents for ``exact_match`` tasks) turn this from a blind, optimistic
        log judge into a **state- and reference-grounded** one — the key precision fix.
        """
        precheck_config = config if _reference_enabled() else None
        self.last_precheck_failure = _deterministic_failure(
            precheck_config, actions, observations, final_state)
        if self.last_precheck_failure is not None:
            return 0.0
        if _percheck_enabled() and precheck_config:
            atomic_score = self._predict_percheck(
                instruction,
                actions,
                observations,
                precheck_config,
                final_state,
                final_answer,
                gold_files,
            )
            whole_weight = min(
                1.0,
                max(0.0, _float_env(
                    "TTS_OFFICEBENCH_VERIFIER_WHOLE_WEIGHT", 0.0)),
            )
            if whole_weight <= 0:
                return atomic_score
            whole_score = self._predict_whole(
                instruction,
                actions,
                observations,
                config,
                final_state,
                final_answer,
                gold_files,
                samples=1,
            )
            if self.last_percheck_breakdown is not None:
                self.last_percheck_breakdown.update({
                    "atomic_score": atomic_score,
                    "whole_score": whole_score,
                    "whole_weight": whole_weight,
                })
            return (1.0 - whole_weight) * atomic_score + whole_weight * whole_score
        self.last_percheck_breakdown = None
        return self._predict_whole(
            instruction,
            actions,
            observations,
            config,
            final_state,
            final_answer,
            gold_files,
        )

    def _predict_whole(
        self,
        instruction: str,
        actions: List[Dict],
        observations: List[str],
        config: Optional[Dict],
        final_state: Optional[Dict],
        final_answer: Optional[str],
        gold_files: Optional[Dict],
        *,
        samples: Optional[int] = None,
    ) -> float:
        prompt = self._build_prompt(instruction, actions, observations, config, final_state,
                                    final_answer, gold_files)
        python_context = self._python_context(
            instruction, actions, observations, config, final_state, final_answer, gold_files)
        k = samples if samples is not None else _verifier_samples()
        votes = 0
        seedable = self._sample_seed is not None and hasattr(self._chat, "seed")
        original_seed = getattr(self._chat, "seed", None)
        query = self._query
        try:
            for i in range(k):
                # K==1 -> deterministic (temp 0); K>1 -> sampled, reproducible independent votes.
                temp = 0.0 if k == 1 else 0.7
                if seedable:
                    self._chat.seed = self._sample_seed + i
                vote_prompt = prompt
                if k > 1:
                    vote_prompt += (
                        "\n\nINDEPENDENT AUDIT FOCUS:\n"
                        f"{_AUDIT_LENSES[i % len(_AUDIT_LENSES)]}\n"
                        "Still judge the complete task and return the required verdict format."
                    )
                if officebench_wm_tool_mode() == "python":
                    result = run_python_prediction_loop(
                        lambda text, budget: self._chat.chat(
                            self._system(_SYSTEM), text,
                            temperature=temp, max_tokens=budget),
                        query,
                        vote_prompt,
                        "VERDICT: SUCCESS  or  VERDICT: FAILURE",
                        context=python_context,
                        max_turns=self.MAX_TURNS,
                        max_tokens=600,
                        output_kind="verdict",
                    )
                    self._record_python_result(result)
                    verdict = result.output
                elif query is None:
                    verdict = self._chat.chat(
                        self._system(_SYSTEM), vote_prompt,
                        temperature=temp, max_tokens=8)
                else:
                    from .officebench_wm_predict import run_agentic_read_loop

                    terminal = run_agentic_read_loop(
                        lambda text, budget: self._chat.chat(
                            self._system(_SYSTEM), text,
                            temperature=temp, max_tokens=budget),
                        query,
                        vote_prompt,
                        "VERDICT: SUCCESS  or  VERDICT: FAILURE",
                        max_turns=self.MAX_TURNS,
                        max_tokens=600,
                    )
                    verdict = terminal[1] if terminal else ""
                if _is_success(verdict):
                    votes += 1
        finally:
            if seedable:
                self._chat.seed = original_seed
        return votes / float(k)

    def _predict_percheck(
        self,
        instruction: str,
        actions: List[Dict],
        observations: List[str],
        config: Dict,
        final_state: Optional[Dict],
        final_answer: Optional[str],
        gold_files: Optional[Dict],
    ) -> float:
        checks = _atomic_requirements(config)
        if not checks:
            self.last_percheck_breakdown = None
            return 0.0

        met = unmet = unknown = 0
        seedable = self._sample_seed is not None and hasattr(self._chat, "seed")
        original_seed = getattr(self._chat, "seed", None)
        try:
            for index, check in enumerate(checks):
                if seedable:
                    self._chat.seed = self._sample_seed + index
                check_config = {"evaluation": [check]}
                prompt = self._build_prompt(
                    instruction,
                    actions,
                    observations,
                    check_config,
                    final_state,
                    final_answer,
                    gold_files,
                    status_check=check,
                )
                # status = self._parse_status(self._chat.chat(
                #     self._system(_STATUS_SYSTEM),
                #     prompt,
                #     temperature=0.0,
                #     max_tokens=16,
                # ))
                extra_body = None
                if (
                    getattr(self._chat, "backend", "") == "openai"
                    and "qwen" in str(getattr(self._chat, "model", "")).lower()
                ):
                    extra_body = {
                        "chat_template_kwargs": {
                            "enable_thinking": False,
                        }
                    }

                raw_status = self._chat.chat(
                    self._system(_STATUS_SYSTEM),
                    prompt,
                    temperature=0.0,
                    max_tokens=16,
                    extra_body=extra_body,
                )

                if not raw_status.strip():
                    raise ValueError(
                        "OfficeBench per-check verifier returned empty response"
                    )

                status = self._parse_status(raw_status)
                if status == "MET":
                    met += 1
                elif status == "UNMET":
                    unmet += 1
                else:
                    unknown += 1
        finally:
            if seedable:
                self._chat.seed = original_seed

        unknown_credit = min(
            1.0,
            max(0.0, _float_env(
                "TTS_OFFICEBENCH_VERIFIER_UNKNOWN_CREDIT", 0.25)),
        )
        self.last_percheck_breakdown = {
            "met": met,
            "unmet": unmet,
            "unknown": unknown,
            "total": len(checks),
            "unknown_credit": unknown_credit,
        }
        return (met + unknown_credit * unknown) / len(checks)

    @staticmethod
    def _parse_status(response: str) -> str:
        text = (response or "").strip().upper()
        tail = text.rsplit("STATUS:", 1)[-1] if "STATUS:" in text else text
        if "UNMET" in tail:
            return "UNMET"
        if "UNKNOWN" in tail:
            return "UNKNOWN"
        if "MET" in tail:
            return "MET"
        return "UNKNOWN"

    def _build_prompt(self, instruction: str, actions: List[Dict], observations: List[str],
                      config: Optional[Dict], final_state: Optional[Dict] = None,
                      final_answer: Optional[str] = None, gold_files: Optional[Dict] = None,
                      status_check: Optional[Dict] = None) -> str:
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
        if status_check is None:
            lines.append(
                "\nUsing the PREDICTED resulting file state (not the agent's claims), did the agent "
                "correctly and completely accomplish the task? Answer with exactly one word: SUCCESS "
                "or FAILURE.")
        else:
            lines.append(
                "\nONE ATOMIC GROUND-TRUTH CHECK:\n"
                f"{json.dumps(status_check, ensure_ascii=True, sort_keys=True)}\n\n"
                "Classify only this check as MET when the evidence proves it passes, UNMET when the "
                "evidence proves it fails, or UNKNOWN when the predicted state/write log is insufficient. "
                "Do not fail it because of unrelated task requirements. End with exactly one line: "
                "STATUS: MET, STATUS: UNMET, or STATUS: UNKNOWN.")
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

_STATUS_SYSTEM = (
    "You are a strict evaluator for one atomic OfficeBench ground-truth check. Use the predicted "
    "resulting state, trajectory, and gold evidence. Classify only the supplied check as MET, UNMET, "
    "or UNKNOWN. Never infer MET from the agent's claim alone, and never mark unrelated omissions "
    "against this check. Reply with exactly `STATUS: MET`, `STATUS: UNMET`, or `STATUS: UNKNOWN`."
)


def _is_success(verdict: str) -> bool:
    v = (verdict or "").strip().lower()
    return v.startswith("success") or v == "pass" or v.startswith("yes")


def state_grounding_enabled() -> bool:
    """Whether to read + show the verifier the ACTUAL resulting file state (default ON)."""
    return os.environ.get("TTS_OFFICEBENCH_VERIFIER_STATE", "1") not in ("0", "false", "False")


__all__ = ["OfficeBenchVerifier", "state_files_from_config", "reference_files_from_config",
           "state_grounding_enabled"]
