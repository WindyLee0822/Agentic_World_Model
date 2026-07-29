"""Restricted Python-only tool protocol for tau2 world-model predictions."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple


_PYTHON_PREFIX = "PYTHON:"
_PYTHON_READ_PREFIX = "PYTHON_READ:"
_PYTHON_FINAL_PREFIX = "PYTHON_FINAL:"


def _env_enabled(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value not in ("0", "false", "False")


def _env_int(name: str, default: int) -> int:
    try:
        return max(1, int(os.environ.get(name, str(default))))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return max(0.05, float(os.environ.get(name, str(default))))
    except (TypeError, ValueError):
        return default


def tau2_wm_tool_mode() -> str:
    """Return the tau2 WM tool protocol; restricted Python is the default."""
    value = str(os.environ.get("TTS_TAU2_WM_TOOL_MODE", "python")).strip().lower()
    return "text" if value in ("text", "legacy", "read") else "python"


def prepare_tau2_python_tool() -> bool:
    """Start the shared restricted worker before a tau2 rollout needs it."""
    if tau2_wm_tool_mode() != "python":
        return False
    if not _env_enabled("TTS_TAU2_WM_PYTHON", True):
        return False
    try:
        from ..llm.python_tool import shared_python_tool

        return shared_python_tool().prepare()
    except Exception:
        return False


def _extract_prefixed_python(text: str, prefix: str) -> Optional[str]:
    stripped = (text or "").lstrip()
    if not stripped.startswith(prefix):
        return None
    code = stripped[len(prefix):].strip()
    if code.startswith("```"):
        first_line, _, remainder = code.partition("\n")
        if first_line.strip().lower() in ("```", "```python", "```py"):
            code = remainder
        if code.rstrip().endswith("```"):
            code = code.rstrip()[:-3]
    return code.strip()


def _extract_python_program(text: str) -> Optional[str]:
    code = _extract_prefixed_python(text, _PYTHON_FINAL_PREFIX)
    if code is not None:
        return code
    return _extract_prefixed_python(text, _PYTHON_PREFIX)


def _extract_python_action(text: str) -> Tuple[Optional[str], Optional[str]]:
    code = _extract_prefixed_python(text, _PYTHON_READ_PREFIX)
    if code is not None:
        return "read", code
    code = _extract_python_program(text)
    if code is not None:
        return "final", code
    return None, None


def _contains_legacy_text_tool(text: str) -> bool:
    return any(
        line.strip().upper().startswith(("READ ", "LIST ", "CALL ", "TRACE "))
        for line in (text or "").splitlines()
    )


def _json_size_hint(value: Any, stop_after: int) -> int:
    try:
        total = 0
        encoder = json.JSONEncoder(ensure_ascii=True, sort_keys=True, default=str)
        for chunk in encoder.iterencode(value):
            total += len(chunk)
            if total > stop_after:
                break
        return total
    except Exception:
        return stop_after + 1


def _bounded_json_text(value: Any, limit: int) -> str:
    try:
        text = json.dumps(value, ensure_ascii=True, sort_keys=True, default=str)
    except Exception:
        text = repr(value)
    if len(text) <= limit:
        return text
    return text[:limit] + f"... [truncated {len(text) - limit} chars]"


def _parse_json(text: str) -> Optional[Any]:
    try:
        return json.loads((text or "").strip())
    except Exception:
        match = re.search(r"(\{.*\}|\[.*\])", text or "", re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except Exception:
                return None
    return None


def _normalize_output(output: str, output_kind: str) -> Optional[str]:
    if output_kind == "text":
        text = (output or "").strip()
        if text.upper().startswith("PREDICT:"):
            text = text[len("PREDICT:"):].strip()
        return text or None

    direct = (output or "").strip().upper()
    if direct == "SUCCESS":
        return "SUCCESS"
    if direct in ("FAIL", "FAILURE"):
        return "FAILURE"
    for line in reversed((output or "").splitlines()):
        match = re.fullmatch(
            r"VERDICT\s*[:=]\s*(SUCCESS|FAIL|FAILURE)", line.strip().upper())
        if match:
            return "SUCCESS" if match.group(1) == "SUCCESS" else "FAILURE"
    payload = None
    for line in reversed((output or "").splitlines()):
        try:
            candidate = json.loads(line.strip())
        except Exception:
            continue
        if isinstance(candidate, dict):
            payload = candidate
            break
    if payload is None:
        payload = _parse_json(output)
    if not isinstance(payload, dict):
        return None
    verdict = str(payload.get("verdict", "")).upper()
    if verdict == "SUCCESS":
        return "SUCCESS"
    if verdict in ("FAIL", "FAILURE"):
        return "FAILURE"
    success = payload.get("success")
    if isinstance(success, bool):
        return "SUCCESS" if success else "FAILURE"
    return None


def _python_protocol(
    context: Dict[str, Any],
    *,
    required: bool,
    context_chars: int,
    output_kind: str,
    allow_reads: bool,
) -> str:
    keys = ", ".join(sorted(map(str, context))) or "(none)"
    routing = (
        f"The copied structured input is larger than the direct-output reliability threshold "
        f"(at least {context_chars} characters). You MUST use PYTHON_FINAL for the final prediction."
        if required else
        "Return semantic predictions directly. Use PYTHON_FINAL only when exact filtering, copying, "
        "counting, comparison, or transformation is genuinely needed."
    )
    if output_kind == "verdict":
        output_instruction = (
            'End by calling `print_json({"verdict": "SUCCESS"})` or '
            '`print_json({"verdict": "FAILURE"})`; alternatively print exactly '
            "`VERDICT: SUCCESS` or `VERDICT: FAILURE` as the final line."
        )
    else:
        output_instruction = (
            "Print the complete tool result. Use `print(value)` for arbitrary text, or "
            "`print_json(value)` for a JSON record/list."
        )
    read_instruction = ""
    if allow_reads:
        read_instruction = (
            "\nTo gather environment evidence, start a turn with `PYTHON_READ:` followed by a program "
            "that prints exactly `{\"queries\":[...]}` via `print_json`. Each query must be one of:\n"
            '  {"op":"read","entity_type":"<type>","entity_id":"<id>"}\n'
            '  {"op":"list","entity_type":"<type>"}\n'
            '  {"op":"call","name":"<read-only tool>","args":{...}}\n'
            "The host validates record types and tool names, executes only read-only operations, and "
            "copies results into `data['reads']` and `data['facts']`. The program never receives the "
            "tau2 environment, database, toolkit, network, files, or process environment.\n"
        )
    final_routing = (
        "For the final answer, make the FIRST non-whitespace characters of your response exactly "
        "`PYTHON_FINAL:` followed by a Python program."
        if required else
        "You may return the final result directly. If you use the helper for the final result, make "
        "the FIRST non-whitespace characters exactly `PYTHON_FINAL:` followed by a Python program."
    )
    return (
        "\n\nPURE COMPUTATION OPTION:\n"
        f"{routing}{read_instruction}{final_routing} The program receives "
        f"a copied JSON-like dictionary named `data` with inputs under: {keys}. It may use ordinary "
        "list/dict/set operations, `json`/`datetime`, date helpers, and `print_json`; files, network, "
        "environment access, subprocesses, and tau2 APIs are unavailable. "
        f"{output_instruction} Print only the complete requested result. The legacy `PYTHON:` prefix "
        "is accepted as an alias for `PYTHON_FINAL:`."
    )


@dataclass
class Tau2PythonLoopResult:
    output: str = ""
    valid: bool = False
    python_calls: int = 0
    python_failures: int = 0
    python_forced: int = 0
    python_mode_misses: int = 0


def _run_python(
    code: str,
    context: Dict[str, Any],
    stats: Tau2PythonLoopResult,
    *,
    allow_bare_container_output: bool = False,
):
    from ..llm.python_tool import shared_python_tool

    stats.python_calls += 1
    result = shared_python_tool().run(
        code,
        context,
        timeout_s=_env_float("TTS_TAU2_WM_PYTHON_TIMEOUT_S", 1.0),
        max_input_chars=_env_int("TTS_TAU2_WM_PYTHON_MAX_INPUT_CHARS", 200_000),
        max_output_chars=_env_int("TTS_TAU2_WM_PYTHON_MAX_OUTPUT_CHARS", 20_000),
        max_code_chars=_env_int("TTS_TAU2_WM_PYTHON_MAX_CODE_CHARS", 12_000),
        max_nodes=_env_int("TTS_TAU2_WM_PYTHON_MAX_AST_NODES", 2_000),
        max_steps=_env_int("TTS_TAU2_WM_PYTHON_MAX_STEPS", 100_000),
        allow_bare_container_output=allow_bare_container_output,
    )
    if result.error:
        stats.python_failures += 1
    return result


def _parse_read_requests(
    output: str,
    *,
    entity_types: set[str],
    read_tools: set[str],
) -> Optional[List[Dict[str, Any]]]:
    payload = _parse_json(output)
    if not isinstance(payload, dict) or not isinstance(payload.get("queries"), list):
        return None
    parsed = []
    for query in payload["queries"]:
        if not isinstance(query, dict):
            continue
        op = str(query.get("op", "")).strip().lower()
        if op == "read":
            entity_type = str(query.get("entity_type", "")).strip()
            entity_id = str(query.get("entity_id", "")).strip()
            if entity_type in entity_types and entity_id:
                parsed.append({
                    "op": op,
                    "entity_type": entity_type,
                    "entity_id": entity_id[:500],
                })
        elif op == "list":
            entity_type = str(query.get("entity_type", "")).strip()
            if entity_type in entity_types:
                parsed.append({"op": op, "entity_type": entity_type})
        elif op == "call":
            name = str(query.get("name", "")).strip()
            args = query.get("args") or {}
            if name in read_tools and isinstance(args, dict):
                parsed.append({"op": op, "name": name, "args": args})
    return parsed or None


def _json_like(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        if isinstance(value, str):
            parsed = _parse_json(value)
            return parsed if parsed is not None else value
        return value
    if isinstance(value, dict):
        if "model" in value:
            return _json_like(value["model"])
        return {str(key): _json_like(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_like(item) for item in value]
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        try:
            return _json_like(dump())
        except Exception:
            pass
    return str(value)


def _bounded_read_value(value: Any) -> Any:
    copied = _json_like(value)
    limit = _env_int("TTS_TAU2_WM_PYTHON_READ_RESULT_CHARS", 8_000)
    try:
        text = json.dumps(copied, ensure_ascii=True, sort_keys=True, default=str)
    except Exception:
        text = str(copied)
    if len(text) <= limit:
        return copied
    return text[:limit] + f"... [truncated {len(text) - limit} chars]"


def _execute_read(query_interface, request: Dict[str, Any], label: str) -> Any:
    op = request["op"]
    if op == "read":
        ref = (request["entity_type"], request["entity_id"])
        rows = query_interface.query([ref], label)
        return _bounded_read_value(rows.get(ref))
    if op == "list":
        return _bounded_read_value(
            list(query_interface.list_ids(request["entity_type"]))[:60])
    return _bounded_read_value(query_interface.call(request["name"], request["args"]))


def _resolve_final(
    response: str,
    *,
    complete: Callable[[str, int], str],
    prompt: str,
    context: Dict[str, Any],
    max_tokens: int,
    required: bool,
    output_kind: str,
    stats: Tau2PythonLoopResult,
) -> str:
    def recover_direct(error: str) -> str:
        recovered = complete(
            f"{prompt}\n\nThe restricted Python attempt could not produce a valid result:\n"
            f"{error[:1_000]}\n\nReturn ONLY the final result directly in the requested format. "
            "Do not emit PYTHON_READ, PYTHON_FINAL, PYTHON, READ, LIST, CALL, or TRACE, and do "
            "not explain the failure.",
            max_tokens,
        )
        mode, _ = _extract_python_action(recovered)
        if mode is None and not _contains_legacy_text_tool(recovered):
            normalized = _normalize_output(recovered, output_kind)
            if normalized is not None:
                stats.valid = True
                return normalized
        stats.valid = True
        return (
            "FAILURE"
            if output_kind == "verdict"
            else "ERROR: world-model prediction could not produce a valid tool result"
        )

    code = _extract_python_program(response)
    if code is None:
        if _extract_prefixed_python(response, _PYTHON_READ_PREFIX) is not None:
            return recover_direct("a final response cannot contain PYTHON_READ")
        if not required:
            normalized = _normalize_output(response, output_kind)
            if normalized is not None:
                stats.valid = True
                return normalized
            return recover_direct(f"direct output is not a valid {output_kind} result")
        stats.python_mode_misses += 1
        response = complete(
            f"{prompt}\n\nYour previous response incorrectly used direct output. Return ONLY a "
            "PYTHON_FINAL program that computes and prints the complete result.",
            max_tokens,
        )
        code = _extract_python_program(response)
        if code is None:
            normalized = _normalize_output(response, output_kind)
            if normalized is not None:
                stats.valid = True
                return normalized
            return recover_direct("the required PYTHON_FINAL retry was invalid")

    result = _run_python(code, context, stats)
    normalized = None if result.error else _normalize_output(result.output, output_kind)
    error = result.error
    if error is None and normalized is None:
        error = f"program output is not a valid {output_kind} result"
        stats.python_failures += 1
    if error is None:
        stats.valid = True
        return normalized or ""
    if not _env_enabled("TTS_TAU2_WM_PYTHON_REPAIR", True):
        return recover_direct(error)

    repair_context = _bounded_json_text(
        context,
        _env_int("TTS_TAU2_WM_PYTHON_REPAIR_CONTEXT_CHARS", 12_000),
    )
    if result.fatal:
        repair_instruction = (
            "The Python helper is unavailable. Return the prediction directly in the requested format."
        )
    elif required:
        repair_instruction = "Return one corrected PYTHON_FINAL program; direct output is not allowed."
    else:
        repair_instruction = (
            "Return either the direct prediction or one corrected PYTHON_FINAL program."
        )
    repaired = complete(
        f"{prompt}\n\nYour previous PYTHON program failed safely with:\n{error}\n\n"
        f"FAILED PROGRAM:\n{code[:4_000]}\n\nCOPIED INPUT (bounded):\n{repair_context}\n\n"
        f"{repair_instruction} Do not explain the failure.",
        max_tokens,
    )
    repaired_code = _extract_python_program(repaired)
    if repaired_code is None:
        normalized = _normalize_output(repaired, output_kind)
        if normalized is not None:
            stats.valid = True
            return normalized
        return recover_direct("the Python repair response was invalid")
    repaired_result = _run_python(repaired_code, context, stats)
    if repaired_result.error:
        return recover_direct(repaired_result.error)
    normalized = _normalize_output(repaired_result.output, output_kind)
    if normalized is None:
        stats.python_failures += 1
        return recover_direct(f"repaired program output is not a valid {output_kind} result")
    stats.valid = True
    return normalized


def run_tau2_python_prediction_loop(
    complete: Callable[[str, int], str],
    query_interface,
    header: str,
    final_instr: str,
    *,
    context: Dict[str, Any],
    max_turns: int = 6,
    max_tokens: int = 1500,
    output_kind: str = "text",
    force_long: bool = False,
) -> Tau2PythonLoopResult:
    """Run bounded Python-emitted reads followed by a direct or Python final result."""
    stats = Tau2PythonLoopResult()
    if not _env_enabled("TTS_TAU2_WM_PYTHON", True):
        direct_prompt = (
            f"{header}\n\nThe restricted Python helper is disabled. Return the final result directly; "
            "do not emit PYTHON_READ, PYTHON_FINAL, READ, LIST, CALL, or TRACE.\n\n"
            f"FINAL OUTPUT FORMAT:\n{final_instr}"
        )
        response = complete(direct_prompt, max_tokens)
        mode, _ = _extract_python_action(response)
        if mode is not None or _contains_legacy_text_tool(response):
            stats.python_mode_misses += 1
            response = complete(
                f"{direct_prompt}\n\nYour previous response used a disabled tool protocol. "
                "Return only the direct final result.",
                max_tokens,
            )
        normalized = _normalize_output(response, output_kind)
        if normalized is not None:
            stats.output = normalized
            stats.valid = True
        else:
            stats.output = (
                "FAILURE"
                if output_kind == "verdict"
                else "ERROR: world-model prediction could not produce a valid tool result"
            )
            stats.valid = True
        return stats

    threshold = _env_int("TTS_TAU2_WM_PYTHON_FORCE_CHARS", 2_000)
    context_chars = _json_size_hint(context, threshold)
    required = bool(force_long and context_chars > threshold)
    if required:
        stats.python_forced = 1

    protocol = _python_protocol(
        context,
        required=required,
        context_chars=context_chars,
        output_kind=output_kind,
        allow_reads=query_interface is not None,
    )
    model_prompt = f"{header}{protocol}\n\nFINAL OUTPUT FORMAT:\n{final_instr}"
    reads = list(context.get("reads") or [])
    base_facts = context.get("facts")
    facts = dict(base_facts) if isinstance(base_facts, dict) else {}

    def current_context() -> Dict[str, Any]:
        return {**context, "reads": list(reads), "facts": dict(facts)}

    turns = max(1, _env_int("TTS_TAU2_WM_PYTHON_READ_TURNS", max_turns))
    max_reads = _env_int("TTS_TAU2_WM_PYTHON_READS_PER_TURN", 4)
    visible_limit = _env_int("TTS_TAU2_WM_PYTHON_READ_CONTEXT_CHARS", 12_000)
    entity_types = {str(value) for value in context.get("entity_types") or []}
    read_tools = {str(value) for value in context.get("read_tools") or []}

    for turn in range(turns):
        response = complete(
            f"{model_prompt}\n\nCOPIED READ RESULTS SO FAR:\n"
            f"{_bounded_json_text(reads, visible_limit)}\n\n"
            f"TOOL FACTS AND ERRORS:\n{_bounded_json_text(facts, visible_limit)}\n\n"
            f"Python tool turn {turn + 1}/{turns}:",
            max_tokens,
        )
        mode, code = _extract_python_action(response)
        if mode == "read" and code is not None:
            result = _run_python(
                code, current_context(), stats, allow_bare_container_output=True)
            if result.error:
                facts[f"python_read_error_{turn + 1}"] = result.error
                if result.fatal:
                    break
                continue
            requests = _parse_read_requests(
                result.output, entity_types=entity_types, read_tools=read_tools)
            if requests is None:
                stats.python_failures += 1
                facts[f"python_read_error_{turn + 1}"] = (
                    "PYTHON_READ output must contain a non-empty list of valid read-only queries.")
                continue
            for index, request in enumerate(requests[:max_reads]):
                try:
                    read_result = _execute_read(
                        query_interface, request, f"python{turn}:{index}")
                except Exception as exc:
                    read_result = f"ERROR: {type(exc).__name__}: {str(exc)[:300]}"
                record = {**request, "result": read_result}
                reads.append(record)
                target = request.get(
                    "entity_type", request.get("name", "unknown"))
                facts[f"{request['op']} {target}"] = read_result
            continue
        if mode == "final":
            stats.output = _resolve_final(
                response,
                complete=complete,
                prompt=model_prompt,
                context=current_context(),
                max_tokens=max_tokens,
                required=required,
                output_kind=output_kind,
                stats=stats,
            )
            return stats
        if _contains_legacy_text_tool(response):
            facts[f"python_tool_error_{turn + 1}"] = (
                "READ/LIST/CALL/TRACE are disabled in Python mode. Use PYTHON_READ.")
            continue
        stats.output = _resolve_final(
            response,
            complete=complete,
            prompt=model_prompt,
            context=current_context(),
            max_tokens=max_tokens,
            required=required,
            output_kind=output_kind,
            stats=stats,
        )
        return stats

    response = complete(
        f"{model_prompt}\n\nYou are out of read turns. Return the final result directly or with "
        "PYTHON_FINAL; do not request more reads.",
        max_tokens,
    )
    mode, _ = _extract_python_action(response)
    if mode != "read" and not _contains_legacy_text_tool(response):
        stats.output = _resolve_final(
            response,
            complete=complete,
            prompt=model_prompt,
            context=current_context(),
            max_tokens=max_tokens,
            required=required,
            output_kind=output_kind,
            stats=stats,
        )
    if not stats.valid:
        stats.output = (
            "FAILURE"
            if output_kind == "verdict"
            else "ERROR: world-model prediction could not produce a valid tool result"
        )
        stats.valid = True
    return stats


__all__ = [
    "Tau2PythonLoopResult",
    "prepare_tau2_python_tool",
    "run_tau2_python_prediction_loop",
    "tau2_wm_tool_mode",
]
