"""Qwen-specific tau2 agent prompt and tool-call recovery.

The stock tau2 prompt is intentionally model-neutral. Qwen benefits from a
more explicit state-machine contract for long, policy-heavy customer-service
tasks, especially when native tool calling is used through vLLM.
"""

from __future__ import annotations

import os
import threading
from copy import deepcopy
from json import JSONDecodeError
from typing import Any, Dict

QWEN_TAU2_AGENT_NAME = "tts_qwen_agent"

QWEN_TAU2_AGENT_INSTRUCTION = """
You are a customer service agent. Resolve every part of the user's request
according to the policy below.

Before each response, reason silently over the policy, the user's stated
constraints, completed subgoals, and the latest tool results.

Follow this operating procedure:
1. Maintain a checklist of every currently confirmed subgoal. The user's latest
   intent supersedes earlier statements. Do not turn alternatives, conditional
   preferences, or superseded requests into extra actions.
2. Do not stop after gathering information. After a read, either perform the
   next policy-allowed action, ask for specifically missing information, or
   explain why the remaining request cannot be completed.
3. Before a state-changing tool call, verify all policy prerequisites from
   tool results. Never guess identifiers, eligibility, prices, payment methods,
   dates, or current status.
4. If the requested action, item, or conditional preference is ambiguous, ask
   one focused clarification before writing. Ask for required confirmation when
   the policy requires it, and do not ask again after the user has supplied it.
5. After a state-changing tool succeeds, mark that subgoal complete. Never
   repeat the same write or restart the workflow unless the tool returned an
   error showing that the write did not happen.
6. Reuse facts already returned by tools. Do not repeat reads unless a prior
   write may have changed the value or the earlier result was incomplete.
7. For requests with several items, reservations, orders, or modifications,
   handle each one exactly once and preserve the user's requested ordering and
   conditional preferences.
8. If the request is unsupported, use the policy's fallback or transfer path;
   do not invent a tool action.
9. Before finishing, accurately communicate requested prices, refunds,
   durations, limitations, and the outcome of each confirmed subgoal. A
   successful tool call alone does not satisfy a request for information.

On each turn, output exactly one of:
- a concise natural-language message to the user; or
- one or more structured tool calls when they are independent.

Use the provided tool interface for tool calls. Supply valid JSON arguments,
but never print JSON, XML, or tool-call syntax as ordinary message text. Do not
mix user-facing text with tool calls in the same turn.
""".strip()

_REGISTER_LOCK = threading.Lock()


def is_qwen_model(model: str | None) -> bool:
    return "qwen" in (model or "").lower()


def qwen_agent_profile(model: str | None) -> str:
    profile = os.environ.get("TTS_TAU2_QWEN_AGENT_PROFILE", "off").strip().lower()
    if profile in {"0", "false", "no", "off", "baseline", "auto"}:
        return "off"
    if profile in {"1", "true", "yes", "on", "policy"}:
        return "policy" if is_qwen_model(model) else "off"
    if profile == "prompt":
        return "prompt" if is_qwen_model(model) else "off"
    raise ValueError(
        "TTS_TAU2_QWEN_AGENT_PROFILE must be auto|off|prompt|policy; "
        f"got {profile!r}"
    )


def qwen_agent_profile_enabled(model: str | None) -> bool:
    return qwen_agent_profile(model) != "off"


def qwen_agent_thinking_enabled(model: str | None) -> bool:
    return qwen_agent_profile(model) == "policy"


def qwen_chat_template_args(
    llm_args: Dict[str, Any] | None,
    *,
    enable_thinking: bool,
) -> Dict[str, Any]:
    """Return request args with a per-call Qwen thinking override."""
    result = deepcopy(llm_args or {})
    extra_body = dict(result.get("extra_body") or {})
    chat_template_kwargs = dict(extra_body.get("chat_template_kwargs") or {})
    chat_template_kwargs["enable_thinking"] = enable_thinking
    extra_body["chat_template_kwargs"] = chat_template_kwargs
    result["extra_body"] = extra_body
    return result


def _is_empty_assistant_error(error: Exception) -> bool:
    message = str(error).lower()
    return (
        "must have either content or tool_calls" in message
        or "must have content or tool calls" in message
        or "has_content_or_tool_calls" in message
    )


def _is_recoverable_qwen_response_error(error: Exception) -> bool:
    return isinstance(error, JSONDecodeError) or _is_empty_assistant_error(error)


def ensure_qwen_tau2_agent_registered() -> str:
    """Register the Qwen-specific agent lazily and return its tau2 name."""
    from tau2.registry import registry

    if registry.get_agent_factory(QWEN_TAU2_AGENT_NAME) is not None:
        return QWEN_TAU2_AGENT_NAME

    with _REGISTER_LOCK:
        if registry.get_agent_factory(QWEN_TAU2_AGENT_NAME) is not None:
            return QWEN_TAU2_AGENT_NAME

        from loguru import logger
        from tau2.agent.llm_agent import LLMAgent, SYSTEM_PROMPT
        from tau2.data_model.message import MultiToolMessage, UserMessage
        from tau2.utils.llm_utils import generate

        class QwenTau2Agent(LLMAgent):
            @property
            def system_prompt(self) -> str:
                return SYSTEM_PROMPT.format(
                    domain_policy=self.domain_policy,
                    agent_instruction=QWEN_TAU2_AGENT_INSTRUCTION,
                )

            def _generate_next_message(self, message, state):
                if isinstance(message, UserMessage) and message.is_audio:
                    raise ValueError(
                        "User message cannot be audio. Use VoiceLLMAgent instead."
                    )
                if isinstance(message, MultiToolMessage):
                    state.messages.extend(message.tool_messages)
                else:
                    state.messages.append(message)
                messages = state.system_messages + state.messages
                enable_thinking = qwen_agent_thinking_enabled(self.llm)

                def generate_response(enable_thinking: bool, call_name: str):
                    return generate(
                        model=self.llm,
                        tools=self.tools,
                        messages=messages,
                        call_name=call_name,
                        **qwen_chat_template_args(
                            self.llm_args,
                            enable_thinking=enable_thinking,
                        ),
                    )

                try:
                    response = generate_response(enable_thinking, "agent_response")
                    response.validate()
                except Exception as error:
                    if not _is_recoverable_qwen_response_error(error):
                        raise
                    logger.warning(
                        "Qwen returned an empty response or malformed tool arguments; "
                        "retrying this turn with thinking disabled: "
                        f"{type(error).__name__}"
                    )
                    return generate_response(
                        False,
                        "agent_response_no_thinking_fallback",
                    )
                return response

        def create_qwen_tau2_agent(tools, domain_policy, **kwargs):
            return QwenTau2Agent(
                tools=tools,
                domain_policy=domain_policy,
                llm=kwargs.get("llm"),
                llm_args=kwargs.get("llm_args"),
            )

        registry.register_agent_factory(
            create_qwen_tau2_agent,
            QWEN_TAU2_AGENT_NAME,
        )
    return QWEN_TAU2_AGENT_NAME


def tau2_agent_name_for_model(model: str | None) -> str:
    if qwen_agent_profile_enabled(model):
        return ensure_qwen_tau2_agent_registered()
    return "llm_agent"


__all__ = [
    "QWEN_TAU2_AGENT_INSTRUCTION",
    "QWEN_TAU2_AGENT_NAME",
    "ensure_qwen_tau2_agent_registered",
    "is_qwen_model",
    "qwen_agent_profile",
    "qwen_agent_profile_enabled",
    "qwen_agent_thinking_enabled",
    "qwen_chat_template_args",
    "tau2_agent_name_for_model",
]
