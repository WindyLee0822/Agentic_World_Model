"""Qwen-AgentWorld client and native trajectory chat formatting.

This module intentionally uses its own OpenAI-compatible endpoint.
It must NOT modify the global local-LLM configuration used by the
Qwen3.5 benchmark agent.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Optional

from openai import OpenAI


_THINK_RE = re.compile(
    r"<think>.*?</think>",
    re.IGNORECASE | re.DOTALL,
)

_PREDICTED_OBS_RE = re.compile(
    r"<predicted_observation>\s*(.*?)\s*</predicted_observation>",
    re.IGNORECASE | re.DOTALL,
)

_ENV_OBS_RE = re.compile(
    r"\*\*Environment Observation:\*\*\s*(.*)",
    re.IGNORECASE | re.DOTALL,
)


def build_agentworld_system_prompt(
    tool_docs: Dict[str, str],
    *,
    template_file: Optional[str] = None,
) -> str:
    """Fill Qwen-AgentWorld's official MCP prompt with τ² tool definitions."""

    tool_definitions = "\n\n".join(
        f"### {name}\n{doc}"
        for name, doc in sorted((tool_docs or {}).items())
    )

    if not template_file:
        raise ValueError(
            "AgentWorld requires the official MCP system prompt file."
        )

    text = Path(template_file).read_text(encoding="utf-8")

    if "{tool_definitions}" not in text:
        raise ValueError(
            f"{template_file} does not contain {{tool_definitions}}"
        )

    text = text.replace(
        "{tool_definitions}",
        tool_definitions,
    )

    # τ² does not provide AgentWorld-format demonstrations.
    text = text.replace(
        "{demonstrations}",
        "",
    )

    return text.strip()


def format_agentworld_action(
    turn: int,
    name: str,
    arguments: dict,
) -> str:
    """Format one tool action in AgentWorldBench MCP style."""

    payload = {
        "name": str(name),
        "arguments": dict(arguments or {}),
    }

    return (
        f"### Turn {int(turn)}\n"
        "**Action:**\n"
        "```json\n"
        f"{json.dumps(payload, ensure_ascii=False, sort_keys=True)}\n"
        "```"
    )


def extract_agentworld_observation(raw: str) -> str:
    """Remove reasoning/format markers before exposing result to τ² agent."""

    text = _THINK_RE.sub("", raw or "").strip()

    match = _PREDICTED_OBS_RE.search(text)
    if match:
        return match.group(1).strip()

    match = _ENV_OBS_RE.search(text)
    if match:
        return match.group(1).strip()

    return text


class AgentWorldSession:
    """Stateful AgentWorld trajectory.

    Message structure:

        system
        user       action_1
        assistant  observation_1
        user       action_2
        assistant  observation_2
        ...
    """

    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        api_key: str = "EMPTY",
        system_prompt: str,
        temperature: float = 0.6,
        top_p: float = 0.95,
        top_k: int = 20,
        max_tokens: int = 32768,
        max_history_chars: Optional[int] = None,
        context_overflow_retries: int = 0,
        enable_thinking: Optional[bool] = None,
    ):
        self.model = model
        self.base_url = base_url
        self.api_key = api_key

        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.max_tokens = max_tokens
        self.max_history_chars = max_history_chars
        self.context_overflow_retries = max(0, int(context_overflow_retries))
        self.enable_thinking = enable_thinking

        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key or "EMPTY",
        )

        self.messages = []

        if system_prompt:
            self.messages.append(
                {
                    "role": "system",
                    "content": system_prompt,
                }
            )

        self.total_tokens = 0
        self._turn = 0

    def _bounded_messages(self):
        """Return system prompt plus a recent complete interaction suffix."""
        if not self.max_history_chars or self.max_history_chars <= 0:
            return list(self.messages)
        system = [m for m in self.messages if m.get("role") == "system"]
        history = [m for m in self.messages if m.get("role") != "system"]
        kept = []
        used = 0
        # Keep whole user/assistant pairs so the chat template remains valid.
        for index in range(len(history) - 2, -1, -2):
            pair = [dict(message) for message in history[index:index + 2]]
            size = sum(len(str(message.get("content", ""))) for message in pair)
            if not kept and size > self.max_history_chars:
                # A single oversized predicted list must not defeat the bound.
                # The fixed initial snapshot remains in the system prompt, and
                # the action is retained; only its verbose observation is
                # summarized for subsequent transition calls.
                pair[-1]["content"] = (
                    "**Environment Observation:**\n"
                    "[Earlier oversized read response omitted from chat history; "
                    "reconstruct state from the fixed initial snapshot and actions.]"
                )
                size = sum(len(str(message.get("content", ""))) for message in pair)
            if kept and used + size > self.max_history_chars:
                break
            kept[0:0] = pair
            used += size
        return [*system, *kept]

    def set_system_prompt(self, system_prompt: str) -> None:
        """Set the system prompt before the first simulated action."""

        # Once AgentWorld has begun a trajectory, changing the system
        # prompt would invalidate its simulated state.
        if any(
            message.get("role") != "system"
            for message in self.messages
        ):
            raise RuntimeError(
                "cannot change AgentWorld system prompt "
                "after simulation has started"
            )

        self.system_prompt = str(system_prompt or "")

        self.messages = (
            [
                {
                    "role": "system",
                    "content": self.system_prompt,
                }
            ]
            if self.system_prompt
            else []
        )

    def predict_tool(
        self,
        name: str,
        arguments: dict,
        *,
        reference_context: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ):

        self._turn += 1
        turn = self._turn

        action_prompt = format_agentworld_action(
            turn,
            name,
            arguments,
        )
        request_prompt = action_prompt

        if reference_context:

            request_prompt += (
                "\n\n"
                "<static_tool_reference>\n"
                + reference_context
                + "\n</static_tool_reference>"
            )

        # IMPORTANT:
        # Current action is appended temporarily as the last USER message.
        bounded_messages = self._bounded_messages()
        if len(bounded_messages) != len(self.messages):
            self.messages = bounded_messages
        request_messages = [
            *bounded_messages,
            {
                "role": "user",
                "content":
                    request_prompt,
            },
        ]

        requested_max_tokens = (
            self.max_tokens
            if max_tokens is None
            else max(1, int(max_tokens))
        )

        def complete(output_tokens: int):
            return self.client.chat.completions.create(
                model=self.model,
                messages=request_messages,
                max_tokens=output_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                extra_body={
                    "top_k": self.top_k,
                    **({"chat_template_kwargs": {"enable_thinking": self.enable_thinking}}
                       if getattr(self, "enable_thinking", None) is not None else {}),
                },
            )

        # vLLM's overflow diagnostic reports that the prompt contains *at
        # least* N tokens.  That lower bound is determined by the requested
        # output allowance, so a single retry can reveal a larger lower bound
        # and overflow again.  Reduce the allowance monotonically until the
        # real prompt fits.  This changes only transport capacity; the prompt,
        # sampled state, and predicted action result are unchanged.
        output_tokens = requested_max_tokens
        overflow_retries = max(0, int(getattr(self, "context_overflow_retries", 0)))
        for attempt in range(overflow_retries + 1):
            try:
                response = complete(output_tokens)
                break
            except Exception as exc:
                message = str(exc)
                match = re.search(
                    r"maximum context length is\s+(\d+).*?prompt contains at least\s+(\d+)",
                    message,
                    re.IGNORECASE | re.DOTALL,
                )
                if not match or attempt == overflow_retries:
                    raise
                context_tokens, prompt_tokens = map(int, match.groups())
                safe_from_bound = context_tokens - prompt_tokens - 1024
                retry_tokens = min(
                    output_tokens - 512,
                    safe_from_bound,
                )
                # The prompt itself may be close to the window.  Retain a
                # small but usable JSON response budget and let the next vLLM
                # diagnostic expose a tighter prompt lower bound if needed.
                retry_tokens = max(256, retry_tokens)
                if retry_tokens >= output_tokens:
                    retry_tokens = max(256, output_tokens // 2)
                if retry_tokens == output_tokens:
                    raise
                output_tokens = retry_tokens

        usage = getattr(response, "usage", None)

        self.total_tokens += int(
            getattr(usage, "total_tokens", 0) or 0
        )

        raw = (
            response.choices[0].message.content
            or ""
        ).strip()

        observation = extract_agentworld_observation(raw)

        self.messages.append(
            {
                "role": "user",
                "content": action_prompt,
            }
        )

        # IMPORTANT:
        # preserve exactly what AgentWorld generated in its own history.
        self.messages.append(
            {
                "role": "assistant",
                "content": raw,
            }
        )

        return observation

    def observe_tool(
        self,
        name: str,
        arguments: dict,
        observation,
    ) -> None:
        """Insert an actually observed tool interaction into AgentWorld history."""

        turn = (
            1
            + sum(
                1
                for message
                in self.messages
                if (
                    message.get(
                        "role"
                    )
                    == "assistant"
                )
            )
        )

        action_prompt = (
            format_agentworld_action(
                turn,
                name,
                arguments,
            )
        )

        if isinstance(
            observation,
            (dict, list),
        ):

            observation_text = json.dumps(
                observation,
                ensure_ascii=False,
                default=str,
            )
        else:
            observation_text = str(
                observation
            )

        self.messages.append(
            {
                "role": "user",
                "content":
                    action_prompt,
            }
        )

        self.messages.append(
            {
                "role": "assistant",
                "content": observation_text,
            }
        )
