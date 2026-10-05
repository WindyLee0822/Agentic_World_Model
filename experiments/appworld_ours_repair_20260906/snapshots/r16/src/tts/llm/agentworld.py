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
    ):
        self.model = model
        self.base_url = base_url
        self.api_key = api_key

        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.max_tokens = max_tokens

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
    ):

        turn = (
            1
            + sum(
                1
                for message in self.messages
                if message["role"] == "assistant"
            )
        )

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
        request_messages = [
            *self.messages,
            {
                "role": "user",
                "content":
                    request_prompt,
            },
        ]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=request_messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            extra_body={
                "top_k": self.top_k,
            },
        )

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
                for message in self.messages
                if message.get("role")
                == "assistant"
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
                "content": action_prompt,
            }
        )

        self.messages.append(
            {
                "role": "assistant",
                "content": observation_text,
            }
        )