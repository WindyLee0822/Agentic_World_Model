from __future__ import annotations

import sys
import types
from json import JSONDecodeError

import pytest

from tts.env.tau2_qwen_agent import (
    QWEN_TAU2_AGENT_INSTRUCTION,
    ensure_qwen_tau2_agent_registered,
    is_qwen_model,
    qwen_agent_profile,
    qwen_agent_profile_enabled,
    qwen_agent_thinking_enabled,
    qwen_chat_template_args,
)


def test_qwen_model_detection_and_profile_gate(monkeypatch):
    assert is_qwen_model("openai/Qwen3.5-35B-A3B")
    assert is_qwen_model("Qwen-AgentWorld-35B-A3B")
    assert not is_qwen_model("azure/gpt-5.4")

    monkeypatch.setenv("TTS_TAU2_QWEN_AGENT_PROFILE", "auto")
    assert not qwen_agent_profile_enabled("openai/Qwen3.5-35B-A3B")
    assert not qwen_agent_profile_enabled("azure/gpt-5.4")

    monkeypatch.setenv("TTS_TAU2_QWEN_AGENT_PROFILE", "off")
    assert not qwen_agent_profile_enabled("openai/Qwen3.5-35B-A3B")

    monkeypatch.setenv("TTS_TAU2_QWEN_AGENT_PROFILE", "prompt")
    assert qwen_agent_profile("openai/Qwen3.5-35B-A3B") == "prompt"
    assert not qwen_agent_thinking_enabled("openai/Qwen3.5-35B-A3B")

    monkeypatch.setenv("TTS_TAU2_QWEN_AGENT_PROFILE", "policy")
    assert qwen_agent_profile("openai/Qwen3.5-35B-A3B") == "policy"
    assert qwen_agent_thinking_enabled("openai/Qwen3.5-35B-A3B")
    assert qwen_agent_profile("azure/gpt-5.4") == "off"

    monkeypatch.setenv("TTS_TAU2_QWEN_AGENT_PROFILE", "invalid")
    with pytest.raises(ValueError):
        qwen_agent_profile_enabled("openai/Qwen3.5-35B-A3B")


def test_qwen_chat_template_args_preserve_existing_extra_body():
    original = {
        "seed": 7,
        "extra_body": {
            "priority": 3,
            "chat_template_kwargs": {"custom": "value"},
        },
    }
    result = qwen_chat_template_args(original, enable_thinking=True)

    assert result["seed"] == 7
    assert result["extra_body"]["priority"] == 3
    assert result["extra_body"]["chat_template_kwargs"] == {
        "custom": "value",
        "enable_thinking": True,
    }
    assert original["extra_body"]["chat_template_kwargs"] == {"custom": "value"}


def test_qwen_tau2_prompt_targets_observed_failure_modes():
    prompt = " ".join(QWEN_TAU2_AGENT_INSTRUCTION.lower().split())

    assert "every currently confirmed subgoal" in prompt
    assert "latest intent supersedes" in prompt
    assert "repeat the same write" in prompt
    assert "do not stop after gathering information" in prompt
    assert "never guess identifiers" in prompt
    assert "ask one focused clarification" in prompt
    assert "successful tool call alone" in prompt
    assert "structured tool calls" in prompt
    assert "do not mix user-facing text with tool calls" in prompt


def test_qwen_agent_retries_empty_reasoning_only_response(monkeypatch):
    calls = []
    monkeypatch.setenv("TTS_TAU2_QWEN_AGENT_PROFILE", "policy")

    class FakeRegistry:
        def __init__(self):
            self.factories = {}

        def get_agent_factory(self, name):
            return self.factories.get(name)

        def register_agent_factory(self, factory, name):
            self.factories[name] = factory

    class FakeLLMAgent:
        def __init__(self, tools, domain_policy, llm, llm_args):
            self.tools = tools
            self.domain_policy = domain_policy
            self.llm = llm
            self.llm_args = llm_args

    class FakeUserMessage:
        is_audio = False

    class FakeMultiToolMessage:
        pass

    class FakeAssistantMessage:
        def __init__(self, valid):
            self.valid = valid

        def validate(self):
            if not self.valid:
                raise ValueError(
                    "AssistantMessage must have either content or tool_calls"
                )

    def fake_generate(**kwargs):
        calls.append(kwargs)
        return FakeAssistantMessage(valid=len(calls) > 1)

    registry = FakeRegistry()
    fake_modules = {
        "loguru": types.SimpleNamespace(
            logger=types.SimpleNamespace(warning=lambda *_args, **_kwargs: None)
        ),
        "tau2": types.ModuleType("tau2"),
        "tau2.registry": types.SimpleNamespace(registry=registry),
        "tau2.agent": types.ModuleType("tau2.agent"),
        "tau2.agent.llm_agent": types.SimpleNamespace(
            LLMAgent=FakeLLMAgent,
            SYSTEM_PROMPT="{domain_policy}\n{agent_instruction}",
        ),
        "tau2.data_model": types.ModuleType("tau2.data_model"),
        "tau2.data_model.message": types.SimpleNamespace(
            MultiToolMessage=FakeMultiToolMessage,
            UserMessage=FakeUserMessage,
        ),
        "tau2.utils": types.ModuleType("tau2.utils"),
        "tau2.utils.llm_utils": types.SimpleNamespace(generate=fake_generate),
    }
    for name, module in fake_modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    name = ensure_qwen_tau2_agent_registered()
    agent = registry.get_agent_factory(name)(
        tools=[],
        domain_policy="policy",
        llm="openai/Qwen3.5-35B-A3B",
        llm_args={"temperature": 1},
    )
    state = types.SimpleNamespace(system_messages=[], messages=[])

    response = agent._generate_next_message(FakeUserMessage(), state)

    assert response.valid
    assert len(calls) == 2
    assert calls[0]["extra_body"]["chat_template_kwargs"]["enable_thinking"] is True
    assert calls[1]["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False


def test_qwen_agent_retries_malformed_tool_arguments(monkeypatch):
    calls = []
    monkeypatch.setenv("TTS_TAU2_QWEN_AGENT_PROFILE", "prompt")

    class FakeRegistry:
        def __init__(self):
            self.factory = None

        def get_agent_factory(self, _name):
            return self.factory

        def register_agent_factory(self, factory, _name):
            self.factory = factory

    class FakeLLMAgent:
        def __init__(self, tools, domain_policy, llm, llm_args):
            self.tools = tools
            self.domain_policy = domain_policy
            self.llm = llm
            self.llm_args = llm_args

    class FakeUserMessage:
        is_audio = False

    class FakeMultiToolMessage:
        pass

    class FakeAssistantMessage:
        def validate(self):
            return None

    def fake_generate(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise JSONDecodeError("Expecting value", "", 0)
        return FakeAssistantMessage()

    registry = FakeRegistry()
    fake_modules = {
        "loguru": types.SimpleNamespace(
            logger=types.SimpleNamespace(warning=lambda *_args, **_kwargs: None)
        ),
        "tau2": types.ModuleType("tau2"),
        "tau2.registry": types.SimpleNamespace(registry=registry),
        "tau2.agent": types.ModuleType("tau2.agent"),
        "tau2.agent.llm_agent": types.SimpleNamespace(
            LLMAgent=FakeLLMAgent,
            SYSTEM_PROMPT="{domain_policy}\n{agent_instruction}",
        ),
        "tau2.data_model": types.ModuleType("tau2.data_model"),
        "tau2.data_model.message": types.SimpleNamespace(
            MultiToolMessage=FakeMultiToolMessage,
            UserMessage=FakeUserMessage,
        ),
        "tau2.utils": types.ModuleType("tau2.utils"),
        "tau2.utils.llm_utils": types.SimpleNamespace(generate=fake_generate),
    }
    for name, module in fake_modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    name = ensure_qwen_tau2_agent_registered()
    agent = registry.get_agent_factory(name)(
        tools=[],
        domain_policy="policy",
        llm="openai/Qwen3.5-35B-A3B",
        llm_args={"temperature": 1},
    )
    state = types.SimpleNamespace(system_messages=[], messages=[])

    agent._generate_next_message(FakeUserMessage(), state)

    assert len(calls) == 2
    assert calls[0]["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False
    assert calls[1]["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False
