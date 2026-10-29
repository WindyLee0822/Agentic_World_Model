"""Qwen dispatch and isolation from the existing GPT/DeepSeek implementation."""

from pathlib import Path
import subprocess
import sys

import pytest

from tts_qwen.routing import use_qwen


@pytest.mark.parametrize("argv,default,expected", [
    (["--llm-agent", "Qwen3.5-35B-A3B"], "", True),
    (["--llm-agent=Qwen-AgentWorld-35B-A3B"], "", True),
    ([], "Qwen3.5-35B-A3B", True),
    (["--llm-agent", "gpt-5.4"], "Qwen3.5-35B-A3B", False),
    (["--llm-agent", "deepseek-v4-pro"], "Qwen3.5-35B-A3B", False),
    (["--llm-agent", "gpt-5.4", "--wm-model", "Qwen"], "", False),
    (["--llm-user", "Qwen"], "gpt-5.4", False),
])
def test_solver_selects_implementation(argv, default, expected):
    assert use_qwen(argv, default) is expected


@pytest.mark.parametrize("model,qwen", [
    ("Qwen3.5-35B-A3B", True), ("gpt-5.4", False),
    ("deepseek-v4-pro", False),
])
def test_cli_dispatch(model, qwen):
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, str(root / "experiments/run.py"),
         "--llm-agent", model, "--help"],
        capture_output=True, text=True, check=True,
    )
    assert "--agent {llm,llm_wm,agent_world}" in result.stdout
    assert "--appworld-qwen-repair" not in result.stdout


def test_packages_do_not_share_mutable_implementation_modules():
    from tts.env import officebench_wm_predict as original
    from tts_qwen.env import officebench_wm_predict as qwen
    from tts.config import Config
    from tts_qwen.config import Config as QwenConfig

    assert original is not qwen
    assert Config is not QwenConfig
