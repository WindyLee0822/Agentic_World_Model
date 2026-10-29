"""Paper method mapping and backend configuration without inference calls."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from awm.cli import parse_args, validate_resume
from tts_qwen.routing import use_qwen


@pytest.mark.parametrize("qwen", [False, True])
@pytest.mark.parametrize("method,agent,label", [
    ("b2", "llm", "real_env_bon"), ("ours", "llm_wm", "awm"),
])
def test_paper_modes_and_defaults(qwen, method, agent, label, monkeypatch):
    monkeypatch.delenv("TTS_TAU2_WM_TOOL_MODE", raising=False)
    _, args = parse_args(["--env", "tau2", "--method", method, "--agent", agent], qwen=qwen)
    assert args.paper_method == label
    assert args.N == "1,2,4,8,16"
    assert args.temperature == 1.0
    assert args.domains == "airline,retail"
    assert args.hard_cap == 6
    assert args.predict_all_actions
    assert not args.allow_partial


def test_trained_baseline_uses_qwen_backend_and_separate_endpoint():
    argv = ["--env", "appworld", "--method", "ours", "--agent", "agent_world",
            "--llm-agent", "Qwen3.5-35B-A3B", "--wm-model", "Qwen-AgentWorld-35B-A3B",
            "--wm-base-url", "http://localhost:8001/v1"]
    assert use_qwen(argv)
    _, args = parse_args(argv, qwen=True)
    assert (args.methods, args.agent) == ("agentworld", "llm_wm")
    assert args.paper_method == "qwen_agentworld"
    assert args.paper_agent == "agent_world"
    assert Path(args.wm_system_prompt_file).is_file()
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("qwen_paper_runner", root / "experiments/run_qwen.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    config = runner._cfg_for("appworld", SimpleNamespace(index=0), args.methods, 4, args)
    assert config.solver.llm_agent == "Qwen3.5-35B-A3B"
    assert config.simulator.world_model_model == "Qwen-AgentWorld-35B-A3B"
    assert config.simulator.world_model_base_url == "http://localhost:8001/v1"


@pytest.mark.parametrize("extra", [
    ["--method", "b1"], ["--method", "b0"], ["--method", "b2", "--agent", "llm_wm"],
    ["--method", "ours", "--agent", "llm"], ["--agent", "scripted"],
    ["--depth-sweep"], ["--agent-self-aggregate"], ["--exact-clean-reads"],
    ["--appworld-agent-reference"], ["--N", "0"], ["--N", "1,1"],
    ["--workers", "0"], ["--wm-model", "Qwen-AgentWorld"],
])
def test_rejects_removed_or_invalid_arguments(extra):
    with pytest.raises(SystemExit) as error:
        parse_args(["--env", "appworld", *extra])
    assert error.value.code == 2


def test_trained_baseline_requires_endpoint():
    with pytest.raises(SystemExit):
        parse_args(["--env", "officebench", "--agent", "agent_world"], qwen=True)


def test_resume_accepts_matching_legacy_records_and_rejects_mixed_methods():
    parser, args = parse_args(["--env", "appworld", "--method", "ours"])
    validate_resume([{"method": "ours"}], args, parser)
    for record in ({"method": "b2"}, {"method": "agentworld"},
                   {"method": "ours", "paper_method": "qwen_agentworld"}):
        with pytest.raises(SystemExit):
            validate_resume([record], args, parser)


def test_paper_domains_reject_unrelated_benchmarks():
    with pytest.raises(SystemExit):
        parse_args(["--env", "tau2", "--domains", "telecom"])


@pytest.mark.parametrize("package", ["tts", "tts_qwen"])
def test_removed_methods_rejected_by_library(package):
    from importlib import import_module
    from conftest import make_feasible_infeasible_env

    Config = import_module(f"{package}.config").Config
    run_decision = import_module(f"{package}.harness").run_decision
    for method in ("b0", "b1"):
        with pytest.raises(KeyError):
            run_decision(make_feasible_infeasible_env(), Config(method=method))
