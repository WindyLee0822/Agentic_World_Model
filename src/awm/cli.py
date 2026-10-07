"""One public CLI for the paper's three supported execution modes.

Backend-specific settings live in profiles.py; they are not experiment axes.
The trained baseline retains its internal ``agentworld`` record identifier so
existing checkpoints cannot be confused with AWM checkpoints.
"""

import argparse
import os
from pathlib import Path

from .profiles import backend_defaults


def positive_int(value):
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def rollout_budgets(value):
    try:
        values = [positive_int(item.strip()) for item in value.split(",")]
    except (ValueError, argparse.ArgumentTypeError):
        raise argparse.ArgumentTypeError("use positive rollout counts, e.g. 1,2,4,8,16")
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("rollout counts must be unique")
    return ",".join(map(str, sorted(values)))


def parse_args(argv=None, *, qwen=False):
    p = argparse.ArgumentParser(
        description="Agentic World Modeling: AWM, real-environment BoN ceiling, and Qwen-AgentWorld.",
        allow_abbrev=False,
    )
    run = p.add_argument_group("experiment")
    run.add_argument("--env", choices=["tau2", "appworld", "officebench"])
    run.add_argument("--method", "--methods", dest="methods", choices=["b2", "ours"], default="ours",
                     help="b2: real-environment ceiling; ours: AWM or Qwen-AgentWorld")
    run.add_argument("--agent", choices=["llm", "llm_wm", "agent_world"], default=None,
                     help="b2 requires llm; ours defaults to llm_wm; agent_world selects Qwen-AgentWorld")
    run.add_argument("--N", type=rollout_budgets, default="1,2,4,8,16", help="Best-of-N budgets")
    run.add_argument("--temperature", type=float, default=1.0, help="solver sampling temperature (paper: 1.0)")
    run.add_argument("--max-steps", type=positive_int, default=100)
    run.add_argument("--seed-base", type=int, default=0)
    run.add_argument("--tool-mode", choices=["python", "text"], default="python",
                     help="AWM prediction tools; text is the paper's without-Python ablation")

    models = p.add_argument_group("model endpoints")
    models.add_argument("--llm-agent", help="solver model/deployment name")
    models.add_argument("--llm-user", help="tau2 user simulator model (default: solver model)")
    models.add_argument("--local", action="store_true", help="use an OpenAI-compatible model server")
    models.add_argument("--local-base-url", help="solver endpoint (with --local)")
    models.add_argument("--local-api-key", help="defaults to OPENAI_API_KEY or EMPTY")
    models.add_argument("--skip-local-healthcheck", action="store_true")
    models.add_argument("--wm-model", help="trained Qwen-AgentWorld model name")
    models.add_argument("--wm-base-url", help="trained Qwen-AgentWorld endpoint")
    models.add_argument("--wm-api-key", default=os.environ.get("AWM_WM_API_KEY", "EMPTY"))
    models.add_argument("--wm-system-prompt-file", help="override the bundled AgentWorld MCP prompt")

    data = p.add_argument_group("benchmark selection")
    data.add_argument("--tasks", type=positive_int, help="limit tasks per subset; omitted means all")
    data.add_argument("--task-indices", help="comma-separated indices within each subset")
    data.add_argument("--splits", help="AppWorld splits (default: test_normal,test_challenge)")
    data.add_argument("--levels", help="AppWorld difficulty or OfficeBench app count (default: 1,2,3)")
    data.add_argument("--domains", help="tau2 domains (default: airline,retail)")
    data.add_argument("--list-subsets", action="store_true")

    execution = p.add_argument_group("execution and results")
    execution.add_argument("--workers", type=positive_int, default=1)
    execution.add_argument("--max-tasks-per-child", type=int, default=0)
    execution.add_argument("--rollout-retries", type=int, default=2)
    execution.add_argument("--cell-retries", type=int, default=2)
    execution.add_argument("--out", default="experiments/results")
    execution.add_argument("--resume", action="store_true")
    execution.add_argument("--report", metavar="RECORDS_JSONL")
    execution.add_argument("--keep-outputs", action="store_true")
    execution.add_argument("--verbose", action="store_true")

    args = p.parse_args(argv)
    if not args.env and not args.report:
        p.error("--env is required (unless --report is used)")
    args.agent = args.agent or ("llm" if args.methods == "b2" else "llm_wm")
    if (args.methods, args.agent) not in {
        ("b2", "llm"), ("ours", "llm_wm"), ("ours", "agent_world"),
    }:
        p.error("supported pairs: b2+llm, ours+llm_wm, ours+agent_world")
    if args.temperature < 0:
        p.error("--temperature must be nonnegative")
    for key in ("rollout_retries", "cell_retries", "max_tasks_per_child"):
        if getattr(args, key) < 0:
            p.error(f"--{key.replace('_', '-')} must be nonnegative")
    args.paper_method = {"llm": "real_env_bon", "llm_wm": "awm", "agent_world": "qwen_agentworld"}[args.agent]
    args.paper_agent = args.agent
    if args.agent == "agent_world":
        if not qwen:
            p.error("agent_world requires the Qwen runner (experiments/run_qwen.py)")
        if not args.report and not args.list_subsets and (not args.wm_model or not args.wm_base_url):
            p.error("agent_world requires --wm-model and --wm-base-url")
        args.wm_system_prompt_file = args.wm_system_prompt_file or str(
            Path(__file__).resolve().parents[1] / "tts_qwen/env/prompts/agentworld_mcp_system_prompt.txt")
        args.methods, args.agent = "agentworld", "llm_wm"
    elif args.wm_model or args.wm_base_url or args.wm_system_prompt_file:
        p.error("--wm-* endpoint options are only for --agent agent_world")

    args.splits = args.splits or "test_normal,test_challenge"
    args.levels = args.levels or "1,2,3"
    args.domains = args.domains or "airline,retail"
    if args.env == "tau2" and not set(args.domains.split(",")) <= {"airline", "retail"}:
        p.error("the paper's tau2 evaluation supports airline and retail domains")
    for key, value in backend_defaults(qwen=qwen).items():
        if not hasattr(args, key):
            setattr(args, key, value)
    args.appworld_wm_tool_mode = args.officebench_wm_tool_mode = args.tool_mode
    os.environ["TTS_TAU2_WM_TOOL_MODE"] = args.tool_mode
    return p, args


def validate_resume(records, args, parser):
    """Keep each result directory restricted to one paper method."""
    for record in records:
        if record.get("method") != args.methods:
            parser.error("--resume method mismatch; use a separate --out directory for each method/agent pair")
        label = record.get("paper_method")
        if label is not None and label != args.paper_method:
            parser.error("--resume paper-method mismatch; use a matching method/agent pair")
