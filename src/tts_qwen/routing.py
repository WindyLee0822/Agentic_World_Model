"""Select the preserved Qwen implementation without changing other model runs."""

import argparse
from typing import Sequence


def use_qwen(argv: Sequence[str], default_model: str = "") -> bool:
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--llm-agent")
    parser.add_argument("--agent")
    args, _ = parser.parse_known_args(argv)
    # The solver selects the implementation; a Qwen WM/user alone must not
    # redirect GPT or DeepSeek solver runs.
    return args.agent == "agent_world" or "qwen" in (args.llm_agent or default_model).lower()
