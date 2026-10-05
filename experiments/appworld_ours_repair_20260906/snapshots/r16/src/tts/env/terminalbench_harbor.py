"""Harbor custom-agent adapter for Terminal-Bench 2."""

from __future__ import annotations

import json
import shlex
from pathlib import Path
from typing import Any, Optional

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

from ..llm import make_chat_client
from .terminalbench_agent import (
    TerminalCommandResult,
    run_terminalbench_tts,
    terminalbench_settings,
)
from .terminalbench_wm import standalone_cd_target


def _chat_model(model_name: Optional[str]) -> Optional[str]:
    if not model_name:
        return None
    provider, separator, model = model_name.partition("/")
    if separator and provider.lower() in {"azure", "openai"}:
        return model
    return model_name


class _HarborExecutor:
    def __init__(self, environment: BaseEnvironment, *, timeout_sec: int) -> None:
        self.environment = environment
        self.timeout_sec = timeout_sec
        self._initial_cwd: Optional[str] = None

    async def initial_cwd(self) -> str:
        if self._initial_cwd is None:
            result = await self.environment.exec(
                command="pwd",
                timeout_sec=self.timeout_sec,
            )
            if result.return_code != 0:
                raise RuntimeError(
                    f"failed to resolve Terminal-Bench working directory: {result.stderr}")
            self._initial_cwd = (result.stdout or "/").strip().splitlines()[-1]
        return self._initial_cwd

    async def execute(self, command: str, cwd: str) -> TerminalCommandResult:
        cd_target = standalone_cd_target(command)
        actual_command = command
        if cd_target is not None:
            actual_command = f"cd -- {shlex.quote(cd_target)} && pwd"
        try:
            result = await self.environment.exec(
                command=actual_command,
                cwd=cwd,
                timeout_sec=self.timeout_sec,
            )
        except TimeoutError:
            return TerminalCommandResult(
                stderr=f"command timed out after {self.timeout_sec}s",
                return_code=124,
                cwd=cwd,
            )
        next_cwd = cwd
        stdout = result.stdout or ""
        if cd_target is not None and result.return_code == 0 and stdout.strip():
            next_cwd = stdout.strip().splitlines()[-1]
        return TerminalCommandResult(
            stdout=stdout,
            stderr=result.stderr or "",
            return_code=result.return_code,
            cwd=next_cwd,
        )


class HarborTerminalBenchAgent(BaseAgent):
    """Run this repository's TTS command loop inside a Harbor task environment."""

    @staticmethod
    def name() -> str:
        return "tts-terminalbench"

    def version(self) -> str:
        return "1.0.0"

    def __init__(
        self,
        logs_dir: Path,
        model_name: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(logs_dir=logs_dir, model_name=model_name, **kwargs)
        self._settings = terminalbench_settings(kwargs)

    async def setup(self, environment: BaseEnvironment) -> None:
        return None

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        executor = _HarborExecutor(
            environment,
            timeout_sec=self._settings["command_timeout"],
        )
        model = _chat_model(self.model_name)

        def chat_factory(seed: int, temperature: float):
            kwargs = {"seed": seed, "temperature": temperature}
            if model:
                kwargs["model"] = model
            return make_chat_client(**kwargs)

        decision = await run_terminalbench_tts(
            instruction,
            executor,
            chat_factory,
            settings=self._settings,
        )
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        (self.logs_dir / "tts-terminalbench.json").write_text(
            json.dumps(decision.to_dict(), indent=2),
            encoding="utf-8",
        )
        context.metadata = {
            **(context.metadata or {}),
            "tts_terminalbench": {
                "method": decision.method,
                "selected_index": decision.selected_index,
                "n_rollouts": len(decision.rollouts),
                "total_tokens": decision.total_tokens,
            },
        }


__all__ = ["HarborTerminalBenchAgent"]
