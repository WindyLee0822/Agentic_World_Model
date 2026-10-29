"""Legacy `tb` custom-agent adapter for Terminal-Bench 1."""

from __future__ import annotations

import asyncio
import json
import re
import shlex
import uuid
from pathlib import Path
from typing import Any, Optional

from terminal_bench.agents.base_agent import AgentResult, BaseAgent
from terminal_bench.terminal.tmux_session import TmuxSession

from ..llm import make_chat_client
from .terminalbench_agent import (
    TerminalCommandResult,
    run_terminalbench_tts,
    terminalbench_settings,
)


def _chat_model(model_name: Optional[str]) -> Optional[str]:
    if not model_name:
        return None
    provider, separator, model = model_name.partition("/")
    if separator and provider.lower() in {"azure", "openai"}:
        return model
    return model_name


class _LegacyExecutor:
    def __init__(self, session: TmuxSession, *, timeout_sec: int) -> None:
        self.session = session
        self.timeout_sec = timeout_sec
        self._initial_cwd: Optional[str] = None

    async def initial_cwd(self) -> str:
        if self._initial_cwd is None:
            result = await self._execute("pwd", None)
            if result.return_code != 0:
                raise RuntimeError("failed to resolve Terminal-Bench working directory")
            self._initial_cwd = result.cwd or "/"
        return self._initial_cwd

    async def execute(self, command: str, cwd: str) -> TerminalCommandResult:
        return await self._execute(command, cwd)

    async def _execute(
        self,
        command: str,
        cwd: Optional[str],
    ) -> TerminalCommandResult:
        marker = f"__TTS_{uuid.uuid4().hex}__"
        stripped = command.rstrip().rstrip(";")
        if cwd is not None:
            stripped = f"cd -- {shlex.quote(cwd)} && {stripped}"
        wrapped = (
            f"{stripped}; __tts_rc=$?; "
            f"printf '\\n{marker}RC:%s\\n' \"$__tts_rc\"; "
            f"printf '{marker}PWD:'; pwd; printf '{marker}END\\n'"
        )
        try:
            self.session.send_keys(
                [wrapped, "Enter"],
                block=True,
                max_timeout_sec=float(self.timeout_sec),
            )
        except TimeoutError:
            return TerminalCommandResult(
                stderr=f"command timed out after {self.timeout_sec}s",
                return_code=124,
                cwd=cwd or self._initial_cwd,
            )
        output = self.session.get_incremental_output()
        rc_matches = list(re.finditer(
            re.escape(marker) + r"RC:(-?\d+)",
            output,
        ))
        pwd_matches = list(re.finditer(
            re.escape(marker) + r"PWD:([^\r\n]*)",
            output,
        ))
        end_matches = list(re.finditer(
            re.escape(marker) + r"END",
            output,
        ))
        rc_match = rc_matches[-1] if rc_matches else None
        pwd_match = pwd_matches[-1] if pwd_matches else None
        end_match = end_matches[-1] if end_matches else None
        cleaned = output
        if rc_match and end_match and rc_match.start() < end_match.end():
            cleaned = output[:rc_match.start()] + output[end_match.end():]
        cleaned = cleaned.strip()
        return TerminalCommandResult(
            stdout=cleaned,
            return_code=int(rc_match.group(1)) if rc_match else 0,
            cwd=(pwd_match.group(1).strip()
                 if pwd_match and pwd_match.group(1).strip()
                 else cwd or self._initial_cwd),
        )


class LegacyTerminalBenchAgent(BaseAgent):
    """Run the same TTS command loop through the Terminal-Bench 1 tmux API."""

    @staticmethod
    def name() -> str:
        return "tts-terminalbench"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._model_name = _chat_model(kwargs.get("model_name"))
        self._settings = terminalbench_settings(kwargs)

    def perform_task(
        self,
        instruction: str,
        session: TmuxSession,
        logging_dir: Path | None = None,
    ) -> AgentResult:
        executor = _LegacyExecutor(
            session,
            timeout_sec=self._settings["command_timeout"],
        )

        def chat_factory(seed: int, temperature: float):
            kwargs: dict[str, Any] = {"seed": seed, "temperature": temperature}
            if self._model_name:
                kwargs["model"] = self._model_name
            return make_chat_client(**kwargs)

        decision = asyncio.run(run_terminalbench_tts(
            instruction,
            executor,
            chat_factory,
            settings=self._settings,
        ))
        if logging_dir is not None:
            logging_dir.mkdir(parents=True, exist_ok=True)
            (logging_dir / "tts-terminalbench.json").write_text(
                json.dumps(decision.to_dict(), indent=2),
                encoding="utf-8",
            )
        return AgentResult(
            total_input_tokens=decision.total_tokens,
            total_output_tokens=0,
        )


__all__ = ["LegacyTerminalBenchAgent"]
