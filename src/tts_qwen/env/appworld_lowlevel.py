"""Low-level helpers for the REAL AppWorld benchmark (package ``appworld``, Python 3.11 env).

Provides availability / data-root checks and task-id loading.  The trajectory-level wrapper
:class:`~tts_qwen.env.real_appworld.AppWorldTaskEnv` opens the genuine ``appworld.AppWorld`` directly
(running the official ReAct code agent and scoring with ``AppWorld.evaluate()``); no synthetic
state-reading layer lives here.
"""

from __future__ import annotations

import os
from typing import List, Optional


def _default_root() -> Optional[str]:
    return os.environ.get("APPWORLD_ROOT") or os.environ.get("APPWORLD_DATA_ROOT")


def real_appworld_available(root: Optional[str] = None) -> bool:
    try:
        import appworld  # noqa: F401
    except Exception:
        return False
    root = root or _default_root()
    return bool(root) and os.path.isdir(os.path.join(root, "data", "tasks"))


def _ensure_root(root: Optional[str]) -> None:
    root = root or _default_root()
    if not root:
        raise RuntimeError("AppWorld data root not set; export APPWORLD_ROOT (see `appworld download data`).")
    from appworld import update_root  # type: ignore

    update_root(root)


def load_real_appworld_task_ids(split: str, level: Optional[int] = None, root: Optional[str] = None) -> List[str]:
    from appworld import load_task_ids  # type: ignore

    _ensure_root(root)
    return list(load_task_ids(split, difficulty=level))


__all__ = [
    "real_appworld_available",
    "load_real_appworld_task_ids",
]
