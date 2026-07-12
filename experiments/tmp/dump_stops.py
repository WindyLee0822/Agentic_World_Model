"""Dump the agent's no-code (stall/refusal) turns for baseline rollouts, to reveal
WHY a rollout stopped early. Baseline only (no sandbox), fast-ish."""
import os
import re
import sys

sys.path.insert(0, "/home/t-wendili/dw/src")

import tts.llm as _llm
import tts.env.real_appworld as _ra
from tts.config import Config, SolverConfig, SimulatorConfig
from tts.budget import Budget
from tts.env.real_appworld import AppWorldTaskEnv

CODE_RE = re.compile(r"```python\n", re.IGNORECASE)
TASK_IDS = sys.argv[1].split(",") if len(sys.argv) > 1 else ["9871968_1", "b3bdcc1_2", "1c4bd27_1"]
MODEL = os.environ.get("WM_MODEL", "gpt-5.2")
BASE = "tts_sandbox_dump"

_orig_make = _llm.make_chat_client
_LOG = []


def _mk(*a, **kw):
    c = _orig_make(*a, **kw)
    if kw.get("seed") is not None:
        oc = c.complete
        def complete(messages, **ckw):
            r = oc(messages, **ckw)
            _LOG.append(r or "")
            return r
        c.complete = complete
    return c


_llm.make_chat_client = _mk
_ra.make_chat_client = _mk

os.environ["TTS_APPWORLD_EXP_BASE"] = BASE
os.environ.pop("TTS_APPWORLD_SANDBOX_PROMPT", None)

for tid in TASK_IDS:
    _LOG.clear()
    env = AppWorldTaskEnv(task_id=tid, experiment_name=BASE)
    cfg = Config(solver=SolverConfig(agent="llm_wm", llm_agent=MODEL, temperature=0.0, max_steps=40),
                 simulator=SimulatorConfig(kind="recursive"))
    try:
        traj, predicted, _ = env.run_agent_rollout_wm(cfg, "ours", 1, Budget(), None)
    except Exception as e:
        print(f"{tid}: ERROR {e}")
        continue
    print(f"\n===== {tid}: {len(traj.steps)} code steps, {len(_LOG)} agent turns =====")
    print(f"instruction: {env._instruction[:300]}")
    for i, r in enumerate(_LOG):
        if not CODE_RE.search(r):
            print(f"  [no-code turn {i}] {r[:500].strip()!r}")
    # also show last turn always
    if _LOG:
        print(f"  [LAST turn] {_LOG[-1][:400].strip()!r}")
