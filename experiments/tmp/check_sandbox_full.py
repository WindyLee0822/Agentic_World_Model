"""A/B: does the sandbox preamble stop gpt-5.2 refusing consequential actions
mid-trajectory? Runs the FULL WM rollout (same path as the experiment) for each
task, baseline vs sandbox, capturing the AGENT's responses, and reports:
  - whether any agent turn REFUSED (no ```python``` block + refusal language)
  - the WM success verdict (predicted_confidence)
  - the real committed outcome (success) via env.evaluate on the realized code

Only the agent chat client is logged (it is the one created with a seed);
the predictor client (seed=None) is left alone.
"""
import os
import re
import sys

sys.path.insert(0, "/home/t-wendili/dw/src")

import tts.llm as _llm
from tts.config import Config, SolverConfig, SimulatorConfig
from tts.env.real_appworld import AppWorldTaskEnv

CODE_RE = re.compile(r"```python\n", re.IGNORECASE)
REFUSE_RE = re.compile(
    r"\b(can'?t|cannot|won'?t|unable to|i'?m not able|i am not able|not comfortable|"
    r"refus|decline|i (?:would|will) not|against (?:my|the) (?:policy|guidelines)|"
    r"as an ai|i'?m sorry|i apologi|cannot assist|can'?t help)\b",
    re.IGNORECASE,
)

TASK_IDS = sys.argv[1].split(",") if len(sys.argv) > 1 else [
    "9871968_1", "9871968_2", "20c1328_2", "1c4bd27_1", "b3bdcc1_2", "b05f69a_1",
]
MODEL = os.environ.get("WM_MODEL", "gpt-5.2")
BASE = "tts_sandbox_probe"

_orig_make = _llm.make_chat_client
_LOG = []


def _logging_make_chat_client(*a, **kw):
    client = _orig_make(*a, **kw)
    if kw.get("seed") is not None:  # the AGENT client (predictor passes no seed)
        orig_complete = client.complete

        def complete(messages, **ckw):
            resp = orig_complete(messages, **ckw)
            _LOG.append(resp or "")
            return resp

        client.complete = complete
    return client


_llm.make_chat_client = _logging_make_chat_client
# patch the name already imported into real_appworld's module namespace if present
import tts.env.real_appworld as _ra
if hasattr(_ra, "make_chat_client"):
    _ra.make_chat_client = _logging_make_chat_client


def refusal_turns(responses):
    out = []
    for r in responses:
        if not CODE_RE.search(r) and REFUSE_RE.search(r):
            out.append(r)
    return out


def run_one(env, sandbox: bool, seed: int):
    if sandbox:
        os.environ["TTS_APPWORLD_SANDBOX_PROMPT"] = "1"
    else:
        os.environ.pop("TTS_APPWORLD_SANDBOX_PROMPT", None)
    _LOG.clear()
    cfg = Config(
        solver=SolverConfig(agent="llm_wm", llm_agent=MODEL, temperature=0.0, max_steps=40),
        simulator=SimulatorConfig(kind="recursive"),
    )
    from tts.config import BudgetConfig
    from tts.budget import Budget
    traj, predicted, _ = env.run_agent_rollout_wm(cfg, "ours", seed, Budget(), None)
    responses = list(_LOG)
    refusals = refusal_turns(responses)
    # real committed outcome: replay realized code on a fresh real world
    code = [s.action.args.get("code", "") for s in traj.steps]
    real = None
    try:
        ri = env.evaluate(traj)
        real = bool(ri.success)
    except Exception as e:  # noqa: BLE001
        real = f"eval-err:{type(e).__name__}"
    return {
        "n_steps": len(traj.steps),
        "refusals": len(refusals),
        "first_refusal": (refusals[0][:160].replace(chr(10), " ") if refusals else None),
        "wm_conf": round(float(predicted.reward), 3),
        "real_success": real,
    }


def main():
    print(f"model={MODEL}  tasks={TASK_IDS}\n")
    os.environ["TTS_APPWORLD_EXP_BASE"] = BASE
    fixed = 0
    for tid in TASK_IDS:
        try:
            env = AppWorldTaskEnv(task_id=tid, experiment_name=BASE)
            b = run_one(env, sandbox=False, seed=1)
            env2 = AppWorldTaskEnv(task_id=tid, experiment_name=BASE)
            s = run_one(env2, sandbox=True, seed=1)
        except Exception as e:  # noqa: BLE001
            print(f"{tid}: ERROR {type(e).__name__}: {e}")
            continue
        flip = (b["refusals"] > 0 and s["refusals"] == 0)
        note = "  <== REFUSAL REMOVED" if flip else ""
        if flip:
            fixed += 1
        print(f"{tid}:")
        print(f"   baseline: steps={b['n_steps']} refusals={b['refusals']} "
              f"wm_conf={b['wm_conf']} real={b['real_success']}")
        if b["first_refusal"]:
            print(f"       refusal: {b['first_refusal']!r}")
        print(f"   sandbox : steps={s['n_steps']} refusals={s['refusals']} "
              f"wm_conf={s['wm_conf']} real={s['real_success']}{note}")
        if s["first_refusal"]:
            print(f"       refusal: {s['first_refusal']!r}")
    print(f"\nrefusals removed by sandbox: {fixed}/{len(TASK_IDS)}")


if __name__ == "__main__":
    main()
