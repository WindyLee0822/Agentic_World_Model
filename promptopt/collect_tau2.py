"""Collect labeled τ² verifier examples — cheaply and without any LLM call.

For each usable τ² task we take the gold action sequence and generate corrupted variants; each
candidate's TRUE label comes from ``env.evaluate`` (the official ``EvaluationType.ENV`` DB/assertion
check — no NL-assertion judge, so no LLM).  The exact verifier inputs (facts / gold / candidate /
tool docs) are captured by running a *flat* world model over the candidate under
:func:`promptopt.patching.capture_tau2` (reads only, still no LLM).  The result is a labeled dataset
the verifier-prompt optimizer can iterate on offline.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Sequence

from .dataset import LabeledExample
from .patching import capture_tau2


def _corruptions(gold: List, rng: random.Random) -> List[List]:
    """Return distinct candidate action lists (including gold) likely to span both labels."""
    cands: List[List] = [list(gold)]                        # gold -> expected success
    if gold:
        cands.append(list(gold[:-1]))                       # drop last -> often fail
        if len(gold) > 2:
            cands.append(list(gold[: len(gold) // 2]))      # truncate -> fail
            drop = rng.randrange(len(gold))
            cands.append([a for i, a in enumerate(gold) if i != drop])   # drop one -> often fail
            cands.append(list(gold) + [gold[rng.randrange(len(gold))]])  # duplicate -> maybe fail
    # de-dup by (name,args) signature (args JSON-serialized so nested values stay hashable)
    import json
    seen, out = set(), []
    for c in cands:
        sig = tuple((a.name, json.dumps(a.args or {}, sort_keys=True, default=str)) for a in c)
        if sig not in seen:
            seen.add(sig)
            out.append(c)
    return out


def collect_tau2_task(env, domain: str, task_id: str, rng: random.Random,
                      max_per_task: int = 5) -> List[LabeledExample]:
    from tts.budget import Budget
    from tts.config import SimulatorConfig
    from tts.trajectory import Step, Trajectory

    gold = [s.action for s in env.gold_trajectory().steps]
    if not gold:
        return []
    examples: List[LabeledExample] = []
    for ci, cand in enumerate(_corruptions(gold, rng)[:max_per_task]):
        traj = Trajectory(steps=[Step(action=a) for a in cand])
        try:
            truth = bool(env.evaluate(traj).success)
        except Exception:
            continue
        # capture verifier slots via a flat WM (reads only; predict_success is stubbed by the hook)
        sink: List[dict] = []
        try:
            wm = env.world_model(SimulatorConfig(kind="flat"), Budget(max_spawns=1024))
            with capture_tau2(sink):
                wm.evaluate(traj)
        except Exception:
            continue
        if not sink:
            continue
        slots = sink[-1]["slots"]
        examples.append(LabeledExample(
            env="tau2", subset=domain, task_id=f"{task_id}#c{ci}", seed=ci,
            data=slots, true_label=truth,
            meta={"n_actions": len(cand), "n_gold": len(gold)}))
    return examples


def collect_tau2(domains: Sequence[str] = ("airline", "telecom"), tasks_per_domain: int = 4,
                 max_per_task: int = 5, seed: int = 0, log=print) -> List[LabeledExample]:
    from tts.env.datasets import TaskSpec
    from tts.env.registry import make_env_from_spec

    rng = random.Random(seed)
    out: List[LabeledExample] = []
    for domain in domains:
        got = 0
        for idx in range(tasks_per_domain * 3):  # over-scan; some tasks yield nothing usable
            if got >= tasks_per_domain:
                break
            try:
                env = make_env_from_spec(TaskSpec("tau2", domain, idx, domain=domain), usable_only=True)
            except Exception as exc:  # noqa: BLE001
                log(f"[tau2/{domain}] task {idx}: skip ({type(exc).__name__})")
                continue
            exs = collect_tau2_task(env, domain, f"{domain}-{idx}", rng, max_per_task=max_per_task)
            if exs:
                out.extend(exs)
                got += 1
                pos = sum(1 for e in exs if e.true_label)
                log(f"[tau2/{domain}] task {idx}: +{len(exs)} examples ({pos} success)")
    return out


__all__ = ["collect_tau2", "collect_tau2_task"]
