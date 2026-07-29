"""CLI for WM verifier prompt optimization.

Examples
--------
Collect labeled data (τ² is cheap & LLM-free; OfficeBench needs Docker + the agent LLM)::

    python -m promptopt.run collect --env tau2 --out promptopt/store/tau2_data.jsonl
    python -m promptopt.run collect --env officebench --out promptopt/store/ob_data.jsonl

Optimize the verifier prompt on the collected subset (advanced model rewrites the prompt)::

    python -m promptopt.run optimize --env tau2 --data promptopt/store/tau2_data.jsonl --rounds 4

Inspect / evaluate::

    python -m promptopt.run eval --env tau2 --data promptopt/store/tau2_data.jsonl --template best
    python -m promptopt.run show-best --env tau2
"""

from __future__ import annotations

import argparse
import os
import sys

# make both the repo root and src/ importable when run as a script
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (_ROOT, os.path.join(_ROOT, "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from promptopt.dataset import (LabeledExample, SubsetSpec, load_examples, save_examples,  # noqa: E402
                               append_examples, slice_counts, label_balance)
from promptopt.evaluate import evaluate_template                                          # noqa: E402
from promptopt import templates as T                                                       # noqa: E402
from promptopt import optimize as O                                                        # noqa: E402


def _cmd_collect(a):
    examples = []
    if a.env in ("tau2", "both"):
        from promptopt.collect_tau2 import collect_tau2
        examples += collect_tau2(domains=tuple(a.domains), tasks_per_domain=a.tasks,
                                 max_per_task=a.max_per_task, seed=a.seed)
    if a.env in ("officebench", "both"):
        from promptopt.collect_officebench import collect_officebench
        examples += collect_officebench(apps=tuple(a.apps), tasks_per_app=a.tasks,
                                        seeds_per_task=a.seeds, model=a.model,
                                        max_steps=a.max_steps)
    if not examples:
        print("No examples collected.")
        return
    if a.append and os.path.exists(a.out):
        append_examples(a.out, examples)
        examples = load_examples(a.out)
    else:
        save_examples(a.out, examples)
    print(f"Saved {len(examples)} examples -> {a.out}")
    print("slices:", slice_counts(examples), "labels:", label_balance(examples))
    miss = SubsetSpec().missing(examples)
    print("subset requirement:", "SATISFIED" if not miss else f"MISSING {miss}")


def _make_llm(kind, model):
    from promptopt.llm import make_llm
    return make_llm(kind, model=model) if kind == "real" else make_llm(kind)


def _cmd_optimize(a):
    examples = load_examples(a.data, env=a.env if a.env != "both" else None)
    envs = ["tau2", "officebench"] if a.env == "both" else [a.env]
    verifier = _make_llm(a.llm, a.model)
    optimizer = _make_llm(a.llm, a.optimizer_model or a.model)
    cfg = O.OptimizerConfig(rounds=a.rounds, k=a.k, verifier_temperature=a.temperature,
                            objective=a.objective, n_error_examples=a.n_errors,
                            max_workers=a.workers, seed=a.seed)
    for env in envs:
        env_exs = [e for e in examples if e.env == env]
        if not env_exs:
            print(f"[{env}] no examples in {a.data}; skipping")
            continue
        spec = SubsetSpec()
        miss = spec.missing(env_exs)
        if miss.get(env if env == "officebench" else "tau2"):
            print(f"[{env}] WARNING: subset not fully covered: {miss}")
        res = O.optimize(env, env_exs, verifier, optimizer, cfg)
        O.save_result(res, a.store)
        print(f"[{env}] saved best (improved={res.improved}) -> {a.store}")


def _cmd_eval(a):
    examples = load_examples(a.data, env=a.env)
    tmpl = T.baseline(a.env) if a.template == "baseline" else (O.load_best(a.env, a.store) or T.baseline(a.env))
    llm = _make_llm(a.llm, a.model)
    m = evaluate_template(tmpl, examples, llm, k=a.k, temperature=a.temperature, max_workers=a.workers)
    print(f"[{a.env}] template={a.template} :: {m.summary()}")


def _cmd_show_best(a):
    res_path = os.path.join(a.store, f"{a.env}_report.json")
    tmpl = O.load_best(a.env, a.store)
    if tmpl is None:
        print(f"No best template stored for {a.env} in {a.store}")
        return
    if os.path.exists(res_path):
        with open(res_path, encoding="utf-8") as fh:
            print(fh.read())
    print("\n=== BEST TEMPLATE ===")
    print(tmpl.to_json())


def main(argv=None):
    p = argparse.ArgumentParser(prog="promptopt.run", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    store_default = O.default_store()

    c = sub.add_parser("collect", help="sample trajectories + record real-env true labels")
    c.add_argument("--env", choices=["tau2", "officebench", "both"], default="tau2")
    c.add_argument("--out", default=os.path.join(store_default, "data.jsonl"))
    c.add_argument("--append", action="store_true", help="append to --out instead of overwrite")
    c.add_argument("--domains", nargs="+", default=["airline", "telecom"], help="τ² domains")
    c.add_argument("--apps", nargs="+", type=int, default=[1, 2, 3], help="OfficeBench app-levels")
    c.add_argument("--tasks", type=int, default=4, help="tasks per domain/app")
    c.add_argument("--max-per-task", dest="max_per_task", type=int, default=5, help="τ² candidates/task")
    c.add_argument("--seeds", type=int, default=2, help="OfficeBench rollouts/task")
    c.add_argument("--max-steps", dest="max_steps", type=int, default=20, help="OfficeBench agent steps")
    c.add_argument("--model", default=None, help="agent/predictor model (OfficeBench)")
    c.add_argument("--seed", type=int, default=0)
    c.set_defaults(func=_cmd_collect)

    o = sub.add_parser("optimize", help="reflection loop; keep best (≥ baseline)")
    o.add_argument("--env", choices=["tau2", "officebench", "both"], default="tau2")
    o.add_argument("--data", required=True)
    o.add_argument("--store", default=store_default)
    o.add_argument("--llm", choices=["real", "mock"], default="real")
    o.add_argument("--model", default=None, help="verifier model")
    o.add_argument("--optimizer-model", dest="optimizer_model", default=None, help="advanced model")
    o.add_argument("--rounds", type=int, default=4)
    o.add_argument("--k", type=int, default=1, help="verifier self-consistency samples")
    o.add_argument("--temperature", type=float, default=0.7)
    o.add_argument("--objective", default="balanced_accuracy",
                   choices=["balanced_accuracy", "accuracy", "f1"])
    o.add_argument("--n-errors", dest="n_errors", type=int, default=8)
    o.add_argument("--workers", type=int, default=4)
    o.add_argument("--seed", type=int, default=0)
    o.set_defaults(func=_cmd_optimize)

    e = sub.add_parser("eval", help="score a template (baseline|best) on a dataset")
    e.add_argument("--env", choices=["tau2", "officebench"], required=True)
    e.add_argument("--data", required=True)
    e.add_argument("--template", choices=["baseline", "best"], default="baseline")
    e.add_argument("--store", default=store_default)
    e.add_argument("--llm", choices=["real", "mock"], default="real")
    e.add_argument("--model", default=None)
    e.add_argument("--k", type=int, default=1)
    e.add_argument("--temperature", type=float, default=0.7)
    e.add_argument("--workers", type=int, default=4)
    e.set_defaults(func=_cmd_eval)

    s = sub.add_parser("show-best", help="print the stored best template + report")
    s.add_argument("--env", choices=["tau2", "officebench"], required=True)
    s.add_argument("--store", default=store_default)
    s.set_defaults(func=_cmd_show_best)

    a = p.parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    main()
