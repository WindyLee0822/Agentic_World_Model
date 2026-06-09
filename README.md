# Trajectory-level, Read-Grounded Test-Time Scaling over Real, Non-Forkable Environments

A research prototype of LLM-agent **test-time scaling (TTS)** over **one real, stateful,
non-forkable environment** — the genuine **τ²-bench** and **AppWorld** benchmarks, executed and
scored through their **official** runners and the **official agent prompts**. A single rollout
(**B0**) is the real official agent (τ²: the official LLM agent + user simulator; AppWorld: the
official ReAct code agent) acting on the real environment and scored by the official evaluator —
so B0 reproduces the official benchmark number. The read-grounded TTS methods then draw `N`
trajectories and, instead of acting on the shared environment, each rollout's *world-simulator
subagent* predicts the outcome by reading the **real backing database** (privileged, read-only)
and, when one bounded read is insufficient, **recursively spawning region-scoped probe
subagents**. After all `N` trajectories are scored, exactly one is selected and committed to the
real environment through a single chokepoint.

```
                          +--------------------------------------------------------+
                          |   ONE REAL, NON-FORKABLE TASK ENVIRONMENT (per task)   |
                          |   tau2 airline (Pydantic FlightDB) | AppWorld (REPL)   |
                          +------+--------------------------------------+----------+
   restricted view (instruction |                                       | privileged READ-ONLY
   + tool names; NO db, NO step,|                                       | EntityStore (deep copy /
   NO commit)                   v                                       v query_only=ON)
        +-------------------------------------+      +-----------------------------------------+
        |  Real official agent rollouts x N   |      |  World-simulators (per rollout) x N      |
        |  - τ²: official LLM agent + user sim | a_t  |  - read entities the trajectory touches  |
        |  - AppWorld: official ReAct agent    +----->|  - recursively read child refs:          |
        |  - official prompt; official scoring | obs |        reservation -> user + flights      |
        +-------------------------------------+      |  - stop rule:                            |
                                                    |      uncertainty: read-coverage < tau    |
                                                    |      fixed:       depth < L              |
                                                    |  - replay candidate + gold on the SHADOW |
                                                    |    (real tool fns / read-grounded check) |
                                                    |  - predict reward; NEVER writes real env |
                                                    +-------------------+----------------------+
                                                        predicted reward_i, spawn_tree_i,
                                                        reads_i, depth_i, tokens_i, cost_i
                                                                        |
                                       +-------------------------------v------------------------+
                                       |  success_score ranker  +  Selector (argmax over N)     |
                                       +-------------------------------+------------------------+
                                                                      | the one selected trajectory
                                       +-------------------------------v------------------------+
                                       |  CommitChokepoint.commit(trajectory)  -> single writer |
                                       +-------------------------------+------------------------+
                                                                      v
                                       official evaluation (tau2 evaluate_simulation / AppWorld
                                       AppWorld.evaluate) -> grounded success/reward
```

## How a single trajectory coheres to the official benchmark

* **τ²-bench (airline).** A trajectory is a sequence of real airline tool-calls. The **real
  agent** is the official τ² LLM customer-service agent (the verbatim official prompt,
  `tau2.agent.llm_agent.AGENT_INSTRUCTION` / `SYSTEM_PROMPT`) conversing with the official user
  simulator over the real environment; stepping uses the official
  `environment.get_response(ToolCall(...))` (the tool execution that mutates the Pydantic
  `FlightDB`). The terminal reward is the official `evaluate_simulation(...)` over the whole
  simulation (`EvaluationType.ALL`), so a single rollout (B0) is the genuine benchmark score. The
  read-grounded world model (B1/Ours) replays tool-call sequences and compares DB-hashes against
  the gold `evaluation_criteria.actions` (a gold sequence scores **1.0**, a perturbed one **0.0**).
* **AppWorld.** Official AppWorld tasks are solved with **free-form Python** in a stateful REPL
  and expose no replayable gold *action list*. A trajectory here is the realized sequence of
  code blocks emitted by the **official ReAct code agent** (the verbatim official prompt +
  loop, see `env/appworld_react_agent.py`): each block runs through the real `AppWorld.execute`
  (real app SQLite), and the terminal reward is the official `AppWorld.evaluate()` (Task Goal
  Completion — every hidden test passes). **All splits are scored locally**: every task ships its
  *test checkers* (`evaluation.py` + `test_data.json` + `answer.json`), which `AppWorld.evaluate()`
  reads under `ground_truth_mode="minimal"`, so the official `test_normal` / `test_challenge`
  splits are evaluated the same way as `dev` / `train` (only the reference *solution*, used by
  `full` mode, is withheld for the test splits).

## Where recursion bites (the read-grounded world model)

The world model (used by the read-grounded TTS methods on τ²) predicts a trajectory's reward
**without touching the real env** by materialising a *shadow* of the entities the trajectory
touches:

* **flat (B1)** reads only `node_read_limit` seed entities — one bounded pass, no recursion. For
  a multi-entity action (τ²: `book`/`update` need the user + flights) it under-reads, cannot
  certify the outcome, and mis-predicts.
* **recursive (Ours)** follows `child_references` to close over the touched subgraph
  (`reservation → user + flights`), so the prediction is correct. The stop rule is configurable:
  `uncertainty` (descend while read-coverage `< tau`) or `fixed` (descend to depth `L`). Every
  read is budgeted; every node is logged in a spawn tree.
* **B2 (snapshot, baseline only)** deep-copies the full τ² state and scores with the oracle — the
  forking upper bound that **Ours matches without forking**. (AppWorld is non-forkable, so B2/Ours
  there fall back to best-of-`N` over the realized official rewards.)

## Non-negotiable constraints — and how they're enforced

| Constraint | Enforcement |
|---|---|
| One real environment, no forking (except B2) | Sampling reads the real `EntityStore`; only B2 calls `fork()`; asserted in tests. |
| Simulator never writes | World model is built from a read-only `EntityStore` + a `ShadowEvaluator` only — never the `TaskEnvironment`; `simulator_writes` is logged and asserted `0`. |
| Two roles, different privileges | Solver gets a `RestrictedView` (instruction + tools); only world models get the privileged `EntityStore`. |
| Exactly one commit | `CommitChokepoint` commits one trajectory, at most once (`n_commits == 1`). |
| No full state in context | Each node reads ≤ `node_read_limit` entities; the flat baseline structurally under-reads. |
| Decoupling | No component imports a concrete env/LLM class — only the ABCs in `tts/abcs.py`. |

## Baselines (same harness, swap `method`)

* **B0** — a single **real official-agent** rollout (official prompt + loop), officially scored;
  this is the benchmark's own number (no TTS, no simulation).
* **B1** — sample `N`, predict reward with the **flat** world model, select best.
* **B2** — sample `N`, oracle reward per rollout, select best (τ²: forked snapshots; baseline only).
* **Ours** — sample `N`, **recursive** read-grounded prediction (`uncertainty` | `fixed`), select best.

The solver is selected with `--agent {llm,scripted}`: `llm` (default) is the real official-prompt
agent (τ²: the official LLM agent + user simulator; AppWorld: the official ReAct code agent), so
**B0 reproduces the official benchmark**. `scripted` is an offline, LLM-free plan-following stub
used only to exercise the env-agnostic machinery (recursion/selection/budget) in unit tests.

**Fair comparison.** Agent rollouts are generated by a single, **method-independent** path
(`run_agent_rollout`, which depends only on `config.solver`, never on `config.method`), so B1, B2
and Ours are prompted **identically**. With the shared rollout pool the runner keeps (the per-task
seed is method-independent), all methods over a task **select from the same pool** of agent
trajectories — the only thing that varies is the selector, not the prompt. This is enforced
offline in `tests/test_fair_comparison.py`.

## Layout

```
src/tts/
  abcs.py            EntityStore / RestrictedView / Agent / WorldModel / ShadowEvaluator / TaskEnvironment
  trajectory.py      Step / Trajectory / RewardInfo / RolloutResult
  worldmodel.py      ReadGroundedWorldModel (flat + recursive), budgeted/cached reads, spawn tree
  agents.py          ScriptedAgent (offline stub), LLMAgent (interactive adapter)
  llm/               ChatClient (openai/azure) + azure.py (litellm Azure-AAD env) for the real agents
  harness.py         run_decision: B0/B1/B2/Ours; real-agent rollouts -> select -> commit
  selection.py       success_score rankers + Selector
  commit.py          CommitChokepoint (single writer)
  budget.py cache.py config.py   kill-switch, read cache + selective invalidation, all knobs
  simulator/spawn_tree.py        parent->child / region / depth / confidence / tokens
  env/
    real_tau2.py        Tau2TaskEnv (official agent loop + AirlineShadowEvaluator + Tau2EntityStore)
    real_appworld.py    AppWorldTaskEnv (official ReAct agent + AppWorld.evaluate; real tasks)
    appworld_react_agent.py   the official AppWorld ReAct code-agent loop (verbatim prompt)
    prompts/            bundled official AppWorld ReAct prompt
    datasets.py         subset taxonomy + official sizes + build_dataset (+ dev/train local splits)
    registry.py         make_env_from_spec / make_env (all-tasks vs usable-task selection)
experiments/run.py   TTS curve vs N + depth sweep vs L + spawn-tree dump, per-subset, JSON/CSV
configs/             b0 b1 b2 ours_uncertainty ours_fixed base
tests/               single-commit, never-writes, privilege, recursion modes, cache/budget, TTS, datasets, real-env
```

## Two conda envs (the real benchmarks can't coexist)

AppWorld pins **pydantic v1**; τ²-bench needs **pydantic v2 + Python ≥3.12** — so they live in
separate environments.

```bash
# τ²-bench  (Python 3.12)
source /home/t-wendili/miniconda3/etc/profile.d/conda.sh && conda activate dw
python -m pytest

# AppWorld  (Python 3.11, needs APPWORLD_ROOT)
APPWORLD_ROOT=/home/t-wendili/env-tts PYTHONPATH=src \
  /home/t-wendili/miniconda3/envs/env/bin/python -m pytest
```

## Run

The real agent talks to the official benchmark model (default `gpt-4.1`). For the Azure setup
here: `export AZURE_OPENAI_ENDPOINT=...; export AZURE_OPENAI_API_VERSION=2024-12-01-preview;
export AZURE_AUTH_MODE=gcr` (an `az login` identity provides the bearer token).

```bash
# real τ²-bench B0 = official agent + user simulator (conda env dw)
python experiments/run.py --env tau2 --domains airline --methods b0 --N 1 --tasks 15

# real AppWorld B0 = official ReAct code agent, scored by AppWorld.evaluate (conda env env)
# all splits score locally; test_normal/test_challenge are the official benchmark splits
# --workers 8 runs 8 tasks in parallel (process pool) for a near-linear speed-up
APPWORLD_ROOT=/home/t-wendili/env-tts PYTHONPATH=src python experiments/run.py \
  --env appworld --splits test_normal,test_challenge --levels 1,2,3 --methods b0 --N 1 --workers 8

# read-grounded TTS methods (τ², offline scripted solver + world model)
python experiments/run.py --env tau2 --agent scripted --methods b1,b2,ours --tasks 12 --N 1,2,4,8

# recursion-depth curve / subset sizes / re-aggregate a saved run
python experiments/run.py --env tau2 --agent scripted --tasks 12 --depth-sweep
python experiments/run.py --env tau2 --list-subsets
python experiments/run.py --report experiments/results/records.jsonl
```

Key knobs: `--agent {llm,scripted}`, `--llm-agent`/`--llm-user`/`--temperature`/`--max-steps`,
`--workers` (parallel tasks — see below), `--all-tasks` (τ² official set vs mutating-only),
`--recursion-mode {uncertainty,fixed}`, `--tau`, `--L`, `--branching`, `--node-read-limit`,
`--hard-cap`, `--N`, `--success-score`, `--no-cache`, `--no-selective-invalidation`. Subset
selection: `--splits`/`--levels` (AppWorld — all splits are scored locally, default
`test_normal,test_challenge`), `--domains`/`--subsets` (τ²); per-subset metrics are reported
automatically.
A progress bar over processed tasks is shown on stderr (uses `tqdm` if installed, otherwise a
plain `k/total (pct)` counter).

### Parallel workers (`--workers`)

The dominant cost is the agent's LLM rollouts — each task is many sequential network calls, so
runs are I/O-bound. `--workers N` runs **N tasks in parallel** for a near-linear speed-up (each
task still runs all its B0/B1/B2/Ours cells together, with its own rollout pool, so the
fair-comparison guarantee is unchanged). AppWorld uses a **process pool** (it is not thread-safe:
thread-affine sqlite + a module-global ground-truth loader); τ² uses a lighter **thread pool**.
Measured: τ² B0 over 4 tasks 91 s → 43 s (`--workers 4`); AppWorld B0 over 4 tasks ~100 s → 30 s
(`--workers 4`). Verified identical results on the deterministic scripted path. Default is `1`
(sequential).

**Temp-file cleanup.** Every AppWorld task open writes an output-dir tree to disk (logs + an
on-disk copy of the task SQLite DBs). To keep large parallel runs from filling the disk, each
world's directory is deleted as soon as that task finishes (a `finally` around `close()`), and the
runner additionally namespaces a run's outputs with a unique base and sweeps any orphans left by
hard-killed workers at the end. Pass `--keep-outputs` (or set `TTS_KEEP_APPWORLD_OUTPUTS=1`) to
retain them for debugging. τ² writes nothing to disk.

**Rate limits (HTTP 429 "Too Many Requests").** Many parallel workers hammering one Azure/OpenAI
deployment can trip its rate limit. The agent's chat client (`tts.llm.chat_client.ChatClient`)
absorbs this automatically: the underlying SDK retries 429s honouring the server's `Retry-After`
hint, and the client adds an outer backoff-with-jitter backstop on a **dedicated** retry budget
(so throttling never consumes the normal error-retry allowance and a transient 429 doesn't fail
the rollout). Tunable via env vars: `TTS_LLM_RATE_LIMIT_RETRIES` (default `8`),
`TTS_LLM_RATE_LIMIT_BASE_DELAY` (`2`s), `TTS_LLM_RATE_LIMIT_MAX_DELAY` (`60`s), and
`TTS_LLM_RATE_LIMIT_JITTER` (`1`s). If you still see 429s, lower `--workers` or raise these.

## Representative results (real benchmarks)

**B0 reproduces the official benchmark** (single real official-agent rollout, `gpt-4.1`):

```
benchmark          B0 (measured)            official gpt-4.1
τ²-bench airline    Pass^1 = 0.533 (n=15)    ≈ 0.52–0.56
AppWorld           TGC   = 0.75   (dev/L1)   ≈ 0.732  (ReAct, test_normal)
```

τ² uses the whole official airline task set (`--all-tasks`, default for `--agent llm`). AppWorld
scores **every** split locally via `AppWorld.evaluate()` (`ground_truth_mode="minimal"`) — including
the official `test_normal` / `test_challenge` benchmark splits — so you can report the leaderboard
splits directly. Both land squarely on the official numbers, confirming the real agent + official
prompt are faithful.

The read-grounded TTS methods keep the clean separation **B0 ≤ B1 < Ours ≈ B2** (success vs `N`)
on τ² with the scripted solver + world model: Ours matches the forking oracle (B2) **without
forking**, while the flat baseline (B1) under-reads multi-entity trajectories and stays at the base
rate; the depth sweep shows success and cost both rising with `L`. (Numbers are produced by
`experiments/run.py`; exact values depend on `--tasks`, the model, and seeds.)

## Tests

```bash
python -m pytest          # dw: 41 passed, 3 skipped (appworld + LLM-gated B0 skipped)
# env: 39 passed, 5 skipped (set APPWORLD_ROOT; LLM-gated B0 runs when AZURE_OPENAI_ENDPOINT is set)
```

Covers: single-commit invariant; simulator-never-writes; privileged-vs-restricted access;
uncertainty stop rule (low coverage ⇒ spawn, high ⇒ not); fixed-depth = `L`; hard-cap; cache
hit/miss; budget kill-switch; TTS separation; **B1/B2/Ours share one agent-rollout pool (same
prompting strategy)**; dataset taxonomy; and guarded real-env tests (τ² official gold = 1.0;
recursive predicts better than flat; real τ²/AppWorld B0 via the official agent loop).

## Extending

Add a benchmark by implementing `TaskEnvironment` + `EntityStore` + `ShadowEvaluator` for it and
registering it in `env/registry.py`; the world model, recursion modes, selection, budget,
caching, harness, and `run.py` are all env-agnostic. Swap the `ScriptedAgent` for `LLMAgent`
(wire a model via `tts/llm`) to drive the official agent loop.
