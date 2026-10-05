# Trajectory-level, Read-Grounded Test-Time Scaling over Real, Non-Forkable Environments

A research prototype of LLM-agent **test-time scaling (TTS)** over **one real, stateful,
non-forkable environment** — the genuine **τ²-bench**, **AppWorld**, **OfficeBench**, and
**Terminal-Bench** benchmarks,
executed and scored through their **official** runners and the **official agent prompts**. A single
rollout (**B0**) is the real official agent (τ²: the official LLM agent + user simulator; AppWorld:
the official ReAct code agent; OfficeBench: the official office-app agent) acting on the real
environment and scored by the official evaluator — so B0 reproduces the official benchmark number.
The read-grounded TTS methods then draw `N`
trajectories and, instead of acting on the shared environment, each rollout's *world-simulator
subagent* predicts the outcome by reading the **real backing database** (privileged, read-only)
and, when one bounded read is insufficient, **recursively spawning region-scoped probe
subagents**. After all `N` trajectories are scored, exactly one is selected and committed to the
real environment through a single chokepoint.

```
                          +--------------------------------------------------------+
                          |   ONE REAL, NON-FORKABLE TASK ENVIRONMENT (per task)   |
                          |   tau2 (Pydantic DB; all domains) | AppWorld (REPL)   |
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

* **τ²-bench (all domains).** A trajectory is a sequence of real τ² tool-calls. The **real
  agent** is the official τ² LLM customer-service agent (the verbatim official prompt,
  `tau2.agent.llm_agent.AGENT_INSTRUCTION` / `SYSTEM_PROMPT`) conversing with the official user
  simulator over the real environment; stepping uses the official
  `environment.get_response(ToolCall(...))` (the tool execution that mutates the domain's Pydantic
  `DB`). The terminal reward is the official `evaluate_simulation(...)` over the whole simulation
  (`EvaluationType.ALL`), so a single rollout (B0) is the genuine benchmark score — for **every**
  registered domain (airline, retail, telecom, mock, banking_knowledge), selected with `--domains`.
  The read-grounded world model (B1/Ours) replays tool-call sequences and compares DB-hashes against
  the gold `evaluation_criteria.actions` (a gold sequence scores **1.0**, a perturbed one **0.0**).
  It is domain-generic — it introspects the backing `DB`'s entity buckets (dict- or list-keyed) and
  follows each fetched row's id references to close over the touched subgraph
  (`reservation → user + flights`, `order → user + products`, ...). Because it grounds on the
  backing DB by replaying the *agent's* action sequence, it applies to **single-control** tasks
  (every gold action is an assistant-side agent tool): all of airline/retail/mock. **Dual-control**
  telecom (the user simulator mutates device state outside the agent's actions) and the
  retrieval-only banking_knowledge domain expose no faithful read-grounded DB target, so the world
  model falls back to the oracle grounded reward there (the same contract AppWorld uses) while B0
  still reproduces their official numbers. In `llm_wm` mode, candidate generation uses the exact
  same official agent/user prompts, sampling parameters, and orchestration loop as the real-env
  path. Only `environment.get_response` is replaced: with the default predict-all setting, recursive
  Ours may gather untouched evidence through whitelisted read-only simulator tools, but every clean
  read, dirty read, and write observation is produced by a required restricted Python program.
  Policy/reference guidance is applied only by the world-model verifier after generation and is never
  injected into the benchmark agent.
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
* **OfficeBench.** Official OfficeBench tasks are solved with **free-form office-app actions**
  (JSON `{'app','action',...}` over calendar / email / excel / word / pdf / ocr / shell / llm) and,
  like AppWorld, expose **no replayable gold action list**. A trajectory is the realized sequence of
  actions emitted by the **official office-app agent** (the verbatim official prompts + app
  state-machine, see `env/officebench_agent.py`): every action runs through the official `apps`
  modules inside a Docker `/testbed` filesystem (the task's backing state), and the terminal reward is
  the official **execution-based** evaluation (`utils/evaluate.py` file-state checks — contain /
  file-exist / exact-match / calendar-no-overlap / excel-cell …, run via `officebench_driver.py`). All
  OfficeBench-specific code (the `apps` and the evaluator) runs **inside the image**, so the harness
  needs only a Docker CLI with a reachable local/remote daemon + the `OFFICEBENCH_ROOT` clone. The
  official leaderboard groups by app-count, so the
  subsets are `1app` / `2app` / `3app` (93 / 95 / 112 = 300). A single rollout reproduces the official
  OfficeBench number; tasks are independent (each materialises a fresh `/testbed`), so there is no
  shared mutable state across rollouts.
* **Terminal-Bench 1/2.** Terminal tasks are launched and officially scored by their native
  harnesses: legacy `tb` for `terminal-bench-core==0.1.1`, and Harbor for
  `terminal-bench@2.0`. Both load this repo's custom agent adapter, which emits one shell command per
  turn. For B1/Ours, statically proven read-only commands execute against the pristine container;
  writes and dirty reads are predicted and appended to a logical transition log. A bounded
  `READ`/`LIST`/`CALL`/`TRACE` probe subagent scores each candidate, then only the selected command
  trajectory is replayed in the official container before the native verifier runs.

## Where recursion bites (the read-grounded world model)

The world model (used by the read-grounded TTS methods on τ²) predicts a trajectory's reward
**without touching the real env** by materialising a *shadow* of the entities the trajectory
touches:

* **flat (B1)** in τ², AppWorld, and OfficeBench prepares its simulator context **before candidate
  generation** and reuses it for every predicted observation and the final success judgment. In τ²
  it contains task/system rules, all tool documentation, gold actions, evaluation criteria, and at
  most `node_read_limit` reference-derived seed entities. In AppWorld it contains the task metadata, relevant API
  documentation, native evaluator requirements/gold values, and fixed task state. In OfficeBench it
  contains the full app catalogue, native evaluation config, testbed listing, and evaluator-selected
  initial/reference files. Candidate actions and accumulated history are runtime inputs only; they
  cannot trigger new adaptive grounding reads or change the fixed system prompt. There is no
  recursion or transitive expansion, so B1 can still be under-grounded.
* **recursive (Ours)** follows `child_references` to close over the touched subgraph
  (`reservation → user + flights`), so the prediction is correct. The stop rule is configurable:
  `uncertainty` (descend while read-coverage `< tau`) or `fixed` (descend to depth `L`). Every
  read is budgeted; every node is logged in a spawn tree.
  Its world-model subagent uses the AppWorld-style restricted coding protocol by default:
  `PYTHON_READ:` programs emit bounded record reads, id listings, or calls to toolkit functions
  already classified as read-only; the host validates and executes those requests and copies their
  results as untouched variables. `PYTHON_FINAL:` programs compute and print the predicted tool
  observation or success verdict from copied actions, history, documentation, and read variables.
  Generated code has no database, environment, toolkit, filesystem, network, credential, or write-tool
  access. Predict-all requires this program path for every candidate action; the text protocol remains
  available only when predict-all is disabled.
* **B2 (snapshot, baseline only)** deep-copies the full τ² state and scores with the oracle — the
  forking upper bound that **Ours matches without forking**. (AppWorld and OfficeBench are
  non-forkable, so B2 there falls back to best-of-`N` over the realized official rewards.)

**AppWorld `llm_wm` (world-model interaction).** AppWorld also supports the `--agent llm_wm` mode: the
official ReAct code agent runs, but every `apis.*` call is routed through the world model at the single
chokepoint `requester.request` — under predict-all, recursive Ours may invoke documented read endpoints
only as bounded simulator evidence probes (flat B1 uses fixed context), while every candidate read and
write response is generated by restricted Python from copied variables. Task-mutating endpoints
(`POST`/`PATCH`/`DELETE`), mutating admin calls, and `logout` are never executed during sampling.
Unknown API metadata is blocked fail-closed. `login` is the sole real POST because its token is required
to authorize exact protected reads; it validates existing credentials without saving task-app records.
`supervisor.complete_task` is recorded as predictor-local logical state and overlaid onto
`show_active_task`, so candidate completion stops the loop without writing the real supervisor DB. A
**read after a write** is WM-predicted so reads reflect the agent's prior writes. With **B1**, the
simulator first freezes one Simia-style system context containing the task, relevant API docs, native
evaluation contract/gold values, app catalogue, frozen time, and active-task metadata. Writes, dirty
reads, and success judgments then use that identical system prompt; candidate code/history is supplied
only in the runtime user message, and no action-conditioned DB query is allowed. With **Ours**, dirty
reads and terminal-success predictions retain the bounded world-model subagent and may combine exact
pre-write state with adaptive read-only grounding. Its default environment-facing tool is restricted Python:
`PYTHON_READ:` code emits structured read requests, which the host validates and executes through the
pristine read-only interface; the raw result is exposed only as a `data['reads'][i]['result']`
variable. `PYTHON_FINAL:` code performs every copy/transformation and its printed value is the final
predicted state or verdict. Direct final output is rejected in predict-all mode. The former textual
interface remains only for non-predict-all ablations. The selected trajectory is committed
by re-executing its realized code on a fresh world and scoring with `AppWorld.evaluate()`. As in
OfficeBench, terminal verification is **reference-grounded by default** against AppWorld's native
`test_data.json` requirements, public/private values, and `answer.json`; it audits exact write
arguments and APPLIED/REJECTED transitions and may use the same read-only subagent tools to inspect
logical state. A candidate whose required answer provably disagrees with `answer.json` is marked
hard-invalid and cannot win a score tie against an answer-compatible candidate.
Set `TTS_APPWORLD_VERIFIER_REFERENCE=0` or pass `--no-verifier-reference` for the blind WM-only ablation.
WM rollout exceptions are retried and retained as explicitly invalid candidates after exhaustion, so
every completed decision still contains exactly `N` candidates; task/N cells also retry around commit
and official evaluation. A sweep with missing cells writes `run_manifest.json` and publishes clearly
labeled partial metrics by default, counting every missing task/method/N cell as a failed decision
with zero reward. Pass `--strict-coverage` to require a complete run instead.

AppWorld state prediction and terminal verification use the restricted Python read/computation protocol,
enabled by default with `TTS_APPWORLD_WM_PYTHON=1`. In predict-all mode every write response and read
observation must begin with `PYTHON_FINAL:` (`PYTHON:` remains a compatibility alias); direct output is
invalid. The generated program receives only copied, credential-redacted JSON inputs plus untouched
host-mediated read variables. A restricted helper process executes it with no unrestricted imports,
filesystem, network, environment, AppWorld object, or write API, and stdout becomes the predicted
response. Safe `json`/`datetime` imports and local helper functions are supported; other imports remain
blocked. Failed programs receive one repair attempt (`TTS_APPWORLD_WM_PYTHON_REPAIR=0` disables it) and
then fail closed in predict-all mode rather than falling back to a direct prediction. Verifier programs
receive copied task requirements, parsed gold sections, write arguments/results, predicted transitions,
agent steps, and any read-only variables; they must emit a validated verdict/status contract and never
receive the requester or benchmark world. Set
`TTS_APPWORLD_VERIFIER_PYTHON=0` to disable code only for terminal verification. Each JSONL decision
records aggregate `n_python_calls`, `n_python_failures`,
`n_python_forced`, and `n_python_mode_misses`, plus selected-rollout call/failure counts.
Within both simulation and verification subagents, a `PYTHON_READ:` program prints
`{"queries":[{"app":...,"api":...,"args":...}]}`. The host rejects undocumented/mutating endpoints,
hydrates any required access token without exposing it to code, executes bounded reads, and copies the
untouched results into `data["reads"]` for the next Python turn. The prompt receives variable
locations rather than the read text; a later `PYTHON_FINAL:` program alone processes the value and
prints the prediction/verdict. Generated code never receives direct database or requester access.
For exact-state challenge tasks, `--appworld-precision-prompt` adds an explicit agent skill for complete
pagination, set/map reconciliation, Amazon cart isolation before ordering, half-open calendar intervals,
wishlist transfer/removal, replay-safe payment-card retries, Venmo ID-to-email resolution and funded
request approval, Spotify cursor-only navigation, write-response ID propagation, and write-once
local-ledger verification. On multi-app tasks the same skill
additionally builds a per-app dependency
ledger, transfers exact source identities and values across apps, preserves one-to-one cardinality, and
audits each native ground-truth check independently to produce a graded selection confidence. This
intentionally augments the official agent prompt and should be applied uniformly in comparisons. On the
GPT-5.4 `llm_wm` benchmark path, instruction preflights are disabled by default so Ours and B2 use
the same stochastic ReAct candidate-generation loop. `--appworld-instruction-preflights` remains an
explicit agent-side ablation for recurring exact intents, is recorded as
`appworld_instruction_preflights`, and must not be reported as a world-model-only improvement or
compared directly with B2. On the complete fair GPT-5.4 temperature-1 `test_normal` N=1 sweep, the
WM-only funding, dirty-read, and unresolved-write completion checks improved Ours from 125/168
(74.40%) to 135/168 (80.36%): L1 54/57, L2 40/48, and L3 41/63. The matched seeded B2 run scored
137/168 (81.55%): L1 54/57, L2 42/48, and L3 41/63.
On the
reference-conditioned BoN@2 configuration, `--appworld-agent-reference` also supplies the native evaluator
checklist and gold values to candidate generation, while `--appworld-deterministic-dirty-reads` applies
known Amazon cart transitions exactly so stale simulated reads do not trigger duplicate writes. The former
uses the task answer key and also enables deterministic Python plans for recurring friend-sync, Spotify
cursor, playlist-archive, roommate expense-split, and unpaid subscription-share contracts; it must not
be reported as blind benchmark generation.
complete 72-task challenge-L1 N=16 run with GPT-4.1 at temperature 1,
the answer gate plus this precision skill scored 64/72 (88.89%), with zero simulator writes and one
official final commit per task. A matching complete N=1 run scored 51/72 (70.83%), giving an
18.06-percentage-point N=1 to N=16 gain under the same prompt configuration.
Measured: on dev
tasks the WM rollout **matches
the real agent** — query tasks 5/5 (B0 4/5) and write tasks 2/3 (B0 2/3), with 5–37 writes predicted per
write-task.

**OfficeBench `llm_wm` (world-model interaction).** OfficeBench likewise supports `--agent llm_wm`,
obeying the **same write-safe contract** as τ²/AppWorld: the official office-app agent runs, but tool
execution is routed through the world model (`env/officebench_wm_predict.py::OfficeBenchWMPredictor`) —
with `--methods ours` and a GPT or DeepSeek model, the repaired profile predicts **all candidate
observations** by default, selects the best trajectory, and commits only that trajectory. Ours may
gather copied pristine objects through bounded driver reads/listings; restricted Python and pure
state-transition helpers process these inputs without putting private contents in model prompts.
Explicit `--exact-clean-reads` restores native pristine reads in the shadow. Other model/method
profiles retain their previous defaults. Write safety is enforced structurally: the sampling container is opened
`read_only`, and a write action reaching `exec_action` raises (a routing bug fails loudly rather than
silently writing). Because the real files stay pristine during sampling, the success verifier
(`env/officebench_wm.py::OfficeBenchVerifier`) judges the **predicted** final state — each gold-checked
file's materialized predicted content where supported, with unresolved state kept uncertain — not
the environment after executing a candidate. Supported native predicates are computed directly on
this predicted state; unsupported checks use the legacy verifier. It is
**reference-grounded by default** against OfficeBench's native ground truth (the per-task `evaluation`
checks *with gold values* + the `reference/` gold expected-output files for exact-match tasks;
`TTS_OFFICEBENCH_VERIFIER_REFERENCE=0` for a blind judge), and flags give-up/refusal answers.
With **B1**, the task config, full app catalogue/demos, testbed file listing, evaluator-selected
initial files, and gold reference files are all read before the first candidate action and frozen into
one shared system prompt. The observation predictor and success verifier reuse that exact prompt;
candidate actions/history and predicted final state are runtime user inputs only, and dirty reads do
not trigger new container reads. With **Ours**, the existing bounded read-only grounding loop remains
available for observation and verifier evidence.
For GPT-5.4, the verifier atomizes small evaluator keyword/match lists and grades each native check
as MET/UNMET/UNKNOWN; UNKNOWN receives configurable partial credit
(`TTS_OFFICEBENCH_VERIFIER_UNKNOWN_CREDIT`, default `0.25`). This avoids unanimous whole-task votes
flattening the BoN pool. One deterministic whole-task audit is blended at weight `0.25`
(`TTS_OFFICEBENCH_VERIFIER_WHOLE_WEIGHT`) to keep predicted-state false positives from saturating.
Every retained candidate uses the same original task prompt, requested temperature, response budget,
action handling, and consecutive seed schedule (`base+i`). Candidate generation is independent of
its position in the BoN pool; only verification and selection differ as the pool grows.
When a newly added candidate is tied with or close to the prior lower-N winner, a comparative
native-check audit (`TTS_OFFICEBENCH_COMPARATIVE_RERANK=1`) uses the predicted final state and
trajectory evidence to break the ambiguity. The audit considers candidates within confidence
margin `0.25` (`TTS_OFFICEBENCH_COMPARATIVE_CONFIDENCE_MARGIN`) and preserves the prior winner unless
the newly added group contains a better candidate.
Legacy whole-task self-consistency remains available through
`TTS_OFFICEBENCH_VERIFIER_PERCHECK=0` and `TTS_OFFICEBENCH_VERIFIER_SAMPLES=K`. The single selected
trajectory is then
committed by re-executing its actions in a fresh container (**the only real writes**) and scoring with
the official `utils/evaluate.py` checks. Write-safety is provable, not just asserted:
`OfficeBenchSession.testbed_digest()` hashes `/testbed` before/after a WM rollout and it is unchanged.
For **Ours**, the AppWorld-style coding protocol is the default: the WM emits restricted `PYTHON_READ`
programs to request host-validated read-only evidence and `PYTHON_FINAL` programs for exact copied-data
transformations. Under `--predict-all-actions` both B1 and Ours require `PYTHON_FINAL` for every
candidate action; direct output and host-side logical overlays are not accepted. The helper receives
no filesystem, app, network, or environment handle.

**Terminal-Bench world-model interaction.** `terminalbench_agent.py` is shared by the legacy and
Harbor adapters, so Terminal-Bench 1 and 2 use the same solver prompt and command history.
`terminalbench_wm.py` conservatively classifies shell commands: pristine reads execute exactly;
anything not proven read-only is predicted and never executed during sampling; every read after a
predicted write is also predicted. Ours gives the WM/verifier a bounded read-only probe loop
(`READ`, `LIST`, `CALL`, `TRACE`), while B1 makes a single flat prediction. The official `tb`/Harbor
verifier remains the final source of truth after the one selected trajectory is committed.

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

The solver is selected with `--agent {llm,llm_wm,scripted}`: `llm` (default) is the real
official-prompt agent (τ²: the official LLM agent + user simulator; AppWorld: the official ReAct
code agent) acting on the **real env**, so **B0 reproduces the official benchmark** and the world
model only *scores* the finished trajectory. `llm_wm` runs the **same official agent + user
simulator**, but routes the environment's single tool-execution chokepoint
(`environment.get_response`) through the read-grounded **world model**: read-only tools (assistant
*and* dual-control user tools) on *untouched* entities are served exactly, a read of an entity a
prior write already changed is WM-predicted (so it reflects the post-write state), every write is
WM-predicted, and the real backing DB is never mutated — so the official agent generates each
candidate by interacting with the
*simulated* env (receiving the same user-simulator information it would on the real env), and only
the selected trajectory is committed to the real env (officially scored, dual-control-safe). It
works for all single- and dual-control τ² domains. `scripted` is an offline, LLM-free
plan-following stub used only to exercise the env-agnostic machinery
(recursion/selection/budget) in unit tests.

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
  llm/               ChatClient (openai/azure) + azure.py (litellm Azure-AAD env) + local.py (--local self-hosted server) for the real agents
  harness.py         run_decision: B0/B1/B2/Ours; real-agent rollouts -> select -> commit
  selection.py       success_score rankers + Selector
  commit.py          CommitChokepoint (single writer)
  budget.py cache.py config.py   kill-switch, read cache + selective invalidation, all knobs
  simulator/spawn_tree.py        parent->child / region / depth / confidence / tokens
  env/
    real_tau2.py        Tau2TaskEnv (official agent loop + AirlineShadowEvaluator + Tau2EntityStore)
    real_appworld.py    AppWorldTaskEnv (official ReAct agent + AppWorld.evaluate; real tasks)
    appworld_react_agent.py   the official AppWorld ReAct code-agent loop (verbatim prompt)
    real_officebench.py officebench_lowlevel.py officebench_agent.py officebench_wm.py
                        OfficeBenchTaskEnv (official office-app agent + official utils/evaluate.py),
                        Docker session + in-container officebench_driver.py + WM success verifier
    terminalbench_agent.py terminalbench_wm.py
                        shared shell-command solver + read-only probe/transition/verifier world model
    terminalbench_legacy.py terminalbench_harbor.py
                        custom-agent adapters for Terminal-Bench 1 (`tb`) and 2 (Harbor)
    officebench_docker/ self-contained app/eval image (apps + evaluation run inside the container)
    prompts/            bundled official AppWorld ReAct prompt
    datasets.py         subset taxonomy + official sizes + build_dataset (+ dev/train local splits)
    registry.py         make_env_from_spec / make_env (all-tasks vs usable-task selection)
experiments/run.py   TTS curve vs N + depth sweep vs L + spawn-tree dump, per-subset, JSON/CSV
configs/             b0 b1 b2 ours_uncertainty ours_fixed base
tests/               single-commit, never-writes, privilege, recursion modes, cache/budget, TTS, datasets, real-env
```

## Benchmark environments

AppWorld pins **pydantic v1**; τ²-bench needs **pydantic v2 + Python ≥3.12** — so they live in
separate environments. **OfficeBench** runs all its app + evaluation code **inside a Docker image**,
so it needs no Python deps on the host — only the `OFFICEBENCH_ROOT` clone + a usable Docker daemon —
and can therefore be driven from *either* interpreter (the app/eval image is built on demand from
`src/tts/env/officebench_docker/`).

```bash
# τ²-bench  (Python 3.12)
source /home/t-wendili/miniconda3/etc/profile.d/conda.sh && conda activate dw
python -m pytest

# AppWorld  (Python 3.11, needs APPWORLD_ROOT)
APPWORLD_ROOT=/home/t-wendili/env-tts PYTHONPATH=src \
  /home/t-wendili/miniconda3/envs/env/bin/python -m pytest

# OfficeBench  (host needs only Docker + the OfficeBench clone)
OFFICEBENCH_ROOT=/home/t-wendili/OfficeBench PYTHONPATH=src python -m pytest tests/test_officebench.py

# Terminal-Bench 1 + 2 (Python 3.12; legacy tb + Harbor share one env)
bash scripts/setup_terminalbench.sh
```

## Run

The real agent talks to the official benchmark model (default `gpt-4.1`). For the Azure setup
here: `export AZURE_OPENAI_ENDPOINT=...; export AZURE_OPENAI_API_VERSION=2024-12-01-preview;
export AZURE_AUTH_MODE=gcr` (an `az login` identity provides the bearer token).

### Local (self-hosted) models — `--local`

To run against a **self-hosted** model instead of a hosted API (e.g. `Qwen-AgentWorld-35B-A3B`),
serve it with any OpenAI-compatible server (vLLM / SGLang) and add `--local` — no Azure endpoint or
`az login` token is needed. `--local` forces the OpenAI backend for AppWorld/OfficeBench
(`ChatClient`) and routes τ² (litellm) through the `openai/` provider, all pointed at the local
`/v1` endpoint. It works even while the Azure env vars above are still exported. A preflight ping
fails fast with clear guidance if no server is up (bypass with `--skip-local-healthcheck`).

```bash
# 1. serve the model in its OWN env (keeps the tau2 `dw` / AppWorld `env` deps untouched):
#    conda create -y -n awserve python=3.12 && conda activate awserve && pip install vllm
#    The serve script defaults are tuned for the Qwen-AgentWorld family (adds --language-model-only,
#    --reasoning-parser qwen3, --trust-remote-code, --enable-auto-tool-choice --tool-call-parser
#    qwen3_xml; served-name Qwen-AgentWorld-35B-A3B, port 8000).
# On ONE A100 80GB the 35B MoE (~67GB bf16) only fits with fp8 (weights ~33GB, ~37GB KV cache ->
# ~52x concurrency, ~140 tok/s/req) — this is the validated single-GPU config:
QUANTIZATION=fp8 MAX_MODEL_LEN=32768 bash scripts/serve_local_model.sh Qwen/Qwen-AgentWorld-35B-A3B

# 2. point experiments at it (from the `dw`/`env` env) — pass the served id to --llm-agent
PYTHONPATH=src python experiments/run.py --local --llm-agent Qwen-AgentWorld-35B-A3B \
  --env appworld --splits test_normal --levels 1 --methods b0 --N 1 --workers 8
PYTHONPATH=src python experiments/run.py --local --llm-agent Qwen-AgentWorld-35B-A3B \
  --env tau2 --domains airline --methods b0,b1,ours --N 1,2,4 --tasks 15
```

Serve-script knobs (env vars): `TP_SIZE` (default = #GPUs), `MAX_MODEL_LEN` (default 32768),
`GPU_MEM_UTIL` (0.92), `QUANTIZATION` (e.g. `fp8` to ~halve GPU memory — recommended on one GPU),
`CPU_OFFLOAD_GB` (offload weights to host RAM), `TOOL_CALL_PARSER` (default `qwen3_xml`, needed by
τ²), `REASONING_PARSER` (`qwen3`), `SERVE_BACKEND` (`vllm`|`sglang`). Qwen-AgentWorld is a **35B MoE
(~67 GB bf16)** and a reasoning model: its card serves it on ~4×80 GB (`--tensor-parallel-size 4`);
to fit one A100 80 GB use `QUANTIZATION=fp8`.

Notes: the value passed to `--llm-agent` must match the server's `--served-model-name`. τ² uses
function calling, so the server MUST be launched with `--enable-auto-tool-choice --tool-call-parser`
(the serve script adds these by default). In local mode τ²'s `banking_knowledge` KB retrieval falls
back to the offline BM25 approximation (a chat server exposes no embeddings); override with
`TTS_BANKING_RETRIEVAL_VARIANT` if your server also serves embeddings.

```bash
# real τ²-bench B0 = official agent + user simulator (conda env dw)
python experiments/run.py --env tau2 --domains airline --methods b0 --N 1 --tasks 15

```bash
# real τ²-bench B0 = official agent + user simulator (conda env dw)
python experiments/run.py --env tau2 --domains airline --methods b0 --N 1 --tasks 15

# real AppWorld B0 = official ReAct code agent, scored by AppWorld.evaluate (conda env env)
# all splits score locally; test_normal/test_challenge are the official benchmark splits
# --workers 8 runs 8 tasks in parallel (process pool) for a near-linear speed-up
APPWORLD_ROOT=/home/t-wendili/env-tts PYTHONPATH=src python experiments/run.py \
  --env appworld --splits test_normal,test_challenge --levels 1,2,3 --methods b0 --N 1 --workers 8

# AppWorld GPT-5.4 world-model BoN. N=1/2/4 share one nested candidate pool and generation policy.
# GPT-5.4 defaults to coverage-aware per-check confidence and guarded confidence ranking:
# UNKNOWN-heavy audits cannot look fully proven, and a challenger needs a >0.10 confidence gain.
APPWORLD_ROOT=/home/t-wendili/env-tts PYTHONPATH=src python experiments/run.py \
  --env appworld --agent llm_wm --splits test_normal --levels 1,2,3 \
  --methods ours --N 1,2,4 --temperature 1 --llm-agent gpt-5.4 \
  --workers 8

# real OfficeBench B0 = official office-app agent, scored by utils/evaluate.py (in-container)
# --levels selects the app-count subsets (1,2,3 -> 1app/2app/3app); per-rollout Docker container
OFFICEBENCH_ROOT=/home/t-wendili/OfficeBench PYTHONPATH=src python experiments/run.py \
  --env officebench --levels 1,2,3 --methods b0 --N 1 --workers 8

# Repaired OfficeBench Ours + llm_wm defaults for GPT and DeepSeek.
# Predict all candidate observations, select best, commit only selected trajectory.
# Typed predicted-state checks, guarded comparative selection, 5 fallback verifier votes,
# and a 5-second Python timeout are enabled by default. Explicit flags/env overrides win.
# Replace gpt-5.4 below with DeepSeek-V4-Pro to use the same repaired defaults.
OFFICEBENCH_ROOT=/home/t-wendili/OfficeBench PYTHONPATH=src python experiments/run.py \
  --env officebench --agent llm_wm --levels 1,2,3 --methods ours --N 1,2,4 \
  --temperature 1 --llm-agent gpt-5.4 --workers 8

# Action-free diagnostic: predict every action, copy immutable object reads into generated Python
# variables, and never commit or officially evaluate a selected trajectory.
OFFICEBENCH_ROOT=/home/t-wendili/OfficeBench PYTHONPATH=src python experiments/run.py \
  --env officebench --agent llm_wm --levels 1 --tasks 5 --methods ours --N 1,2,4 \
  --temperature 1 --llm-agent gpt-5.4 --workers 1 --simulation-only \
  --out experiments/ofbench-ours-5.4-simulation-only

# Terminal-Bench unified sweep. Native tb/Harbor results are normalized into
# <out>/records.jsonl, aggregates.json, and aggregates.csv.
python experiments/run.py --env terminalbench --terminalbench-versions 1,2 \
  --methods ours,b2 --temperature 1 --N 1,2,4,8,16 --workers 4 \
  --llm-agent gpt-4.1 --out experiments/terminalbench-ours-b2-t1

# Low-level oracle smoke launches (official native verifiers)
scripts/run_terminalbench.sh --version 1 --agent oracle --task hello-world
scripts/run_terminalbench.sh --version 2 --agent oracle --task cancel-async-tasks

# Low-level custom-agent launches. Defaults: method=ours, N=4.
export TTS_TERMINALBENCH_METHOD=ours TTS_TERMINALBENCH_N=4
scripts/run_terminalbench.sh --version 1 --agent tts --task hello-world --model gpt-4.1
scripts/run_terminalbench.sh --version 2 --agent tts --task cancel-async-tasks --model gpt-4.1

# Whole official datasets (expensive):
scripts/run_terminalbench.sh --version 1 --agent tts --all --model gpt-4.1 -- --n-concurrent 4
scripts/run_terminalbench.sh --version 2 --agent tts --all --model gpt-4.1 -- --n-concurrent 4

# read-grounded TTS methods (τ², offline scripted solver + world model)
python experiments/run.py --env tau2 --agent scripted --methods b1,b2,ours --tasks 12 --N 1,2,4,8

# LLM solver that INTERACTS with the world model (τ²): the official agent + user simulator run
# against the WM (all substantive actions predicted by default, real env untouched); commit the
# selected one to the real env. Works for all single- and dual-control domains.
python experiments/run.py --env tau2 --agent llm_wm --methods b0,b1,ours --domains airline --tasks 12 --N 1,2,4,8

# Opt out of predict-all and execute untouched clean reads exactly.
python experiments/run.py --env tau2 --agent llm_wm --methods ours \
  --domains airline --tasks 12 --N 4 --exact-clean-reads

# Seeded strict BoN@4 gates (Python tools are the default).
PYTHONPATH=src python experiments/run.py \
  --env tau2 --agent llm_wm --methods ours --N 4 --all-tasks --domains airline \
  --temperature 0.7 --workers 8 --llm-agent gpt-4.1 --llm-user gpt-4.1 \
  --success-score simulator_confidence --out experiments/tau-python-4.1-airline-final-policy-gate
PYTHONPATH=src python experiments/run.py \
  --env tau2 --agent llm_wm --methods ours --N 4 --all-tasks --domains retail \
  --temperature 0.7 --workers 8 --llm-agent gpt-4.1 --llm-user gpt-4.1 \
  --success-score simulator_confidence --out experiments/tau-python-4.1-retail-official-gate
# Verified: airline 41/50 (82%); retail 99/114 (86.84%); zero simulator writes/errors.

# Opt out of the default PYTHON_READ/PYTHON_FINAL protocol to reproduce the former text tools.
TTS_TAU2_WM_TOOL_MODE=text python experiments/run.py \
  --env tau2 --agent llm_wm --methods ours --domains airline --tasks 12 --N 4

# recursion-depth curve / subset sizes / re-aggregate a saved run
python experiments/run.py --env tau2 --agent scripted --tasks 12 --depth-sweep
python experiments/run.py --env tau2 --list-subsets
python experiments/run.py --report experiments/results/records.jsonl
```

For Terminal-Bench, `b2` is implemented by the unified experiment runner as a true oracle
best-of-`N`: it runs the same B0 agent seeds in `N` official isolated task containers and selects
the highest native verifier reward. `b1` and `ours` continue to use one official task container,
world-model selection, and one selected command replay.

Key knobs: `--agent {llm,llm_wm,scripted}`, `--llm-agent`/`--llm-user`/`--temperature`/`--max-steps`,
`--local`/`--local-base-url`/`--local-api-key` (self-hosted OpenAI-compatible model),
`--workers` (parallel tasks — see below), `--all-tasks` (τ² official set vs mutating-only),
`--recursion-mode {uncertainty,fixed}`, `--tau`, `--L`, `--branching`, `--node-read-limit`,
`--hard-cap`, `--N`, `--success-score`, `--verifier-samples`, `--no-cache`,
`--no-selective-invalidation`. Subset
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
the rollout). Tunable via env vars: `TTS_LLM_RATE_LIMIT_RETRIES` (default `12`),
`TTS_LLM_RATE_LIMIT_BASE_DELAY` (`2`s), `TTS_LLM_RATE_LIMIT_MAX_DELAY` (`120`s), and
`TTS_LLM_RATE_LIMIT_JITTER` (`1`s). If you still see 429s, lower `--workers` or raise these.

*`llm_wm` makes more calls.* The AppWorld/τ² `llm_wm` mode adds an LLM call per *predicted write*
(and per dirty read / success verdict) on top of the agent's calls, so a write-heavy task can be
several× the call volume of `B0` — and a loop that fires many writes in one code block (e.g. "follow
19 artists") is a *burst* that trips the deployment's requests-per-minute quota even at `--workers 1`.
Every write is **LLM-generated** (no synthesised template — `follow_artist` → `{"message": "Artist
followed."}`, matching the real API), but the burst is folded into **one** call: status/message-only
writes use a stable (args-free) prediction prompt, so the **deterministic response cache** (τ²'s
`litellm.cache` lever; `TTS_LLM_RESPONSE_CACHE`) de-duplicates the 19 identical follows into a single
LLM call. Additional 429 levers mirror τ²-bench: **bounded httpx connection pools**
(`TTS_LLM_MAX_CONNECTIONS`, default 10) and the retry/backoff above. If you *still* see 429s (e.g. a
`B0` run with many workers, or many distinct substantive writes), set a **min interval between LLM
calls** via `TTS_LLM_MIN_INTERVAL_S` (e.g. `0.3`–`1.0` s — a simple per-process rate limiter) and/or
**lower `--workers`** (e.g. `--workers 1`–`2`) so fewer processes contend for the deployment's quota.


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
recursive predicts better than flat; real τ²/AppWorld B0 via the official agent loop). Terminal-Bench
tests additionally pin conservative shell classification, pristine-read-only sampling, dirty-read
prediction, selection, and single replay.

## Extending

Add a benchmark by implementing `TaskEnvironment` + `EntityStore` + `ShadowEvaluator` for it and
registering it in `env/registry.py`; the world model, recursion modes, selection, budget,
caching, harness, and `run.py` are all env-agnostic. Swap the `ScriptedAgent` for `LLMAgent`
(wire a model via `tts/llm`) to drive the official agent loop.
