# Agentic World Modeling (AWM)

Implementation of **Agentic World Modeling for Agent Test-Time Scaling Without
Environment Replica**, accompanying [the paper](AWM__ICLR_2027_%20%2815%29.pdf).

AWM predicts the observation produced by a hypothetical agent action through
multi-turn, read-only evidence acquisition. A simulator subagent inspects relevant
records, uses restricted Python to compute an observation, and returns a compact
result to the task agent. Candidate trajectories do not apply their hypothetical
mutations to the shared environment. The simulator scores completed candidates;
the selected trajectory is replayed and evaluated in the real benchmark.

## Methods in this release

| Paper method | `--method` | `--agent` | Candidate execution and selection |
| --- | --- | --- | --- |
| Real-environment BoN ceiling | `b2` | `llm` | Execute each candidate independently in the real benchmark; select using native evaluation. This is the reference ceiling and requires isolated executions. |
| AWM | `ours` | `llm_wm` | Generate candidates through the read-grounded simulator; select using predicted outcomes, then replay one candidate. |
| WM-Training / Qwen-AgentWorld | `ours` | `agent_world` | Use the trained Qwen-AgentWorld model as the simulator, then replay the selected candidate. Qwen backend only. |

`--agent` defaults to `llm` for `b2` and `llm_wm` for `ours`. Other combinations
are rejected. `N=1` supplies the single-rollout comparison without a separate
method. `--methods` remains an alias for `--method`, but accepts only one method
per invocation; use separate output directories for the three modes.

The paper also compares against SIMIA. That baseline is **not included in this
cleaned release**: the old B1 fixed-context path was removed. B0, Terminal-Bench,
trajectory self-aggregation, and historical tuning flags are also removed.

## Benchmarks and models

| Paper benchmark | CLI name | Evaluation subsets | Paper task count |
| --- | --- | --- | --- |
| τ-bench | `tau2` | Airline / Retail | 50 / 114 |
| AppWorld | `appworld` | Levels 1 / 2 / 3, pooled across test splits | 129 / 198 / 258 |
| OfficeBench | `officebench` | One / two / three applications | 93 / 95 / 112 |

The customer-service adapter uses the **tau2-bench package and registry**. The
paper calls this evaluation τ-bench; the implementation's environment name is
`tau2`. Use the intended benchmark installation and task version when comparing
with the paper. Task discovery reads the installed benchmark data.

The model backbones are GPT-5.4, DeepSeek-V4-Pro, and Qwen3.5-35B-A3B. For the
trained baseline, keep Qwen3.5-35B-A3B as the task agent and serve
Qwen-AgentWorld-35B-A3B at the separate world-model endpoint.

## Repository layout

```text
experiments/run.py          Main runner; routes Qwen solver models automatically
experiments/run_qwen.py     Explicit Qwen entrypoint, including custom model aliases
src/awm/cli.py              Shared paper CLI and method validation
src/awm/profiles.py         Fixed backend runtime defaults
src/tts/                   GPT-5.4 / DeepSeek implementation
src/tts_qwen/              Imported Qwen implementation and provenance
  env/                    Benchmark adapters, simulators, and read-only tools
  llm/                    Model clients and restricted Python execution
  harness.py              Candidate sampling, selection, and commit orchestration
configs/                   Typed-library examples for AWM and the BoN ceiling
scripts/                   Local model serving and Qwen AppWorld launcher
aml/                       Reusable Azure ML jobs, environments, and launch helpers
tests/                     CLI, simulator, isolation, selection, and adapter tests
```

Both backends use the same public arguments. Qwen's benchmark-specific simulation
and repair logic remains separate from GPT/DeepSeek's. A solver name containing
`qwen` selects `tts_qwen`; `--agent agent_world` also selects that backend.
For a custom Qwen served name, invoke `experiments/run_qwen.py` directly. A Qwen
user-simulator name alone does not redirect a GPT/DeepSeek solver.

Historical research outputs under `experiments/`, data, figures, and other
research artifacts are retained. They are not executable configurations for the
cleaned CLI. `src/tts_qwen/SOURCE.json` records the original import's provenance;
its hashes describe the source before this paper-focused refactor.

## Installation

Use separate Python environments for benchmarks with incompatible dependencies.
Install this project into each environment that runs experiments:

```bash
python -m pip install -e '.[dev,llm,progress]'
```

- **AppWorld:** use its supported Python environment, install the pinned
  `appworld` extra (`python -m pip install -e '.[appworld]'`), and install/download
  its benchmark data. Set `APPWORLD_ROOT` to the prepared AppWorld root.
- **τ-bench adapter:** install your tau2-bench checkout into its supported Python
  environment with `python -m pip install -e /path/to/tau2-bench`. Configure its
  dataset location as required by that checkout. Empty `tau2-bench/` or
  `tau-bench/` directories in this repository are not installations.
- **OfficeBench:** set `OFFICEBENCH_ROOT` to a checkout containing `tasks/`,
  `apps/`, and `utils/evaluate.py`. The default backend requires Docker. The
  application image builds from `src/tts/env/officebench_docker/` or its Qwen
  counterpart. AML also supports a prepared local backend; see `aml/environment/`.

List benchmark subsets before launching a full run:

```bash
python experiments/run.py --env appworld --list-subsets
python experiments/run.py --env tau2 --list-subsets
python experiments/run.py --env officebench --list-subsets
```

## Model endpoints

For Azure-hosted deployments, configure the existing client with your endpoint,
deployment name, and authentication. For API-key authentication:

```bash
export AZURE_OPENAI_ENDPOINT='https://YOUR-RESOURCE.openai.azure.com/'
export AZURE_OPENAI_API_VERSION='2024-12-01-preview'
export AZURE_AUTH_MODE=api_key
export WM_MODEL=gpt-5.4
# Set AZURE_OPENAI_API_KEY through your normal secret-management mechanism.
```

`--llm-agent` overrides the solver model/deployment name. Use the deployment name
for your GPT-5.4 or DeepSeek-V4-Pro endpoint. An OpenAI-compatible endpoint can
instead be selected with `--local --local-base-url URL`, including remote servers.
Its key comes from `OPENAI_API_KEY` unless explicitly overridden.

For Qwen, start a compatible vLLM server separately. The repository includes
`scripts/serve_local_model.sh`; serving dependencies and model weights are not
installed by the experiment runner. The examples below assume the task model
is available at `http://localhost:8000/v1` and the trained world model at
`http://localhost:8001/v1`.

## Run experiments

### AWM: GPT-5.4 or DeepSeek-V4-Pro

```bash
python experiments/run.py \
  --env appworld --method ours --agent llm_wm \
  --llm-agent gpt-5.4 --N 1,2,4,8,16 \
  --temperature 1.0 --max-steps 40 --workers 8 \
  --out experiments/results/gpt54_appworld_awm
```

Replace `gpt-5.4` with your DeepSeek deployment name for DeepSeek-V4-Pro.
Use `--env tau2 --domains airline,retail` for the customer-service tasks, or
`--env officebench --levels 1,2,3` for office automation.

### Real-environment BoN ceiling

```bash
python experiments/run.py \
  --env appworld --method b2 --agent llm \
  --llm-agent gpt-5.4 --N 1,2,4,8,16 \
  --temperature 1.0 --max-steps 40 --workers 8 \
  --out experiments/results/gpt54_appworld_ceiling
```

Unlike AWM, this mode executes **every candidate** in an isolated benchmark
execution and uses real evaluation to select the best. It is an experimental
reference ceiling, not a replica-free deployment method.

### AWM: Qwen

```bash
python experiments/run_qwen.py \
  --env appworld --method ours --agent llm_wm \
  --local --local-base-url http://localhost:8000/v1 \
  --llm-agent Qwen3.5-35B-A3B --N 1,2,4,8,16 \
  --temperature 1.0 --max-steps 40 --workers 8 \
  --out experiments/results/qwen_appworld_awm
```

`scripts/run_qwen_appworld.sh` is a convenience launcher for this mode.
To run the Qwen ceiling, change the pair to `--method b2 --agent llm`.

### Qwen-AgentWorld trained baseline

```bash
python experiments/run_qwen.py \
  --env appworld --method ours --agent agent_world \
  --local --local-base-url http://localhost:8000/v1 \
  --llm-agent Qwen3.5-35B-A3B \
  --wm-model Qwen-AgentWorld-35B-A3B \
  --wm-base-url http://localhost:8001/v1 \
  --N 1,2,4,8,16 --temperature 1.0 --max-steps 40 --workers 8 \
  --out experiments/results/qwen_appworld_agentworld
```

The bundled MCP system prompt is selected automatically.
`--wm-system-prompt-file` overrides it; `AWM_WM_API_KEY` supplies the separate
world-model credential. The same mode is available for `tau2` and `officebench`.

## Evaluation protocol and outputs

The CLI defaults to the paper's rollout budgets `1,2,4,8,16` and solver sampling
temperature `1.0`. It selects both AppWorld test splits and all three levels,
both customer-service domains, and all three OfficeBench application counts.
Omitting `--tasks` evaluates the full selected dataset; `--tasks 2 --N 1` is a
small smoke run, with two tasks **per subset**. Backend prediction loops use a
six-turn read budget by default. Model and server support may constrain sampling
settings; the CLI does not claim to override unsupported provider parameters.

Use matching models, task selections, temperatures, step limits, and seeds across
comparisons. Run only one method/agent pair per output directory. Each run writes:

- `records.jsonl`: task/rollout-budget records with native success, reward,
  simulator scores, timing, token accounting, and available read/commit telemetry.
- `aggregates.json` and `aggregates.csv`: aggregate success and cost metrics.
- Coverage/checkpoint artifacts used to detect missing cells and resume runs.

For checkpoint compatibility, the trained baseline retains the internal record
method `agentworld`. New records also include `paper_method` (`real_env_bon`,
`awm`, or `qwen_agentworld`) and `paper_agent`. A simulator confidence score is
not the reported benchmark success rate. Failed/missing cells are tracked;
full coverage is required for a successful run exit.

```bash
# Add --resume to the same invocation and output directory after interruption.
python experiments/run.py --report experiments/results/gpt54_appworld_awm/records.jsonl
```

The runtime profiles preserve the integrated implementations' benchmark-specific
verification and repair settings, including reference-grounded verification in
some GPT/DeepSeek paths and the Qwen AppWorld profile. Those details matter for
reproduction: this cleanup does not establish that fresh runs reproduce every
paper table or that all backends use identical verifier information.

## Ablations and scope

`--tool-mode text` selects the existing without-Python prediction path; the default
is `python`. The paper's simulator-side scaling parameter **M** counts independent
predictions of each transition. This CLI does not expose an M sweep. The deleted
trajectory self-aggregation option combined complete trajectories and was not an
implementation of that experiment. The non-adaptive and without-subagent studies
are also not exposed as supported launch modes here.

## Tests

```bash
python -m pytest tests/test_paper_cli.py tests/test_qwen_routing.py
python -m pytest
```

Unit tests can run without model inference. Benchmark integration tests require
the relevant installations and data. Some existing tests fail without tau2-bench;
the integrated GPT/DeepSeek code also has pre-existing AppWorld write-predictor
and OfficeBench verifier failures. These should not be mistaken for successful
end-to-end reproduction. No model evaluations were run as part of the cleanup.
