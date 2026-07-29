# Running `dw` on Azure ML

This folder adds everything needed to run the `dw` test-time-scaling harness
(`experiments/run.py`) on **real AppWorld**, **tau2-bench**, or
**OfficeBench** on the `M365Research` Azure ML workspace. **No core repo code is
changed** — only this `aml/` folder plus the repo-root `.amlignore`.

There is one job file per benchmark (they need different conda envs / inputs);
all share the runtime wrapper `aml/src/run_dw.sh`.

## Files

| File | Purpose |
|------|---------|
| `job.yml` | AML command job for **AppWorld** (env `conda.appworld.yml`). |
| `job.appworld.wm.yml` | Dedicated **AppWorld interactive-WM sweep**: GPT-5.4, temperature 1, `N=1,2,4,8,16`. |
| `job.appworld.bon2.yml` | Complete **AppWorld Python-only multi-app BoN@2 gate**: GPT-4.1, L2/L3, graded per-check selection, strict requested thresholds. |
| `job.appworld.bon4.yml` | Complete **AppWorld Python-only WM BoN@4 gate**: GPT-4.1, all six split/level subsets, strict requested thresholds. |
| `job.tau2.yml` | AML command job for **tau2-bench** (env `conda.yml`). |
| `job.qwen.tau2.yml` | Single-node **Qwen + tau2** job: starts vLLM and runs the harness on one 80GB GPU. |
| `job.qwen.appworld.yml` | **Qwen + AppWorld** job configured for BF16 tensor/expert parallelism on two H100s. |
| `job.officebench.yml` | AML command job for **OfficeBench** using a remote Docker VM over SSH. |
| `src/run_dw.sh` | Runtime wrapper on the compute node: sets Azure auth, prepares the benchmark, runs `experiments/run.py`, saves a downloadable summary + log. |
| `src/provision_officebench_docker.py` | Creates/reuses the OfficeBench Docker VM, verifies SSH/Docker, stores credentials in the workspace Key Vault, and grants the AML identity access. |
| `src/hydrate_appworld_bundles.py` | Replaces AppWorld source-archive Git LFS pointers with the pinned, SHA-256-verified encrypted bundles before `appworld install`. |
| `src/summarize.py` | Builds a 4-decimal `summary.json`/`summary.txt` from `records.jsonl` (pure stdlib). |
| `environment/conda.yml` | **tau2 + officebench** env (Python 3.12, pydantic 2): `tau2[knowledge]` from git, `openai`, `azure-identity`, `azure-keyvault-secrets`, `pyyaml`, `tqdm`. |
| `environment/conda.qwen.yml` | **Qwen + tau2** GPU env with vLLM 0.24.0 and the tau2 dependencies. |
| `environment/conda.qwen.appworld.yml` | **Qwen + AppWorld** GPU env with vLLM and the pinned AppWorld source. |
| `environment/conda.appworld.yml` | **appworld** env (Python 3.11): pinned AppWorld `0.2.0.dev0` source matching the mounted `0.2.0` data asset, `openai<2`, `azure-identity`, `pyyaml`, `tqdm`. |
| `environment/conda.terminalbench.yml` | **Terminal-Bench 1 + 2** env (Python 3.12): pinned legacy `terminal-bench` and Harbor harnesses. Requires Docker-capable compute. |
| `submit.sh` | Wrapper around `az ml job create`; automatically provisions and injects remote Docker for OfficeBench. |
| `submit_qwen.sh` | Qwen job submitter; validates/reuses a one-GPU AML cluster before calling `az ml job create`. |
| `../.amlignore` | Keeps the uploaded code snapshot small (excludes `data/`, results, `.git`, caches). |

## Environment profiles

`dw` keeps AppWorld in a separate profile because its unreleased source commit
must match the mounted data asset's DB version (`0.2.0`); tau2 and OfficeBench
use the Python 3.12 profile. Each job file is already wired to the right env +
inputs:

| Benchmark | Job file | `inputs.env` | conda env |
|-----------|----------|--------------|-----------|
| AppWorld | `job.yml` | `appworld` | `conda.appworld.yml` |
| tau2 | `job.tau2.yml` | `tau2` | `conda.yml` |
| OfficeBench | `job.officebench.yml` | `officebench` | `conda.yml` (no py deps; needs Docker) |

OfficeBench's app + scoring code runs inside a per-rollout **Docker container**.
Standard AML command compute cannot start nested Docker, so the AML container
uses the Docker CLI to connect over SSH to a separate Linux VM with Docker. The
OfficeBench clone remains an AML data input; `docker build` and `docker cp`
stream the required files to the remote daemon.

## Azure OpenAI auth (differs by benchmark)

- **AppWorld** uses `dw`'s own `ChatClient`, which supports **managed identity**
  on AML (`AZURE_AUTH_MODE=aml` -> `DefaultAzureCredential`). Works out of the box.
- **tau2** drives its agent/user models through **litellm**, and `dw`'s litellm
  AAD token is fetched via the `az` CLI / `AzureCliCredential` — **not** managed
  identity. On AML the reliable path is an **API key**: set `inputs.aoai_api_key`
  (ideally from a secret), e.g. `--set inputs.aoai_api_key=$AOAI_KEY`. With an
  empty key the wrapper falls back to managed identity and runs
  `az login --identity`, which only works if `az` is present on the compute and
  the resource accepts AAD.
- The wrapper sets `AZURE_AUTH_MODE=api_key` automatically when a key is given.

## Prerequisites (already satisfied on this box)

- `az login` is done (subscription `M365 Research`).
- The `az ml` CLI extension v2 is installed.

If you ever move to a fresh machine:

```bash
az login
az extension add -n ml      # or: az extension update -n ml
```

## Submit a job

From the repo root (`/home/t-wendili/dw`). Pick the benchmark with `AML_JOB_FILE`
(default = AppWorld):

```bash
bash aml/submit.sh                                       # AppWorld
AML_JOB_FILE=aml/job.appworld.wm.yml bash aml/submit.sh  # AppWorld interactive WM sweep
AML_JOB_FILE=aml/job.appworld.bon2.yml bash aml/submit.sh # AppWorld multi-app Python-only BoN@2 gate
AML_JOB_FILE=aml/job.appworld.bon4.yml bash aml/submit.sh # AppWorld Python-only BoN@4 gate
AML_JOB_FILE=aml/job.tau2.yml bash aml/submit.sh         # tau2  (add inputs.aoai_api_key)
AML_JOB_FILE=aml/job.officebench.yml bash aml/submit.sh  # OfficeBench (configure remote Docker below)
```

### Local Qwen on one AML GPU

The normal jobs above use the CPU-only `D15v2-NCUS-hipri` cluster and call a
hosted model API. The Qwen job instead reuses the existing dedicated
`NC40adsv5-EUS-hipri` target (`Standard_NC40ads_H100_v5`, one H100 80GB),
starts vLLM inside the command job, waits for `/v1/models`, and runs tau2 against
`127.0.0.1:8000`. Model-server logs are published as `vllm.log`.
The default `Qwen/Qwen3.5-35B-A3B` checkpoint is the official post-trained
variant derived from `Qwen/Qwen3.5-35B-A3B-Base`; it supports instruction
following, reasoning, and the Qwen XML chat/tool protocol.
The runtime keeps the Hugging Face cache in the AML job workspace and requires
at least 100 GiB free before downloading the roughly 70 GB source checkpoint.

```bash
bash aml/submit_qwen.sh --stream

# Small smoke test
bash aml/submit_qwen.sh --stream --set \
  inputs.workers=2 inputs.extra_args="--tasks 2"
```

Override the compute through `AML_QWEN_COMPUTE` and `AML_QWEN_VM_SIZE`. For
example, an existing A100 target can be selected with:

```bash
AML_QWEN_COMPUTE=NC24ads-GWC-hipri \
AML_QWEN_VM_SIZE=Standard_NC24ads_A100_v4 \
bash aml/submit_qwen.sh --stream
```

The submitter requires a pre-existing cluster by default. Set
`AML_QWEN_CREATE_COMPUTE=1` only when the subscription has quota for the
requested SKU. Job inputs
such as `model`, `max_model_len`, `quantization`, `language_model_only`,
`enable_thinking`, `domains`, `methods`, `n`, and `extra_args` can be changed with the same
`--set inputs.*=...` syntax used by `submit.sh`.
Use `inputs.enable_thinking=0` to disable Qwen reasoning. Do not place
`--default-chat-template-kwargs` in `vllm_extra_args`; embedded JSON quotes are
not stable through AML command interpolation.

For BF16 on the existing two-H100 target, the AppWorld job uses tensor
parallelism size 2 plus vLLM expert parallelism. Its default selection is one
`test_normal/L1` task:

```bash
AML_QWEN_JOB_FILE=aml/job.qwen.appworld.yml \
bash aml/submit_qwen.sh --stream
```

`submit_qwen.sh` inherits the compute target from the selected job YAML. If the
two-H100 East US target is temporarily out of capacity, retry on the existing
dedicated two-A100 Germany West Central target:

```bash
AML_QWEN_JOB_FILE=aml/job.qwen.appworld.yml \
AML_QWEN_COMPUTE=NC48ads-GWC-A100-v4 \
bash aml/submit_qwen.sh --stream
```

The dedicated AppWorld WM job runs `--methods ours --agent llm_wm
--temperature 1 --N 1,2,4,8,16` with the `gpt-5.4` Azure OpenAI deployment.
It targets all six test split/level subsets (`test_normal,test_challenge`,
levels `1,2,3`), uses the OfficeBench-style reference-grounded verifier by
default, and includes native `answer.json` hard-invalid filtering. It does not
enable `--appworld-precision-prompt` or `--appworld-instruction-preflights`, so
Ours keeps the same stochastic ReAct agent/candidate-generation loop as B2 and
differs only in world-model simulation, verification, and selection. Both
flags remain explicit agent-side ablations and are not valid for a direct B2
comparison. A complete fair GPT-5.4 temperature-1 `test_normal` N=1 run scored
135/168 (80.36%), compared with 137/168 (81.55%) for the matched seeded B2
run; Ours tied B2 on L1 (54/57) and L3 (41/63), with L2 at 40/48 versus
42/48. The job has an extended 7200-second stall guard because the `N=16`
interactive-WM cell is substantially slower. Completed cells are checkpointed
individually, so a later cell failure or hang no longer deletes earlier `N`
results; `run_manifest.json` reports complete, partial, failed, and abandoned
coverage. Each WM candidate and whole task/N cell is retried twice by default.
If cells still remain missing, the run publishes clearly labeled partial
metrics; re-run the same output with `--resume` to complete coverage, or pass
`--strict-coverage` to reject the partial run.
Submit it and stream logs with:

```bash
AML_JOB_FILE=aml/job.appworld.wm.yml bash aml/submit.sh --stream
```

The AppWorld BoN@4 gate job uses the same shared agent and WM implementation for
levels 1, 2, and 3, explicitly selects the Python-only
`PYTHON_READ`/`PYTHON_FINAL` protocol, and runs `ours` at `N=4` over
`test_normal,test_challenge`. After complete coverage, the job fails unless:

- L1 is above 89% / 78% for normal / challenge,
- L2 is above 68% / 55%,
- L3 is above 60% / 50%.

Submit it with:

```bash
AML_JOB_FILE=aml/job.appworld.bon4.yml bash aml/submit.sh --stream
```

The multi-app BoN@2 gate targets levels 2 and 3 with the same Python-only WM.
It enables the precision prompt's cross-app dependency ledger, supplies the
native evaluator contract to reference-conditioned candidate generation, applies
known Amazon cart dirty reads deterministically, and ranks the two candidates
with graded per-check reference confidence. This is explicitly answer-key
conditioned rather than a blind benchmark run. The job fails unless
complete official coverage is above 70% / 60% for normal / challenge at L2 and
above 63% / 60% at L3:

```bash
AML_JOB_FILE=aml/job.appworld.bon2.yml bash aml/submit.sh --stream
```

## OfficeBench: automatic remote Docker VM

OfficeBench cannot use the Docker daemon of a standard AML command-job node.
`aml/submit.sh` now solves that automatically: for `job.officebench.yml` it
creates or reuses an `officebench-docker` Ubuntu VM, installs Docker, verifies
the SSH route, stores the private key and pinned host key in the workspace Key
Vault, grants the AML managed identity read access, and injects the four remote
Docker inputs before submitting the command job.

The submitting identity needs permission to create/start a VM in the resource
group, write secrets in the workspace Key Vault, and grant the AML managed
identity secret-read access. Submit a one-task smoke test with no Docker
configuration:

```bash
AML_JOB_FILE=aml/job.officebench.yml bash aml/submit.sh --stream --set \
  inputs.methods="b0" inputs.n="1" inputs.workers=1 \
  inputs.extra_args="--tasks 1"
```

The VM and its SSH key are reused on later submissions, preserving Docker's
image cache. Override its defaults with
`AML_OFFICEBENCH_DOCKER_VM_NAME`, `AML_OFFICEBENCH_DOCKER_VM_SIZE`,
`AML_OFFICEBENCH_DOCKER_RESOURCE_GROUP`, `AML_OFFICEBENCH_DOCKER_VM_USER`, or
`AML_OFFICEBENCH_DOCKER_SSH_KEY_PATH`. If tenant policy prevents service
principal lookup through Microsoft Graph, set `AML_MANAGED_IDENTITY_OBJECT_ID`
to the managed identity's principal/object ID. The default VM exposes SSH
through a Standard public IP and accepts key authentication only; use an
existing private-network host through the explicit overrides when public SSH is
not allowed. Deallocate the VM between runs to stop compute charges while
preserving its disk:

```bash
az vm deallocate -g Singularity -n officebench-docker
```

To use an existing Docker host instead, pass the four `inputs.docker_*`
overrides shown in `job.officebench.yml`, or set
`AML_OFFICEBENCH_DOCKER_HOST`, `AML_OFFICEBENCH_DOCKER_KEY_VAULT_URL`,
`AML_OFFICEBENCH_DOCKER_SSH_KEY_SECRET`, and optionally
`AML_OFFICEBENCH_DOCKER_SSH_KNOWN_HOSTS_SECRET`. Set
`AML_OFFICEBENCH_AUTO_PROVISION=0` to require explicit configuration. Never
expose an unauthenticated Docker TCP socket such as port 2375.

The AML wrapper installs a standalone Docker CLI if the command image does not
already contain one. Before starting workers, each job builds
`officebench-tts:latest` on the remote daemon; Docker reuses cached layers, and
the up-front build avoids a first-task build race between workers.
The remote SSH user only needs Docker access—the OfficeBench clone and this repo
remain AML inputs on the client side. The generated SSH config multiplexes
Docker commands over persistent connections; keep the default `workers: 8` at
or below the Docker VM's SSH `MaxSessions` limit.

**Configure a run by editing the `inputs:` block in the chosen job file**
(`methods`, `n`, `workers`, `max_steps`, `agent`, `extra_args`,
`rollout_retries`, `cell_retries`,
`aoai_deployment`, plus `splits`/`levels` for AppWorld, `domains` for tau2,
`levels` = app-counts for OfficeBench). AppWorld defaults: `--methods
b0,b1,b2,ours --N 1,2,4,8 --splits test_normal --levels 1 --workers 8
--max-steps 40`.

`inputs.extra_args` is forwarded verbatim to `experiments/run.py`, so add any
harness flag there, e.g. `--tasks 4`, `--no-verifier-reference`,
`--verifier-percheck`, `--sandbox-prompt`, `--appworld-precision-prompt`,
`--appworld-agent-reference`, `--appworld-deterministic-dirty-reads`,
`--task-indices 0,3,15`, `--agent llm_wm`, or `--strict-coverage`.

Metrics from interrupted or partially failed sweeps are published by default.
Both the core runner and AML summarizer return success, while `summary.json`
sets `"partial": true`, embeds the full coverage manifest, and retains each
aggregate's expected `decisions`, actual `recorded_decisions`, and
`imputed_failures` counts. Every missing task/method/N cell contributes a failed
decision (and zero reward), so incomplete runs cannot inflate success metrics.
`summary.txt` begins with a prominent partial-results warning. Use
`--strict-coverage` to make incomplete coverage a hard error. The former
`--allow-partial` flag remains accepted for compatibility but is no longer
required.

### One-off overrides (without editing job.yml)

Extra args are forwarded to `az ml job create`; CLI `--set` wins:

```bash
bash aml/submit.sh --set \
  display_name=dw-smoke \
  inputs.methods="b0,ours" \
  inputs.n="1,2" \
  inputs.workers=4 \
  inputs.extra_args="--tasks 4"
```

### Stream logs

```bash
bash aml/submit.sh --stream
```

Or open the Studio URL printed on submit.

## Troubleshooting

- **`apps.bundle is a Git LFS pointer and not a bundle file`** → GitHub's
  AppWorld source archive contains pointer stubs. The runtime wrapper now
  hydrates both encrypted bundles from the pinned commit and verifies their
  LFS-declared SHA-256 and size before running `appworld install`.
- **`database version ... is not compatible`** → the AppWorld code and mounted
  data asset do not match. The AppWorld profile pins the upstream `0.2.0.dev0`
  commit matching `tts_appworld_data`; the runtime now fails once up front
  instead of writing an empty results file after every task fails. Rebuild the
  AML environment from `environment/conda.appworld.yml`.
- **`Request is invalid and/or missing fields`** at submit → an input resolved
  to an **empty string**, which AML rejects. The job files use the literal
  `none` as the empty sentinel for `extra_args`/`aoai_api_key` (the wrapper maps
  `none` -> empty). So never set an input to `""` — if overriding, pass a real
  value or `none`, e.g. `--set inputs.aoai_api_key=none` (not
  `--set inputs.aoai_api_key=$KEY` when `$KEY` is unset, which expands to empty).
- **`No such file or directory: .../data/tau2/domains/<domain>/tasks.json`** →
  tau2-bench's `data/` dir is **not** packaged in its pip wheel (it lives at the
  repo root), so the non-editable git install can't find it. The wrapper fixes
  this by sparse-cloning `data/tau2` from the pinned tau2 commit on the compute
  and exporting `TAU2_DATA_DIR` (≈146 MB, a few seconds). To avoid the per-run
  download, mount your own copy and pass it via `--tau2-data` (or pre-set
  `TAU2_DATA_DIR`); see the tau2 caveat below.

- **`No such file or directory: 'az'`** (tau2 on AML) → the compute image has no
  `az` CLI, but `dw`'s litellm AAD path (`azure.py:fetch_aad_token`) shells out
  to `az`. (AppWorld is unaffected — like `../TTS` it uses `DefaultAzureCredential`,
  which needs no `az`.) The wrapper now **installs `az` at runtime and runs
  `az login --identity`** so tau2 authenticates as the compute's managed identity.
  If the runtime install fails (no apt/root on the image), either bake `az` into
  a **custom AML environment image** (reliable, cache-once) or fall back to an
  API key via `inputs.aoai_api_key`.

- **`AzureCliCredential.get_token failed: Please run 'az login'`** (tau2 on AML)
  → `az login --identity` didn't establish an account. Common cause: **az CLI
  2.71+ renamed the user-assigned MI flag from `--username` to `--client-id`**.
  The wrapper now tries `--client-id` first (then `--username`, then
  system-assigned) and verifies the token, logging the az error if it still
  fails. If it fails, confirm the compute's managed identity has
  `Cognitive Services OpenAI User` on the AOAI resource, or pass
  `inputs.aoai_api_key`. (`../TTS` avoids all of this by using
  `DefaultAzureCredential` in Python — no `az` — but `dw`'s tau2/litellm path is
  hardwired to `AzureCliCredential`, so `az` is required for tau2.)

- **`[hang-guard] no task finished in 1200s; abandoning N wedged task(s)`** →
  the sweep's stall-guard fired because **no task completed within
  `TTS_RUN_STALL_TIMEOUT_S`** (default 1200s in core). Usually the config is just
  slow, not stuck: `ours` + `llm_wm` + large `--N` (e.g. `1,2,4,8,16`) runs many
  recursive world-model rollouts per task, and a high `--workers` against one
  AOAI deployment triggers 429 rate-limiting that stalls everything. Fixes:
  - raise the guard via `inputs.stall_timeout` (the wrapper exports it;
    tau2 defaults to 7200s), and
  - **lower `inputs.workers`** — this path is rate-limit-bound, not CPU-bound, so
    64 workers is counter-productive; 8–16 lets tasks actually finish.
  Records are checkpointed, so nothing already done is lost.

- **`[officebench][error] no usable Docker daemon`** → the job was submitted
  directly or without injected remote-Docker inputs. Submit through
  `AML_JOB_FILE=aml/job.officebench.yml bash aml/submit.sh`; it provisions and
  injects the Docker VM automatically. For an explicit host, test the same route
  with `DOCKER_HOST=ssh://USER@HOST docker info`. If the job times out, check the
  VM firewall/VNet route; if SSH succeeds but Docker fails, add the SSH user to
  the VM's `docker` group and reconnect.

## Outputs

The wrapper writes results to **two** places (identical content):

1. The named `output_dir` (an AML `uri_folder` output — the authoritative copy,
   stored on the workspace datastore).
2. `./outputs/` in the job — which AML **auto-captures as job artifacts**,
   visible in Studio under *Outputs + logs* and downloadable without knowing the
   output name.

Files written:

- `records.jsonl` — one record per (task, method, N) cell (the harness checkpoint).
- `summary.json` — aggregated metrics by (env, subset, method, N), **4 decimals**.
- `summary.txt` — the success-vs-N TTS curve as a readable table, **4 decimals**.
- `bon4_gate.json` / `bon4_gate.txt` — strict six-subset threshold result for
  `job.appworld.bon4.yml`.
- `bon2_gate.json` / `bon2_gate.txt` — strict multi-app threshold result for
  `job.appworld.bon2.yml`.
- `run.log` — the full harness stdout/stderr (the printed report, tee'd to file).
- `metadata.json` — the run configuration.

(`dw`'s harness only *prints* its metrics to stdout, so the wrapper regenerates a
downloadable 4-decimal `summary.*` from `records.jsonl` — mirroring `../TTS`'s
`.summary.json`.)

Download after the run — either works:

```bash
# grabs the ./outputs artifacts + logs (no --output-name needed)
az ml job download --name <job_name> -g Singularity -w M365Research --all

# or just the named uri_folder output
az ml job download --name <job_name> -g Singularity -w M365Research \
  --output-name output_dir
```

## Changing defaults

- Edit the chosen job file's `inputs:` for durable defaults (n, workers,
  methods, model, and the per-env knobs splits/levels/domains).
- Override per-submit with `--set inputs.<name>=<value>`.
- The model is the AOAI deployment `gpt-4.1`; change `inputs.aoai_deployment`
  (the harness picks it up via `WM_MODEL`) to use another deployment.

## How it maps to the working `../TTS` example

The `../TTS` repo runs `bon/run_bon.py` as three parallel AML jobs. `dw` instead
runs **all** methods (`b0/b1/b2/ours`) and all `N` in a single
`experiments/run.py` invocation, so each benchmark is **one command job**. The
Azure plumbing (workspace, compute, managed identity, AOAI endpoint, data asset)
is identical and reused as-is.

## Notes / caveats

- **One job file per benchmark** (already wired to the right conda env + inputs):
  `job.yml` (AppWorld), `job.tau2.yml` (tau2), `job.officebench.yml`
  (OfficeBench). The two conda envs are mutually exclusive (pydantic 1 vs 2).
- **tau2 auth**: tau2's agent/user models run through **litellm**, whose AAD
  token in `dw` comes from the `az` CLI / `AzureCliCredential`, **not**
  `DefaultAzureCredential`. Since the AML image has no `az`, the wrapper installs
  it and runs `az login --identity` so tau2 authenticates as the compute's
  managed identity (this requires the identity to be assigned to the compute and
  granted `Cognitive Services OpenAI User` on the AOAI resource). If you'd rather
  not depend on `az`, pass `inputs.aoai_api_key` (key auth works for both `dw`'s
  client and litellm, no `az` needed). For reliability, prefer baking `az` into a
  custom AML environment image over the per-run install.
- **tau2 data**: the pip wheel omits tau2's `data/` dir, so the wrapper
  sparse-clones `data/tau2` (domains + user_simulator, ≈146 MB) from the pinned
  commit at runtime and sets `TAU2_DATA_DIR`. To skip the per-run download,
  register a `uri_folder` data asset from a tau2 `data/` dir and pass it through:
  add a `uri_folder` input to `job.tau2.yml` and forward it as `--tau2-data`, or
  set `TAU2_DATA_DIR` on the compute. Keep the fetched commit in sync with the
  tau2 version in `conda.yml` (both pinned to the same SHA in `run_dw.sh`).
- **OfficeBench remote Docker**: standard AML command jobs cannot start a Docker
  daemon. `aml/submit.sh` automatically provisions/reuses an SSH Docker VM and
  stores its credentials in the workspace Key Vault; the job's managed identity
  retrieves them at runtime. A direct run on a Docker-capable Compute Instance/VM
  still works too: run `bash aml/src/run_dw.sh --env officebench ...` without
  `--docker-host`. Register the OfficeBench clone as a `uri_folder` data asset
  and point `inputs.officebench_root` at it.
- AML jobs default to managed identity for `dw`'s own client (the wrapper forces
  `AZURE_AUTH_MODE=aml`); supplying `inputs.aoai_api_key` switches it to key auth.
- The `gpt-4.1` deployment + managed identity `258f6dab-...` and the
  `aidaihub2294673195` AOAI endpoint are reused from the proven `../TTS` example
  because those are the ones authorized on the AML compute.
