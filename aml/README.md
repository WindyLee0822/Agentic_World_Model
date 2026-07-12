# Running `dw` on Azure ML

This folder adds everything needed to run the `dw` test-time-scaling harness
(`experiments/run.py`) on **real AppWorld**, **tau2-bench**, or (sketch)
**OfficeBench** on the `M365Research` Azure ML workspace. **No core repo code is
changed** — only this `aml/` folder plus the repo-root `.amlignore`.

There is one job file per benchmark (they need different conda envs / inputs);
all share the runtime wrapper `aml/src/run_dw.sh`.

## Files

| File | Purpose |
|------|---------|
| `job.yml` | AML command job for **AppWorld** (env `conda.appworld.yml`). |
| `job.tau2.yml` | AML command job for **tau2-bench** (env `conda.yml`). |
| `job.officebench.yml` | AML command job for **OfficeBench** — *sketch*, needs Docker-capable compute. |
| `src/run_dw.sh` | Runtime wrapper on the compute node: sets Azure auth, prepares the benchmark, runs `experiments/run.py`, saves a downloadable summary + log. |
| `src/summarize.py` | Builds a 4-decimal `summary.json`/`summary.txt` from `records.jsonl` (pure stdlib). |
| `environment/conda.yml` | **tau2 + officebench** env (Python 3.12, pydantic 2): `tau2[knowledge]` from git, `openai`, `azure-identity`, `pyyaml`, `tqdm`. |
| `environment/conda.appworld.yml` | **appworld** env (Python 3.11, pydantic <2): stock `appworld`, `openai<2`, `azure-identity`, `pyyaml`, `tqdm`. |
| `submit.sh` | Thin wrapper around `az ml job create` (pick the job file with `AML_JOB_FILE`). |
| `../.amlignore` | Keeps the uploaded code snapshot small (excludes `data/`, results, `.git`, caches). |

## Environment profiles (mutually exclusive)

`dw` targets three benchmarks, but their deps **cannot share one conda env**:
AppWorld pins `pydantic<2` (Python 3.11) while tau2 needs `pydantic>=2`
(Python 3.12). Each job file is already wired to the right env + inputs:

| Benchmark | Job file | `inputs.env` | conda env |
|-----------|----------|--------------|-----------|
| AppWorld | `job.yml` | `appworld` | `conda.appworld.yml` |
| tau2 | `job.tau2.yml` | `tau2` | `conda.yml` |
| OfficeBench | `job.officebench.yml` | `officebench` | `conda.yml` (no py deps; needs Docker) |

OfficeBench has no Python deps: its app + scoring code runs inside a per-rollout
**Docker container**, so the compute host needs the `docker` CLI, a usable
Docker daemon, and the `OFFICEBENCH_ROOT` clone (see caveats).

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
AML_JOB_FILE=aml/job.tau2.yml bash aml/submit.sh         # tau2  (add inputs.aoai_api_key)
AML_JOB_FILE=aml/job.officebench.yml bash aml/submit.sh  # OfficeBench (needs Docker compute)
```

**Configure a run by editing the `inputs:` block in the chosen job file**
(`methods`, `n`, `workers`, `max_steps`, `agent`, `extra_args`,
`aoai_deployment`, plus `splits`/`levels` for AppWorld, `domains` for tau2,
`levels` = app-counts for OfficeBench). AppWorld defaults: `--methods
b0,b1,b2,ours --N 1,2,4,8 --splits test_normal --levels 1 --workers 8
--max-steps 40`.

`inputs.extra_args` is forwarded verbatim to `experiments/run.py`, so add any
harness flag there, e.g. `--tasks 4`, `--verifier-reference`, `--sandbox-prompt`,
`--agent llm_wm`, `--splits test_normal,test_challenge`.

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
- **OfficeBench needs Docker**: it runs apps + scoring inside a container, and a
  standard AML command job **cannot start a Docker daemon**. The wrapper checks
  `docker info` and exits 3 with guidance if absent. Realistic options: run
  `bash aml/src/run_dw.sh --env officebench ...` on an AML **Compute Instance** /
  VM where you control Docker, or attach Docker-capable compute (privileged /
  docker-in-docker / mounted `/var/run/docker.sock`). Also register the
  OfficeBench clone as a `uri_folder` data asset and point `inputs.officebench_root`
  at it. tau2 and AppWorld run on standard AML compute.
- AML jobs default to managed identity for `dw`'s own client (the wrapper forces
  `AZURE_AUTH_MODE=aml`); supplying `inputs.aoai_api_key` switches it to key auth.
- The `gpt-4.1` deployment + managed identity `258f6dab-...` and the
  `aidaihub2294673195` AOAI endpoint are reused from the proven `../TTS` example
  because those are the ones authorized on the AML compute.
