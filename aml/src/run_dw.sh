#!/usr/bin/env bash
# Runtime wrapper executed on AML compute. Wires up Azure-OpenAI auth, prepares
# the selected benchmark, then runs the `dw` harness (experiments/run.py). The
# *core* repo code is never modified; everything Azure-specific lives here.
#
# Supports three envs (pick via --env, and match the conda_file in the job yaml):
#   appworld   -> mounts the AppWorld 0.2 data asset, `appworld install`
#   tau2       -> tau2-bench via litellm (pydantic 2 env); needs api-key or az-login auth
#   officebench-> runs apps/eval inside Docker; needs a Docker daemon + OFFICEBENCH_ROOT
set -euo pipefail

env_name="appworld"
appworld_data=""
officebench_root=""
docker_host=""
docker_key_vault_url=""
docker_ssh_key_secret=""
docker_ssh_known_hosts_secret=""
tau2_data=""
output_dir=""
methods="b0,b1,b2,ours"
n="1,2,4,8"
agent="llm"
appworld_wm_tool_mode="python"
splits="test_normal"
levels="1"
domains="airline"
workers="8"
max_steps="40"
rollout_retries="2"
cell_retries="2"
temperature="0.7"
aoai_endpoint=""
aoai_deployment="gpt-4.1"
aoai_api_version="2024-12-01-preview"
aoai_api_key=""
managed_identity_client_id=""
extra_args=""
stall_timeout=""
benchmark_gate=""

# tau2-bench commit whose data/ subtree is fetched at runtime (must match the
# tau2 version pinned in aml/environment/conda.yml).
tau2_sha="1746a25db265724f6fed2260934bb67f1514fad7"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env) env_name="$2"; shift 2 ;;
    --appworld-data) appworld_data="$2"; shift 2 ;;
    --officebench-root) officebench_root="$2"; shift 2 ;;
    --docker-host) docker_host="$2"; shift 2 ;;
    --docker-key-vault-url) docker_key_vault_url="$2"; shift 2 ;;
    --docker-ssh-key-secret) docker_ssh_key_secret="$2"; shift 2 ;;
    --docker-ssh-known-hosts-secret) docker_ssh_known_hosts_secret="$2"; shift 2 ;;
    --tau2-data) tau2_data="$2"; shift 2 ;;
    --output-dir) output_dir="$2"; shift 2 ;;
    --methods) methods="$2"; shift 2 ;;
    --n) n="$2"; shift 2 ;;
    --agent) agent="$2"; shift 2 ;;
    --appworld-wm-tool-mode) appworld_wm_tool_mode="$2"; shift 2 ;;
    --splits) splits="$2"; shift 2 ;;
    --levels) levels="$2"; shift 2 ;;
    --domains) domains="$2"; shift 2 ;;
    --workers) workers="$2"; shift 2 ;;
    --max-steps) max_steps="$2"; shift 2 ;;
    --rollout-retries) rollout_retries="$2"; shift 2 ;;
    --cell-retries) cell_retries="$2"; shift 2 ;;
    --temperature) temperature="$2"; shift 2 ;;
    --aoai-endpoint) aoai_endpoint="$2"; shift 2 ;;
    --aoai-deployment) aoai_deployment="$2"; shift 2 ;;
    --aoai-api-version) aoai_api_version="$2"; shift 2 ;;
    --aoai-api-key) aoai_api_key="$2"; shift 2 ;;
    --managed-identity-client-id) managed_identity_client_id="$2"; shift 2 ;;
    --extra-args) extra_args="$2"; shift 2 ;;
    --stall-timeout) stall_timeout="$2"; shift 2 ;;
    --benchmark-gate) benchmark_gate="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

# AML command jobs reject empty-string inputs ("Request is invalid and/or missing
# fields"), so the job YAMLs use the literal "none" as the empty sentinel.
[[ "$extra_args" == "none" ]] && extra_args=""
[[ "$aoai_api_key" == "none" ]] && aoai_api_key=""
[[ "$domains" == "none" ]] && domains=""
[[ "$tau2_data" == "none" ]] && tau2_data=""
[[ "$stall_timeout" == "none" ]] && stall_timeout=""
[[ "$docker_host" == "none" ]] && docker_host=""
[[ "$docker_key_vault_url" == "none" ]] && docker_key_vault_url=""
[[ "$docker_ssh_key_secret" == "none" ]] && docker_ssh_key_secret=""
[[ "$docker_ssh_known_hosts_secret" == "none" ]] && docker_ssh_known_hosts_secret=""
[[ "$benchmark_gate" == "none" ]] && benchmark_gate=""

# Hang-guard window (run.py TTS_RUN_STALL_TIMEOUT_S): the sweep abandons all
# in-flight tasks if NONE completes within this many seconds. Heavy configs
# (ours + llm_wm + large N, or rate-limited high --workers) need a big value,
# else the guard wrongly abandons slow-but-progressing tasks. Default 3600s.
[[ -n "$stall_timeout" ]] && export TTS_RUN_STALL_TIMEOUT_S="$stall_timeout"
: "${TTS_RUN_STALL_TIMEOUT_S:=3600}"
export TTS_RUN_STALL_TIMEOUT_S
echo "[run] TTS_RUN_STALL_TIMEOUT_S=${TTS_RUN_STALL_TIMEOUT_S}s"

if [[ -z "$output_dir" ]]; then
  echo "output-dir is required" >&2; exit 2
fi
if [[ "$env_name" == "appworld" && -z "$appworld_data" ]]; then
  echo "appworld-data is required when --env appworld" >&2; exit 2
fi
if [[ "$env_name" == "officebench" && -z "$officebench_root" ]]; then
  echo "officebench-root is required when --env officebench" >&2; exit 2
fi

mkdir -p "$output_dir"

install_docker_cli() {
  if command -v docker >/dev/null 2>&1; then
    return
  fi
  local version="${TTS_DOCKER_CLI_VERSION:-27.5.1}"
  local machine archive_arch tmp_dir
  machine="$(uname -m)"
  case "$machine" in
    x86_64) archive_arch="x86_64" ;;
    aarch64|arm64) archive_arch="aarch64" ;;
    *)
      echo "[officebench][docker][error] unsupported CPU architecture for Docker CLI: $machine" >&2
      exit 3
      ;;
  esac
  command -v curl >/dev/null 2>&1 || {
    echo "[officebench][docker][error] curl is required to install the Docker CLI." >&2
    exit 3
  }
  command -v tar >/dev/null 2>&1 || {
    echo "[officebench][docker][error] tar is required to install the Docker CLI." >&2
    exit 3
  }
  tmp_dir="$(mktemp -d)"
  echo "[officebench][docker] installing Docker CLI ${version}"
  curl -fsSL "https://download.docker.com/linux/static/stable/${archive_arch}/docker-${version}.tgz" \
    | tar -xz -C "$tmp_dir"
  mkdir -p "$HOME/.local/bin"
  install -m 0755 "$tmp_dir/docker/docker" "$HOME/.local/bin/docker"
  rm -rf "$tmp_dir"
  export PATH="$HOME/.local/bin:$PATH"
}

install_ssh_client() {
  if command -v ssh >/dev/null 2>&1; then
    return
  fi
  local sudo_cmd=()
  if [[ "$(id -u)" -ne 0 ]]; then
    command -v sudo >/dev/null 2>&1 || {
      echo "[officebench][docker][error] openssh-client is required for an ssh:// Docker host." >&2
      exit 3
    }
    sudo_cmd=(sudo)
  fi
  echo "[officebench][docker] installing openssh-client"
  "${sudo_cmd[@]}" apt-get update >/tmp/officebench_apt.log 2>&1
  "${sudo_cmd[@]}" apt-get install -y openssh-client >>/tmp/officebench_apt.log 2>&1 || {
    echo "[officebench][docker][error] could not install openssh-client." >&2
    sed 's/^/  apt: /' /tmp/officebench_apt.log >&2 || true
    exit 3
  }
}

configure_officebench_docker() {
  install_docker_cli
  if [[ -n "$docker_host" ]]; then
    [[ "$docker_host" == ssh://* ]] && install_ssh_client
    export DOCKER_HOST
    DOCKER_HOST="$(
      python aml/src/configure_remote_docker.py \
        --docker-host "$docker_host" \
        --vault-url "$docker_key_vault_url" \
        --ssh-key-secret "$docker_ssh_key_secret" \
        --known-hosts-secret "$docker_ssh_known_hosts_secret" \
        --managed-identity-client-id "$managed_identity_client_id"
    )"
    echo "[officebench][docker] using remote daemon at $DOCKER_HOST"
  elif [[ -n "$docker_ssh_key_secret" || -n "$docker_ssh_known_hosts_secret" ]]; then
    echo "[officebench][docker][error] Docker SSH secrets were set without --docker-host." >&2
    exit 3
  fi
  # The standalone Docker CLI does not include the buildx plugin. The legacy
  # builder is sufficient for the small OfficeBench runtime image.
  : "${DOCKER_BUILDKIT:=0}"
  export DOCKER_BUILDKIT
}

# --- Azure OpenAI auth ------------------------------------------------------
# dw uses TWO Azure auth paths:
#   * its own ChatClient (world model / verifier / AppWorld agent) honours
#     AZURE_AUTH_MODE (aml -> DefaultAzureCredential = managed identity).
#   * tau2 drives agent/user models through litellm, whose AAD token in dw is
#     fetched via AzureCliCredential/`az` (NOT managed identity). So on AML, tau2
#     needs either an API key (recommended) or a working `az login`.
export AZURE_OPENAI_ENDPOINT="$aoai_endpoint"
export AZURE_OPENAI_API_VERSION="$aoai_api_version"
export WM_MODEL="$aoai_deployment"

if [[ "${TTS_LOCAL_LLM:-0}" == "1" ]]; then
  # A sibling vLLM process on this AML node serves every model call. Avoid
  # installing/logging into Azure CLI or attempting managed-identity auth.
  export AZURE_AUTH_MODE="openai"
  echo "[auth] using local OpenAI-compatible model server"
elif [[ -n "$aoai_api_key" ]]; then
  # Key auth works uniformly for ChatClient AND tau2/litellm, with no az/MI deps.
  # dw's ChatClient reads AZURE_OPENAI_API_KEY; litellm (tau2) reads AZURE_API_KEY.
  export AZURE_OPENAI_API_KEY="$aoai_api_key"
  export AZURE_API_KEY="$aoai_api_key"
  export AZURE_AUTH_MODE="api_key"
  echo "[auth] using Azure OpenAI API key"
else
  # Managed identity for dw's ChatClient.
  export AZURE_AUTH_MODE="aml"
  export AZURE_CLIENT_ID="$managed_identity_client_id"
  export DEFAULT_IDENTITY_CLIENT_ID="$managed_identity_client_id"
  export AZURE_MANAGED_IDENTITY_CLIENT_ID="$managed_identity_client_id"
  echo "[auth] using managed identity (AZURE_AUTH_MODE=aml)"
  if [[ "$env_name" == "tau2" ]]; then
    # Unlike dw's AppWorld path (DefaultAzureCredential, no CLI — same as ../TTS),
    # tau2's litellm AAD token comes from dw's fetch_aad_token, which shells out to
    # `az`. The AML image has no `az`, so install it, then log in as the compute's
    # managed identity (AzureCliCredential / `az account get-access-token` read it).
    if ! command -v az >/dev/null 2>&1; then
      echo "[auth] installing az CLI (tau2's litellm AAD path needs it)"
      _sudo=""; [[ "$(id -u)" -ne 0 ]] && command -v sudo >/dev/null 2>&1 && _sudo="sudo"
      { curl -fsSL https://aka.ms/InstallAzureCLIDeb | $_sudo bash; } >/tmp/az_install.log 2>&1 \
        || echo "[auth][warn] az CLI install failed (see /tmp/az_install.log). Bake az into a custom AML environment image, or pass --aoai-api-key." >&2
    fi
    if command -v az >/dev/null 2>&1; then
      # Log in as the compute's managed identity so AzureCliCredential /
      # `az account get-access-token` (dw's fetch_aad_token) work.
      # NOTE: az CLI 2.71+ takes the user-assigned identity via --client-id;
      # older CLIs used --username. Try the new flag first, then fall back.
      echo "[auth] az login --identity for tau2 litellm AAD"
      _mi_ok=0
      if [[ -n "$managed_identity_client_id" ]]; then
        az login --identity --client-id "$managed_identity_client_id" >/tmp/az_login.log 2>&1 && _mi_ok=1
        [[ "$_mi_ok" -eq 0 ]] && az login --identity --username "$managed_identity_client_id" >>/tmp/az_login.log 2>&1 && _mi_ok=1
      fi
      # Fall back to a system-assigned / default identity.
      [[ "$_mi_ok" -eq 0 ]] && az login --identity >>/tmp/az_login.log 2>&1 && _mi_ok=1
      # Verify a token can actually be minted for Azure OpenAI.
      if [[ "$_mi_ok" -eq 1 ]] && \
         az account get-access-token --resource https://cognitiveservices.azure.com/ -o none >>/tmp/az_login.log 2>&1; then
        echo "[auth] az managed-identity login OK"
      else
        echo "[auth][error] az managed-identity login/token failed (see below). tau2 will 401." >&2
        sed 's/^/  az: /' /tmp/az_login.log >&2 || true
        echo "  Ensure the identity has 'Cognitive Services OpenAI User' on the AOAI resource, or pass --aoai-api-key." >&2
      fi
    else
      echo "[auth][error] az CLI unavailable; tau2 litellm cannot authenticate via managed identity." >&2
    fi
  fi
fi

# --- Benchmark-specific setup ----------------------------------------------
env_args=()
case "$env_name" in
  appworld)
    # The mounted data asset IS AppWorld's data/ dir (tasks/, base_dbs/, ...).
    rm -rf data
    ln -s "$appworld_data" data
    export APPWORLD_ROOT="$(pwd)"
    # GitHub source archives preserve AppWorld's encrypted bundles as Git LFS
    # pointer files. Hydrate the exact objects from the pinned commit and verify
    # their pointer-declared SHA-256/size before asking AppWorld to unpack them.
    python aml/src/hydrate_appworld_bundles.py
    appworld install
    # AppWorld rejects tasks generated by an incompatible data/DB version. Check
    # once up front instead of allowing every task to fail and producing an
    # empty records file with hundreds of repeated errors.
    python - <<'PY'
import json
from pathlib import Path

from appworld.common import constants

specs_path = next(Path("data/tasks").glob("*/specs.json"), None)
if specs_path is None:
    raise SystemExit("[appworld][error] no task specs found under data/tasks")

data_version = str(json.loads(specs_path.read_text(encoding="utf-8"))["db_version"])
compatible_versions = tuple(
    str(version)
    for version in getattr(constants, "COMPATIBLE_DB_VERSIONS", [constants.DB_VERSION])
)
if data_version not in compatible_versions:
    raise SystemExit(
        "[appworld][error] incompatible code/data versions: "
        f"mounted data uses {data_version}, but this AppWorld accepts "
        f"{', '.join(compatible_versions)}"
    )
print(
    "[appworld] compatible code/data versions: "
    f"data={data_version}, accepted={','.join(compatible_versions)}"
)
PY
    [[ -n "$splits" ]] && env_args+=(--splits "$splits")
    [[ -n "$levels" ]] && env_args+=(--levels "$levels")
    ;;
  tau2)
    # tau2's data/ dir is NOT packaged in the pip wheel (it lives at the repo
    # root), so the non-editable git install can't find domain tasks/DBs. Point
    # TAU2_DATA_DIR at a copy of just the text-domain data.
    if [[ -n "$tau2_data" ]]; then
      export TAU2_DATA_DIR="$tau2_data"          # caller mounted a data asset
    elif [[ -z "${TAU2_DATA_DIR:-}" ]]; then
      tau2_src="$PWD/.tau2-data"
      if [[ ! -f "$tau2_src/data/tau2/domains/airline/tasks.json" ]]; then
        command -v git >/dev/null 2>&1 || {
          echo "[tau2][error] git not found; cannot fetch tau2 data — mount one via --tau2-data." >&2
          exit 4; }
        echo "[tau2] fetching domain data (sparse checkout of tau2-bench@${tau2_sha:0:7} data/tau2)"
        rm -rf "$tau2_src"
        git clone --filter=blob:none --no-checkout \
          https://github.com/sierra-research/tau2-bench.git "$tau2_src"
        git -C "$tau2_src" sparse-checkout set data/tau2/domains data/tau2/user_simulator
        git -C "$tau2_src" checkout "$tau2_sha"
      fi
      export TAU2_DATA_DIR="$tau2_src/data"
    fi
    echo "[tau2] TAU2_DATA_DIR=${TAU2_DATA_DIR}"
    [[ -n "$domains" ]] && env_args+=(--domains "$domains")
    ;;
  officebench)
    # OfficeBench runs apps + scoring inside a Docker container built on demand.
    export OFFICEBENCH_ROOT="$officebench_root"
    export TTS_OFFICEBENCH_RUN_ID="${TTS_OFFICEBENCH_RUN_ID:-${AZUREML_RUN_ID:-aml-$(date +%s)-$$}}"
    echo "[officebench] container run id: $TTS_OFFICEBENCH_RUN_ID"
    configure_officebench_docker
    docker_info=""
    if ! docker_info="$(timeout "${TTS_DOCKER_INFO_TIMEOUT_S:-60}" docker info 2>&1)"; then
      echo "[officebench][error] no usable Docker daemon." >&2
      echo "  Standard AML command-job compute cannot run nested Docker. Submit with" >&2
      echo "  aml/submit.sh so it provisions/injects the remote Docker VM, configure" >&2
      echo "  --docker-host ssh://USER@HOST explicitly, or run the wrapper directly" >&2
      echo "  on a Docker-capable compute instance." >&2
      printf '%s\n' "$docker_info" | tail -n 20 | sed 's/^/  docker: /' >&2
      exit 3
    fi
    if [[ -z "${TTS_OFFICEBENCH_IMAGE:-}" ]]; then
      # Build once before worker processes start. Without this, a new daemon can
      # receive one concurrent build per worker on the first task.
      echo "[officebench][docker] preparing officebench-tts:latest"
      docker build -t officebench-tts:latest src/tts/env/officebench_docker
    elif ! docker image inspect "$TTS_OFFICEBENCH_IMAGE" >/dev/null 2>&1; then
      echo "[officebench][docker][error] configured image not found: $TTS_OFFICEBENCH_IMAGE" >&2
      exit 3
    fi
    # OfficeBench uses --levels as app-counts (1/2/3 apps).
    [[ -n "$levels" ]] && env_args+=(--levels "$levels")
    ;;
  *)
    echo "Unknown env: $env_name" >&2; exit 2 ;;
esac

read -r -a extra_arr <<< "$extra_args"

cmd=(
  python experiments/run.py
  --env "$env_name"
  --methods "$methods"
  --N "$n"
  --agent "$agent"
  --appworld-wm-tool-mode "$appworld_wm_tool_mode"
  --workers "$workers"
  --max-steps "$max_steps"
  --rollout-retries "$rollout_retries"
  --cell-retries "$cell_retries"
  --temperature "$temperature"
  --out "$output_dir"
)
[[ ${#env_args[@]} -gt 0 ]] && cmd+=("${env_args[@]}")
[[ ${#extra_arr[@]} -gt 0 ]] && cmd+=("${extra_arr[@]}")

echo "Running: ${cmd[*]}"
# Capture the harness stdout/stderr (incl. the printed metrics report) to a file
# in the output dir, so the run log is downloadable — not just in the AML system
# log. `set -o pipefail` (already on) makes the tee'd exit status the python one.
run_status=0
"${cmd[@]}" 2>&1 | tee "$output_dir/run.log" || run_status=$?

# dw only PRINTS its metrics (2 decimals); write a downloadable 4-decimal summary
# from the checkpointed records (mirrors ../TTS's .summary.json) AND print the
# 4-decimal tables to the console. Tee to run.log so the saved log has them too.
summary_status=0
summary_args=()
for arg in "${extra_arr[@]}"; do
  case "$arg" in
    --allow-partial|--strict-coverage) summary_args+=("$arg") ;;
  esac
done
python "$(dirname "$0")/summarize.py" \
  "$output_dir/records.jsonl" "$output_dir" "${summary_args[@]}" 2>&1 \
  | tee -a "$output_dir/run.log" || summary_status=$?

gate_status=0
if [[ "$benchmark_gate" == "appworld_bon4" ]]; then
  python "$(dirname "$0")/check_appworld_bon4.py" \
    "$output_dir/records.jsonl" "$output_dir" 2>&1 \
    | tee -a "$output_dir/run.log" || gate_status=$?
elif [[ "$benchmark_gate" == "appworld_bon2" ]]; then
  python "$(dirname "$0")/check_appworld_bon2.py" \
    "$output_dir/records.jsonl" "$output_dir" 2>&1 \
    | tee -a "$output_dir/run.log" || gate_status=$?
elif [[ -n "$benchmark_gate" ]]; then
  echo "[gate][error] unknown benchmark gate: $benchmark_gate" | tee -a "$output_dir/run.log"
  gate_status=2
fi

python - <<PY
import json
from pathlib import Path

out = Path("$output_dir")
payload = {
    "env": "$env_name",
    "methods": "$methods".split(","),
    "N": "$n",
    "agent": "$agent",
    "appworld_wm_tool_mode": "$appworld_wm_tool_mode",
    "splits": "$splits".split(",") if "$splits" else [],
    "levels": "$levels".split(",") if "$levels" else [],
    "domains": "$domains".split(",") if "$domains" else [],
    "workers": int("$workers"),
    "max_steps": int("$max_steps"),
    "rollout_retries": int("$rollout_retries"),
    "cell_retries": int("$cell_retries"),
    "temperature": float("$temperature"),
    "aoai_deployment": "$aoai_deployment",
    "auth_mode": (
        "local" if "${TTS_LOCAL_LLM:-0}" == "1"
        else ("api_key" if "$aoai_api_key" else "aml")
    ),
    "records": str(out / "records.jsonl"),
    "run_manifest": str(out / "run_manifest.json"),
    "extra_args": "$extra_args",
    "benchmark_gate": "$benchmark_gate",
}
(out / "metadata.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
print("Wrote", out / "metadata.json")
PY

# Mirror the results into ./outputs, which AML auto-captures as job artifacts
# (downloadable from Studio's "Outputs + logs" and via `az ml job download`
# WITHOUT needing --output-name). The named uri_folder output still holds the
# authoritative copy.
out_abs="$(cd "$output_dir" && pwd)"
mkdir -p ./outputs
outputs_abs="$(cd ./outputs && pwd)"
if [[ "$out_abs" != "$outputs_abs" ]]; then
  for f in records.jsonl run_manifest.json summary.json summary.txt metadata.json run.log bon2_gate.json bon2_gate.txt bon4_gate.json bon4_gate.txt; do
    if [[ -f "$output_dir/$f" ]]; then
      cp -f "$output_dir/$f" "./outputs/$f"
    fi
  done
  echo "[run] mirrored results to ./outputs (AML job artifacts)"
fi

final_status=$run_status
[[ "$final_status" -eq 0 && "$summary_status" -ne 0 ]] && final_status=$summary_status
[[ "$final_status" -eq 0 && "$gate_status" -ne 0 ]] && final_status=$gate_status
exit $final_status
