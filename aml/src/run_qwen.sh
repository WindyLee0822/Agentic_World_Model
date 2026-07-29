#!/usr/bin/env bash
# Start a local vLLM server and run the existing AML benchmark wrapper against
# it on the same GPU node.
set -euo pipefail

model="Qwen/Qwen3.5-35B-A3B"
served_name="Qwen3.5-35B-A3B"
port="8000"
tp_size="1"
max_model_len="32768"
gpu_mem_util="0.92"
quantization="fp8"
tool_call_parser="qwen3_xml"
language_model_only="1"
enable_thinking="auto"
vllm_extra_args=""
server_timeout="3600"
min_cache_free_gb="100"
output_dir=""
env_name="tau2"
appworld_data=""
domains="airline"
splits="test_normal"
levels="1"
methods="ours"
n="1"
agent="llm_wm"
workers="8"
max_steps="50"
rollout_retries="2"
cell_retries="2"
temperature="0.7"
stall_timeout="7200"
extra_args=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) model="$2"; shift 2 ;;
    --served-name) served_name="$2"; shift 2 ;;
    --port) port="$2"; shift 2 ;;
    --tp-size) tp_size="$2"; shift 2 ;;
    --max-model-len) max_model_len="$2"; shift 2 ;;
    --gpu-mem-util) gpu_mem_util="$2"; shift 2 ;;
    --quantization) quantization="$2"; shift 2 ;;
    --tool-call-parser) tool_call_parser="$2"; shift 2 ;;
    --language-model-only) language_model_only="$2"; shift 2 ;;
    --enable-thinking) enable_thinking="$2"; shift 2 ;;
    --vllm-extra-args) vllm_extra_args="$2"; shift 2 ;;
    --server-timeout) server_timeout="$2"; shift 2 ;;
    --min-cache-free-gb) min_cache_free_gb="$2"; shift 2 ;;
    --output-dir) output_dir="$2"; shift 2 ;;
    --env) env_name="$2"; shift 2 ;;
    --appworld-data) appworld_data="$2"; shift 2 ;;
    --domains) domains="$2"; shift 2 ;;
    --splits) splits="$2"; shift 2 ;;
    --levels) levels="$2"; shift 2 ;;
    --methods) methods="$2"; shift 2 ;;
    --n) n="$2"; shift 2 ;;
    --agent) agent="$2"; shift 2 ;;
    --workers) workers="$2"; shift 2 ;;
    --max-steps) max_steps="$2"; shift 2 ;;
    --rollout-retries) rollout_retries="$2"; shift 2 ;;
    --cell-retries) cell_retries="$2"; shift 2 ;;
    --temperature) temperature="$2"; shift 2 ;;
    --stall-timeout) stall_timeout="$2"; shift 2 ;;
    --extra-args) extra_args="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

[[ "$extra_args" == "none" ]] && extra_args=""
[[ "$quantization" == "none" ]] && quantization=""
[[ "$tool_call_parser" == "none" ]] && tool_call_parser=""
[[ "$vllm_extra_args" == "none" ]] && vllm_extra_args=""

case "${enable_thinking,,}" in
  auto) [[ "$env_name" == "tau2" ]] && enable_thinking="0" || enable_thinking="1" ;;
  0|false|no) enable_thinking="0" ;;
  1|true|yes) enable_thinking="1" ;;
  *)
    echo "enable-thinking must be auto|0|1; got: $enable_thinking" >&2
    exit 2
    ;;
esac
if [[ "$vllm_extra_args" == *"--default-chat-template-kwargs"* ]]; then
  echo "Do not pass --default-chat-template-kwargs through --vllm-extra-args." >&2
  echo "Use --enable-thinking 0|1; AML shell interpolation strips embedded JSON quotes." >&2
  exit 2
fi

if [[ -z "$output_dir" ]]; then
  echo "output-dir is required" >&2
  exit 2
fi
case "$env_name" in
  tau2) ;;
  appworld)
    if [[ -z "$appworld_data" ]]; then
      echo "appworld-data is required when --env appworld" >&2
      exit 2
    fi
    ;;
  *)
    echo "run_qwen.sh supports --env tau2|appworld; got: $env_name" >&2
    exit 2
    ;;
esac
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "[qwen][error] this AML job is not running on a GPU compute target." >&2
  exit 3
fi

mkdir -p "$output_dir"
server_log="$output_dir/vllm.log"
base_url="http://127.0.0.1:${port}/v1"

# Keep model downloads in the AML job workspace rather than /tmp, which may be
# backed by the smaller OS/overlay disk. The cache is local to the compute node.
export HF_HOME="${HF_HOME:-$PWD/.cache/huggingface}"
mkdir -p "$HF_HOME"
available_cache_kb="$(df -Pk "$HF_HOME" | awk 'NR == 2 {print $4}')"
required_cache_kb=$((min_cache_free_gb * 1024 * 1024))
if [[ -z "$available_cache_kb" || "$available_cache_kb" -lt "$required_cache_kb" ]]; then
  echo "[qwen][error] model cache needs at least ${min_cache_free_gb} GiB free: $HF_HOME" >&2
  df -h "$HF_HOME" >&2 || true
  exit 3
fi
export TTS_LOCAL_LLM=1

server_pid=""
cleanup() {
  if [[ -n "$server_pid" ]] && kill -0 "$server_pid" 2>/dev/null; then
    echo "[qwen] stopping vLLM server (pid=$server_pid)"
    kill -- "-$server_pid" 2>/dev/null || kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
  fi
  mkdir -p ./outputs
  if [[ -f "$server_log" ]]; then
    cp -f "$server_log" ./outputs/vllm.log 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

serve_env=(
  HOST=127.0.0.1
  TP_SIZE="$tp_size"
  MAX_MODEL_LEN="$max_model_len"
  GPU_MEM_UTIL="$gpu_mem_util"
  LANG_MODEL_ONLY="$language_model_only"
  TRUST_REMOTE_CODE=1
  REASONING_PARSER=qwen3
  ENABLE_AUTO_TOOL_CHOICE=1
  TOOL_CALL_PARSER="${tool_call_parser:-0}"
  QUANTIZATION="$quantization"
  ENABLE_THINKING="$enable_thinking"
  # The AML runtime image has CUDA libraries but no nvcc for FlashInfer JIT.
  VLLM_USE_FLASHINFER_SAMPLER=0
  EXTRA_ARGS="$vllm_extra_args"
)

echo "[qwen] GPU inventory:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo "[qwen] starting model=$model served_name=$served_name base_url=$base_url"
setsid env "${serve_env[@]}" \
  bash scripts/serve_local_model.sh "$model" "$served_name" "$port" \
  >"$server_log" 2>&1 &
server_pid=$!

deadline=$((SECONDS + server_timeout))
while ! curl -fsS "$base_url/models" >/dev/null 2>&1; do
  if ! kill -0 "$server_pid" 2>/dev/null; then
    echo "[qwen][error] vLLM exited during startup." >&2
    tail -n 200 "$server_log" >&2 || true
    exit 4
  fi
  if (( SECONDS >= deadline )); then
    echo "[qwen][error] vLLM did not become ready within ${server_timeout}s." >&2
    tail -n 200 "$server_log" >&2 || true
    exit 4
  fi
  sleep 10
done
echo "[qwen] vLLM is ready"
tail -n 20 "$server_log" || true

local_args=(
  --local
  --local-base-url "$base_url"
  --local-api-key EMPTY
  --llm-agent "$served_name"
  --llm-user "$served_name"
)
if [[ -n "$extra_args" ]]; then
  read -r -a forwarded_args <<< "$extra_args"
  local_args+=("${forwarded_args[@]}")
fi
printf -v combined_extra_args '%s ' "${local_args[@]}"
combined_extra_args="${combined_extra_args% }"

run_status=0
benchmark_args=(
  --env "$env_name" \
  --output-dir "$output_dir" \
  --methods "$methods" \
  --n "$n" \
  --agent "$agent" \
  --workers "$workers" \
  --max-steps "$max_steps" \
  --rollout-retries "$rollout_retries" \
  --cell-retries "$cell_retries" \
  --temperature "$temperature" \
  --aoai-endpoint "$base_url" \
  --aoai-deployment "$served_name" \
  --aoai-api-version "local" \
  --aoai-api-key "none" \
  --managed-identity-client-id "none" \
  --stall-timeout "$stall_timeout" \
  --extra-args "$combined_extra_args"
)
if [[ "$env_name" == "appworld" ]]; then
  benchmark_args+=(
    --appworld-data "$appworld_data"
    --splits "$splits"
    --levels "$levels"
  )
else
  benchmark_args+=(--domains "$domains")
fi

bash aml/src/run_dw.sh "${benchmark_args[@]}" || run_status=$?

exit "$run_status"
