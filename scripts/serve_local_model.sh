#!/usr/bin/env bash
# Serve a self-hosted model as an OpenAI-compatible endpoint for the dw experiment harness.
#
# The dw harness (experiments/run.py --local) talks to any OpenAI-compatible /v1 server. vLLM and
# SGLang both expose one; this script launches vLLM by default. Once it is up, run experiments with:
#
#   PYTHONPATH=src python experiments/run.py --local \
#     --llm-agent Qwen-AgentWorld-35B-A3B --env appworld --methods b0 --N 1 --tasks 8
#
# Usage:
#   bash scripts/serve_local_model.sh <MODEL_PATH_OR_HF_ID> [SERVED_NAME] [PORT]
#
#   MODEL_PATH_OR_HF_ID  local dir with the weights, or a HuggingFace repo id (required)
#   SERVED_NAME          model id the server advertises (default: Qwen-AgentWorld-35B-A3B)
#                        -> pass the SAME value to `run.py --llm-agent`
#   PORT                 listen port (default: 8000)
#
# Env overrides:
#   SERVE_BACKEND     vllm | sglang        (default: vllm)
#   TP_SIZE           tensor-parallel size (default: number of visible GPUs, else 1)
#   MAX_MODEL_LEN     context length       (default: 32768)
#   GPU_MEM_UTIL      vLLM --gpu-memory-utilization (default: 0.92)
#   EXTRA_ARGS        extra args forwarded verbatim to the server
#
# Examples:
#   # serve local weights on a single A100 80GB
#   bash scripts/serve_local_model.sh /models/Qwen-AgentWorld-35B-A3B
#
#   # two GPUs, longer context, custom port
#   TP_SIZE=2 MAX_MODEL_LEN=65536 bash scripts/serve_local_model.sh /models/qwen aw-35b 8010
set -euo pipefail

MODEL="${1:-}"
SERVED_NAME="${2:-Qwen-AgentWorld-35B-A3B}"
PORT="${3:-8000}"
HOST="${HOST:-0.0.0.0}"
BACKEND="${SERVE_BACKEND:-vllm}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.92}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
# Model-specific toggles. Qwen-AgentWorld ships visual module *definitions* but language-only
# weights, so --language-model-only is REQUIRED (else vLLM tries to build the vision tower and
# fails); it is also a reasoning model (<think> blocks) served best with the qwen3 reasoning parser.
# Default ON for the Qwen-AgentWorld family; set to 0 for other models.
REASONING_PARSER="${REASONING_PARSER:-qwen3}"
LANG_MODEL_ONLY="${LANG_MODEL_ONLY:-1}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-1}"
# Tool/function calling (required by τ², which uses tool_choice="auto"). Qwen-AgentWorld's chat
# template emits XML-style <function=..><parameter=..> calls -> the qwen3_xml parser. vLLM rejects
# tool_choice requests unless BOTH are set. Harmless for AppWorld/OfficeBench (they don't tool-call).
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-qwen3_xml}"
ENABLE_AUTO_TOOL_CHOICE="${ENABLE_AUTO_TOOL_CHOICE:-1}"
CPU_OFFLOAD_GB="${CPU_OFFLOAD_GB:-0}"   # offload N GB of weights to host RAM (helps single-GPU fit)
QUANTIZATION="${QUANTIZATION:-}"        # e.g. fp8 to roughly halve the GPU memory footprint

if [[ -z "${MODEL}" ]]; then
  echo "error: MODEL (local weights dir or HF repo id) is required." >&2
  echo "usage: bash scripts/serve_local_model.sh <MODEL> [SERVED_NAME] [PORT]" >&2
  exit 2
fi

# Default tensor-parallel size to the number of visible GPUs.
if [[ -z "${TP_SIZE:-}" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    TP_SIZE="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"
  fi
  TP_SIZE="${TP_SIZE:-1}"
  [[ "${TP_SIZE}" -ge 1 ]] 2>/dev/null || TP_SIZE=1
fi

# Assemble the model-specific / memory flags shared by both backends.
VLLM_FLAGS=()
SGLANG_FLAGS=()
[[ "${LANG_MODEL_ONLY}" == "1" ]] && VLLM_FLAGS+=(--language-model-only)
[[ "${TRUST_REMOTE_CODE}" == "1" ]] && { VLLM_FLAGS+=(--trust-remote-code); SGLANG_FLAGS+=(--trust-remote-code); }
[[ -n "${REASONING_PARSER}" && "${REASONING_PARSER}" != "0" ]] && { VLLM_FLAGS+=(--reasoning-parser "${REASONING_PARSER}"); SGLANG_FLAGS+=(--reasoning-parser "${REASONING_PARSER}"); }
if [[ "${ENABLE_AUTO_TOOL_CHOICE}" == "1" && -n "${TOOL_CALL_PARSER}" && "${TOOL_CALL_PARSER}" != "0" ]]; then
  VLLM_FLAGS+=(--enable-auto-tool-choice --tool-call-parser "${TOOL_CALL_PARSER}")
  SGLANG_FLAGS+=(--tool-call-parser "${TOOL_CALL_PARSER}")
fi
[[ "${CPU_OFFLOAD_GB}" != "0" ]] && VLLM_FLAGS+=(--cpu-offload-gb "${CPU_OFFLOAD_GB}")
[[ -n "${QUANTIZATION}" ]] && VLLM_FLAGS+=(--quantization "${QUANTIZATION}")

echo "[serve] backend=${BACKEND} model=${MODEL} served-name=${SERVED_NAME}"
echo "[serve] host=${HOST} port=${PORT} tp=${TP_SIZE} max_model_len=${MAX_MODEL_LEN} gpu_mem_util=${GPU_MEM_UTIL}"
echo "[serve] model flags: reasoning_parser=${REASONING_PARSER} language_model_only=${LANG_MODEL_ONLY} trust_remote_code=${TRUST_REMOTE_CODE} tool_parser=${TOOL_CALL_PARSER} cpu_offload_gb=${CPU_OFFLOAD_GB} quant=${QUANTIZATION:-none}"
echo "[serve] once up, run experiments with:  --local --llm-agent ${SERVED_NAME}"

case "${BACKEND}" in
  vllm)
    if ! python -c "import vllm" >/dev/null 2>&1; then
      echo "[serve] vLLM not installed. Install it with:  pip install vllm" >&2
      exit 3
    fi
    exec python -m vllm.entrypoints.openai.api_server \
      --model "${MODEL}" \
      --served-model-name "${SERVED_NAME}" \
      --host "${HOST}" --port "${PORT}" \
      --tensor-parallel-size "${TP_SIZE}" \
      --max-model-len "${MAX_MODEL_LEN}" \
      --gpu-memory-utilization "${GPU_MEM_UTIL}" \
      "${VLLM_FLAGS[@]}" \
      ${EXTRA_ARGS}
    ;;
  sglang)
    if ! python -c "import sglang" >/dev/null 2>&1; then
      echo "[serve] SGLang not installed. Install it with:  pip install 'sglang[all]'" >&2
      exit 3
    fi
    exec python -m sglang.launch_server \
      --model-path "${MODEL}" \
      --served-model-name "${SERVED_NAME}" \
      --host "${HOST}" --port "${PORT}" \
      --tp-size "${TP_SIZE}" \
      --context-length "${MAX_MODEL_LEN}" \
      "${SGLANG_FLAGS[@]}" \
      ${EXTRA_ARGS}
    ;;
  *)
    echo "error: unknown SERVE_BACKEND='${BACKEND}' (use vllm | sglang)." >&2
    exit 2
    ;;
esac
