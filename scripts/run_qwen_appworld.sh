#!/usr/bin/env bash
# AWM on all AppWorld test tasks using an existing Qwen server.
set -euo pipefail
task_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$task_root"
export APPWORLD_ROOT="${APPWORLD_ROOT:-$task_root}"
exec "${QWEN_APPWORLD_PYTHON:-python}" experiments/run_qwen.py \
  --env appworld --method ours --agent llm_wm \
  --local --local-base-url "${QWEN_APPWORLD_URL:-http://localhost:8000/v1}" \
  --llm-agent "${QWEN_MODEL:-Qwen3.5-35B-A3B}" \
  --N 1,2,4,8,16 --temperature 1.0 --max-steps 40 \
  --out experiments/results/qwen_appworld_awm "$@"
