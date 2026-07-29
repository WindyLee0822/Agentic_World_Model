#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
env_name="${TTS_TERMINALBENCH_ENV:-tts-terminalbench}"
bench_root="${TERMINALBENCH_ROOT:-$repo_root/.benchmarks/terminalbench}"
version=""
agent="tts"
task=""
all=0
model="${WM_MODEL:-${AZURE_OPENAI_DEPLOYMENT:-gpt-4o-mini}}"
extra=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --version) version="$2"; shift 2 ;;
    --agent) agent="$2"; shift 2 ;;
    --task) task="$2"; shift 2 ;;
    --all) all=1; shift ;;
    --model) model="$2"; shift 2 ;;
    --env-name) env_name="$2"; shift 2 ;;
    --root) bench_root="$2"; shift 2 ;;
    --) shift; extra+=("$@"); break ;;
    *) extra+=("$1"); shift ;;
  esac
done

if [[ "$version" != "1" && "$version" != "2" ]]; then
  echo "--version must be 1 or 2" >&2
  exit 2
fi
if [[ "$agent" != "tts" && "$agent" != "oracle" ]]; then
  echo "--agent must be tts or oracle" >&2
  exit 2
fi

export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"

if [[ "$version" == "1" ]]; then
  if [[ "$all" -eq 1 ]]; then
    args=(tb run --dataset "terminal-bench-core==0.1.1")
  else
    task="${task:-hello-world}"
    args=(
      tb run
      --dataset-path "$bench_root/terminal-bench-1/tasks"
      --task-id "$task"
    )
  fi
  if [[ "$agent" == "oracle" ]]; then
    args+=(--agent oracle)
  else
    args+=(
      --agent-import-path tts.env.terminalbench_legacy:LegacyTerminalBenchAgent
      --model "$model"
    )
  fi
else
  if [[ "$all" -eq 1 ]]; then
    args=(harbor run --dataset "terminal-bench@2.0")
  else
    task="${task:-cancel-async-tasks}"
    args=(
      harbor trials start
      -p "$bench_root/terminal-bench-2/$task"
    )
  fi
  if [[ "$agent" == "oracle" ]]; then
    args+=(--agent oracle)
  else
    args+=(
      --agent tts.env.terminalbench_harbor:HarborTerminalBenchAgent
      --model "$model"
    )
  fi
fi

args+=("${extra[@]}")
printf 'Running: conda run -n %q' "$env_name"
printf ' %q' "${args[@]}"
printf '\n'
conda run -n "$env_name" "${args[@]}"
