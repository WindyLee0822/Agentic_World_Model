#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"
source get_ready.sh

TTS_TERMINALBENCH_AGENT_MAX_TOKENS=8000 \
TTS_LLM_MIN_INTERVAL_S=0.5 \
PYTHONUNBUFFERED=1 \
python experiments/run.py \
  --env terminalbench \
  --terminalbench-versions 1,2 \
  --methods ours,b2 \
  --temperature 1 \
  --N 1,2,4,8,16 \
  --workers 2 \
  --max-steps 40 \
  --llm-agent gpt-5.4 \
  --resume \
  --out experiments/terminal-full-ours-b2-5.4-cwd-fixed
