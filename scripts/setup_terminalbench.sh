#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
env_name="${TTS_TERMINALBENCH_ENV:-tts-terminalbench}"
bench_root="${TERMINALBENCH_ROOT:-$repo_root/.benchmarks/terminalbench}"
smoke=0

tb1_url="https://github.com/harbor-framework/terminal-bench.git"
tb1_sha="91e10457b5410f16c44364da1a34cb6de8c488a5"
tb2_url="https://github.com/harbor-framework/terminal-bench-2.git"
tb2_sha="69671fbaac6d67a7ef0dfec016cc38a64ef7a77c"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-name) env_name="$2"; shift 2 ;;
    --root) bench_root="$2"; shift 2 ;;
    --smoke) smoke=1; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

command -v conda >/dev/null 2>&1 || {
  echo "conda is required to create the Terminal-Bench environment." >&2
  exit 2
}
command -v git >/dev/null 2>&1 || {
  echo "git is required to download the Terminal-Bench datasets." >&2
  exit 2
}
command -v docker >/dev/null 2>&1 || {
  echo "Docker is required by both Terminal-Bench harnesses." >&2
  exit 2
}
docker info >/dev/null 2>&1 || {
  echo "Docker is installed, but its daemon is not usable by this user." >&2
  exit 2
}

if ! conda env list | awk '{print $1}' | grep -Fxq "$env_name"; then
  conda env create -n "$env_name" -f "$repo_root/aml/environment/conda.terminalbench.yml"
fi

conda run -n "$env_name" python -m pip install \
  -e "$repo_root[terminalbench,llm,progress]"

mkdir -p "$bench_root"

checkout_at() {
  local url="$1"
  local dest="$2"
  local sha="$3"
  if [[ ! -d "$dest/.git" ]]; then
    git clone --filter=blob:none --no-checkout "$url" "$dest"
  fi
  git -C "$dest" fetch --depth 1 origin "$sha"
  git -C "$dest" checkout --detach "$sha"
}

checkout_at "$tb1_url" "$bench_root/terminal-bench-1" "$tb1_sha"
checkout_at "$tb2_url" "$bench_root/terminal-bench-2" "$tb2_sha"

conda run -n "$env_name" tb --help >/dev/null
conda run -n "$env_name" harbor --help >/dev/null

printf '%s\n' \
  "Terminal-Bench environment ready." \
  "  conda env: $env_name" \
  "  TB1 tasks: $bench_root/terminal-bench-1/tasks" \
  "  TB2 tasks: $bench_root/terminal-bench-2" \
  "  runner:    $repo_root/scripts/run_terminalbench.sh"

if [[ "$smoke" -eq 1 ]]; then
  "$repo_root/scripts/run_terminalbench.sh" --version 1 --agent oracle --task hello-world
  "$repo_root/scripts/run_terminalbench.sh" --version 2 --agent oracle --task cancel-async-tasks
fi
