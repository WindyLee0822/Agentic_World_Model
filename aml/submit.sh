#!/usr/bin/env bash
# Convenience wrapper around `az ml job create` for the dw jobs.
#
# Requires: `az login` (already done on this box) and the `az ml` CLI extension
# (already installed). No Python SDK needed.
#
# Pick the job (env) with AML_JOB_FILE (default aml/job.yml = AppWorld):
#   bash aml/submit.sh                                   # AppWorld
#   AML_JOB_FILE=aml/job.appworld.wm.yml bash aml/submit.sh  # AppWorld interactive WM
#   AML_JOB_FILE=aml/job.tau2.yml bash aml/submit.sh     # tau2
#   AML_JOB_FILE=aml/job.officebench.yml bash aml/submit.sh   # OfficeBench
#
# Configure a run by editing `inputs:` in the chosen job file, or override per
# submit (extra args are forwarded verbatim to `az ml job create`):
#   bash aml/submit.sh --stream
#   AML_JOB_FILE=aml/job.tau2.yml bash aml/submit.sh --set inputs.aoai_api_key=$AOAI_KEY
set -euo pipefail

SUBSCRIPTION="${AML_SUBSCRIPTION:-acc09744-1ee3-4242-b375-93421c63af0c}"
RESOURCE_GROUP="${AML_RESOURCE_GROUP:-Singularity}"
WORKSPACE="${AML_WORKSPACE:-M365Research}"

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
job_file="${AML_JOB_FILE:-$repo_root/aml/job.yml}"
[[ "$job_file" = /* ]] || job_file="$repo_root/$job_file"

if [[ "$(basename "$job_file")" == "job.officebench.yml" ]]; then
  if ! az ml environment show \
    --name tts-officebench-local \
    --version 20260803 \
    --resource-group "$RESOURCE_GROUP" \
    --workspace-name "$WORKSPACE" \
    --subscription "$SUBSCRIPTION" >/dev/null 2>&1; then
    echo "Registering the OfficeBench no-Docker AML environment." >&2
    az ml environment create \
      --file "$repo_root/aml/environment/officebench-local.yml" \
      --resource-group "$RESOURCE_GROUP" \
      --workspace-name "$WORKSPACE" \
      --subscription "$SUBSCRIPTION"
  fi
fi

echo "Submitting $job_file to workspace=$WORKSPACE (rg=$RESOURCE_GROUP, sub=$SUBSCRIPTION)"
exec az ml job create \
  --file "$job_file" \
  --resource-group "$RESOURCE_GROUP" \
  --workspace-name "$WORKSPACE" \
  --subscription "$SUBSCRIPTION" \
  "$@"
