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

officebench_overrides=()
if [[ "$(basename "$job_file")" == "job.officebench.yml" ]]; then
  explicit_docker_host=0
  managed_identity_client_id="${AML_MANAGED_IDENTITY_CLIENT_ID:-258f6dab-3e80-4254-83e3-bfb24a9ae8d9}"
  for arg in "$@"; do
    if [[ "$arg" == inputs.docker_host=* ]]; then
      explicit_docker_host=1
    elif [[ "$arg" == inputs.managed_identity_client_id=* ]]; then
      managed_identity_client_id="${arg#*=}"
    fi
  done

  if [[ "$explicit_docker_host" -eq 0 ]]; then
    if [[ -n "${AML_OFFICEBENCH_DOCKER_HOST:-}" ]]; then
      if [[ "$AML_OFFICEBENCH_DOCKER_HOST" == ssh://* ]] && {
        [[ -z "${AML_OFFICEBENCH_DOCKER_KEY_VAULT_URL:-}" ]] ||
        [[ -z "${AML_OFFICEBENCH_DOCKER_SSH_KEY_SECRET:-}" ]];
      }; then
        echo "AML_OFFICEBENCH_DOCKER_KEY_VAULT_URL and AML_OFFICEBENCH_DOCKER_SSH_KEY_SECRET are required for an ssh:// Docker host." >&2
        exit 2
      fi
      docker_host="$AML_OFFICEBENCH_DOCKER_HOST"
      docker_vault="${AML_OFFICEBENCH_DOCKER_KEY_VAULT_URL:-none}"
      docker_key_secret="${AML_OFFICEBENCH_DOCKER_SSH_KEY_SECRET:-none}"
      docker_known_hosts_secret="${AML_OFFICEBENCH_DOCKER_SSH_KNOWN_HOSTS_SECRET:-none}"
    elif [[ "${AML_OFFICEBENCH_AUTO_PROVISION:-1}" != "0" ]]; then
      echo "OfficeBench requires Docker; provisioning/reusing its remote Docker VM." >&2
      provisioned="$(
        python "$repo_root/aml/src/provision_officebench_docker.py" \
          --subscription "$SUBSCRIPTION" \
          --resource-group "$RESOURCE_GROUP" \
          --workspace "$WORKSPACE" \
          --managed-identity-client-id "$managed_identity_client_id" \
          --managed-identity-object-id "${AML_MANAGED_IDENTITY_OBJECT_ID:-}" \
          --vm-resource-group "${AML_OFFICEBENCH_DOCKER_RESOURCE_GROUP:-$RESOURCE_GROUP}" \
          --vm-name "${AML_OFFICEBENCH_DOCKER_VM_NAME:-officebench-docker}" \
          --vm-size "${AML_OFFICEBENCH_DOCKER_VM_SIZE:-Standard_D16s_v5}" \
          --vm-user "${AML_OFFICEBENCH_DOCKER_VM_USER:-officebench}" \
          --ssh-key-path "${AML_OFFICEBENCH_DOCKER_SSH_KEY_PATH:-$HOME/.ssh/officebench_aml}" \
          --output lines
      )"
      mapfile -t docker_settings <<< "$provisioned"
      if [[ "${#docker_settings[@]}" -ne 4 ]]; then
        echo "OfficeBench Docker provisioner returned an invalid result." >&2
        exit 2
      fi
      docker_host="${docker_settings[0]}"
      docker_vault="${docker_settings[1]}"
      docker_key_secret="${docker_settings[2]}"
      docker_known_hosts_secret="${docker_settings[3]}"
    else
      echo "OfficeBench cannot run on standard AML command compute without a remote Docker daemon." >&2
      echo "Enable automatic provisioning (default), set AML_OFFICEBENCH_DOCKER_HOST, or pass --set inputs.docker_host=ssh://USER@HOST." >&2
      exit 2
    fi

    officebench_overrides=(
      --set
      "inputs.docker_host=$docker_host"
      "inputs.docker_key_vault_url=$docker_vault"
      "inputs.docker_ssh_key_secret=$docker_key_secret"
      "inputs.docker_ssh_known_hosts_secret=$docker_known_hosts_secret"
    )
  fi
fi

echo "Submitting $job_file to workspace=$WORKSPACE (rg=$RESOURCE_GROUP, sub=$SUBSCRIPTION)"
exec az ml job create \
  --file "$job_file" \
  --resource-group "$RESOURCE_GROUP" \
  --workspace-name "$WORKSPACE" \
  --subscription "$SUBSCRIPTION" \
  "${officebench_overrides[@]}" \
  "$@"
