#!/usr/bin/env bash
# Submit the local-Qwen AML job. This mirrors submit.sh, but also creates or
# reuses a dedicated one-GPU compute cluster because the existing dw AML
# cluster is CPU-only.
set -euo pipefail

SUBSCRIPTION="${AML_SUBSCRIPTION:-acc09744-1ee3-4242-b375-93421c63af0c}"
RESOURCE_GROUP="${AML_RESOURCE_GROUP:-Singularity}"
WORKSPACE="${AML_WORKSPACE:-M365Research}"

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
job_file="${AML_QWEN_JOB_FILE:-$repo_root/aml/job.qwen.tau2.yml}"
[[ "$job_file" = /* ]] || job_file="$repo_root/$job_file"

if [[ "$(basename "$job_file")" == "job.qwen.officebench.yml" ]]; then
  if ! az ml environment show \
    --name tts-officebench-qwen-local \
    --version 20260806 \
    --resource-group "$RESOURCE_GROUP" \
    --workspace-name "$WORKSPACE" \
    --subscription "$SUBSCRIPTION" >/dev/null 2>&1; then
    echo "Registering the OfficeBench Qwen AML environment." >&2
    az ml environment create \
      --file "$repo_root/aml/environment/officebench-qwen-local.yml" \
      --resource-group "$RESOURCE_GROUP" \
      --workspace-name "$WORKSPACE" \
      --subscription "$SUBSCRIPTION"
  fi
fi

job_compute="$(
  sed -nE 's/^[[:space:]]*compute:[[:space:]]*azureml:([^[:space:]#]+).*/\1/p' \
    "$job_file" | head -n 1
)"
if [[ -z "$job_compute" && -z "${AML_QWEN_COMPUTE:-}" ]]; then
  echo "Could not determine a compute target from $job_file." >&2
  echo "Set AML_QWEN_COMPUTE to an existing GPU AML compute name." >&2
  exit 2
fi

compute_name="${AML_QWEN_COMPUTE:-$job_compute}"
vm_size="${AML_QWEN_VM_SIZE:-}"
compute_tier="${AML_QWEN_COMPUTE_TIER:-dedicated}"

if compute_info="$(
  az ml compute show \
    --name "$compute_name" \
    --resource-group "$RESOURCE_GROUP" \
    --workspace-name "$WORKSPACE" \
    --subscription "$SUBSCRIPTION" \
    --query "join('|', [type, size, provisioning_state])" -o tsv 2>/dev/null
)"; then
  IFS='|' read -r existing_type existing_size provisioning_state <<< "$compute_info"
  if [[ "${existing_type,,}" != "amlcompute" ]]; then
    echo "AML compute '$compute_name' is type '$existing_type', not AmlCompute." >&2
    exit 2
  fi
  if [[ -n "$vm_size" && "${existing_size,,}" != "${vm_size,,}" ]]; then
    echo "AML compute '$compute_name' uses '$existing_size', expected '$vm_size'." >&2
    echo "Set AML_QWEN_VM_SIZE=$existing_size to explicitly use it, or choose another AML_QWEN_COMPUTE." >&2
    exit 2
  fi
  vm_size="$existing_size"
  echo "Using existing AML GPU compute: $compute_name ($existing_size, state=$provisioning_state)"
else
  if [[ "${AML_QWEN_CREATE_COMPUTE:-0}" == "0" ]]; then
    echo "AML compute '$compute_name' does not exist and AML_QWEN_CREATE_COMPUTE=0." >&2
    echo "Choose an existing GPU target with AML_QWEN_COMPUTE/AML_QWEN_VM_SIZE," >&2
    echo "or set AML_QWEN_CREATE_COMPUTE=1 after obtaining quota for the requested SKU." >&2
    exit 2
  fi
  if [[ -z "$vm_size" ]]; then
    echo "AML_QWEN_VM_SIZE is required when creating compute '$compute_name'." >&2
    exit 2
  fi
  echo "Creating AML GPU compute: $compute_name ($vm_size, tier=$compute_tier)"
  az ml compute create \
    --name "$compute_name" \
    --type AmlCompute \
    --size "$vm_size" \
    --min-instances 0 \
    --max-instances 1 \
    --idle-time-before-scale-down 300 \
    --tier "$compute_tier" \
    --resource-group "$RESOURCE_GROUP" \
    --workspace-name "$WORKSPACE" \
    --subscription "$SUBSCRIPTION" \
    --only-show-errors \
    --output none
fi

echo "Submitting $job_file to workspace=$WORKSPACE on compute=$compute_name"
exec az ml job create \
  --file "$job_file" \
  --resource-group "$RESOURCE_GROUP" \
  --workspace-name "$WORKSPACE" \
  --subscription "$SUBSCRIPTION" \
  --set "compute=azureml:$compute_name" \
  "$@"
