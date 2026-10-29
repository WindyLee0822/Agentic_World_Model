"""Submit an AML YAML job with the compute-instance managed identity (no az login)."""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, "/home/azureuser/.azure/cliextensions-local/ml")

from azure.ai.ml import MLClient, load_job
from azure.ai.ml.entities import UserIdentityConfiguration
from azure.identity import ManagedIdentityCredential


def _coerce(value: str, current):
    existing = getattr(current, "_data", current)
    if isinstance(existing, bool):
        return value.lower() in {"1", "true", "yes"}
    if isinstance(existing, int) and not isinstance(existing, bool):
        return int(value)
    if isinstance(existing, float):
        return float(value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True)
    parser.add_argument("--compute")
    parser.add_argument("--name")
    parser.add_argument("--set", nargs="*", default=[])
    parser.add_argument("--stream", action="store_true")
    args = parser.parse_args()

    job = load_job(args.file)
    if args.compute:
        job.compute = args.compute.removeprefix("azureml:")
    if args.name:
        job.name = args.name
    for assignment in args.set:
        if "=" not in assignment:
            raise SystemExit(f"invalid --set value {assignment!r}; expected inputs.NAME=VALUE")
        path, value = assignment.split("=", 1)
        if not path.startswith("inputs."):
            raise SystemExit(f"managed-identity submit currently supports inputs.* overrides, got {path!r}")
        key = path.removeprefix("inputs.")
        if key not in job.inputs:
            raise SystemExit(f"unknown job input {key!r}")
        current = job.inputs[key]
        coerced = _coerce(value, current)
        if hasattr(current, "_data"):
            current._data = coerced
        else:
            job.inputs[key] = coerced

    client_id = os.environ.get(
        "AML_MANAGED_IDENTITY_CLIENT_ID",
        os.environ.get("DEFAULT_IDENTITY_CLIENT_ID", ""),
    )
    if not client_id:
        raise SystemExit("AML_MANAGED_IDENTITY_CLIENT_ID is required")
    client = MLClient(
        ManagedIdentityCredential(client_id=client_id),
        os.environ.get("AML_SUBSCRIPTION", "acc09744-1ee3-4242-b375-93421c63af0c"),
        os.environ.get("AML_RESOURCE_GROUP", "external-collaborations"),
        os.environ.get("AML_WORKSPACE", "M365Research-External-EUS"),
    )
    job.identity = UserIdentityConfiguration()
    created = client.jobs.create_or_update(job)
    print(f"JOB_NAME={created.name}")
    print(f"STATUS={created.status}")
    print(f"STUDIO_URL={created.studio_url}")
    if args.stream:
        client.jobs.stream(created.name)


if __name__ == "__main__":
    main()
