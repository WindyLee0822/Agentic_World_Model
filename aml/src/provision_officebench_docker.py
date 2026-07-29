#!/usr/bin/env python3
"""Provision the remote Docker daemon used by OfficeBench AML jobs."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProvisionedDocker:
    docker_host: str
    key_vault_url: str
    ssh_key_secret: str
    ssh_known_hosts_secret: str

    def as_dict(self) -> dict[str, str]:
        return {
            "docker_host": self.docker_host,
            "key_vault_url": self.key_vault_url,
            "ssh_key_secret": self.ssh_key_secret,
            "ssh_known_hosts_secret": self.ssh_known_hosts_secret,
        }


def _log(message: str) -> None:
    print(f"[officebench][provision] {message}", file=sys.stderr)


def _run(
    args: list[str],
    *,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(
        args,
        capture_output=True,
        text=True,
        check=False,
    )
    if check and proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()
        raise RuntimeError(f"{' '.join(args[:4])} failed: {detail[-2000:]}")
    return proc


def _az(args: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return _run(["az", *args, "--only-show-errors"], check=check)


def _az_text(args: list[str]) -> str:
    proc = _az([*args, "--output", "tsv"])
    return proc.stdout.strip()


def _require_commands() -> None:
    missing = [
        name
        for name in ("az", "ssh", "ssh-keygen", "ssh-keyscan")
        if shutil.which(name) is None
    ]
    if missing:
        raise RuntimeError(f"missing required command(s): {', '.join(missing)}")


def _validate_args(args: argparse.Namespace) -> None:
    if not re.fullmatch(r"[a-z_][a-z0-9_-]{0,31}", args.vm_user):
        raise RuntimeError(
            "vm_user must be a lowercase Linux username containing only letters, "
            "digits, underscores, or hyphens"
        )
    if not 1 <= args.ssh_port <= 65535:
        raise RuntimeError("ssh_port must be between 1 and 65535")


def _ensure_ssh_key(path: Path) -> tuple[Path, Path]:
    private_key = path.expanduser().resolve()
    public_key = Path(str(private_key) + ".pub")
    if private_key.is_file() and public_key.is_file():
        return private_key, public_key
    if private_key.is_file():
        public = _run(["ssh-keygen", "-y", "-f", str(private_key)]).stdout
        public_key.write_text(public.rstrip("\n") + "\n", encoding="utf-8")
        private_key.chmod(0o600)
        return private_key, public_key
    public_key.unlink(missing_ok=True)
    private_key.parent.mkdir(parents=True, exist_ok=True)
    _log(f"creating SSH key {private_key}")
    _run(
        [
            "ssh-keygen",
            "-q",
            "-t",
            "ed25519",
            "-N",
            "",
            "-f",
            str(private_key),
            "-C",
            "officebench-aml",
        ]
    )
    private_key.chmod(0o600)
    return private_key, public_key


def _workspace_location(args: argparse.Namespace) -> str:
    return _az_text(
        [
            "ml",
            "workspace",
            "show",
            "--name",
            args.workspace,
            "--resource-group",
            args.resource_group,
            "--subscription",
            args.subscription,
            "--query",
            "location",
        ]
    )


def _workspace_key_vault(args: argparse.Namespace) -> tuple[str, str, str]:
    vault_id = _az_text(
        [
            "ml",
            "workspace",
            "show",
            "--name",
            args.workspace,
            "--resource-group",
            args.resource_group,
            "--subscription",
            args.subscription,
            "--query",
            "key_vault",
        ]
    )
    if not vault_id:
        raise RuntimeError("the AML workspace does not report a default Key Vault")
    vault_name = vault_id.rstrip("/").split("/")[-1]
    return vault_id, vault_name, f"https://{vault_name}.vault.azure.net"


def _ensure_vm(
    args: argparse.Namespace,
    public_key: Path,
    location: str,
) -> str:
    vm_rg = args.vm_resource_group or args.resource_group
    group_exists = _az_text(
        [
            "group",
            "exists",
            "--name",
            vm_rg,
            "--subscription",
            args.subscription,
        ]
    ).lower() == "true"
    if not group_exists:
        _log(f"creating resource group {vm_rg}")
        _az(
            [
                "group",
                "create",
                "--name",
                vm_rg,
                "--location",
                location,
                "--subscription",
                args.subscription,
                "--output",
                "none",
            ]
        )
    show = _az(
        [
            "vm",
            "show",
            "--name",
            args.vm_name,
            "--resource-group",
            vm_rg,
            "--subscription",
            args.subscription,
            "--output",
            "none",
        ],
        check=False,
    )
    if show.returncode != 0:
        _log(f"creating VM {vm_rg}/{args.vm_name} ({args.vm_size})")
        _az(
            [
                "vm",
                "create",
                "--name",
                args.vm_name,
                "--resource-group",
                vm_rg,
                "--subscription",
                args.subscription,
                "--location",
                location,
                "--image",
                args.vm_image,
                "--size",
                args.vm_size,
                "--admin-username",
                args.vm_user,
                "--authentication-type",
                "ssh",
                "--ssh-key-values",
                str(public_key),
                "--public-ip-sku",
                "Standard",
                "--nsg-rule",
                "SSH",
                "--output",
                "none",
            ]
        )
    else:
        _log(f"reusing VM {vm_rg}/{args.vm_name}")
        _az(
            [
                "vm",
                "start",
                "--name",
                args.vm_name,
                "--resource-group",
                vm_rg,
                "--subscription",
                args.subscription,
                "--output",
                "none",
            ]
        )
        _az(
            [
                "vm",
                "user",
                "update",
                "--name",
                args.vm_name,
                "--resource-group",
                vm_rg,
                "--subscription",
                args.subscription,
                "--username",
                args.vm_user,
                "--ssh-key-value",
                str(public_key),
                "--output",
                "none",
            ]
        )

    _log("installing and starting Docker on the VM")
    vm_user = shlex.quote(args.vm_user)
    install_script = f"""set -eu
retry() {{
  attempt=1
  while ! "$@"; do
    if [ "$attempt" -ge 3 ]; then
      return 1
    fi
    sleep $((attempt * 5))
    attempt=$((attempt + 1))
  done
}}
if ! command -v docker >/dev/null 2>&1; then
  export DEBIAN_FRONTEND=noninteractive
  retry apt-get update
  retry apt-get install -y ca-certificates curl
  retry curl -fsSL https://get.docker.com -o /tmp/get-docker.sh
  retry sh /tmp/get-docker.sh
fi
usermod -aG docker {vm_user}
systemctl enable --now docker
docker info >/dev/null
echo OFFICEBENCH_DOCKER_READY
"""
    run_command = _az(
        [
            "vm",
            "run-command",
            "invoke",
            "--name",
            args.vm_name,
            "--resource-group",
            vm_rg,
            "--subscription",
            args.subscription,
            "--command-id",
            "RunShellScript",
            "--scripts",
            install_script,
        ]
    )
    try:
        response = json.loads(run_command.stdout)
        messages = "\n".join(
            str(item.get("message", ""))
            for item in response.get("value", [])
            if isinstance(item, dict)
        )
    except (AttributeError, json.JSONDecodeError, TypeError) as exc:
        raise RuntimeError("Azure returned an invalid VM run-command response") from exc
    if "OFFICEBENCH_DOCKER_READY" not in messages:
        raise RuntimeError(f"Docker installation failed on the VM: {messages[-2000:]}")

    public_ip = _az_text(
        [
            "vm",
            "show",
            "--name",
            args.vm_name,
            "--resource-group",
            vm_rg,
            "--subscription",
            args.subscription,
            "--show-details",
            "--query",
            "publicIps",
        ]
    )
    if not public_ip:
        raise RuntimeError(
            f"VM {vm_rg}/{args.vm_name} has no public IP; configure a reachable "
            "AML-to-VM private network or provision the default public endpoint"
        )
    return public_ip.split(",")[0].strip()


def _scan_host_key(host: str, port: int = 22) -> str:
    last_error = ""
    for attempt in range(12):
        proc = _run(
            ["ssh-keyscan", "-T", "10", "-p", str(port), host],
            check=False,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            return proc.stdout
        last_error = (proc.stderr or proc.stdout or "").strip()
        time.sleep(min(5 + attempt, 15))
    raise RuntimeError(f"could not read the SSH host key for {host}:{port}: {last_error}")


def _verify_ssh(
    host: str,
    user: str,
    private_key: Path,
    known_hosts_path: Path,
    port: int = 22,
) -> None:
    last_error = ""
    command = [
        "ssh",
        "-p",
        str(port),
        "-i",
        str(private_key),
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=15",
        "-o",
        "StrictHostKeyChecking=yes",
        "-o",
        f"UserKnownHostsFile={known_hosts_path}",
        f"{user}@{host}",
        "docker",
        "info",
    ]
    for attempt in range(12):
        proc = _run(command, check=False)
        if proc.returncode == 0:
            return
        last_error = (proc.stderr or proc.stdout or "").strip()
        time.sleep(min(5 + attempt, 15))
    raise RuntimeError(f"Docker is not reachable over SSH: {last_error[-2000:]}")


def _store_secret(
    subscription: str,
    vault_name: str,
    secret_name: str,
    path: Path,
) -> None:
    try:
        _az(
            [
                "keyvault",
                "secret",
                "set",
                "--vault-name",
                vault_name,
                "--subscription",
                subscription,
                "--name",
                secret_name,
                "--file",
                str(path),
                "--encoding",
                "utf-8",
                "--output",
                "none",
            ]
        )
    except RuntimeError as exc:
        raise RuntimeError(
            f"could not write secret {secret_name!r} to Key Vault {vault_name!r}; "
            "an RBAC-enabled vault requires a data-plane role such as "
            "'Key Vault Secrets Officer' for the submitting identity"
        ) from exc


def _managed_identity_object_id(args: argparse.Namespace) -> str:
    if args.managed_identity_object_id:
        return args.managed_identity_object_id
    try:
        object_id = _az_text(
            [
                "identity",
                "list",
                "--subscription",
                args.subscription,
                "--query",
                f"[?clientId=='{args.managed_identity_client_id}'].principalId | [0]",
            ]
        )
    except RuntimeError:
        object_id = ""
    if object_id:
        return object_id
    try:
        object_id = _az_text(
            [
                "ad",
                "sp",
                "show",
                "--id",
                args.managed_identity_client_id,
                "--query",
                "id",
            ]
        )
    except RuntimeError as exc:
        raise RuntimeError(
            "could not resolve the AML managed identity object ID; set "
            "AML_MANAGED_IDENTITY_OBJECT_ID to avoid Microsoft Graph lookup"
        ) from exc
    if not object_id:
        raise RuntimeError("the AML managed identity object ID could not be resolved")
    return object_id


def _grant_secret_access(
    args: argparse.Namespace,
    vault_id: str,
    vault_name: str,
) -> None:
    if not args.managed_identity_client_id:
        raise RuntimeError("managed_identity_client_id is required for OfficeBench Key Vault access")
    rbac_enabled = _az_text(
        [
            "keyvault",
            "show",
            "--name",
            vault_name,
            "--subscription",
            args.subscription,
            "--query",
            "properties.enableRbacAuthorization",
        ]
    ).lower() == "true"
    object_id = _managed_identity_object_id(args)
    if rbac_enabled:
        assigned = _az_text(
            [
                "role",
                "assignment",
                "list",
                "--assignee",
                object_id,
                "--scope",
                vault_id,
                "--subscription",
                args.subscription,
                "--query",
                "[?roleDefinitionName=='Key Vault Secrets User'] | length(@)",
            ]
        )
        try:
            assignment_count = int(assigned or "0")
        except ValueError:
            assignment_count = 0
        if assignment_count == 0:
            _az(
                [
                    "role",
                    "assignment",
                    "create",
                    "--assignee-object-id",
                    object_id,
                    "--assignee-principal-type",
                    "ServicePrincipal",
                    "--role",
                    "Key Vault Secrets User",
                    "--scope",
                    vault_id,
                    "--subscription",
                    args.subscription,
                    "--output",
                    "none",
                ]
            )
    else:
        _az(
            [
                "keyvault",
                "set-policy",
                "--name",
                vault_name,
                "--subscription",
                args.subscription,
                "--object-id",
                object_id,
                "--secret-permissions",
                "get",
                "--output",
                "none",
            ]
        )


def provision(args: argparse.Namespace) -> ProvisionedDocker:
    _validate_args(args)
    _require_commands()
    private_key, public_key = _ensure_ssh_key(Path(args.ssh_key_path))
    location = args.location or _workspace_location(args)
    vault_id, vault_name, vault_url = _workspace_key_vault(args)
    host = _ensure_vm(args, public_key, location)
    known_hosts = _scan_host_key(host, args.ssh_port)

    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as handle:
        handle.write(known_hosts.rstrip("\n") + "\n")
        known_hosts_path = Path(handle.name)
    try:
        _verify_ssh(
            host,
            args.vm_user,
            private_key,
            known_hosts_path,
            args.ssh_port,
        )
        _log(f"storing SSH credentials in Key Vault {vault_name}")
        _store_secret(
            args.subscription,
            vault_name,
            args.ssh_key_secret,
            private_key,
        )
        _store_secret(
            args.subscription,
            vault_name,
            args.ssh_known_hosts_secret,
            known_hosts_path,
        )
    finally:
        known_hosts_path.unlink(missing_ok=True)

    _grant_secret_access(args, vault_id, vault_name)
    return ProvisionedDocker(
        docker_host=f"ssh://{args.vm_user}@{host}:{args.ssh_port}",
        key_vault_url=vault_url,
        ssh_key_secret=args.ssh_key_secret,
        ssh_known_hosts_secret=args.ssh_known_hosts_secret,
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subscription", required=True)
    parser.add_argument("--resource-group", required=True)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--managed-identity-client-id", required=True)
    parser.add_argument("--managed-identity-object-id", default="")
    parser.add_argument("--location", default="")
    parser.add_argument("--vm-resource-group", default="")
    parser.add_argument("--vm-name", default="officebench-docker")
    parser.add_argument("--vm-size", default="Standard_D16s_v5")
    parser.add_argument("--vm-image", default="Ubuntu2204")
    parser.add_argument("--vm-user", default="officebench")
    parser.add_argument("--ssh-port", type=int, default=22)
    parser.add_argument(
        "--ssh-key-path",
        default=os.path.expanduser("~/.ssh/officebench_aml"),
    )
    parser.add_argument(
        "--ssh-key-secret",
        default="officebench-docker-ssh-key",
    )
    parser.add_argument(
        "--ssh-known-hosts-secret",
        default="officebench-docker-known-hosts",
    )
    parser.add_argument("--output", choices=("json", "lines"), default="json")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        result = provision(args)
    except RuntimeError as exc:
        print(f"[officebench][provision][error] {exc}", file=sys.stderr)
        return 2
    if args.output == "lines":
        print(result.docker_host)
        print(result.key_vault_url)
        print(result.ssh_key_secret)
        print(result.ssh_known_hosts_secret)
    else:
        print(json.dumps(result.as_dict()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
