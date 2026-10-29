#!/usr/bin/env python3
"""Configure Docker's SSH transport for an AML command job."""

from __future__ import annotations

import argparse
import stat
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlsplit


def _fetch_secret(vault_url: str, secret_name: str, managed_identity_client_id: str) -> str:
    try:
        from azure.core.exceptions import AzureError
        from azure.identity import DefaultAzureCredential
        from azure.keyvault.secrets import SecretClient
    except ImportError as exc:
        raise RuntimeError(
            "azure-identity and azure-keyvault-secrets are required to fetch Docker SSH credentials"
        ) from exc

    credential = DefaultAzureCredential(
        managed_identity_client_id=managed_identity_client_id or None
    )
    client = SecretClient(vault_url=vault_url, credential=credential)
    last_error: AzureError | None = None
    value = None
    for attempt in range(12):
        try:
            value = client.get_secret(secret_name).value
            break
        except AzureError as exc:
            last_error = exc
            if attempt == 11:
                break
            time.sleep(min(5 + attempt * 2, 30))
    if value is None and last_error is not None:
        raise RuntimeError(
            f"could not fetch Key Vault secret {secret_name!r} after waiting for access propagation"
        ) from last_error
    if not value:
        raise RuntimeError(f"Key Vault secret {secret_name!r} is empty")
    return value


def _write_text(path: Path, value: str, mode: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = value.replace("\r\n", "\n").replace("\r", "\n")
    path.write_text(normalized.rstrip("\n") + "\n", encoding="utf-8")
    path.chmod(mode)


def _scan_host_key(hostname: str, port: int) -> str:
    try:
        proc = subprocess.run(
            ["ssh-keyscan", "-T", "15", "-p", str(port), hostname],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(
            "ssh-keyscan is required when no known-hosts Key Vault secret is configured"
        ) from exc
    if proc.returncode != 0 or not proc.stdout.strip():
        raise RuntimeError(
            f"could not read the SSH host key for {hostname}:{port}: {proc.stderr.strip()}"
        )
    return proc.stdout


def configure_ssh(
    docker_host: str,
    vault_url: str,
    ssh_key_secret: str,
    known_hosts_secret: str,
    managed_identity_client_id: str,
    *,
    home: Path | None = None,
) -> str:
    """Create an isolated SSH alias and return the effective ``DOCKER_HOST``."""
    parsed = urlsplit(docker_host)
    if parsed.scheme != "ssh" or not parsed.hostname:
        raise ValueError("remote Docker SSH host must look like ssh://user@hostname[:port]")

    hostname = parsed.hostname
    port = parsed.port or 22
    username = parsed.username
    ssh_dir = (home or Path.home()) / ".ssh"
    key_path = ssh_dir / "officebench_docker"
    known_hosts_path = ssh_dir / "officebench_docker_known_hosts"
    config_path = ssh_dir / "config"

    if ssh_key_secret:
        if not vault_url:
            raise ValueError("docker_key_vault_url is required with docker_ssh_key_secret")
        key = _fetch_secret(vault_url, ssh_key_secret, managed_identity_client_id)
        _write_text(key_path, key, stat.S_IRUSR | stat.S_IWUSR)
    elif not key_path.is_file():
        print(
            "[officebench][docker][warn] no SSH key secret configured; "
            f"expecting an existing key at {key_path}",
            file=sys.stderr,
        )

    if known_hosts_secret:
        if not vault_url:
            raise ValueError(
                "docker_key_vault_url is required with docker_ssh_known_hosts_secret"
            )
        known_hosts = _fetch_secret(
            vault_url, known_hosts_secret, managed_identity_client_id
        )
    else:
        print(
            "[officebench][docker][warn] no pinned known-hosts secret configured; "
            "trusting the host key returned by ssh-keyscan for this job",
            file=sys.stderr,
        )
        known_hosts = _scan_host_key(hostname, port)
    _write_text(known_hosts_path, known_hosts, stat.S_IRUSR | stat.S_IWUSR)

    lines = [
        "Host officebench-docker",
        f"    HostName {hostname}",
        f"    Port {port}",
        "    BatchMode yes",
        f"    IdentityFile {key_path}",
        "    IdentitiesOnly yes",
        "    ControlMaster auto",
        f"    ControlPath {ssh_dir}/officebench_docker-%C",
        "    ControlPersist 300",
        "    StrictHostKeyChecking yes",
        f"    UserKnownHostsFile {known_hosts_path}",
    ]
    if username:
        lines.insert(3, f"    User {username}")
    existing = config_path.read_text(encoding="utf-8") if config_path.exists() else ""
    _write_text(
        config_path,
        "\n".join(lines) + ("\n\n" + existing if existing.strip() else ""),
        stat.S_IRUSR | stat.S_IWUSR,
    )
    return "ssh://officebench-docker"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--docker-host", required=True)
    parser.add_argument("--vault-url", default="")
    parser.add_argument("--ssh-key-secret", default="")
    parser.add_argument("--known-hosts-secret", default="")
    parser.add_argument("--managed-identity-client-id", default="")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        if args.docker_host.startswith("ssh://"):
            docker_host = configure_ssh(
                args.docker_host,
                args.vault_url,
                args.ssh_key_secret,
                args.known_hosts_secret,
                args.managed_identity_client_id,
            )
        else:
            if args.ssh_key_secret or args.known_hosts_secret:
                raise ValueError("SSH secrets can only be used with an ssh:// Docker host")
            docker_host = args.docker_host
    except (RuntimeError, ValueError) as exc:
        print(f"[officebench][docker][error] {exc}", file=sys.stderr)
        return 2
    print(docker_host)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
