from __future__ import annotations

import importlib.util
import stat
import sys
from pathlib import Path

import yaml


def _load_configurator():
    path = (
        Path(__file__).parents[1]
        / "aml"
        / "src"
        / "configure_remote_docker.py"
    )
    spec = importlib.util.spec_from_file_location("configure_remote_docker", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_provisioner():
    path = (
        Path(__file__).parents[1]
        / "aml"
        / "src"
        / "provision_officebench_docker.py"
    )
    spec = importlib.util.spec_from_file_location("provision_officebench_docker", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_configure_remote_docker_writes_key_and_pinned_host(tmp_path, monkeypatch):
    configurator = _load_configurator()
    secrets = {
        "ssh-key": "-----BEGIN OPENSSH PRIVATE KEY-----\nkey\n-----END OPENSSH PRIVATE KEY-----",
        "known-hosts": "[docker.example]:2222 ssh-ed25519 AAAATEST",
    }
    monkeypatch.setattr(
        configurator,
        "_fetch_secret",
        lambda _vault, name, _client_id: secrets[name],
    )
    monkeypatch.setattr(
        configurator,
        "_scan_host_key",
        lambda *_args: (_ for _ in ()).throw(AssertionError("must use the pinned host key")),
    )

    effective = configurator.configure_ssh(
        "ssh://officebench@docker.example:2222",
        "https://vault.vault.azure.net",
        "ssh-key",
        "known-hosts",
        "managed-identity-client-id",
        home=tmp_path,
    )

    ssh_dir = tmp_path / ".ssh"
    key_path = ssh_dir / "officebench_docker"
    known_hosts_path = ssh_dir / "officebench_docker_known_hosts"
    config = (ssh_dir / "config").read_text(encoding="utf-8")
    assert effective == "ssh://officebench-docker"
    assert key_path.read_text(encoding="utf-8").startswith(
        "-----BEGIN OPENSSH PRIVATE KEY-----"
    )
    assert stat.S_IMODE(key_path.stat().st_mode) == 0o600
    assert known_hosts_path.read_text(encoding="utf-8").startswith(
        "[docker.example]:2222"
    )
    assert "HostName docker.example" in config
    assert "Port 2222" in config
    assert "User officebench" in config
    assert f"IdentityFile {key_path}" in config
    assert "ControlMaster auto" in config
    assert "ControlPersist 300" in config
    assert "StrictHostKeyChecking yes" in config


def test_configure_remote_docker_rejects_credentials_for_tcp(capsys):
    configurator = _load_configurator()

    status = configurator.main(
        [
            "--docker-host",
            "tcp://docker.example:2376",
            "--vault-url",
            "https://vault.vault.azure.net",
            "--ssh-key-secret",
            "ssh-key",
        ]
    )

    assert status == 2
    assert "SSH secrets can only be used with an ssh:// Docker host" in capsys.readouterr().err


def test_officebench_aml_job_uses_local_no_docker_backend():
    root = Path(__file__).parents[1]
    job = yaml.safe_load((root / "aml" / "job.officebench.yml").read_text(encoding="utf-8"))
    inputs = job["inputs"]
    command = job["command"]

    assert job["environment"] == "azureml:tts-officebench-local:20260803"
    assert job["environment_variables"]["TTS_OFFICEBENCH_BACKEND"] == "local"
    assert job["environment_variables"]["TTS_OFFICEBENCH_WM_PYTHON_TIMEOUT_S"] == "5"
    assert inputs["agent"] == "llm_wm"
    assert inputs["aoai_deployment"] == "gpt-5.4"
    assert inputs["temperature"] == 1
    assert inputs["workers"] == 8
    assert inputs["extra_args"] == "--exact-clean-reads"
    assert not any(key.startswith("docker_") for key in inputs)
    assert "--docker-host" not in command


def test_officebench_runner_bypasses_docker_in_local_mode():
    runner = (
        Path(__file__).parents[1] / "aml" / "src" / "run_dw.sh"
    ).read_text(encoding="utf-8")

    officebench_case = runner.index("  officebench)")
    local_check = runner.index('TTS_OFFICEBENCH_BACKEND:-docker', officebench_case)
    docker_configure = runner.index("configure_officebench_docker", local_check)
    assert local_check < docker_configure
    assert "using local no-Docker backend" in runner
    assert "TTS_OFFICEBENCH_MAX_WORKERS:-64" in runner
    assert "capping workers" in runner
    assert "local backend requires proot" in runner
    assert "tts-no-network sandbox launcher" in runner
    assert "TTS_OFFICEBENCH_RUN_ID" in runner
    assert 'timeout "${TTS_DOCKER_INFO_TIMEOUT_S:-60}" docker info' in runner
    assert "docker build -t officebench-tts:latest" in runner


def test_officebench_provisioner_stores_verified_ssh_credentials(tmp_path, monkeypatch):
    provisioner = _load_provisioner()
    private_key = tmp_path / "officebench_aml"
    public_key = tmp_path / "officebench_aml.pub"
    private_key.write_text("PRIVATE\n", encoding="utf-8")
    public_key.write_text("PUBLIC\n", encoding="utf-8")
    stored = {}

    args = provisioner._parse_args(
        [
            "--subscription",
            "sub",
            "--resource-group",
            "rg",
            "--workspace",
            "ws",
            "--managed-identity-client-id",
            "client-id",
            "--ssh-key-path",
            str(private_key),
        ]
    )
    monkeypatch.setattr(provisioner, "_require_commands", lambda: None)
    monkeypatch.setattr(
        provisioner, "_ensure_ssh_key", lambda _path: (private_key, public_key)
    )
    monkeypatch.setattr(provisioner, "_workspace_location", lambda _args: "eastus")
    monkeypatch.setattr(
        provisioner,
        "_workspace_key_vault",
        lambda _args: (
            "/subscriptions/sub/resourceGroups/rg/providers/Microsoft.KeyVault/vaults/ws-kv",
            "ws-kv",
            "https://ws-kv.vault.azure.net",
        ),
    )
    monkeypatch.setattr(
        provisioner,
        "_ensure_vm",
        lambda _args, _public_key, _location: "203.0.113.10",
    )
    monkeypatch.setattr(
        provisioner,
        "_scan_host_key",
        lambda _host, _port: "203.0.113.10 ssh-ed25519 AAAATEST\n",
    )
    monkeypatch.setattr(
        provisioner,
        "_verify_ssh",
        lambda host, user, key, known_hosts, port: stored.update(
            verified=(host, user, key, known_hosts.read_text(encoding="utf-8"), port)
        ),
    )
    monkeypatch.setattr(
        provisioner,
        "_store_secret",
        lambda _subscription, _vault, name, path: stored.update(
            {name: path.read_text(encoding="utf-8")}
        ),
    )
    monkeypatch.setattr(
        provisioner,
        "_grant_secret_access",
        lambda _args, vault_id, vault_name: stored.update(
            access=(vault_id, vault_name)
        ),
    )

    result = provisioner.provision(args)

    assert result.docker_host == "ssh://officebench@203.0.113.10:22"
    assert result.key_vault_url == "https://ws-kv.vault.azure.net"
    assert stored["officebench-docker-ssh-key"] == "PRIVATE\n"
    assert stored["officebench-docker-known-hosts"].startswith("203.0.113.10 ")
    assert stored["verified"][:3] == (
        "203.0.113.10",
        "officebench",
        private_key,
    )
    assert stored["access"][1] == "ws-kv"


def test_officebench_submit_registers_local_environment_without_provisioning_docker():
    submit = (
        Path(__file__).parents[1] / "aml" / "submit.sh"
    ).read_text(encoding="utf-8")

    assert "az ml environment show" in submit
    assert "tts-officebench-local" in submit
    assert "aml/environment/officebench-local.yml" in submit
    assert "provision_officebench_docker.py" not in submit
    assert "inputs.docker_host" not in submit


def test_provisioner_grants_rbac_by_managed_identity_object_id(monkeypatch):
    provisioner = _load_provisioner()
    args = provisioner._parse_args(
        [
            "--subscription",
            "sub",
            "--resource-group",
            "rg",
            "--workspace",
            "ws",
            "--managed-identity-client-id",
            "client-id",
            "--managed-identity-object-id",
            "object-id",
        ]
    )
    commands = []

    def _az_text(command):
        if command[:2] == ["keyvault", "show"]:
            return "true"
        if command[:3] == ["role", "assignment", "list"]:
            assert command[command.index("--assignee") + 1] == "object-id"
            return "0"
        raise AssertionError(command)

    monkeypatch.setattr(provisioner, "_az_text", _az_text)
    monkeypatch.setattr(
        provisioner,
        "_az",
        lambda command, check=True: commands.append(command),
    )

    provisioner._grant_secret_access(args, "/vault/id", "ws-kv")

    create = next(
        command for command in commands if command[:3] == ["role", "assignment", "create"]
    )
    assert create[create.index("--assignee-object-id") + 1] == "object-id"
    assert create[create.index("--assignee-principal-type") + 1] == "ServicePrincipal"


def test_provisioner_does_not_recreate_existing_resource_group(tmp_path, monkeypatch):
    provisioner = _load_provisioner()
    args = provisioner._parse_args(
        [
            "--subscription",
            "sub",
            "--resource-group",
            "Singularity",
            "--workspace",
            "ws",
            "--managed-identity-client-id",
            "client-id",
        ]
    )
    public_key = tmp_path / "officebench_aml.pub"
    public_key.write_text("PUBLIC\n", encoding="utf-8")
    commands = []

    def _az_text(command):
        if command[:2] == ["group", "exists"]:
            return "true"
        if command[:2] == ["vm", "show"]:
            return "203.0.113.10"
        raise AssertionError(command)

    def _az(command, check=True):
        commands.append(command)
        if command[:2] == ["vm", "show"]:
            return type("Result", (), {"returncode": 0, "stdout": "", "stderr": ""})()
        if command[:3] == ["vm", "run-command", "invoke"]:
            return type(
                "Result",
                (),
                {
                    "returncode": 0,
                    "stdout": (
                        '{"value":[{"message":"[stdout]\\n'
                        'OFFICEBENCH_DOCKER_READY\\n"}]}'
                    ),
                    "stderr": "",
                },
            )()
        return type("Result", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    monkeypatch.setattr(provisioner, "_az_text", _az_text)
    monkeypatch.setattr(provisioner, "_az", _az)

    host = provisioner._ensure_vm(args, public_key, "eastus")

    assert host == "203.0.113.10"
    assert not any(command[:2] == ["group", "create"] for command in commands)
