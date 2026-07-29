from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

import yaml


ROOT = Path(__file__).parents[1]


def test_qwen_job_runs_server_and_tau2_on_one_aml_node():
    job = yaml.safe_load(
        (ROOT / "aml" / "job.qwen.tau2.yml").read_text(encoding="utf-8")
    )

    assert job["compute"] == "azureml:NC40adsv5-EUS-hipri"
    assert job["inputs"]["model"] == "Qwen/Qwen3.5-35B-A3B"
    assert job["inputs"]["quantization"] == "fp8"
    assert job["inputs"]["language_model_only"] == 1
    assert job["inputs"]["enable_thinking"] == 0
    assert job["inputs"]["min_cache_free_gb"] == 100
    assert job["inputs"]["env"] == "tau2"
    assert "aml/src/run_qwen.sh" in job["command"]
    assert "--output-dir" in job["command"]


def test_qwen_environment_contains_vllm_and_tau2():
    environment = yaml.safe_load(
        (ROOT / "aml" / "environment" / "conda.qwen.yml").read_text(
            encoding="utf-8"
        )
    )
    pip_dependencies = next(
        item["pip"] for item in environment["dependencies"] if isinstance(item, dict)
    )

    assert "vllm==0.24.0" in pip_dependencies
    assert any(
        dependency.startswith("tau2[knowledge] @")
        for dependency in pip_dependencies
    )


def test_qwen_appworld_job_uses_two_gpu_bf16_and_one_task():
    job = yaml.safe_load(
        (ROOT / "aml" / "job.qwen.appworld.yml").read_text(encoding="utf-8")
    )

    assert job["compute"] == "azureml:NC80adsv5-EUS-hipri"
    assert job["inputs"]["appworld_data"]["path"].startswith(
        "azureml:tts_appworld_data:"
    )
    assert job["inputs"]["tp_size"] == 2
    assert job["inputs"]["quantization"] == "none"
    assert job["inputs"]["enable_thinking"] == 1
    assert job["inputs"]["vllm_extra_args"] == "--enable-expert-parallel"
    assert job["inputs"]["env"] == "appworld"
    assert job["inputs"]["workers"] == 1
    assert job["inputs"]["extra_args"] == "--tasks 1"


def test_qwen_appworld_environment_contains_vllm_and_pinned_appworld():
    environment = yaml.safe_load(
        (ROOT / "aml" / "environment" / "conda.qwen.appworld.yml").read_text(
            encoding="utf-8"
        )
    )
    pip_dependencies = next(
        item["pip"] for item in environment["dependencies"] if isinstance(item, dict)
    )

    assert "vllm==0.24.0" in pip_dependencies
    assert any(dependency.startswith("appworld @") for dependency in pip_dependencies)
    assert "openai>=2.0.0" in pip_dependencies


def test_qwen_runner_uses_local_server_and_existing_aml_pipeline():
    runner = (ROOT / "aml" / "src" / "run_qwen.sh").read_text(encoding="utf-8")

    assert "scripts/serve_local_model.sh" in runner
    assert 'curl -fsS "$base_url/models"' in runner
    assert "--local-base-url" in runner
    assert "--llm-agent" in runner
    assert "aml/src/run_dw.sh" in runner
    assert "--appworld-data" in runner
    assert 'EXTRA_ARGS="$vllm_extra_args"' in runner
    assert "VLLM_USE_FLASHINFER_SAMPLER=0" in runner
    assert '--enable-thinking) enable_thinking="$2"' in runner
    assert "Do not pass --default-chat-template-kwargs" in runner
    assert 'ENABLE_THINKING="$enable_thinking"' in runner
    assert "TTS_LOCAL_LLM=1" in runner
    assert 'HF_HOME="${HF_HOME:-$PWD/.cache/huggingface}"' in runner
    assert "cp -f \"$server_log\" ./outputs/vllm.log" in runner
    assert 'kill -- "-$server_pid"' in runner

    server = (ROOT / "scripts" / "serve_local_model.sh").read_text(encoding="utf-8")
    assert "ENABLE_THINKING" in server
    assert "VLLM_FLAGS+=(--default-chat-template-kwargs" in server


def test_qwen_submitter_validates_gpu_compute_and_submits_job():
    submitter = (ROOT / "aml" / "submit_qwen.sh").read_text(encoding="utf-8")

    assert "job_compute=" in submitter
    assert 'compute_name="${AML_QWEN_COMPUTE:-$job_compute}"' in submitter
    assert 'vm_size="${AML_QWEN_VM_SIZE:-}"' in submitter
    assert "az ml compute create" in submitter
    assert "az ml job create" in submitter
    assert "join('|', [type, size, provisioning_state])" in submitter
    assert "IFS='|' read -r existing_type existing_size provisioning_state" in submitter
    assert '"compute=azureml:$compute_name"' in submitter


def test_qwen_submitter_inherits_compute_from_selected_job(tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    capture = tmp_path / "az-job-create.txt"
    fake_az = fake_bin / "az"
    fake_az.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
if [[ "$*" == *"ml compute show"* ]]; then
  echo "amlcompute|Standard_NC80adis_H100_v5|Succeeded"
elif [[ "$*" == *"ml job create"* ]]; then
  printf '%s\\n' "$@" > "$AZ_CAPTURE"
else
  exit 1
fi
""",
        encoding="utf-8",
    )
    fake_az.chmod(fake_az.stat().st_mode | stat.S_IXUSR)
    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}:{env['PATH']}"
    env["AZ_CAPTURE"] = str(capture)
    env["AML_QWEN_JOB_FILE"] = str(ROOT / "aml" / "job.qwen.appworld.yml")
    env.pop("AML_QWEN_COMPUTE", None)
    env.pop("AML_QWEN_VM_SIZE", None)

    subprocess.run(
        ["bash", str(ROOT / "aml" / "submit_qwen.sh")],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    submitted_args = capture.read_text(encoding="utf-8")
    assert "compute=azureml:NC80adsv5-EUS-hipri" in submitted_args


def test_shared_aml_runner_skips_azure_auth_for_local_qwen():
    runner = (ROOT / "aml" / "src" / "run_dw.sh").read_text(encoding="utf-8")

    assert 'if [[ "${TTS_LOCAL_LLM:-0}" == "1" ]]' in runner
    assert '"local" if "${TTS_LOCAL_LLM:-0}" == "1"' in runner
