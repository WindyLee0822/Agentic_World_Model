from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest


def _load_hydrator():
    path = Path(__file__).parents[1] / "aml" / "src" / "hydrate_appworld_bundles.py"
    spec = importlib.util.spec_from_file_location("hydrate_appworld_bundles", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _pointer(payload: bytes) -> bytes:
    return (
        "version https://git-lfs.github.com/spec/v1\n"
        f"oid sha256:{hashlib.sha256(payload).hexdigest()}\n"
        f"size {len(payload)}\n"
    ).encode("ascii")


def _load_summarizer():
    path = Path(__file__).parents[1] / "aml" / "src" / "summarize.py"
    spec = importlib.util.spec_from_file_location("aml_summarize", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_bon4_gate():
    path = Path(__file__).parents[1] / "aml" / "src" / "check_appworld_bon4.py"
    spec = importlib.util.spec_from_file_location("aml_check_appworld_bon4", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_bon2_gate():
    path = Path(__file__).parents[1] / "aml" / "src" / "check_appworld_bon2.py"
    spec = importlib.util.spec_from_file_location("aml_check_appworld_bon2", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_appworld_wm_job_keeps_candidate_generation_agent_fair():
    job = (
        Path(__file__).parents[1] / "aml" / "job.appworld.wm.yml"
    ).read_text(encoding="utf-8")

    assert "extra_args: none" in job
    assert "--appworld-instruction-preflights" not in job
    assert "--appworld-precision-prompt" not in job


def test_all_aml_jobs_default_to_non_strict_coverage():
    aml_dir = Path(__file__).parents[1] / "aml"
    for path in aml_dir.glob("job*.yml"):
        defaults = [
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip().startswith("extra_args:")
        ]
        assert all("--strict-coverage" not in line for line in defaults), path


def test_hydrate_appworld_bundle_replaces_verified_lfs_pointer(tmp_path, monkeypatch):
    hydrator = _load_hydrator()
    payload = b"encrypted AppWorld bundle"
    bundle_path = tmp_path / "apps.bundle"
    bundle_path.write_bytes(_pointer(payload))
    monkeypatch.setattr(hydrator, "_download_bytes", lambda url: payload)

    assert hydrator.hydrate_bundle(bundle_path) is True
    assert bundle_path.read_bytes() == payload


def test_hydrate_appworld_bundle_rejects_unverified_download(tmp_path, monkeypatch):
    hydrator = _load_hydrator()
    bundle_path = tmp_path / "apps.bundle"
    pointer = _pointer(b"expected")
    bundle_path.write_bytes(pointer)
    monkeypatch.setattr(hydrator, "_download_bytes", lambda url: b"wrong")

    with pytest.raises(RuntimeError, match="failed Git LFS verification"):
        hydrator.hydrate_bundle(bundle_path)
    assert bundle_path.read_bytes() == pointer


def test_hydrate_appworld_bundle_leaves_real_bundle_untouched(tmp_path, monkeypatch):
    hydrator = _load_hydrator()
    bundle_path = tmp_path / "apps.bundle"
    bundle_path.write_bytes(b"already hydrated")
    monkeypatch.setattr(
        hydrator,
        "_download_bytes",
        lambda url: pytest.fail("ready bundle must not be downloaded"),
    )

    assert hydrator.hydrate_bundle(bundle_path) is False
    assert bundle_path.read_bytes() == b"already hydrated"


def test_appworld_task_index_does_not_wrap_missing_coverage(monkeypatch):
    from tts.env import real_appworld

    monkeypatch.setattr(real_appworld, "real_appworld_available", lambda: True)
    monkeypatch.setattr(
        real_appworld,
        "load_real_appworld_task_ids",
        lambda split, level: ["only-task"],
    )

    with pytest.raises(IndexError, match="out of range"):
        real_appworld.AppWorldTaskEnv(
            split="test_normal",
            level=1,
            index=1,
        )


def test_summarizer_refuses_incomplete_coverage_when_strict(tmp_path, monkeypatch):
    summarizer = _load_summarizer()
    records = tmp_path / "records.jsonl"
    records.write_text(json.dumps({
        "env": "appworld", "subset": "test_challenge/L1", "method": "ours",
        "n_rollouts": 1, "success": True,
    }) + "\n", encoding="utf-8")
    (tmp_path / "run_manifest.json").write_text(json.dumps({
        "expected_tasks": 72,
        "complete_tasks": 71,
        "partial_tasks": 1,
        "missing_tasks": 0,
    }), encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        ["summarize.py", str(records), str(tmp_path), "--strict-coverage"],
    )

    assert summarizer.main() == 2
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["error"] == "incomplete_coverage"


@pytest.mark.parametrize("coverage_args", [[], ["--allow-partial"]])
def test_summarizer_allows_labeled_partial_coverage_by_default(
    tmp_path, monkeypatch, coverage_args
):
    summarizer = _load_summarizer()
    records = tmp_path / "records.jsonl"
    records.write_text(json.dumps({
        "env": "appworld", "subset": "test_challenge/L1", "method": "ours",
        "n_rollouts": 4, "success": True, "reward": 1.0,
    }) + "\n", encoding="utf-8")
    manifest = {
        "env": "appworld",
        "expected_tasks": 2,
        "complete_tasks": 1,
        "partial_tasks": 0,
        "missing_tasks": 1,
        "expected_cells_per_task": 1,
        "expected_cells": [{"method": "ours", "n_rollouts": 4}],
        "expected_decisions": 2,
        "recorded_decisions": 1,
        "missing_decisions": 1,
        "by_subset": {
            "test_challenge/L1": {
                "expected_tasks": 2,
                "complete_tasks": 1,
                "partial_tasks": 0,
                "missing_tasks": 1,
                "expected_decisions": 2,
                "recorded_decisions": 1,
                "missing_decisions": 1,
            }
        },
    }
    (tmp_path / "run_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        ["summarize.py", str(records), str(tmp_path), *coverage_args],
    )

    assert summarizer.main() == 0
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["partial"] is True
    assert summary["coverage"] == manifest
    assert summary["decisions"] == 2
    assert summary["recorded_decisions"] == 1
    assert summary["imputed_failures"] == 1
    assert summary["by_method_n"][0]["decisions"] == 2
    assert summary["by_method_n"][0]["recorded_decisions"] == 1
    assert summary["by_method_n"][0]["imputed_failures"] == 1
    assert summary["by_method_n"][0]["success_rate"] == 0.5
    assert "PARTIAL RESULTS" in (tmp_path / "summary.txt").read_text(encoding="utf-8")


def test_aml_runner_forwards_coverage_override_to_summarizer():
    runner = (
        Path(__file__).parents[1] / "aml" / "src" / "run_dw.sh"
    ).read_text(encoding="utf-8")

    assert "--allow-partial|--strict-coverage" in runner
    assert '"$output_dir/records.jsonl" "$output_dir" "${summary_args[@]}"' in runner


def test_summarizer_counts_entirely_missing_run_as_failures(tmp_path, monkeypatch):
    summarizer = _load_summarizer()
    records = tmp_path / "records.jsonl"
    records.write_text("", encoding="utf-8")
    manifest = {
        "env": "appworld",
        "expected_tasks": 2,
        "complete_tasks": 0,
        "partial_tasks": 0,
        "missing_tasks": 2,
        "expected_cells_per_task": 1,
        "expected_cells": [{"method": "ours", "n_rollouts": 4}],
        "expected_decisions": 2,
        "recorded_decisions": 0,
        "missing_decisions": 2,
        "by_subset": {
            "test_challenge/L1": {
                "expected_tasks": 2,
                "complete_tasks": 0,
                "partial_tasks": 0,
                "missing_tasks": 2,
                "expected_decisions": 2,
                "recorded_decisions": 0,
                "missing_decisions": 2,
            }
        },
    }
    (tmp_path / "run_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["summarize.py", str(records), str(tmp_path)])

    assert summarizer.main() == 0
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["decisions"] == 2
    assert summary["recorded_decisions"] == 0
    assert summary["imputed_failures"] == 2
    assert summary["by_method_n"][0]["success_rate"] == 0.0


def test_summarizer_requires_coverage_manifest(tmp_path, monkeypatch):
    summarizer = _load_summarizer()
    records = tmp_path / "records.jsonl"
    records.write_text(json.dumps({
        "env": "appworld", "subset": "test_challenge/L1", "method": "ours",
        "n_rollouts": 1, "success": True,
    }) + "\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["summarize.py", str(records), str(tmp_path)])

    assert summarizer.main() == 2
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["error"] == "missing_run_manifest"


def _bon4_records(rates, mode="python"):
    gate = _load_bon4_gate()
    records = []
    for subset, rate in rates.items():
        total = gate.EXPECTED_DECISIONS[subset]
        passed = int(rate * total)
        records.extend({
            "env": "appworld",
            "subset": subset,
            "method": "ours",
            "n_rollouts": 4,
            "success": index < passed,
            "appworld_wm_tool_mode": mode,
            "solver_agent": "llm_wm",
            "llm_agent": "gpt-4.1",
        } for index in range(total))
    return records


def test_appworld_bon4_gate_passes_all_strict_thresholds():
    gate = _load_bon4_gate()
    rates = {
        subset: threshold + 0.05
        for subset, threshold in gate.THRESHOLDS.items()
    }

    result = gate.evaluate(_bon4_records(rates))

    assert result["passed"] is True
    assert result["tool_modes"] == ["python"]
    assert result["solver_agents"] == ["llm_wm"]
    assert result["models"] == ["gpt-4.1"]
    assert all(row["passed"] for row in result["rows"])


def test_appworld_bon4_gate_is_strict_and_python_only():
    gate = _load_bon4_gate()
    equal_rates = dict(gate.THRESHOLDS)

    strict = gate.evaluate(_bon4_records(equal_rates))
    text_mode = gate.evaluate(_bon4_records({
        subset: threshold + 0.05
        for subset, threshold in gate.THRESHOLDS.items()
    }, mode="text"))

    assert strict["passed"] is False
    assert text_mode["passed"] is False


def _bon2_records(rates, **overrides):
    gate = _load_bon2_gate()
    records = []
    defaults = {
        "appworld_wm_tool_mode": "python",
        "solver_agent": "llm_wm",
        "llm_agent": "gpt-4.1",
        "appworld_precision_prompt": True,
        "appworld_agent_reference": True,
        "appworld_deterministic_dirty_reads": True,
        "appworld_verifier_percheck": True,
        "success_score": "simulator_confidence",
    }
    defaults.update(overrides)
    for subset, rate in rates.items():
        total = gate.EXPECTED_DECISIONS[subset]
        passed = int(rate * total)
        records.extend({
            "env": "appworld",
            "subset": subset,
            "method": "ours",
            "n_rollouts": 2,
            "success": index < passed,
            **defaults,
        } for index in range(total))
    return records


def test_appworld_bon2_gate_passes_all_strict_multi_app_thresholds():
    gate = _load_bon2_gate()
    rates = {
        subset: threshold + 0.05
        for subset, threshold in gate.THRESHOLDS.items()
    }

    result = gate.evaluate(_bon2_records(rates))

    assert result["passed"] is True
    assert all(row["passed"] for row in result["rows"])


def test_appworld_bon2_gate_is_strict_and_requires_graded_python_configuration():
    gate = _load_bon2_gate()
    strict = gate.evaluate(_bon2_records(dict(gate.THRESHOLDS)))
    binary = gate.evaluate(_bon2_records({
        subset: threshold + 0.05
        for subset, threshold in gate.THRESHOLDS.items()
    }, success_score="simulator_success"))
    no_percheck = gate.evaluate(_bon2_records({
        subset: threshold + 0.05
        for subset, threshold in gate.THRESHOLDS.items()
    }, appworld_verifier_percheck=False))
    no_agent_reference = gate.evaluate(_bon2_records({
        subset: threshold + 0.05
        for subset, threshold in gate.THRESHOLDS.items()
    }, appworld_agent_reference=False))
    stale_dirty_reads = gate.evaluate(_bon2_records({
        subset: threshold + 0.05
        for subset, threshold in gate.THRESHOLDS.items()
    }, appworld_deterministic_dirty_reads=False))

    assert strict["passed"] is False
    assert binary["passed"] is False
    assert no_percheck["passed"] is False
    assert no_agent_reference["passed"] is False
    assert stale_dirty_reads["passed"] is False
