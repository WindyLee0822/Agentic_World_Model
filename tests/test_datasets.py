"""Dataset taxonomy tests for the real benchmarks (no real packages required)."""

from __future__ import annotations

import pytest

from tts.env import (
    APPWORLD_OFFICIAL_SIZE,
    TAU2_OFFICIAL_SIZE,
    build_dataset,
    dataset_size,
    subset_size,
    subsets,
)


def test_appworld_subsets_and_official_sizes():
    subs = subsets("appworld")
    assert subs == [f"{s}/L{l}" for s in ("test_normal", "test_challenge") for l in (1, 2, 3)]
    # official appworld.load_task_ids counts
    assert subset_size("appworld", "test_normal/L1") == 57
    assert subset_size("appworld", "test_challenge/L3") == 195
    assert dataset_size("appworld", splits=["test_normal"]) == 168
    assert dataset_size("appworld", splits=["test_challenge"]) == 417
    assert dataset_size("appworld") == 585


def test_tau2_subsets_and_official_sizes():
    assert set(subsets("tau2")) == {"airline", "retail", "telecom", "mock", "banking_knowledge"}
    # official tau2.registry.get_tasks_loader counts
    assert subset_size("tau2", "airline") == 50
    assert subset_size("tau2", "retail") == 114
    assert subset_size("tau2", "telecom") == 114
    assert subset_size("tau2", "mock") == 10
    assert subset_size("tau2", "banking_knowledge") == 97
    assert dataset_size("tau2") == sum(TAU2_OFFICIAL_SIZE.values())


def test_build_dataset_whole_benchmark_uses_official_sizes():
    specs = build_dataset("appworld")  # no tasks, no filter -> whole benchmark
    from collections import Counter

    counts = Counter(s.subset for s in specs)
    for sub in subsets("appworld"):
        assert counts[sub] == subset_size("appworld", sub)


def test_build_dataset_filters():
    specs = build_dataset("appworld", tasks_per_subset=2, splits=["test_normal"], levels=[1, 3])
    assert {s.subset for s in specs} == {"test_normal/L1", "test_normal/L3"}
    assert len(specs) == 4
    t = build_dataset("tau2", tasks_per_subset=3, domains=["airline", "mock"])
    assert {s.subset for s in t} == {"airline", "mock"} and len(t) == 6


def test_explicit_tasks_caps_size():
    assert len(build_dataset("tau2", tasks_per_subset=2, domains=["airline"])) == 2


def test_taskspec_carries_taxonomy():
    spec = build_dataset("appworld", tasks_per_subset=1, splits=["test_challenge"], levels=[2])[0]
    assert spec.env == "appworld" and spec.split == "test_challenge" and spec.level == 2
    assert spec.subset == "test_challenge/L2"


def test_unknown_selectors_raise():
    with pytest.raises(KeyError):
        subset_size("appworld", "bogus/L1")
    with pytest.raises(KeyError):
        build_dataset("tau2", domains=["nope"])
    with pytest.raises(KeyError):
        subsets("not_an_env")
