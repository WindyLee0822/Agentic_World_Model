"""Real benchmark environments + dataset taxonomy for the trajectory-level TTS harness."""

from __future__ import annotations

from .datasets import (
    APPWORLD_LEVELS,
    APPWORLD_LOCAL_SPLITS,
    APPWORLD_OFFICIAL_SIZE,
    APPWORLD_SPLITS,
    REAL_ENVS,
    TAU2_DOMAINS,
    TAU2_OFFICIAL_SIZE,
    TaskSpec,
    build_dataset,
    dataset_size,
    subset_size,
    subsets,
)
from .registry import available_real_envs, make_env, make_env_from_spec, tau2_usable_size
from .real_tau2 import Tau2TaskEnv, real_tau2_available, real_tau2_subset_size

__all__ = [
    "REAL_ENVS",
    "APPWORLD_SPLITS",
    "APPWORLD_LOCAL_SPLITS",
    "APPWORLD_LEVELS",
    "APPWORLD_OFFICIAL_SIZE",
    "TAU2_DOMAINS",
    "TAU2_OFFICIAL_SIZE",
    "TaskSpec",
    "build_dataset",
    "dataset_size",
    "subset_size",
    "subsets",
    "make_env",
    "make_env_from_spec",
    "available_real_envs",
    "tau2_usable_size",
    "Tau2TaskEnv",
    "real_tau2_available",
    "real_tau2_subset_size",
]
