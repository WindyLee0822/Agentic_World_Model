"""Trajectory-level, read-grounded test-time scaling over real, non-forkable environments.

``N`` parallel *trajectories* are sampled against one real, stateful, non-forkable
:class:`~tts.abcs.TaskEnvironment` (the genuine AppWorld / τ²-bench benchmarks, executed and
scored through their official runners).  Each trajectory is scored by a per-rollout
*world model* that predicts its outcome by reading the **real backing database** (privileged,
read-only) and, when one bounded read is insufficient, recursively spawning region-scoped probe
subagents.  A pluggable ``success_score`` selects one trajectory and its **whole action
sequence** is committed through a single chokepoint.  See ``README.md`` for the diagram.
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.4.0"
