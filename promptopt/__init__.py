"""Reflection-based prompt optimization for the WM success verifiers (OfficeBench + τ²).

This package lives *outside* ``src/tts`` and never edits it.  It optimizes the natural-language
wording of the world-model **success verifier** prompts by:

1. sampling agent trajectories and recording the TRUE label from the real environment,
2. capturing the exact verifier inputs for each trajectory (collection),
3. having an advanced model summarize failure patterns and propose an improved prompt, and
4. keeping the best prompt over an eval subset (all OfficeBench app-levels + ≥2 τ² domains).

The current (baseline) prompt is always retained, so if optimization does not help you keep it.
Use :func:`promptopt.patching.apply_best` (opt-in) to run inference with an optimized prompt without
touching any source file.
"""

from __future__ import annotations

__all__ = ["templates", "dataset", "evaluate", "optimize", "llm", "patching"]
