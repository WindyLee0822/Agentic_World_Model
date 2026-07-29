# WM Verifier Prompt Optimization (OfficeBench + τ²)

Reflection-based **prompt optimization** for the world-model **success verifiers** of OfficeBench
(`OfficeBenchVerifier.predict_success`) and τ² (`_LLMWorldModelPredictor.predict_success`).

This package lives entirely outside `src/tts` and **never edits it**. The current (baseline) prompt is
always preserved; an optimized prompt is only *opt-in*. If optimization does not help, you keep the
current version — nothing in `src/tts` changes.

## Why the verifier?

The verifier turns a trajectory into a single success/fail label. The **true label** (from executing
the trajectory in the real env) is a clean supervised signal, so the verifier prompt is the natural
optimization target. Predictor prompts (write/observation prediction) have no single scalar label and
are out of scope here.

## Algorithm

1. **Sample** trajectories with the agent.
2. **Label** each by executing it in the **real environment** (ground-truth reward).
3. **Capture** the exact verifier inputs for each trajectory (as template *data slots*).
4. An **advanced model** reads the misclassified cases, summarizes the failure patterns, and rewrites
   the prompt wording (a *textual gradient*).
5. **Keep the best** prompt over an eval subset spanning **all OfficeBench app-levels (1/2/3-app)** and
   **≥2 τ² domains**. The baseline is always retained for fallback; a candidate is adopted only if it
   beats the current best.

Only wording is mutated: a template is `system` + ordered `sections`, where `text` sections (the
instructions) are editable and `data` sections (instruction/trajectory/predicted-state/gold-checks)
are fixed. This guarantees the optimizer can reword but never break the data plumbing.

## Layout

| file | role |
|------|------|
| `templates.py` | `PromptTemplate` + **faithful baselines** mirroring the source prompts + render/parse |
| `slots.py` | reproduce the source verifiers' data formatting → template data slots |
| `dataset.py` | `LabeledExample`, JSONL store, `SubsetSpec` (OB app-levels + τ² domains) |
| `patching.py` | capture hook (collection) + `apply_best` (opt-in inference) — monkeypatch, no source edits |
| `collect_tau2.py` | **cheap, LLM-free** τ² collection (corrupt gold → `EvaluationType.ENV` label → flat-WM slot capture) |
| `collect_officebench.py` | OfficeBench collection (`run_agent_rollout_wm` + `commit_trajectory`) |
| `evaluate.py` | score a template → accuracy / balanced-accuracy / F1 + per-slice |
| `optimize.py` | the reflection loop; keep best; persist best + baseline + report |
| `llm.py` | `RealLLM` (wraps `tts.llm.ChatClient`) + deterministic `MockLLM` |
| `run.py` | CLI: `collect` / `optimize` / `eval` / `show-best` |
| `store/` | outputs: datasets, `*_baseline.json`, `*_best.json`, `*_report.json` |

## Usage

```bash
source get_ready.sh          # Azure endpoint + WM_MODEL + az login

# 1) collect labeled data (τ² is cheap & LLM-free; OfficeBench needs Docker + the agent LLM)
python -m promptopt.run collect --env tau2 --domains airline telecom --tasks 4 \
       --out promptopt/store/tau2_data.jsonl
python -m promptopt.run collect --env officebench --apps 1 2 3 --tasks 2 --seeds 2 \
       --out promptopt/store/ob_data.jsonl

# 2) optimize the verifier prompt (advanced model rewrites it; keeps best ≥ baseline)
python -m promptopt.run optimize --env tau2 --data promptopt/store/tau2_data.jsonl --rounds 4

# 3) inspect / evaluate
python -m promptopt.run show-best --env tau2
python -m promptopt.run eval --env tau2 --data promptopt/store/tau2_data.jsonl --template best
```

Use `--optimizer-model` to run the reflection step with a stronger model than the verifier.

## Using an optimized prompt (opt-in, no source edits)

```python
from promptopt.patching import apply_best
with apply_best("tau2"):          # patches the source verifier to render the stored best template
    ...                           # run your normal τ² world-model selection here
# outside the block, the original source prompt is restored — baseline stays the default
```

## Offline test

```bash
python -m pytest promptopt/tests/test_offline_loop.py -q     # deterministic, no network
```

## Example result (real run, gpt-5.4)

τ² (27 examples, airline + telecom), `balanced_accuracy`:

```
baseline: bal_acc=0.850  (fp=6, fn=0)   airline=0.88 telecom=0.64
round 1 : bal_acc=0.900  (fp=4, fn=0)   airline=1.00 telecom=0.64   ACCEPT
```

The advanced model independently added a strict system prompt (τ²'s baseline had none), **structural
normalization of auto-generated ids/timestamps**, toggle-parity / side-effect accounting, and
**user-facing outcome** checks — cutting the verifier's false positives (its rubber-stamping failure
mode).

## Scope

- Optimizes the **one-shot** verifier prompts (`predict_success`). The τ² *agentic* verifier
  (`predict_success_agentic`) and the WM *predictor* prompts are analogous but out of scope.
- τ² true labels use `EvaluationType.ENV` (DB/assertion check — LLM-free). Some domains (e.g. telecom
  dual-control) may label gold as fail under ENV-only scoring; those contribute hard negatives.
