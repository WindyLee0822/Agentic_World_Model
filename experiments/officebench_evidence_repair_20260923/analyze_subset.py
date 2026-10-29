"""Paired task-set comparison; no actions executed during analysis."""
import collections
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW = Path('/tmp/ob-evidence-r2-results/named-outputs/output_dir')
OUT = ROOT / 'analysis'
OUT.mkdir(exist_ok=True)
for path in RAW.iterdir():
    if path.is_file():
        shutil.copy2(path, OUT / path.name)
tasks = set(json.loads((ROOT / 'tasks.json').read_text()))
def rows(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]
old = rows(ROOT / 'baseline_records.jsonl')
new = rows(RAW / 'records.jsonl')
assert {(r['task_id'], r['n_rollouts']) for r in old} == {(r['task_id'], r['n_rollouts']) for r in new}
assert len(new) == 24
assert all(r['predict_all_actions'] and r['officebench_verifier_typed'] and r['llm_agent'] == 'gpt-5.4' for r in new)
def summary(records):
    out = {}
    for n in (1, 2, 4):
        rs = [r for r in records if r['n_rollouts'] == n]
        out[n] = dict(tasks=len(rs), success=sum(r['success'] for r in rs),
            agreement=sum(r['success'] == r['predicted_success'] for r in rs),
            TP=sum(r['success'] and r['predicted_success'] for r in rs),
            FP=sum(not r['success'] and r['predicted_success'] for r in rs),
            FN=sum(r['success'] and not r['predicted_success'] for r in rs),
            TN=sum(not r['success'] and not r['predicted_success'] for r in rs),
            all_zero=sum(not any(r['candidate_confidences']) for r in rs),
            python_calls=sum(r['n_python_calls'] for r in rs),
            python_failures=sum(r['n_python_failures'] for r in rs),
            simulator_writes=sum(r['simulator_writes'] for r in rs),
            commits=sum(r['n_commits'] for r in rs),
            rollout_errors=sum(r['n_rollout_errors'] for r in rs))
    return out

def objects(path, prefix):
    text = path.read_text()
    decoder, cursor, out = json.JSONDecoder(), 0, []
    while (pos := text.find(prefix, cursor)) >= 0:
        cursor = pos + len(prefix)
        try:
            val, length = decoder.raw_decode(text[cursor:])
            cursor += length
            if val.get('task_id') in tasks:
                out.append(val)
        except json.JSONDecodeError:
            pass
    return out

def observations(path):
    traces = objects(path, '[officebench-commit-trace] ')
    by_action = collections.defaultdict(collections.Counter)
    for t in traces:
        a = t['action']
        count = by_action[a.get('app', '?') + '.' + a.get('action', '?')]
        count['total'] += 1
        count['exact'] += t['predicted'] == t['actual']
    return dict(total=len(traces), exact=sum(t['predicted'] == t['actual'] for t in traces),
                by_action=dict(by_action)), [t for t in traces if t['predicted'] != t['actual']]
old_obs, _ = observations(ROOT.parent / 'officebench_repaired_full_20260922/analysis/run.log')
new_obs, mismatches = observations(RAW / 'run.log')
paired = []
old_by_key = {(r['task_id'], r['n_rollouts']): r for r in old}
keys = ['success', 'predicted_success', 'predicted_confidence', 'selected_index',
        'candidate_confidences', 'candidate_verifier_percheck', 'n_python_failures']
for r in sorted(new, key=lambda r: (r['task_id'], r['n_rollouts'])):
    b = old_by_key[r['task_id'], r['n_rollouts']]
    paired.append(dict(task_id=r['task_id'], n=r['n_rollouts'],
                       before={k:b[k] for k in keys}, after={k:r[k] for k in keys}))
result = dict(before=summary(old), after=summary(new),
              observations_before=old_obs, observations_after=new_obs, paired=paired)
(OUT / 'comparison.json').write_text(json.dumps(result, indent=2))
(OUT / 'observation_mismatches.json').write_text(json.dumps(mismatches, indent=2))
print(json.dumps({k:v for k,v in result.items() if k != 'paired'}, indent=2))
