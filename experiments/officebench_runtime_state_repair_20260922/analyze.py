"""Compare completed smoke task/N cells and audit selected-observation fidelity."""
import collections
import json
from pathlib import Path
import sys

base_path = Path('experiments/officebench_contracts_subset_20260922/analysis/records.jsonl')
new_dir = Path(sys.argv[1]); out_dir = Path(sys.argv[2]); out_dir.mkdir(parents=True, exist_ok=True)
new = [json.loads(line) for line in (new_dir/'records.jsonl').read_text().splitlines() if line.strip()]
keys = {(r['subset'], r['task_id'], r['n_rollouts']) for r in new}
old = [r for line in base_path.read_text().splitlines() if (r := json.loads(line)) and (r['subset'], r['task_id'], r['n_rollouts']) in keys]

def summarize(rows):
    out = {}
    for n in sorted({r['n_rollouts'] for r in rows}):
        selected = [r for r in rows if r['n_rollouts'] == n]
        out[n] = {
            'tasks': len(selected), 'successes': sum(bool(r['success']) for r in selected),
            'exhausted_error_candidates': sum(r.get('n_rollout_errors', 0) for r in selected),
            'candidates': sum(r['n_rollouts'] for r in selected),
            'all_error_tasks': sum(r.get('n_rollout_errors') == n for r in selected),
            'all_zero_tasks': sum(not any(r.get('candidate_confidences', [])) for r in selected),
            'true_positive': sum(bool(r['success'] and r['predicted_success']) for r in selected),
            'false_positive': sum(bool(not r['success'] and r['predicted_success']) for r in selected),
            'false_negative': sum(bool(r['success'] and not r['predicted_success']) for r in selected),
            'true_negative': sum(bool(not r['success'] and not r['predicted_success']) for r in selected),
            'python_calls': sum(r.get('n_python_calls', 0) for r in selected),
            'python_failures': sum(r.get('n_python_failures', 0) for r in selected),
            'simulator_writes': sum(r.get('simulator_writes', 0) for r in selected),
            'commits': sum(r.get('n_commits', 0) for r in selected),
        }
    return out

traces = []; errors = []
log = (new_dir/'run.log').read_text(errors='replace')
decoder = json.JSONDecoder()
for prefix, dest in [('[officebench-commit-trace] ', traces), ('[officebench-python-error] ', errors)]:
    cursor = 0
    while (position := log.find(prefix, cursor)) >= 0:
        cursor = position + len(prefix)
        try:
            value, length = decoder.raw_decode(log[cursor:]); dest.append(value); cursor += length
        except json.JSONDecodeError:
            pass
counts = collections.defaultdict(collections.Counter)
for t in traces:
    key = t['action'].get('app', '?') + '.' + t['action'].get('action', '?')
    counts[key]['observations'] += 1
    counts[key]['exact'] += t['predicted'] == t['actual']
    counts[key]['trimmed_exact'] += t['predicted'].strip() == t['actual'].strip()
report = {'previous_same_cells': summarize(old), 'repaired_subset': summarize(new),
          'selected_observation_fidelity': dict(counts),
          'python_error_categories': dict(collections.Counter(e.get('error') for e in errors)),
          'task_outcomes': [{k:r.get(k) for k in ['subset','task_id','n_rollouts','success','predicted_confidence','n_rollout_errors','candidate_errors']} for r in new]}
(out_dir/'metrics.json').write_text(json.dumps(report,indent=2)+'\n')
(out_dir/'observation_mismatches.json').write_text(json.dumps([t for t in traces if t['predicted'].strip()!=t['actual'].strip()],indent=2)+'\n')
(out_dir/'python_errors.json').write_text(json.dumps(errors,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='task_outcomes'},indent=2))
