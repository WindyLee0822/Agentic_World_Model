"""Summarize the completed full run without replaying any candidate action."""
import collections
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).parent
OUT = ROOT / 'analysis'
OUT.mkdir(exist_ok=True)
RAW = Path('/tmp/ob-repaired-full-results/named-outputs/output_dir')
SOURCE = Path('/tmp/officebench-repaired-full-n124-20260922')
for path in RAW.iterdir():
    if path.is_file():
        shutil.copy2(path, OUT / path.name)
shutil.copy2('/tmp/ob-full-status.json', ROOT / 'completed_job.json')
rows = [json.loads(line) for line in (RAW / 'records.jsonl').read_text().splitlines()]
assert len(rows) == 900
assert len({(r['task_id'], r['n_rollouts']) for r in rows}) == 900
assert all(r['predict_all_actions'] and r['llm_agent'] == 'gpt-5.4' for r in rows)

traces, errors = [], []
log = (RAW / 'run.log').read_text()
decoder = json.JSONDecoder()
for prefix, target in [('[officebench-commit-trace] ', traces),
                       ('[officebench-python-error] ', errors)]:
    cursor = 0
    while (position := log.find(prefix, cursor)) >= 0:
        cursor = position + len(prefix)
        try:
            value, length = decoder.raw_decode(log[cursor:])
            target.append(value)
            cursor += length
        except json.JSONDecodeError:
            pass

per_n = {}
for n in (1, 2, 4):
    selected = [r for r in rows if r['n_rollouts'] == n]
    assert len(selected) == 300
    per_n[n] = {
        'tasks': len(selected), 'successes': sum(r['success'] for r in selected),
        'TP': sum(r['success'] and r['predicted_success'] for r in selected),
        'FP': sum(not r['success'] and r['predicted_success'] for r in selected),
        'FN': sum(r['success'] and not r['predicted_success'] for r in selected),
        'TN': sum(not r['success'] and not r['predicted_success'] for r in selected),
        'all_zero_pools': sum(not any(r['candidate_confidences']) for r in selected),
        'python_calls': sum(r['n_python_calls'] for r in selected),
        'python_failures': sum(r['n_python_failures'] for r in selected),
        'error_candidates': sum(r['n_rollout_errors'] for r in selected),
        'commits': sum(r['n_commits'] for r in selected),
        'simulator_writes': sum(r['simulator_writes'] for r in selected),
    }
by_action = collections.defaultdict(collections.Counter)
mismatches = []
for trace in traces:
    action = trace['action']
    name = action.get('app', '?') + '.' + action.get('action', '?')
    by_action[name]['total'] += 1
    by_action[name]['exact'] += trace['predicted'] == trace['actual']
    if trace['predicted'] != trace['actual']:
        mismatches.append(trace)

by_task = collections.defaultdict(dict)
for row in rows:
    by_task[row['task_id']][row['n_rollouts']] = row
n4 = [r for r in rows if r['n_rollouts'] == 4]
false_negatives = [r for r in n4 if r['success'] and not r['predicted_success']]
fn_categories = collections.Counter()
for r in false_negatives:
    audit = r['candidate_verifier_percheck'][r['selected_index']]
    category = ('precheck_or_no_audit' if audit is None else
                'unmet' if audit['unmet'] else 'unknown' if audit['unknown'] else
                'all_met_low_whole')
    fn_categories[category] += 1

regressions = []
for task, runs in by_task.items():
    if runs[1]['success'] and not runs[4]['success']:
        regressions.append({'task_id': task, 'runs': [{
            k: runs[n][k] for k in ['n_rollouts', 'success', 'selected_index',
                'candidate_done', 'candidate_confidences', 'candidate_verifier_percheck',
                'candidate_comparative_rerank']} for n in (1, 2, 4)]})

associations = collections.defaultdict(list)
for r in n4:
    task, subtask = r['task_id'].split('/')
    cfg = json.loads((SOURCE / 'officebench_data/tasks' / task / 'subtasks' / (subtask + '.json')).read_text())
    checks = cfg['evaluation']
    flags = {
        'email': any(c['args'].get('doc_type') == 'email' for c in checks),
        'docx_existence_without_type': any(c['function'] == 'evaluate_file_exist' and
            c['args'].get('file', '').endswith('.docx') and 'doc_type' not in c['args'] for c in checks),
        'excel_cell': any(c['function'] == 'evaluate_excel_cell_value' for c in checks),
        'diff_contain': any(c['function'] == 'evaluate_diff_contain_text' for c in checks),
        'split_negative': any(c['function'] == 'evaluate_not_contain' and
            1 < len(c['args'].get('keywords', [])) <= 4 for c in checks),
    }
    for flag, value in flags.items():
        if value:
            associations[flag].append({'task_id': r['task_id'],
                'false_negative': bool(r['success'] and not r['predicted_success'])})

result = {
    'job': 'dw-officebench-gpt54-repaired-full-n124-20260922',
    'per_n': per_n, 'observations': len(traces), 'exact_observations': len(traces)-len(mismatches),
    'observation_fidelity': (len(traces)-len(mismatches))/len(traces),
    'by_action': dict(by_action), 'n4_false_negative_categories': dict(fn_categories),
    'logged_python_errors': len(errors),
    'errors_by_kind': dict(collections.Counter(e['kind'] for e in errors)),
    'errors_by_message': dict(collections.Counter(e['error'] for e in errors)),
    'n1_to_n4_regressions': regressions,
    'n1_to_n4_improvements': [t for t,d in by_task.items() if not d[1]['success'] and d[4]['success']],
    'associations_not_causal_attributions': dict(associations),
    'diagnostic_mismatch_counts': {
        'repeat_feedback': sum('repeated action' in t['predicted'] for t in mismatches),
        'predict_prefix': sum(t['predicted'].startswith('PREDICT:') for t in mismatches),
        'grounding_refusal': sum('cannot be proven read-only' in t['predicted'] for t in mismatches),
        'llm_missing_key': sum(t['action'].get('app') == 'llm' and '/openai_key.txt' in t['actual'] for t in mismatches),
    },
}
(OUT / 'audit.json').write_text(json.dumps(result, indent=2))
(OUT / 'false_negatives_n4.json').write_text(json.dumps(false_negatives, indent=2))
(OUT / 'false_positives.json').write_text(json.dumps([
    r for r in rows if not r['success'] and r['predicted_success']], indent=2))
print(json.dumps({'per_n': per_n, 'observations': len(traces),
                  'exact_observations': result['exact_observations'],
                  'n4_false_negative_categories': dict(fn_categories)}, indent=2))
