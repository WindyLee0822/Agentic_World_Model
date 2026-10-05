#!/usr/bin/env python3
"""Use already-collected N=1 Ours trajectories, with fixed source priority and no duplicates."""
import hashlib
import json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCES = [
    ('ours_run', 'dw-officebench-ours-gpt54-tokens-n1-20260908'),
    ('ours_run_2app', 'dw-officebench-ours-gpt54-tokens-n1-2app-20260908'),
    ('ours_run_3app', 'dw-officebench-ours-gpt54-tokens-n1-3app-20260908'),
]
EXPECTED = {'1app': 93, '2app': 95, '3app': 112}


def main():
    selected, sources = {}, {}
    reference = [json.loads(line) for line in (HERE / 'b1_run/named-outputs/output_dir/records.jsonl').read_text().splitlines()]
    reference_ids = {r['task_id'] for r in reference if r['n_rollouts'] == 1}
    assert len(reference_ids) == 300
    for folder, job in SOURCES:
        path = HERE / folder / 'named-outputs/output_dir/records.jsonl'
        data = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        group = [r for r in data if r.get('method') == 'ours' and r.get('n_rollouts') == 1 and not r.get('coverage_imputed_failure')]
        assert len({r['task_id'] for r in group}) == len(group)
        assert all(r['task_id'] in reference_ids for r in group)
        assert all(r.get('llm_agent') == 'gpt-5.4' and r.get('solver_max_steps') == 40 and not r.get('predict_all_actions') for r in group)
        assert all(len(r.get('candidate_actions', [])) == 1 for r in group)
        added = 0
        for row in group:
            if row['task_id'] not in selected:
                selected[row['task_id']] = dict(row, source_aml_job=job)
                added += 1
        sources[job] = dict(path=str(path.relative_to(HERE)), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                            available_tasks=len(group), selected_tasks=added, duplicate_tasks_skipped=len(group)-added)
    rows = sorted(selected.values(), key=lambda r: (r['subset'], r['spec_index']))
    counts = Counter(r['subset'] for r in rows)
    assert 0 < len(rows) <= 300 and all(0 < counts[s] <= EXPECTED[s] for s in EXPECTED)
    (HERE / 'ours_records.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    manifest = dict(n_tasks=len(rows), population_n_tasks=300, full_coverage=len(rows)==300,
                    observed_subset_counts=dict(counts), population_subset_counts=EXPECTED,
                    matched_b1_task_ids=True, sources=sources,
                    missing_task_ids=sorted(reference_ids-set(selected)),
                    selection='Completed trajectories available when stopped, not a random sample. Prefer original run, then two-app, then three-app run for duplicates; never select by token cost or success.',
                    weighting='Each observed task in subset s has weight population_count[s] / observed_count[s]. This corrects app-count proportions, not completion-time selection bias.')
    (HERE / 'merged_run_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print('Merged collected Ours trajectories:',len(rows),dict(counts))


if __name__ == '__main__': main()
