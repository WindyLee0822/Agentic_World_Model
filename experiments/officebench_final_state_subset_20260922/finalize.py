from pathlib import Path
import collections,json,shutil,sys
out=Path(__file__).parent
source=Path(sys.argv[1]);analysis=out/'analysis';analysis.mkdir(exist_ok=True)
for name in ['records.jsonl','metadata.json','run.log','run_manifest.json','summary.json','summary.txt','runtime_preflight.json']:
 if (source/name).exists():shutil.copyfile(source/name,analysis/name)
metrics=json.loads((analysis/'metrics.json').read_text());rows=[json.loads(x) for x in (analysis/'records.jsonl').read_text().splitlines() if x]
old=metrics['previous_same_cells'];new=metrics['repaired_subset'];fidelity=metrics['selected_observation_fidelity']
actions=sum(v['observations'] for v in fidelity.values());exact=sum(v['exact'] for v in fidelity.values())
lines=['# Runtime and state repair: four-task subset','','Job: `dw-officebench-gpt54-final-state-n124-20260922`. Four conversion/email tasks, N=1,2,4, GPT-5.4, predict-all, select then commit one candidate.','','## Native task results','','| N | Before | Repaired | All-zero pools | Exhausted candidates |','|---|---|---|---|---|']
for n,v in new.items():
 b=old[n];lines.append(f"| {n} | {b['successes']}/{b['tasks']} | {v['successes']}/{v['tasks']} | {v['all_zero_tasks']}/{v['tasks']} | {v['exhausted_error_candidates']}/{v['candidates']} |")
lines += ['', '## Success-prediction agreement', '', '| N | Correct / tasks | False positives | False negatives |', '|---|---|---|---|']
for n,v in new.items():
 lines.append(f"| {n} | {v['true_positive']+v['true_negative']}/{v['tasks']} | {v['false_positive']} | {v['false_negative']} |")
lines += ['',f'Exact selected-action observation fidelity: **{exact}/{actions} ({exact/actions:.1%})**. Action mixtures differ across runs; these are not paired identical-action trials.','','| Action | Exact / total |','|---|---|']
for action,v in sorted(fidelity.items()):lines.append(f"| {action} | {v['exact']}/{v['observations']} |")
lines += ['',f"Simulator writes: {sum(v['simulator_writes'] for v in new.values())}; real commits: {sum(v['commits'] for v in new.values())}.",'','## Task outcomes','','| Task | N | Success | Confidence | Selected |','|---|---|---|---|---|']
for r in sorted(rows,key=lambda r:(r['task_id'],r['n_rollouts'])):lines.append(f"| {r['task_id']} | {r['n_rollouts']} | {r['success']} | {r['predicted_confidence']} | {r['selected_index']} |")
lines += ['','Native evaluator scores are reported unchanged. A passing file-existence check alone does not establish PDF validity. The runtime preflight independently validates synthetic Word/Excel PDFs, text content, shell listing, and network denial. Per-check selection audits remain predictions.','', 'See `README.md`, `changes.patch`, `tests_main.txt`, `tests_other.txt`, `analysis/metrics.json`, and `analysis/observation_mismatches.json` for implementation and evidence.']
lines += ['', 'Task `3-1/0` has an independently checked evaluator/fixture mismatch: its checks require `Activaty1..4`, whereas the spreadsheet contains `Activity 1..4`. See `evaluator_fixture_mismatch.json`. No evaluator or fixture change was made.']
(out/'report.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines))
