import copy
import importlib.util
import json
from pathlib import Path
root=Path('/home/azureuser/cloudfiles/code/Agentic_World_Model/experiments')
report=root/'appworld_ours_relaxed_20260911/r20'
spec=importlib.util.spec_from_file_location('audit', report/'audit_records.py')
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
reference=[json.loads(x) for x in (root/'appworld_ours_repair_20260906/baseline/records.jsonl').read_text().splitlines()]
seed=json.loads((report.parent/'records.jsonl').read_text().splitlines()[0])
identities=sorted({(x['subset'],x['task_id']) for x in reference if x['env']=='appworld' and x['method']=='ours'})
rows=[{**copy.deepcopy(seed),'subset':s,'task_id':t} for s,t in identities]
def check(r): return audit.evaluate(r,reference,require_setup_predictions=True)
assert check(rows)['record_contract_passed']
assert check(rows)['completion_proven'] is False
assert not check(rows[:-1])['full_coverage']
wrong=copy.deepcopy(rows);wrong[0]['task_id']='invented'
assert len(wrong)==585 and not check(wrong)['full_coverage']
assert not check(rows+[rows[0]])['full_coverage']
for field,value in [('n_rollouts',2),('success_score','oracle'),('simulator_writes',1),('n_commits',2),('predict_all_actions',False),('candidate_grounded_success',[True]*16),('candidate_setup_prediction',[{'enabled':False,'count':0}]*16)]:
 bad=copy.deepcopy(rows);bad[0][field]=value
 assert not check(bad)['record_contract_passed'],field
partial=check([json.loads(x) for x in (report.parent/'records.jsonl').read_text().splitlines()])
assert partial['official_successes']==40 and not partial['full_coverage']
assert partial['success_rate'] is None
(report/'audit-checks.json').write_text(json.dumps({'checks_passed':13,'r19_partial':partial},indent=2)+'\n')
print('13 audit checks passed; R19 remains incomplete (40/44, 541 missing).')
