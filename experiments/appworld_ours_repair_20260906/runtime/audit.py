import pathlib,json,importlib.util,datetime,hashlib
report=pathlib.Path(__file__).resolve().parent.parent
repo=report.parent.parent
spec=importlib.util.spec_from_file_location('goal_gate',repo/'aml/src/check_appworld_ours_goal.py');gate=importlib.util.module_from_spec(spec);spec.loader.exec_module(gate)
reference=[json.loads(l) for l in (report/'baseline/records.jsonl').read_text().splitlines() if l]
for version in ['r14','r16']:
 rows=[json.loads(l) for l in (report/version/'records.jsonl').read_text().splitlines() if l]
 result=gate.evaluate(rows,reference);result['checked_at']=datetime.datetime.now(datetime.timezone.utc).isoformat();(report/f'partial-{version}-gate.json').write_text(json.dumps(result,indent=2)+'\n')
 p=report/f'full-{version}.json';manifest=json.loads(p.read_text());manifest['partial_results']={k:result[k] for k in ['checked_at','observed_decisions','official_successes','full_coverage','contract_violations']};manifest['source_matches_worktree']=all(hashlib.sha256((repo/f).read_bytes()).hexdigest()==h for f,h in manifest['source_sha256'].items());p.write_text(json.dumps(manifest,indent=2)+'\n')
 print(json.dumps({'version':version,'decisions':len(rows),'successes':result['official_successes'],'remaining':585-len(rows),'successes_needed':max(0,352-result['official_successes']),'contract_violations':len(result['contract_violations']),'full_coverage':result['full_coverage'],'goal_passed':result['passed'],'source_matches_worktree':manifest['source_matches_worktree']}),flush=True)
