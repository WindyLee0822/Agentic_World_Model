from pathlib import Path
import json,shutil,hashlib,xml.etree.ElementTree as ET
import yaml
root=Path('/home/azureuser/cloudfiles/code/Agentic_World_Model/experiments/appworld_ours_relaxed_20260911')
out=root/'r30';out.mkdir(exist_ok=True);stage=Path('/tmp/appworld-r30-stage')
base=yaml.safe_load((root/'r29/job.appworld.ours.relaxed-r29-n1-full.yml').read_text())
base['environment_variables']['TTS_APPWORLD_VERIFIER_PYTHON']='1'
rows=[json.loads(x) for x in (root/'r29/sidecar-records.jsonl').read_text().splitlines()]
manifest=[]
for n in [1,2,4]:
 for split,short in [('test_normal','normal'),('test_challenge','challenge')]:
  for level in [1,2,3]:
   job=dict(base);job['environment_variables']=dict(base['environment_variables'])
   name=f'dw-appworld-ours-r30-n{n}-{short}-l{level}-20260915'
   job['display_name']=name;job['name']=name
   job['description']='R30 AppWorld Ours DeepSeek-V4-Pro. Opt-in compatible Python runtime and gold-blind state reconstruction before simulator verification. Predict all actions; no native candidate writes; replay selected candidate only.'
   cmd=base['command'].replace('--n "1"',f'--n "{n}"').replace('--splits "test_normal,test_challenge"',f'--splits "{split}"').replace('--levels "1,2,3"',f'--levels "{level}"').replace('--workers 16','--workers 4')
   # YAML folding can retain whitespace around a line break.
   import re
   cmd=re.sub(r'--levels\s+"1,2,3"',f'--levels "{level}"',cmd)
   assert '--predict-all-actions' in str(job['inputs']) and '--n "'+str(n)+'"' in cmd
   assert f'--levels "{level}"' in cmd and f'--splits "{split}"' in cmd
   job['command']=cmd
   file=stage/'aml'/f'job.appworld.ours.r30.n{n}.{short}.l{level}.yml'
   file.write_text(yaml.safe_dump(job,sort_keys=False));shutil.copy2(file,out/file.name)
   ids=sorted(r['task_id'] for r in rows if r['subset']==f'{split}/L{level}')
   manifest.append({'name':name,'n':n,'split':split,'level':level,'tasks':len(ids),'task_ids':ids,'workers':4,'job_file':str(file)})
for n in [1,2,4]:
 ids=[(j['split'],j['level'],t) for j in manifest if j['n']==n for t in j['task_ids']]
 assert len(ids)==len(set(ids))==585
(out/'job-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
changed=[]
for f in (stage/'src').rglob('*.py'):
 rel=f.relative_to(stage);old=Path('/tmp/appworld-r29-stage')/rel
 if not old.exists() or old.read_bytes()!=f.read_bytes():
  changed.append(str(rel));target=out/'staged_changes'/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,target)
assert set(changed)=={'src/tts/llm/python_tool.py','src/tts/env/appworld_ours_model_driven.py'},changed
(out/'isolation.json').write_text(json.dumps({'base_stage':'/tmp/appworld-r29-stage','changed_source_files':changed,'runtime_compatibility_default':False,'activation':'Only AppWorldOursModelDrivenPredictor passes appworld_compat=True','hashes':{f:hashlib.sha256((stage/f).read_bytes()).hexdigest() for f in changed}},indent=2)+'\n')
shutil.copy2(stage/'tests/test_r30_repairs.py',out/'test_r30_repairs.py')
checks={}
for label,file in [('repairs','/tmp/r30-repairs.xml'),('regressions','/tmp/r30-regressions.xml'),('final_checks','/tmp/r30-final-checks.xml')]:
 a=ET.parse(file).getroot()[0].attrib;assert a['errors']=='0' and a['failures']=='0';checks[label]={k:a[k] for k in ['tests','errors','failures','skipped']};shutil.copy2(file,out/Path(file).name)
checks.update(unique_tests_passed=77,full_coverage_per_n=585,decisions=1755,minimum_candidates=4095,performance_improvement_unmeasured=True)
(out/'validation.json').write_text(json.dumps(checks,indent=2)+'\n')
print(json.dumps({'jobs':len(manifest),'tasks_per_n':585,'changed':changed,'tests':checks}))
