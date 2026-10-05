import sys,os,json
os.chdir("/tmp")
from pathlib import Path
sys.path[:]=[p for p in sys.path if p and not p.startswith(('/mnt/batch/','/home/azureuser/cloudfiles/code/'))]
sys.path.insert(0,'/home/azureuser/.azure/cliextensions-local/ml')
from azure.ai.ml import MLClient,load_job
from azure.ai.ml.entities._assets._artifacts.code import Code
from azure.identity import ManagedIdentityCredential
from azure.core.exceptions import ResourceNotFoundError
root=Path('/home/azureuser/cloudfiles/code/Agentic_World_Model/experiments/appworld_ours_relaxed_20260911/r30')
c=MLClient(ManagedIdentityCredential(client_id=os.environ['DEFAULT_IDENTITY_CLIENT_ID']),'acc09744-1ee3-4242-b375-93421c63af0c','external-collaborations','M365Research-External-EUS')
code_file=root/'code-asset.json'
if code_file.exists():
 code_id=json.loads(code_file.read_text())['id']
else:
 code=c._code.create_or_update(Code(name='appworld-ours-r30-20260915',version='1',path='/tmp/appworld-r30-stage'))
 code_id=code.id;code_file.write_text(json.dumps({'id':code_id},indent=2)+'\n')
print(json.dumps({'code':code_id}),flush=True)
submitted=[]
for spec in json.loads((root/'job-manifest.json').read_text()):
 try:
  j=c.jobs.get(spec['name'])
  assert j.code==code_id, 'Existing job has different code'
 except ResourceNotFoundError:
  j=load_job(spec['job_file']);j.code=code_id
  j=c.jobs.create_or_update(j)
 expected_n='--n "'+str(spec['n'])+'"'
 assert expected_n in j.command and '--agent llm_wm' in j.command and '--methods ours' in j.command
 assert '--splits "'+spec['split']+'"' in j.command
 assert '--levels "'+str(spec['level'])+'"' in j.command
 assert getattr(j.inputs['extra_args'], '_data', j.inputs['extra_args'])=='--predict-all-actions', repr(j.inputs)
 row={'name':j.name,'status':j.status,'code':j.code,'n':spec['n'],'split':spec['split'],'level':spec['level'],'tasks':spec['tasks'],'command_verified':True}
 submitted.append(row);(root/'submissions.json').write_text(json.dumps(submitted,indent=2)+'\n')
 print(json.dumps(row),flush=True)
print('All 18 jobs accepted and their commands verified.',flush=True)
