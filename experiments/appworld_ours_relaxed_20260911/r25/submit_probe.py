import json,os,sys
from pathlib import Path
sys.path[:] = [p for p in sys.path if p and not p.startswith(('/mnt/batch/','/home/azureuser/cloudfiles/code/'))]
sys.path.insert(0,'/home/azureuser/.azure/cliextensions-local/ml')
from azure.ai.ml import MLClient, load_job
from azure.identity import ManagedIdentityCredential
client=MLClient(ManagedIdentityCredential(client_id=os.environ['DEFAULT_IDENTITY_CLIENT_ID']),'acc09744-1ee3-4242-b375-93421c63af0c','external-collaborations','M365Research-External-EUS')
job=load_job('/tmp/appworld-r25-stage/aml/job.appworld.ours.relaxed-r25-n1-probe.yml')
job.name='dw-appworld-ours-relaxed-r25-n1-probe-20260913'
result=client.jobs.create_or_update(job)
p=Path('/home/azureuser/cloudfiles/code/Agentic_World_Model/experiments/appworld_ours_relaxed_20260911/r25/submission.json')
p.write_text(json.dumps({'name':result.name,'status':result.status,'code':result.code},indent=2)+'\n')
print(p.read_text(),flush=True)
