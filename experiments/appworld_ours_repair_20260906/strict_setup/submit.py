import json,os,sys
from pathlib import Path
sys.path.insert(0,'/home/azureuser/.azure/cliextensions-local/ml')
from azure.ai.ml import MLClient, load_job
from azure.identity import ManagedIdentityCredential
client=MLClient(ManagedIdentityCredential(client_id=os.environ['DEFAULT_IDENTITY_CLIENT_ID']),'acc09744-1ee3-4242-b375-93421c63af0c','external-collaborations','M365Research-External-EUS')
job=load_job('/tmp/appworld-strict-stage/aml/job.appworld.ours.strict-r17-full.yml')
job.name='dw-appworld-ours-strict-r17-full-20260907'
result=client.jobs.create_or_update(job)
Path('/tmp/appworld-strict-submission.json').write_text(json.dumps({'name':result.name,'status':result.status,'code':result.code},indent=2)+'\n')
print(Path('/tmp/appworld-strict-submission.json').read_text(),flush=True)
