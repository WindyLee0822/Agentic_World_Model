"""Poll the existing R20 job and atomically audit any available records."""
import importlib.util
import json
import os
from pathlib import Path
import sys
from datetime import datetime, timezone

# Avoid slow negative import lookups on the shared repository filesystem.
sys.path[:] = [item for item in sys.path if item and not item.startswith((
    '/mnt/batch/', '/home/azureuser/cloudfiles/code/'))]
sys.path.insert(0, '/home/azureuser/.azure/cliextensions-local/ml')
from azure.ai.ml import MLClient
from azure.identity import ManagedIdentityCredential
from azure.storage.blob import ContainerClient
from azure.core.exceptions import ResourceNotFoundError

root = Path(__file__).resolve().parent

credential = ManagedIdentityCredential(client_id=os.environ['DEFAULT_IDENTITY_CLIENT_ID'])
client = MLClient(credential, 'acc09744-1ee3-4242-b375-93421c63af0c',
                  'external-collaborations', 'M365Research-External-EUS')
job_name = 'dw-appworld-ours-relaxed-r25-n1-probe-20260913'
job = client.jobs.get(job_name)
status = {'name': job.name, 'status': job.status, 'code': job.code,
          'checked_at': datetime.now(timezone.utc).isoformat()}
container = ContainerClient(
    account_url='https://m365researchex2937532597.blob.core.windows.net',
    container_name='azureml-blobstore-0e758b43-c315-4fe1-891b-a93eb147fd79',
    credential=credential)
prefix='azureml/'+job_name+'/output_dir/.cell-checkpoints/record-'
records=[]
for blob in container.list_blobs(name_starts_with=prefix):
 if blob.size:
  records.append(json.loads(container.download_blob(blob.name).readall()))
if not records and job.status in {'Completed','Failed','Canceled'}:
 try:
  raw=container.download_blob('azureml/'+job_name+'/output_dir/records.jsonl').readall().decode()
  records=[json.loads(line) for line in raw.splitlines() if line.strip()]
  status['record_source']='final records.jsonl'
 except ResourceNotFoundError:
  pass
status['records']=len(records)
status['successes']=sum(bool(r.get('success')) for r in records)
status['diagnostic_only']=True
status['command_checks']={'n1':'--n \"1\"' in job.command, 'task_index7':'--task-indices 7' in job.command, 'normal_only':'--splits \"test_normal\"' in job.command, 'one_worker':'--workers 1' in job.command}
status['expected_tasks']=['634f342_2']
status['expected_n']=1
status['command_checks']['level2']='--levels \"2\"' in job.command or '--levels\n  \"2\"' in job.command
(root/'probe-status.json').write_text(json.dumps(status,indent=2)+'\n')
(root/'probe-records.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
print(json.dumps(status),flush=True)
