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
job_name = 'dw-appworld-ours-relaxed-r20-n16-full-20260912'
job = client.jobs.get(job_name)
status = {'name': job.name, 'status': job.status, 'code': job.code,
          'checked_at': datetime.now(timezone.utc).isoformat()}
container = ContainerClient(
    account_url='https://m365researchex2937532597.blob.core.windows.net',
    container_name='azureml-blobstore-0e758b43-c315-4fe1-891b-a93eb147fd79',
    credential=credential)
out=root/'failures'/'425a494_2';out.mkdir(parents=True,exist_ok=True)
for blob in container.list_blobs(name_starts_with='azureml/'+job_name+'/output_dir/wm_diagnostics/wm-671-'):
    if not blob.size:continue
    content=container.download_blob(blob.name).readall()
    first=json.loads(content.splitlines()[0])
    if first.get('task_id')=='425a494_2':
        (out/blob.name.rsplit('/',1)[-1]).write_bytes(content)
        print(json.dumps({'seed':first.get('seed'),'bytes':len(content),'file':blob.name.rsplit('/',1)[-1]}),flush=True)
