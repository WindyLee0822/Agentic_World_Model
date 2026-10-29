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
job_name = 'dw-appworld-ours-relaxed-r22-n16-full-20260913'
job = client.jobs.get(job_name)
status = {'name': job.name, 'status': job.status, 'code': job.code,
          'checked_at': datetime.now(timezone.utc).isoformat()}
container = ContainerClient(
    account_url='https://m365researchex2937532597.blob.core.windows.net',
    container_name='azureml-blobstore-0e758b43-c315-4fe1-891b-a93eb147fd79',
    credential=credential)
from concurrent.futures import ThreadPoolExecutor
out=root/'verifier_reviews'/'cef9191_2';out.mkdir(parents=True,exist_ok=True)
prefix='azureml/'+job_name+'/output_dir/wm_diagnostics/'
blobs=list(container.list_blobs(name_starts_with=prefix))
groups={}
for blob in blobs:
    if blob.size:
        pid=blob.name.rsplit('/',1)[-1].split('-')[1]
        groups.setdefault(pid,[]).append(blob)
def identify(item):
    pid,items=item
    b=items[0]
    raw=container.download_blob(b.name,offset=0,length=min(b.size,512)).readall()
    first=json.loads(raw.splitlines()[0])
    return pid,first.get('task_id')
with ThreadPoolExecutor(max_workers=8) as pool:
    matches=[pid for pid,task in pool.map(identify,groups.items()) if task=='cef9191_2']
for pid in matches:
    for blob in groups[pid]:
        content=container.download_blob(blob.name).readall()
        first=json.loads(content.splitlines()[0])
        if first.get('task_id')=='cef9191_2':
            (out/blob.name.rsplit('/',1)[-1]).write_bytes(content)
            print(json.dumps({'seed':first.get('seed'),'bytes':len(content),'file':blob.name.rsplit('/',1)[-1]}),flush=True)
