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
job_name = 'dw-appworld-ours-relaxed-r23-n1-probe-20260913'
job = client.jobs.get(job_name)
status = {'name': job.name, 'status': job.status, 'code': job.code,
          'checked_at': datetime.now(timezone.utc).isoformat()}
container = ContainerClient(
    account_url='https://m365researchex2937532597.blob.core.windows.net',
    container_name='azureml-blobstore-0e758b43-c315-4fe1-891b-a93eb147fd79',
    credential=credential)
prefix='azureml/'+job_name+'/output_dir/'
blobs=list(container.list_blobs(name_starts_with=prefix))
(root/'output-inventory.json').write_text(json.dumps([{'name':b.name,'bytes':b.size} for b in blobs],indent=2)+'\n')
for b in blobs:
 rel=b.name[len(prefix):]
 if '/' not in rel:
  print(json.dumps({'name':rel,'bytes':b.size}),flush=True)
  if rel.endswith(('.json','.jsonl','.log')) and b.size < 10_000_000:
   (root/('final-'+rel)).write_bytes(container.download_blob(b.name).readall())
