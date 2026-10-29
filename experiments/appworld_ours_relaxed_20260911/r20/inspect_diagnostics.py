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
prefix='azureml/'+job_name+'/output_dir/wm_diagnostics/'
# Follow all metadata pages so newer workers are not hidden by name ordering.
# Download only the four newest nonempty files, bounded to 32 KB each.
blobs=list(container.list_blobs(name_starts_with=prefix))
summary=[]
for b in sorted((b for b in blobs if b.size), key=lambda b: b.last_modified, reverse=True)[:4]:
    content=container.download_blob(b.name,offset=0,length=min(b.size,32000)).readall()
    events=[]
    for line in content.splitlines():
        try: events.append(json.loads(line))
        except ValueError: pass
    summary.append({'file':b.name.rsplit('/',1)[-1], 'bytes':b.size,
                    'modified':b.last_modified.isoformat(), 'events':events})
(root/'early-diagnostics.json').write_text(json.dumps({'status':status, 'listed_files':len(blobs),'samples':summary},indent=2)+'\n')
print(json.dumps({'status':status['status'],'listed_files':len(blobs),'samples':[{'file':s['file'],'bytes':s['bytes'],'events':len(s['events']),'modified':s['modified']} for s in summary]}),flush=True)
