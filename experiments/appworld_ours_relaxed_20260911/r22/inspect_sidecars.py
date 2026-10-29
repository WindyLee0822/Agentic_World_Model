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
prefix='azureml/'+job_name+'/output_dir/.cell-checkpoints/record-'
rows=[]
for blob in container.list_blobs(name_starts_with=prefix):
    if blob.size:
        rows.append(json.loads(container.download_blob(blob.name).readall()))
snapshot = root / 'sidecar-records.jsonl'
if snapshot.exists():
    old = [json.loads(line) for line in snapshot.read_text().splitlines() if line.strip()]
    old_keys = {(r.get('subset'),r.get('task_id'),r.get('n_rollouts')) for r in old}
    new_keys = {(r.get('subset'),r.get('task_id'),r.get('n_rollouts')) for r in rows}
    if not old_keys <= new_keys:
        raise RuntimeError('Sidecar listing regressed; preserving prior checkpoint snapshot')
temp = root / 'sidecar-records.tmp'
temp.write_text(''.join(json.dumps(r)+'\n' for r in rows))
temp.replace(snapshot)
spec = importlib.util.spec_from_file_location('r20_audit', root / 'audit_records.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)
reference_path = root.parent.parent / 'appworld_ours_repair_20260906/baseline/records.jsonl'
reference = [json.loads(line) for line in reference_path.read_text().splitlines() if line.strip()]
result = audit.evaluate(rows, reference, require_setup_predictions=True)
(root/'sidecar-audit.json').write_text(json.dumps(result,indent=2)+'\n')
status.update(sidecar_records=len(rows), successes=result['official_successes'],
              audit_flags=len(result['contract_violations']),full_coverage=result['full_coverage'],
              success_rate=result['success_rate'], record_source='individual cell checkpoints')
(root/'sidecar-status.json').write_text(json.dumps(status,indent=2)+'\n')
print(json.dumps(status),flush=True)
