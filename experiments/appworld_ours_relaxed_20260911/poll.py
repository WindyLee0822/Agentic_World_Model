"""Poll the existing R18 job and atomically audit any available records."""
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
job_name = 'dw-appworld-ours-relaxed-r19-n16-full-20260911'
job = client.jobs.get(job_name)
status = {'name': job.name, 'status': job.status, 'code': job.code,
          'checked_at': datetime.now(timezone.utc).isoformat()}
container = ContainerClient(
    account_url='https://m365researchex2937532597.blob.core.windows.net',
    container_name='azureml-blobstore-0e758b43-c315-4fe1-891b-a93eb147fd79',
    credential=credential)
try:
    content = container.download_blob('azureml/' + job_name + '/output_dir/records.jsonl').readall()
    if not content.strip() and not (root / 'records.jsonl').exists():
        status.update(records=0, records_available=True, snapshot_changed=False)
    else:
        lines = content.splitlines()
        rows = [json.loads(line) for line in lines if line.strip()]
        count = len(rows)
        changed = not (root / 'records.jsonl').exists() or (root / 'records.jsonl').read_bytes() != content
        if changed:
            (root / 'records.tmp').write_bytes(content)
            (root / 'records.tmp').replace(root / 'records.jsonl')
        status.update(records=count, snapshot_changed=changed)
except ResourceNotFoundError:
    status['records_available'] = False
if (root / 'records.jsonl').exists():
    rows = [json.loads(line) for line in (root / 'records.jsonl').read_text().splitlines() if line.strip()]
    successes = sum(r.get('success') is True for r in rows)
    flags = []
    for r in rows:
        reasons = []
        for field, expected in {'n_rollouts':16, 'method':'ours', 'env':'appworld', 'simulator_writes':0, 'predict_all_actions':True, 'appworld_predict_all_program_only':True, 'appworld_deterministic_grounded_clean_reads':False}.items():
            if r.get(field) != expected: reasons.append(field)
        if r.get('candidate_grounded_success') != [None] * 16: reasons.append('candidate_grounded_success')
        setup = r.get('candidate_setup_prediction', [])
        if len(setup) != 16 or any(x.get('enabled') is not True for x in setup): reasons.append('setup_metadata_needs_review')
        if reasons: flags.append({'task_id':r.get('task_id'), 'fields':reasons})
    keys = {(r.get('subset'),r.get('task_id')) for r in rows}
    status.update(records=len(rows), unique_tasks=len(keys), successes=successes,
                  observed_success_rate=successes/len(rows) if rows else None,
                  audit_flags=len(flags), full_coverage=len(rows)==len(keys)==585)
    (root / 'audit-flags.json').write_text(json.dumps(flags,indent=2)+'\n')
(root / 'live-status.json').write_text(json.dumps(status, indent=2) + '\n')
print(json.dumps(status), flush=True)
