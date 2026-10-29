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
    spec = importlib.util.spec_from_file_location('r20_audit', root / 'audit_records.py')
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    reference_path = root.parent.parent / 'appworld_ours_repair_20260906/baseline/records.jsonl'
    reference = [json.loads(line) for line in reference_path.read_text().splitlines() if line.strip()]
    result = audit.evaluate(rows, reference, require_setup_predictions=True)
    (root / 'result-audit.json').write_text(json.dumps(result, indent=2) + '\n')
    status.update(records=result['observed_decisions'], successes=result['official_successes'],
                  full_coverage=result['full_coverage'], success_rate=result['success_rate'],
                  audit_flags=len(result['contract_violations']),
                  record_contract_passed=result['record_contract_passed'])
(root / 'live-status.json').write_text(json.dumps(status, indent=2) + '\n')
print(json.dumps(status), flush=True)
