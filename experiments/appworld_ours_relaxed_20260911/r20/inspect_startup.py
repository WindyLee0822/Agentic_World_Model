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
expected = {
    'TTS_APPWORLD_PREDICT_ALL_PROGRAM_ONLY':'1',
    'TTS_APPWORLD_OURS_MODEL_DRIVEN_WRITES':'1',
    'TTS_APPWORLD_COMPLETION_GUARD':'0',
    'TTS_APPWORLD_AGENT_REFERENCE':'0',
    'TTS_APPWORLD_DETERMINISTIC_GROUNDED_CLEAN_READS':'0',
}
actual = job.environment_variables or {}
status['environment_matches'] = {key: actual.get(key) == value for key,value in expected.items()}
status['command_has_n16'] = '--n "16"' in job.command
status['command_has_worker_recycling'] = '--max-tasks-per-child 1' in job.command
status['command_has_extended_watchdog'] = '--stall-timeout 129600' in job.command
status['properties'] = {k:v for k,v in (job.properties or {}).items() if k in ('StartTimeUtc','EndTimeUtc','ProcessInfoFile','ProcessStatusFile')}
try:
    blob=container.get_blob_client('azureml/' + job_name + '/output_dir/run.log')
    size=blob.get_blob_properties().size
    status['log_bytes']=size
    status['log_modified']=blob.get_blob_properties().last_modified.isoformat()
    if size:
        tail=blob.download_blob(offset=max(0,size-32000)).readall()
        (root/'startup-tail.log').write_bytes(tail)
except ResourceNotFoundError:
    status['log_available']=False
(root/'startup-audit.json').write_text(json.dumps(status,indent=2)+'\n')
print(json.dumps(status),flush=True)
