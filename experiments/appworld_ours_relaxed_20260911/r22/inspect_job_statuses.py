import json,os,sys
from pathlib import Path
sys.path[:] = [p for p in sys.path if p and not p.startswith(('/mnt/batch/','/home/azureuser/cloudfiles/code/'))]
sys.path.insert(0,'/home/azureuser/.azure/cliextensions-local/ml')
from azure.ai.ml import MLClient, load_job
from azure.identity import ManagedIdentityCredential
client=MLClient(ManagedIdentityCredential(client_id=os.environ['DEFAULT_IDENTITY_CLIENT_ID']),'acc09744-1ee3-4242-b375-93421c63af0c','external-collaborations','M365Research-External-EUS')
from datetime import datetime,timezone
results=[]
for name in ['dw-appworld-ours-relaxed-r20-n16-full-20260912','dw-appworld-ours-relaxed-r21-n16-full-20260913','dw-appworld-ours-relaxed-r22-n16-full-20260913']:
 j=client.jobs.get(name)
 results.append({'name':j.name,'status':j.status,'code':j.code})
report={'checked_at':datetime.now(timezone.utc).isoformat(),'jobs':results}
Path('/home/azureuser/cloudfiles/code/Agentic_World_Model/experiments/appworld_ours_relaxed_20260911/r22/job-statuses.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report),flush=True)
