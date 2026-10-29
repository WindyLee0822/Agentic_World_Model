import json,os,sys
from pathlib import Path
sys.path[:] = [p for p in sys.path if p and not p.startswith(('/mnt/batch/','/home/azureuser/cloudfiles/code/'))]
sys.path.insert(0,'/home/azureuser/.azure/cliextensions-local/ml')
from azure.ai.ml import MLClient, load_job
from azure.identity import ManagedIdentityCredential
client=MLClient(ManagedIdentityCredential(client_id=os.environ['DEFAULT_IDENTITY_CLIENT_ID']),'acc09744-1ee3-4242-b375-93421c63af0c','external-collaborations','M365Research-External-EUS')
root=Path('/home/azureuser/cloudfiles/code/Agentic_World_Model/experiments/appworld_ours_relaxed_20260911/r22')
results=[]
for name,reason in [
 ('dw-appworld-ours-relaxed-r20-n16-full-20260912','Confirmed internal-error observation exposure; superseded by request boundary'),
 ('dw-appworld-ours-relaxed-r21-n16-full-20260913','Confirmed boundary regression changes verifier-error recovery and discards candidates')]:
 before=client.jobs.get(name)
 if before.status not in {'Completed','Failed','Canceled','CancelRequested'}:
  client.jobs.begin_cancel(name)
 after=client.jobs.get(name)
 results.append({'name':name,'before':before.status,'after':after.status,'reason':reason})
 (root/'superseded-jobs.json').write_text(json.dumps(results,indent=2)+'\n')
 print(json.dumps(results[-1]),flush=True)
