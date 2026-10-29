import os,sys,json
sys.path.insert(0,'/home/azureuser/.azure/cliextensions-local/ml')
from azure.ai.ml import MLClient
from azure.identity import ManagedIdentityCredential
client=MLClient(ManagedIdentityCredential(client_id=os.environ['DEFAULT_IDENTITY_CLIENT_ID']),'acc09744-1ee3-4242-b375-93421c63af0c','external-collaborations','M365Research-External-EUS')
if __name__=='__main__':
 for name in ['dw-appworld-ours-verified-r14-full-20260907','dw-appworld-ours-combined-r16-full-20260907']:
  j=client.jobs.get(name)
  print(json.dumps({'name':name,'status':j.status,'code':j.code}),flush=True)
