import os,sys,pathlib,json
from atomic_records import publish
sys.path.insert(0,'/home/azureuser/.azure/cliextensions-local/ml')
from azure.identity import ManagedIdentityCredential
from azure.storage.blob import ContainerClient
client=ContainerClient(account_url='https://m365researchex2937532597.blob.core.windows.net',container_name='azureml-blobstore-0e758b43-c315-4fe1-891b-a93eb147fd79',credential=ManagedIdentityCredential(client_id=os.environ['DEFAULT_IDENTITY_CLIENT_ID']))
base=pathlib.Path(__file__).resolve().parent.parent
failed=False
for version,job in [('r14','dw-appworld-ours-verified-r14-full-20260907'),('r16','dw-appworld-ours-combined-r16-full-20260907'),('baseline','dw-appworld-ours-llm-wm-deepseek-v4-pro-n12-20260904')]:
 root=base/version;root.mkdir(exist_ok=True)
 if version=='baseline' and (root/'records.jsonl').exists():continue
 for name in ['records.jsonl']:
  try:
   content=client.download_blob('azureml/'+job+'/output_dir/'+name).readall()
   gate=base/f'partial-{version}-gate.json'
   minimum=json.loads(gate.read_text())['observed_decisions'] if gate.exists() else 0
   count,changed=publish(root/name,content,minimum)
   print(version,name,len(content),'rows',count,'changed',changed,flush=True)
  except Exception as exc:
   failed=True
   print(version,type(exc).__name__,str(exc)[:300],flush=True)
if failed:sys.exit(1)
