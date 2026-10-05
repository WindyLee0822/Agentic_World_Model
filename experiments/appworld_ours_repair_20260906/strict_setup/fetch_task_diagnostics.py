import sys,os,json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0,'/home/azureuser/.azure/cliextensions-local/ml')
from azure.storage.blob import ContainerClient
from azure.identity import ManagedIdentityCredential
task=sys.argv[1]
c=ContainerClient(account_url='https://m365researchex2937532597.blob.core.windows.net',container_name='azureml-blobstore-0e758b43-c315-4fe1-891b-a93eb147fd79',credential=ManagedIdentityCredential(client_id=os.environ['DEFAULT_IDENTITY_CLIENT_ID']))
p='azureml/dw-appworld-ours-strict-r17-full-20260907/output_dir/wm_diagnostics/'
root=Path('/tmp/appworld-r17-failure-'+task);root.mkdir(exist_ok=True)
def read(b):
 if not b.size:return
 prefix=c.download_blob(b.name,offset=0,length=min(b.size,500)).readall()
 if task.encode() in prefix:
  content=c.download_blob(b.name).readall();(root/b.name.split('/')[-1]).write_bytes(content)
  print(json.dumps({'file':b.name.split('/')[-1],'bytes':len(content)}),flush=True)
with ThreadPoolExecutor(max_workers=8) as pool:list(pool.map(read,c.list_blobs(name_starts_with=p)))
