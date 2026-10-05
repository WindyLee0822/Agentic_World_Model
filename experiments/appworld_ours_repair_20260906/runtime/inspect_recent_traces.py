import os,sys,json,heapq,datetime,concurrent.futures
from pathlib import Path
sys.path.insert(0,'/home/azureuser/.azure/cliextensions-local/ml')
from azure.identity import ManagedIdentityCredential
from azure.storage.blob import ContainerClient
c=ContainerClient(account_url='https://m365researchex2937532597.blob.core.windows.net',container_name='azureml-blobstore-0e758b43-c315-4fe1-891b-a93eb147fd79',credential=ManagedIdentityCredential(client_id=os.environ['DEFAULT_IDENTITY_CLIENT_ID']))
prefix='azureml/dw-appworld-ours-combined-r16-full-20260907/output_dir/wm_diagnostics/'
recent=[]
for b in c.list_blobs(name_starts_with=prefix):
 if b.size<1000:continue
 heapq.heappush(recent,(b.last_modified,b.name,b.size))
 if len(recent)>55:heapq.heappop(recent)
root=Path(__file__).resolve().parent.parent/'recent-traces';root.mkdir(exist_ok=True)
def inspect(item):
 modified,name,size=item
 data=c.download_blob(name).readall();rows=[json.loads(l) for l in data.splitlines() if l]
 ids=sorted(set(str(r.get('task_id')) for r in rows if r.get('task_id')))
 (root/Path(name).name).write_bytes(data)
 words=['401','access_token','download_attachment','file_exists','permission','Python execution failed','WMHardTimeout']
 joined='\n'.join(str(r.get('response_prefix','')) for r in rows)
 return {'file':Path(name).name,'modified':str(modified),'task_ids':ids,'model_calls':len(rows),'model_errors':sum('error_type' in r for r in rows),'mentions':{w:joined.count(w) for w in words}}
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:out=list(pool.map(inspect,sorted(recent,reverse=True)))
(root/'index.json').write_text(json.dumps(out,indent=2)+'\n')
for row in out:
 if any(t.startswith(('998908e','d8e490b','6d59d90')) for t in row['task_ids']):print(json.dumps(row),flush=True)
print('Saved',len(out),'recent traces and index.',flush=True)
