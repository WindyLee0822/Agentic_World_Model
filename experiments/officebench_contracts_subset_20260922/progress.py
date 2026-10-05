import json,sys,collections
from pathlib import Path
s=Path(sys.argv[1]).read_text(errors="replace")
dec=json.JSONDecoder(); traces=[]; errors=[]
for marker,out in [("[officebench-commit-trace] ",traces),("[officebench-python-error] ",errors)]:
 pos=0
 while True:
  i=s.find(marker,pos)
  if i<0: break
  pos=i+len(marker)
  try:
   value,n=dec.raw_decode(s[pos:]);out.append(value);pos+=n
  except ValueError: pass
print(json.dumps({"selected_replay_observations":len(traces),"exact_observations":sum(t["predicted"]==t["actual"] for t in traces),"tasks_with_replay":sorted(set(t["task_id"] for t in traces)),"python_errors":len(errors),"error_categories":dict(collections.Counter(e.get("error") for e in errors))},indent=2))
