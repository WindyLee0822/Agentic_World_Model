import json
from pathlib import Path
from tts.llm.chat_client import ChatClient
from tts.env.officebench_wm_predict import OfficeBenchWMPredictor
client=ChatClient(model='gpt-5.4',backend='azure',azure_endpoint='https://ai-m365researchefficientaieus2212654625773.cognitiveservices.azure.com/',azure_api_version='2024-12-01-preview',azure_auth_mode='gcr',max_retries=1,max_rate_limit_retries=1)
p=OfficeBenchWMPredictor(client);p.program_only=True
result=p.predict_write({'app':'email','action':'send_email','sender':'Alice','recipient':'Bob, Tom','subject':'notice','content':'body'})
report={'observation':result,'match':result=="Failed to send email to ('Bob', 'Tom').",'applied_resources':list(p.applied_writes),'python_calls':p.n_python_calls,'python_failures':p.n_python_failures}
Path(__file__).with_suffix('.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
assert report['match'] and not report['applied_resources']
