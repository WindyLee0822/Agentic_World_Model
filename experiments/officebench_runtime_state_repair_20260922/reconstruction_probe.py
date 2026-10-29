import json
from pathlib import Path
from tts.llm.chat_client import ChatClient
from tts.env.officebench_wm_predict import OfficeBenchWMPredictor
client=ChatClient(model='gpt-5.4',backend='azure',azure_endpoint='https://ai-m365researchefficientaieus2212654625773.cognitiveservices.azure.com/',azure_api_version='2024-12-01-preview',azure_auth_mode='gcr',max_retries=1,max_rate_limit_retries=1)
programs=[]
class Chat:
 def chat(self,system,prompt,**kw):
  assert 'PRIVATE_WORD_4731' not in prompt
  result=client.chat(system,prompt,**kw);programs.append(result);return result
p=OfficeBenchWMPredictor(Chat());p.program_only=True
act={'app':'word','action':'convert_to_pdf','word_file_path':'/testbed/data/new.docx','pdf_file_path':'/testbed/data/new.pdf'}
p.applied_writes['data/new.pdf']=[(act,'Successfully convert /testbed/data/new.docx to /testbed/data/new.pdf')]
p.conversion_sources['data/new.pdf']={'action':act,'path':'data/new.docx','state':{'status':'exact','exists':True,'content':'PRIVATE_WORD_4731'}}
result=p.predict_file_state('data/new.pdf','[file does not exist]')
report={'result':result,'programs':programs,'python_failures':p.n_python_failures,'match':result.strip()=='PRIVATE_WORD_4731'}
Path(__file__).with_suffix('.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
