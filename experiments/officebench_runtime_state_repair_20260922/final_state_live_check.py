"""GPT-5.4 probe with copied private inputs and no candidate action interface."""
import json
from pathlib import Path
from tts.llm.chat_client import ChatClient
from tts.env.officebench_wm_predict import OfficeBenchWMPredictor
from tts.env.officebench_semantics import normalize_agent_observation
out=Path(__file__).with_suffix('.json');report={'checks':[],'native_candidate_actions':0}
try:
 client=ChatClient(model='gpt-5.4',backend='azure',azure_endpoint='https://ai-m365researchefficientaieus2212654625773.cognitiveservices.azure.com/',azure_api_version='2024-12-01-preview',azure_auth_mode='gcr',max_retries=1,max_rate_limit_retries=1)
 class Chat:
  def chat(self,system,prompt,**kw):
   assert 'PRIVATE_OCR_8294' not in prompt and 'PRIVATE_WORD_4731' not in prompt
   response=client.chat(system,prompt,**kw)
   report.setdefault('programs',[]).append(response)
   return response
 class Query:
  available_calls=[]
  def workdir(self):return '/testbed'
  def list(self,path,**kw):return '[]'
  def read_raw(self,path):
   return 'PRIVATE_OCR_8294\nLine two\n' if path.endswith('.png') else '[file does not exist]'
 p=OfficeBenchWMPredictor(Chat());p.program_only=True;p._query=Query()
 source='/testbed/data/new.docx'
 p.predict_write({'app':'word','action':'create_new_file','file_path':source})
 # A copied logical transition, with its contents unavailable in the prediction prompt.
 p.applied_writes['data/new.docx'].append(({'app':'word','action':'write_to_file','file_path':source,'contents':'PRIVATE_WORD_4731'},'Successfully write contents to '+source))
 action={'app':'word','action':'convert_to_pdf','word_file_path':source,'pdf_file_path':'/testbed/data/new.pdf'}
 actual=normalize_agent_observation(action,p.predict_write(action))
 report['checks'].append({'name':'convert_new_logical_input','match':actual=='Successfully convert '+source+' to /testbed/data/new.pdf','observation':actual})
 p.applied_writes['data/new.docx'].append(({'app':'word','action':'write_to_file','file_path':source,'contents':'LATER_SOURCE_EDIT'},'Successfully write contents to '+source))
 actual=p.predict_file_state('data/new.pdf','[file does not exist]')
 report['checks'].append({'name':'conversion_final_state_snapshot','match':actual.strip()=='PRIVATE_WORD_4731','observation':actual})
 action={'app':'email','action':'send_email','sender':'Alice','recipient':'Tom@example.org','subject':'notice','content':'A notification body longer than twenty characters'}
 actual=normalize_agent_observation(action,p.predict_write(action))
 report['checks'].append({'name':'send_original_recipient','match':actual=='Successfully sent email to Tom@example.org.','observation':actual})
 action={'app':'email','action':'list_emails','username':'Tom'}
 actual=normalize_agent_observation(action,p.predict_read(action,None,dirty=True))
 report['checks'].append({'name':'read_after_send','match':'Email ID: notice.eml' in actual and 'Content: A notification body ...' in actual,'observation':actual})
 action={'app':'ocr','action':'recognize_file','file_path':'/testbed/data/private.png'}
 actual=normalize_agent_observation(action,p.predict_read(action,None,dirty=False))
 report['checks'].append({'name':'ocr_preserves_private_string','match':actual=='The text from /testbed/data/private.png is:\nPRIVATE_OCR_8294\nLine two','observation':actual})
 report.update(private_values_hidden=True,python_calls=p.n_python_calls,python_failures=p.n_python_failures)
except Exception as exc:report['error']=type(exc).__name__+': '+str(exc)[:1000]
out.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
assert len(report['checks'])==5 and all(c['match'] for c in report['checks'])
