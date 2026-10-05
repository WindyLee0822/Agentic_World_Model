import json
from pathlib import Path
from tts.llm.chat_client import ChatClient
from tts.env.officebench_wm_predict import OfficeBenchWMPredictor
from tts.env.officebench_semantics import normalize_agent_observation
out=Path('experiments/officebench_contracts_20260922/live_check.json')
report={'checks':[], 'native_candidate_actions':0}
try:
 client=ChatClient(model='gpt-5.4',backend='azure',azure_endpoint='https://ai-m365researchefficientaieus2212654625773.cognitiveservices.azure.com/',azure_api_version='2024-12-01-preview',azure_auth_mode='gcr',max_retries=1,max_rate_limit_retries=1)
 class PrivateChat:
  prompts=[]
  responses=[]
  def chat(self,system,prompt,**kw):
   assert 'PRIVATE_EVENT_7329' not in prompt
   self.prompts.append(prompt)
   response=client.chat(system,prompt,**kw)
   self.responses.append(response)
   return response
 class Query:
  available_calls=[]
  def workdir(self): return '/testbed'
  def read_raw(self,path):
   if path.endswith('Alice.ics'):
    return 'BEGIN:VCALENDAR\r\nVERSION:2.0\r\nBEGIN:VEVENT\r\nSUMMARY:PRIVATE_EVENT_7329\r\nDTSTART:20140101T141000Z\r\nDTEND:20140101T143000Z\r\nEND:VEVENT\r\nEND:VCALENDAR\r\n'
   return '[file does not exist]'
  def list(self,path,**kw): return '[]'
 chat=PrivateChat(); p=OfficeBenchWMPredictor(chat); p.program_only=True; p._query=Query()
 action={'app':'calendar','action':'list_events','username':'Alice'}
 actual=p.predict_read(action,None,dirty=False)
 expected='Successfully list events for Alice:\nSummary: PRIVATE_EVENT_7329\nStart Time: 2014-01-01 14:10:00\nEnd Time: 2014-01-01 14:30:00\nDescription: None\nLocation: None\n'+'-'*50
 report['checks'].append({'name':'calendar_format','match':actual==expected,'predicted':actual,'expected':expected})
 path='/testbed/data/new.xlsx'
 p.predict_write({'app':'excel','action':'create_new_file','file_path':path})
 actual=p.predict_write({'app':'excel','action':'set_cell','file_path':path,'row_idx':1,'column_idx':1,'text':'Agenda'})
 report['checks'].append({'name':'create_then_edit','match':actual=='Successfully write text to '+path,'predicted':actual})
 actual=normalize_agent_observation({'app':'excel','action':'read_file'},p.predict_read({'app':'excel','action':'read_file','file_path':path},None,dirty=True))
 report['checks'].append({'name':'read_after_edit','match':actual=='The following is the table from the excel file:\n(1, 1): Agenda','predicted':actual})
 missing='/testbed/data/missing.xlsx'
 actual=p.predict_write({'app':'excel','action':'set_cell','file_path':missing,'row_idx':1,'column_idx':1,'text':'Agenda'})
 report['checks'].append({'name':'missing_file_not_confused_with_state_record','match':actual=='The file '+missing+' does not exist. Failed to write to the file.','predicted':actual})
 report.update(prompt_values_hidden=True,python_calls=p.n_python_calls,python_failures=p.n_python_failures,generated_programs=chat.responses,repair_feedback=[prompt.split('Your previous PYTHON program failed safely with:')[1].split('FAILED PROGRAM:')[0] for prompt in chat.prompts if 'Your previous PYTHON program failed safely with:' in prompt])
except Exception as exc:
 report['error']=type(exc).__name__+': '+str(exc)[:800]
out.write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
