"""Opt-in, process-scoped OfficeBench completion-contract repair.

No production files/defaults are changed. This profile never executes a simulated action.
"""
import copy,json,time,inspect,re,tokenize,io
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

ACTION_SCHEMA={'type':'object','properties':{'app':{'type':'string'},'action':{'type':'string'}},'required':['app','action'],'additionalProperties':True}

class JsonPolicy:
    def __init__(self,chat,schema=None):self.chat_client=chat;self.schema=schema or ACTION_SCHEMA
    def chat(self,system,user,**kwargs):
        extra=copy.deepcopy(kwargs.pop('extra_body',None) or {})
        extra['chat_template_kwargs']={**extra.get('chat_template_kwargs',{}),'enable_thinking':False}
        extra['structured_outputs']={'json':self.schema}
        kwargs['max_tokens']=max(2048,kwargs.get('max_tokens') or 0)
        result=self.chat_client.chat(system,user,extra_body=extra,**kwargs)
        # Preserve native Python-literal parsing while accepting valid JSON booleans/null.
        try:
            value=json.loads(result)
            if isinstance(value,dict) and isinstance(value.get('app'),str) and isinstance(value.get('action'),str):return repr(value)
        except (ValueError,TypeError):pass
        return result

def install(profile,trace_dir=None):
    assert profile in {'control','budget','json','combined','grounded','source','wmfix'}
    from tts_qwen.env import officebench_agent as agent,officebench_wm_predict as wm,officebench_wm_python as py
    stack=ExitStack()
    def log(kind,payload):
        if trace_dir:
            path=Path(trace_dir);path.mkdir(parents=True,exist_ok=True)
            with (path/'trace.jsonl').open('a') as f:f.write(json.dumps({'kind':kind,'time':time.time(),**payload},default=str)+'\n')
    original_loop=agent.run_officebench_agent
    def loop(session,config,chat_client,**kwargs):
        native=isinstance(kwargs.get('predictor'),wm.OfficeBenchWMPredictor)
        if native:
            assert kwargs.get('predict_all_actions') is True
            assert session.actions_disabled and session.read_only
            if profile in {'json','combined','grounded','source'}:
                chat_client=JsonPolicy(chat_client,action_schema(session.apps_catalogue()) if profile in {'grounded','source'} else None)
        result=original_loop(session,config,chat_client,**kwargs)
        log('trajectory',dict(native_ours=native,actions=result.actions,observations=result.observations,history=result.history))
        return result
    stack.enter_context(patch.object(agent,'run_officebench_agent',loop))
    if profile in {'budget','combined','grounded','source','wmfix'}:
        original_complete=wm.OfficeBenchWMPredictor._complete
        def complete(self,prompt,*,max_tokens):
            if not self.program_only:return original_complete(self,prompt,max_tokens=max_tokens)
            try:
                self.n_llm_calls+=1
                return (self._chat.chat(self._flat_system_prompt or wm._OFFICEBENCH_SIMULATOR_SYSTEM,prompt,
                    temperature=0.0,max_tokens=max(2048,max_tokens),
                    extra_body={'chat_template_kwargs':{'enable_thinking':False}}) or '').strip()
            except Exception as error:
                log('completion_error',dict(error=repr(error)));return ''
        stack.enter_context(patch.object(wm.OfficeBenchWMPredictor,'_complete',complete))
    for name in ['predict_read','predict_write']:
        original=getattr(wm.OfficeBenchWMPredictor,name)
        def traced(self,action,*args,_original=original,_name=name,**kwargs):
            before=(self.n_python_calls,self.n_python_failures)
            result=_original(self,action,*args,**kwargs)
            log('observation',dict(route=_name,action=action,prediction=result,python_calls=self.n_python_calls-before[0],python_failures=self.n_python_failures-before[1]))
            return result
        stack.enter_context(patch.object(wm.OfficeBenchWMPredictor,name,traced))
    original_python=py._run_python
    def python(code,context,stats,**kwargs):
        if profile=='wmfix':code=strip_comments(code)
        result=original_python(code,context,stats,**kwargs)
        if result.error:log('python_error',dict(error=result.error,code=code,context_kind=context.get('kind')))
        return result
    stack.enter_context(patch.object(py,'_run_python',python))
    if profile in {'grounded','source','wmfix'}:
        original_predict=wm.OfficeBenchWMPredictor._predict
        def grounded_predict(self,*args,**kwargs):
            if self.program_only and self._query is not None and kwargs.get('python_context') is not None:
                context=dict(kwargs['python_context'])
                if not hasattr(self,'_repair_inventory'):
                    self._repair_inventory=self._query.list('/testbed',recursive=True)
                context['initial_inventory']=self._repair_inventory
                action=context.get('action') or {}
                catalogue=self._query._session.apps_catalogue()
                context['official_action_contract']=catalogue.get('demo',{}).get(action.get('app'),{}).get(action.get('action'),'')
                if profile in {'source','wmfix'}:
                    app,act=action.get('app',''),action.get('action','')
                    if re.fullmatch(r'[a-z_]+',app) and re.fullmatch(r'[a-z_]+',act):
                        path=Path(self._query._session.root)/'apps'/f'{app}_app'/f'{app}_{act}.py'
                        if path.is_file():context['official_action_source']=path.read_text()[:12000]
                    target=wm.read_target(action,self._current_workdir())
                    paths=[]
                    if context.get('kind') in ('clean_read','dirty_read') and target and app!='shell':
                        target=target if target.startswith('/') else '/testbed/'+target
                        if app=='email':
                            try:
                                entries=json.loads(self._query.list(target))
                                paths=[e['path'] for e in entries if e.get('type')=='file'][:8]
                            except (ValueError,TypeError,KeyError):paths=[]
                        else:paths=[target]
                        context['reads']=[{'op':'read','path':path,'result':self._query.read_raw(path)} for path in paths]
                kwargs['python_context']=context
            return original_predict(self,*args,**kwargs)
        stack.enter_context(patch.object(wm.OfficeBenchWMPredictor,'_predict',grounded_predict))
        # Reuse the native computation/read loop, adding bounded evidence previews only.
        original_protocol=py._python_protocol
        def protocol(context,**kwargs):
            preview={k:context[k] for k in ['initial_inventory','official_action_contract','official_action_source','action','writes'] if k in context}
            text=original_protocol(context,**kwargs)
            return text+'\nCOPIED INPUT PREVIEW (bounded, possibly incomplete; compute the output from full data variables):\n'+py._bounded_json_text(preview,10000)+'\nDo not guess data formats. Read results can be plain text, ICS, EML, or native spreadsheet cell text; use their previews to choose parsing code. When official_action_source is provided, use it to understand exact semantics and wording, but DO NOT import its modules or execute it. Immutable READ of xlsx/docx/pdf/images already provides extracted text, NOT bytes or JSON; process those data variables directly. Never guess a different operand or username from the inventory: a missing requested path must return the native empty/error outcome. If data[\"reads\"] already contains the needed operand, compute PYTHON_FINAL directly; do not read it again. Keep programs concise, with no analysis comments. The initial inventory is pristine; apply prior simulated writes from data["writes"] for current state.\n'
        source=inspect.getsource(py.run_python_prediction_loop)
        needle='"variable": f"data[\'reads\'][{index}][\'result\']",'
        assert source.count(needle)==1
        source=source.replace(needle,needle+'\n                "preview": str(record.get("result", ""))[:2000],')
        namespace=dict(vars(py),_python_protocol=protocol)
        exec(compile(source,'<officebench_evidence_preview>','exec'),namespace)
        stack.enter_context(patch.object(wm,'run_python_prediction_loop',namespace['run_python_prediction_loop']))
    return stack


def action_schema(catalogue):
    variants=[]
    for app,actions in catalogue['actions'].items():
        variants.append({'type':'object','properties':{'app':{'const':app},'action':{'enum':list(dict.fromkeys(list(actions)+['switch_app']))}},'required':['app','action'],'additionalProperties':True})
    variants.append({'type':'object','properties':{'app':{'const':'system'},'action':{'enum':['switch_app','finish_task','got_stuck']}},'required':['app','action'],'additionalProperties':True})
    return {'anyOf':variants}


def strip_comments(code):
    """Drop non-executable commentary before the existing code-size/AST/safety checks."""
    try:
        return tokenize.untokenize((t.type,t.string) for t in tokenize.generate_tokens(io.StringIO(code).readline) if t.type!=tokenize.COMMENT)
    except (tokenize.TokenError,IndentationError):return code
