import json
from pathlib import Path
from types import SimpleNamespace
from appworld.requester import Requester
from tts.env.appworld_wm import AppWorldWMQuery
from tts.env.appworld_ours_model_driven import AppWorldOursModelDrivenPredictor

def main():
    native=object.__new__(Requester)
    native.raise_on_extra_parameters=True
    def request(app,api,**kwargs):
        kwargs.pop('raise_on_failure',None)
        native.raise_if_extra_parameters(app,api,kwargs)
        raise AssertionError('Expected argument validation to reject before API execution')
    fake=SimpleNamespace(allows_grounding=lambda *a:True,hydrate_read_args=lambda a,b,args:dict(args))
    value=AppWorldWMQuery(request,fake).read_raw('file_system.show_file',{'path':'~/documents/personal/songs_to_archive.txt'})
    assert isinstance(value,str) and 'Unexpected parameter' in value
    p=AppWorldOursModelDrivenPredictor(None);p.start_rollout();p.program_only=True
    context={'reads':[{'result':value}]}
    outputs={}
    for name,program in [('raw_print',"print(data['reads'][0]['result'])"),('parse_then_print',"import json\nprint(json.dumps(json.loads(data['reads'][0]['result'])))"),('json_print',"print_json(data['reads'][0]['result'])")]:
        r=p._run_python_prediction(program,context)
        normalized=p._normalize_python_output(r.output,'json') if not r.error else None
        outputs[name]={'error':r.error,'valid_json_prediction':normalized is not None,'decoded_matches_evidence':json.loads(normalized)==value if normalized is not None else False}
    assert not outputs['raw_print']['valid_json_prediction']
    assert outputs['parse_then_print']['error']
    assert outputs['json_print']['decoded_matches_evidence']
    result={'reproduction':'Native argument validation plus actual simulator Python worker; no API endpoint executed','evidence':value,'outputs':outputs,'limitation':'Reproduces the mechanism with observed invalid arguments, not the original unlogged runtime cause.'}
    out=Path('/home/azureuser/cloudfiles/code/Agentic_World_Model/experiments/appworld_ours_relaxed_20260911/r22/failures/634f342_2/read-error-reproduction.json')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
