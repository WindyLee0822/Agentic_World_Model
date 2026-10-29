import json, os
from pathlib import Path
from routing import wrap_setup_predictions
from tts.env.appworld_ours_wm import AppWorldOursWMPredictor
from tts.env.appworld_wm import install_wm_requester
from tts.env.real_appworld import _world_session
from tts.llm import make_chat_client

def main():
    os.environ['AZURE_CLIENT_ID'] = os.environ['DEFAULT_IDENTITY_CLIENT_ID']
    os.environ['APPWORLD_ROOT'] = '/tmp/appworld-strict-data'
    os.environ['TTS_APPWORLD_DETERMINISTIC_GROUNDED_CLEAN_READS'] = '0'
    chat = make_chat_client(backend='azure', model='DeepSeek-V4-Pro', azure_endpoint='https://ai-m365researchefficientaieus2212654625773.cognitiveservices.azure.com/', azure_api_version='2024-12-01-preview', azure_auth_mode='default')
    pred = AppWorldOursWMPredictor(chat, time_budget_s=600, hard_timeout_s=900)
    pred.warm_up()
    pred.prepare_python_tool()
    results = []
    try:
        with _world_session('024c982_1', 'strict_setup_probe', 'native') as world:
            world.timeout_seconds = 600
            original = install_wm_requester(world, pred, predict_all_actions=True)
            wrap_setup_predictions(world, pred)
            pred.start_rollout()
            for app, api, args in [
                ('supervisor', 'show_profile', {}),
                ('supervisor', 'show_account_passwords', {}),
                ('api_docs', 'show_api_doc', {'app_name': 'gmail', 'api_name': 'login'}),
            ]:
                expected = original(app, api, **args)
                before = pred.n_python_calls
                actual = world.requester.request(app, api, **args)
                assert actual == expected, f'{app}.{api} mismatch'
                assert pred.n_python_calls > before
                results.append({'api': f'{app}.{api}', 'matches': True, 'python_calls': pred.n_python_calls-before})
                print(json.dumps(results[-1]), flush=True)
            before = pred.n_python_calls
            output = world.execute("profile = apis.supervisor.show_profile()\npasswords = apis.supervisor.show_account_passwords()\npw = next(p['password'] for p in passwords if p['account_name'] == 'gmail')\nlogin = apis.gmail.login(username=profile['email'], password=pw)\nassert login.get('access_token')\nprint('PREDICTED_LOGIN_OK')")
            assert 'PREDICTED_LOGIN_OK' in output, output
            results.append({'case': 'native_repl_login', 'passed': True, 'python_calls': pred.n_python_calls-before})
            native_before = original('supervisor', 'show_active_task')
            before = pred.n_python_calls
            world.requester.request('supervisor', 'complete_task', status='success')
            generated = world.requester.request('supervisor', 'show_active_task')
            native_after = original('supervisor', 'show_active_task')
            assert native_before == native_after
            assert generated['status'] == 'success', generated
            assert pred.n_python_calls > before
            results.append({'case': 'logical_completion_only', 'passed': True, 'python_calls': pred.n_python_calls-before})
    finally:
        close = getattr(pred._chat, 'close', None)
        if close: close()
        Path('native-model-results.json').write_text(json.dumps(results, indent=2)+'\n')
    print(json.dumps(results), flush=True)

if __name__ == '__main__': main()
