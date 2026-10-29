"""Live model probe with synthetic setup fixtures; not a benchmark score."""

import json
import os
import re
import time
from pathlib import Path
from types import SimpleNamespace

from routing import wrap_setup_predictions
from tts.env.appworld_ours_wm import AppWorldOursWMPredictor
from tts.llm import make_chat_client


def main():
    os.environ['AZURE_CLIENT_ID'] = os.environ['DEFAULT_IDENTITY_CLIENT_ID']
    os.environ['TTS_APPWORLD_DETERMINISTIC_GROUNDED_CLEAN_READS'] = '0'
    chat = make_chat_client(
        backend='azure', model='DeepSeek-V4-Pro',
        azure_endpoint='https://ai-m365researchefficientaieus2212654625773.cognitiveservices.azure.com/',
        azure_api_version='2024-12-01-preview', azure_auth_mode='default',
    )
    predictor = AppWorldOursWMPredictor(chat, time_budget_s=600, hard_timeout_s=900)
    predictor.program_only = True
    predictor.warm_up()
    predictor.prepare_python_tool()
    predictor.start_rollout()
    results = []
    cases = [
        ('gmail', 'login', {'access_token': 'synthetic.opaque-token.123', 'token_type': 'bearer'}),
        ('supervisor', 'show_account_passwords', [{'app': 'gmail', 'password': 'synthetic-test-password'}]),
        ('supervisor', 'show_active_task', {'status': 'complete', 'answer': None}),
    ]
    try:
        for app, api, expected in cases:
            world = SimpleNamespace(requester=SimpleNamespace(request=lambda *a, **k: expected))
            wrap_setup_predictions(world, predictor)
            before = predictor.n_python_calls
            started = time.monotonic()
            try:
                actual = world.requester.request(app, api)
                result = {'api': f'{app}.{api}', 'matches': actual == expected,
                          'python_calls': predictor.n_python_calls - before}
            except Exception as exc:
                result = {'api': f'{app}.{api}', 'matches': False, 'error_type': type(exc).__name__,
                          'error': re.sub(r'eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+',
                                          '<redacted>', str(exc))[:1000]}
            result['seconds'] = time.monotonic() - started
            results.append(result)
            print(json.dumps(result), flush=True)
            Path('live-model-results.json').write_text(json.dumps(results, indent=2) + '\n')
    finally:
        close = getattr(predictor._chat, 'close', None)
        if close:
            close()
    if not all(r['matches'] and r.get('python_calls', 0) for r in results):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
