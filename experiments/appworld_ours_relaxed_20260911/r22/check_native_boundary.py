"""Native AppWorld execution integration; no model calls or benchmark evaluation."""
import importlib.util
import json
import os
from pathlib import Path
import uuid

root = Path(__file__).resolve().parent
os.environ.setdefault('APPWORLD_ROOT', str(root.parents[2]))
from appworld import AppWorld

spec = importlib.util.spec_from_file_location(
    'failure_boundary', root / 'staged_changes/src/tts/env/appworld_ours_failure_boundary.py')
boundary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(boundary)

results = []
for caught in (False, True):
    name = 'r22-boundary-' + uuid.uuid4().hex
    with AppWorld(task_id='024c982_1', experiment_name=name, ground_truth_mode='minimal') as world:
        calls = []
        def fail(*args, **kwargs):
            calls.append(True)
            raise RuntimeError('internal simulator sentinel')
        world.requester.request = fail
        boundary.install_failure_boundary(world)
        code = 'apis.supervisor.show_active_task()'
        if caught:
            code = 'try:\n    ' + code + '\nexcept Exception as exc:\n    print(str(exc))'
        try:
            world.execute(code)
        except boundary.SimulatorObservationFailure:
            assert calls, 'API did not reach guarded requester'
            results.append({'solver_catches_error': caught, 'host_rejected_output': True})
        else:
            raise AssertionError('Native execution returned internal simulator output')

with AppWorld(task_id='024c982_1', experiment_name='r22-boundary-' + uuid.uuid4().hex,
              ground_truth_mode='minimal') as world:
    def reject(*args, **kwargs):
        raise Exception('predicted API rejection sentinel')
    world.requester.request = reject
    boundary.install_failure_boundary(world)
    output = world.execute('apis.supervisor.show_active_task()')
    assert 'predicted API rejection sentinel' in output, output
    results.append({'predicted_api_rejection_preserved': True})

(root / 'native-boundary-checks.json').write_text(json.dumps(results, indent=2) + '\n')
print(json.dumps(results))
