from __future__ import annotations

from tts.env import appworld_react_agent


class _Chat:
    def __init__(self):
        self.calls = 0

    def complete(self, messages, **kwargs):
        self.calls += 1
        return "```python\napis.supervisor.complete_task()\n```"


class _World:
    def __init__(self):
        self.executions = 0
        self.simulated_completion = False

    def execute(self, code):
        self.executions += 1
        self.simulated_completion = True
        return "done"

    def task_completed(self):
        return False


def test_react_loop_stops_on_predictor_local_completion(monkeypatch):
    monkeypatch.setattr(
        appworld_react_agent,
        "build_initial_messages",
        lambda *args, **kwargs: [],
    )
    world = _World()
    chat = _Chat()

    code, outputs = appworld_react_agent.run_react_code_agent(
        world,
        chat,
        completion_check=lambda: world.simulated_completion,
    )

    assert len(code) == len(outputs) == 1
    assert world.executions == chat.calls == 1


def test_official_prompt_does_not_offer_failure_escape_hatch():
    prompt = appworld_react_agent.load_prompt_template()

    assert 'you can call it with status="fail"' in prompt
