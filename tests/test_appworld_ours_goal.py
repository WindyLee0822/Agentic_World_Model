import copy
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "appworld_goal", Path(__file__).resolve().parents[1] / "aml/src/check_appworld_ours_goal.py")
goal = importlib.util.module_from_spec(spec)
spec.loader.exec_module(goal)


def records(successes):
    result = []
    for subset, size in goal.EXPECTED_COUNTS.items():
        for index in range(size):
            result.append({
                "env": "appworld", "method": "ours", "n_rollouts": 2,
                "subset": subset, "task_id": f"{subset}-{index}",
                "success": len(result) < successes, "n_commits": 1,
                "solver_agent": "llm_wm", "predict_all_actions": True,
                "simulation_only": False, "agent_self_aggregate": False,
                "simulator_writes": 0, "appworld_agent_reference": False,
                "appworld_instruction_preflights": False,
                "appworld_preground_clean_reads": False,
                "appworld_deterministic_dirty_reads": False,
                "appworld_deterministic_grounded_clean_reads": False,
                "appworld_predict_all_program_only": True,
                "success_score": "simulator_success", "candidate_grounded_success": [None, None],
            })
    return result


def test_goal_requires_strictly_more_than_sixty_percent():
    reference = records(0)
    assert not goal.evaluate(records(351), reference)["passed"]
    assert goal.evaluate(records(352), reference)["passed"]
    assert goal.evaluate(records(352), records(585))["official_successes"] == 352


def test_successful_subset_or_duplicate_cannot_satisfy_full_goal():
    reference = records(0)
    assert not goal.evaluate(records(585)[:6], reference)["passed"]
    rows = records(585)
    rows[-1] = copy.deepcopy(rows[0])
    result = goal.evaluate(rows, reference)
    assert not result["passed"]
    assert result["duplicates"] and result["missing"]


@pytest.mark.parametrize("field,value", [
    ("predict_all_actions", False), ("simulator_writes", 1),
    ("candidate_grounded_success", [True, False]), ("agent_self_aggregate", True),
    ("appworld_deterministic_grounded_clean_reads", True), ("n_commits", 2),
    ("appworld_preground_clean_reads", True),
    ("appworld_deterministic_dirty_reads", True),
])
def test_success_rate_cannot_override_simulation_contract(field, value):
    rows = records(585)
    rows[0][field] = value
    result = goal.evaluate(rows, records(0))
    assert not result["passed"]
    assert field in result["contract_violations"][0]["fields"]
