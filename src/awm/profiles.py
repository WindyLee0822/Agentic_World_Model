"""Fixed runtime profiles for the two integrated backends.

These settings retain backend-specific scoring and repair behavior; the public
experiment axes are defined once in cli.py.
"""

_DEFAULTS = {'L': 2,
 'all_tasks': False,
 'allow_partial': False,
 'appworld_agent_reference': False,
 'appworld_deterministic_dirty_reads': False,
 'appworld_instruction_preflights': None,
 'appworld_precision_prompt': False,
 'appworld_wm_tool_mode': 'python',
 'branching': 2,
 'hard_cap': 6,
 'no_cache': False,
 'no_selective_invalidation': False,
 'no_verifier_reference': False,
 'node_read_limit': 1,
 'officebench_wm_tool_mode': 'python',
 'p_correct': 0.5,
 'predict_all_actions': True,
 'recursion_mode': 'uncertainty',
 'sandbox_prompt': False,
 'simulation_only': False,
 'subsets': None,
 'success_score': None,
 'task_ids': None,
 'tau': 0.9,
 'tau2_qwen_agent_profile': 'off',
 'verifier_percheck': False,
 'verifier_reference': False,
 'verifier_samples': None}

_QWEN = {'appworld_qwen_repair': 'v3',
 'officebench_wm_repair': 'v2',
 'wm_api_key': 'EMPTY',
 'wm_max_tokens': 32768,
 'wm_temperature': 0.6,
 'wm_top_k': 20,
 'wm_top_p': 0.95}

def backend_defaults(*, qwen=False):
    return {**_DEFAULTS, **(_QWEN if qwen else {})}
