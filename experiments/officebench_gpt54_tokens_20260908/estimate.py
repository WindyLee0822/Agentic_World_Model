#!/usr/bin/env python3
"""Reconstruct GPT-5.4 token scenarios from AML N=1 trajectories (token_assumption.md).

PYTHONPATH=src:/tmp/officebench-token-deps python experiments/officebench_gpt54_tokens_20260908/estimate.py --help
No model calls or saved candidate actions are executed by this script.
"""
from __future__ import annotations
import argparse
import ast
import csv
import hashlib
import inspect
import json
import math
import posixpath
import re
import statistics
from collections import Counter
from functools import lru_cache
from pathlib import Path

import tiktoken
from tts.env.officebench_agent import construct_system_message, build_prompt, _format_history
from tts.env.officebench_wm import OfficeBenchVerifier, state_files_from_config, reference_files_from_config, _SYSTEM, _STATUS_SYSTEM, _atomic_requirements
from tts.env.officebench_wm_predict import _OFFICEBENCH_SIMULATOR_SYSTEM, build_officebench_flat_system_prompt, action_execution_kind, read_target, write_targets
from tts.env.officebench_wm_python import _python_protocol

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SCENARIOS = {'low': (0.5, 1.0, 0), 'central': (1.0, 1.5, 100), 'high': (2.0, 2.5, 400)}
ENC = tiktoken.get_encoding('cl100k_base')
EXPECTED = {'1app': 93, '2app': 95, '3app': 112}

@lru_cache(maxsize=32768)
def tokens(text):
    return len(ENC.encode(str(text), disallowed_special=()))

def dump(x):
    return json.dumps(x, ensure_ascii=True, sort_keys=True, default=str)

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def literal(path, name):
    for n in ast.parse(path.read_text()).body:
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in n.targets):
            return ast.literal_eval(n.value)
    raise ValueError((path, name))

def catalogue(root):
    apps = root / 'apps'
    tree = ast.parse((apps / '__init__.py').read_text())
    assignments = {n.targets[0].id: n.value for n in tree.body if isinstance(n, ast.Assign)}
    available = [ast.literal_eval(k) for k in assignments['AVAILABLE_APPS'].keys]
    demos, actions = {}, {}
    for k, v in zip(assignments['AVAILABLE_ACTIONS'].keys, assignments['AVAILABLE_ACTIONS'].values):
        app = ast.literal_eval(k)
        demos[app] = {ast.literal_eval(a): literal(apps / module.value.id / (module.attr + '.py'), 'DEMO')
                      for a, module in zip(v.keys, v.values)}
        actions[app] = list(demos[app])
    return dict(available_apps=available, intro={a: literal(apps / (a + '_app') / '__init__.py', 'INTRO') for a in available}, demo=demos, actions=actions)

@lru_cache(maxsize=None)
def read_file(path):
    p = Path(path)
    if not p.is_file():
        return '[file does not exist]'
    suffix = p.suffix.lower()
    try:
        if suffix == '.xlsx':
            import openpyxl
            book = openpyxl.load_workbook(p)
            text = ''.join(''.join(f'({c.row}, {c.column}): {c.value if c.value is not None else "[Empty Cell]"}\t' for c in row) + '\n' for row in book.active.iter_rows())
            book.close()
            return text
        if suffix in ('.doc', '.docx'):
            from docx import Document
            return '\n'.join(p.text for p in Document(p).paragraphs)
        if suffix == '.pdf':
            import fitz
            with fitz.open(p) as doc:
                return '\n'.join(page.get_text() for page in doc)
        if suffix in ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'):
            import pytesseract
            from PIL import Image
            return pytesseract.image_to_string(Image.open(p), timeout=30)
        return p.read_text(errors='replace')
    except Exception as e:
        raise RuntimeError(f'Cannot reconstruct pristine file {p}: {type(e).__name__}: {e}') from e

def norm(path):
    path = str(path)
    if path.startswith('/testbed/'):
        return posixpath.normpath(path)
    return posixpath.normpath('/testbed/' + path.lstrip('./'))

def snapshot(root, task_id):
    task, subtask = task_id.split('/')
    folder = root / 'tasks' / task
    config = json.loads((folder / 'subtasks' / (subtask + '.json')).read_text())
    files = {'/testbed/' + p.relative_to(folder / 'testbed').as_posix(): read_file(str(p))
             for p in sorted((folder / 'testbed').rglob('*')) if p.is_file()}
    refs = {p.name: read_file(str(p)) for p in sorted((folder / 'reference').rglob('*')) if p.is_file()}
    state = {s['path']: files.get(norm(s['path']), '[file does not exist]')[:2000] for s in state_files_from_config(config)}
    for path, content in files.items():
        if '/calendar/' in path or '/emails/' in path:
            state[path.removeprefix('/testbed/')] = content[:1500]
    gold = {s['label']: refs.get(Path(s['path']).name, '[file does not exist]')[:2000] for s in reference_files_from_config(config)}
    listing = '\n'.join(p for p in files if p.startswith('/testbed/data/') and p.count('/') <= 6)
    return dict(config=config, files=files, initial_state=state, gold_files=gold, listing=listing)

def parse_actions(text):
    """Split only app.action boundaries, never pipes within saved action arguments."""
    parts = re.split(r' \| (?=[A-Za-z_]\w*\.[A-Za-z_]\w*\()', text)
    out = []
    for part in parts:
        start = part.find('(')
        action = None
        malformed = False
        try:
            obj, _ = json.JSONDecoder().raw_decode(part[start + 1:])
            action = obj['action']
            if not isinstance(action, dict):
                raise ValueError('not an action dict')
        except (ValueError, KeyError, TypeError):
            malformed = True
            # The logger truncates long strings, sometimes truncating the whole JSON.
            # Recover complete string fields; retain the raw step for actor-token counting.
            fields = re.findall(r'"([^"\\]+)"\s*:\s*("(?:[^"\\]|\\.)*")', part)
            action = {k: json.loads(v) for k, v in fields}
            # Retain the visible prefix of the final unterminated string argument.
            tail = re.search(r'"([^"\\]+)"\s*:\s*"((?:[^"\\]|\\.)*)$', part)
            if tail:
                try:
                    action[tail[1]] = json.loads('"' + tail[2] + '"')
                except ValueError:
                    action[tail[1]] = tail[2]
            for key, value in re.findall(r'"([^"\\]+)"\s*:\s*(-?\d+(?:\.\d+)?|true|false|null)(?=\s*[,}])', part):
                if key != 'step': action[key] = json.loads(value)
            prefix = part[:start].split('.')
            if len(prefix) == 2:
                action.setdefault('app', prefix[0]); action.setdefault('action', prefix[1])
        out.append((action, part, malformed))
    return out

def is_read_action(action):
    return action_execution_kind(action) == 'read'

def observation(action, snap):
    app, name = action.get('app'), action.get('action')
    if app == 'system':
        if name == 'switch_app': return f"Successfully switched to app: {action.get('target_app')}"
        return 'Task finished' if name == 'finish_task' else 'Task failed'
    target = read_target(action)
    if app == 'shell' and is_read_action(action):
        return snap['listing']
    if target and norm(target) in snap['files']:
        text = snap['files'][norm(target)]
        if app == 'excel': return 'OBSERVATION: The following is the table from the excel file:\n' + text
        if app == 'word': return 'OBSERVATION: The following is the content from the word file:\n' + text
        return text
    if is_read_action(action):
        sources = [(p, t) for p, t in snap['files'].items() if p.startswith('/testbed/data/')]
        return max(sources, key=lambda pt: tokens(pt[1]))[1] if sources else '[file does not exist]'
    return 'OBSERVATION: The operation was successful.'

def history_tokens(history, scale):
    empty = [(a, '') for a, obs in history]
    base = tokens(_format_history(empty))
    return base + scale * max(0, tokens(_format_history(history)) - base)

def estimate(record, snap, cat, scenario):
    scale, multiplier, extra = SCENARIOS[scenario]
    cfg = snap['config']
    parsed = parse_actions(record['candidate_actions'][0])
    actions = [a for a, _, _ in parsed]
    observations = [observation(a, snap) for a in actions]
    history, current_app = [], None
    actor_in = actor_out = 0.0
    events = []
    dirty = set()
    is_b1 = record['method'] == 'b1'
    fixed = build_officebench_flat_system_prompt(task_config=cfg, catalogue=cat, initial_state=snap['initial_state'], gold_files=snap['gold_files'], testbed_listing=snap['listing'], predict_all_actions=False)
    wm_system = fixed if is_b1 else _OFFICEBENCH_SIMULATOR_SYSTEM
    protocol = _python_protocol({'kind': 'observation', 'action': {}, 'reads': [], 'facts': {}, 'agent_history': []}, required=False, context_chars=0, output_kind='text', allow_reads=not is_b1)
    wm_base = tokens(wm_system) + tokens(protocol) + 320 + 150
    source_evidence = tokens(dump({'files': snap['files'], 'gold_files': snap['gold_files']}))
    for (a, raw, malformed), obs in zip(parsed, observations):
        empty_history = [(action, '') for action, _ in history]
        prompt = build_prompt(current_app, history, cfg, cat)
        stripped = build_prompt(current_app, empty_history, cfg, cat)
        actor_in += tokens(construct_system_message(cfg, cat)) + tokens(stripped) + scale * max(0, tokens(prompt) - tokens(stripped))
        actor_out += tokens(raw if malformed else dump(a)) + extra
        if a.get('app') != 'system' and a.get('app') in cat['available_apps']:
            target = read_target(a)
            predict = not is_read_action(a) or (target and norm(target) in dirty)
            if predict:
                raw_context = scale * source_evidence + history_tokens(history, scale)
                request = tokens(protocol) + 320 + 150 + tokens(dump(a))
                output = (scale * tokens(obs) if is_read_action(a) else tokens(obs)) + extra
                events.append([wm_base + tokens(dump(a)), output, raw_context, request, multiplier])
            for p in write_targets(a):
                dirty.add(norm(p))
        if a.get('action') == 'switch_app':
            current_app = a.get('target_app')
        history.append((a, obs))
    # Changed evaluator files add final-state prediction work. Gold-backed files are
    # exempt only if that run explicitly enabled gold reconstruction.
    final_targets = [s for s in state_files_from_config(cfg) if norm(s['path']) in dirty]
    if record.get('officebench_reconstruct_gold') and snap['gold_files']:
        final_targets = []
    for spec in final_targets:
        request = tokens(protocol) + 320 + 150 + tokens(spec['path'])
        output = scale * tokens(snap['initial_state'].get(spec['path'], '')) + extra
        events.append([tokens(wm_system) + request, output, scale * source_evidence + history_tokens(history, scale), request, multiplier])
    breakdown = (record.get('candidate_verifier_percheck') or [None])[0]
    atomic_count = int((breakdown or {}).get('total', 0))
    whole = int(breakdown is not None)
    # Python executions include some verifier work; this is a sensitivity proxy,
    # not an observed count of prediction/model calls.
    modeled = sum(e[4] for e in events) + whole
    floor = record.get('n_python_calls', 0)
    if events and floor > modeled:
        factor = (floor - whole) / sum(e[4] for e in events)
        for e in events: e[4] *= factor
    elif not events and floor > whole:
        events.append([wm_base, 40 + extra, scale * source_evidence + history_tokens(history, scale), wm_base - tokens(wm_system), floor - whole])
    verifier = OfficeBenchVerifier(None)
    answer = next((a.get('answer') for a in reversed(actions) if a.get('action') == 'finish_task'), None)
    checks = _atomic_requirements(cfg)
    for i in range(atomic_count + whole):
        atomic = i < atomic_count
        check = checks[i % len(checks)] if atomic and checks else None
        check_config = {'evaluation': [check]} if check else cfg
        request = verifier._build_prompt(cfg['task'], actions, observations, check_config, snap['initial_state'], answer, snap['gold_files'], status_check=check)
        stripped = verifier._build_prompt(cfg['task'], actions, [''] * len(actions), check_config, snap['initial_state'], answer, snap['gold_files'], status_check=check)
        request_size = tokens(stripped) + scale * max(0, tokens(request) - tokens(stripped))
        if not atomic: request_size += tokens(protocol)
        system = fixed if is_b1 else (_STATUS_SYSTEM if atomic else _SYSTEM)
        events.append([tokens(system) + request_size, 4 if atomic else 8 + extra,
                       scale * source_evidence + history_tokens(history, scale), request_size, 1.0])
    wm_in = sum(inp * count for inp, out, raw, request, count in events)
    wm_out = sum(out * count for inp, out, raw, request, count in events)
    raw_extra = sum(raw * count for inp, out, raw, request, count in events)
    retained = noctx_extra = 0.0
    for inp, out, raw, request, count in events:
        exchange = request + raw + out
        noctx_extra += count * (raw + retained) + count * (count - 1) / 2 * exchange
        retained += count * exchange
    return dict(input=actor_in + wm_in, output=actor_out + wm_out, total=actor_in + wm_in + actor_out + wm_out,
                actor=actor_in + actor_out, wm=wm_in + wm_out, steps=len(parsed), wm_calls=sum(e[4] for e in events),
                fixed_prompt=tokens(wm_system), raw_only_input=actor_in + wm_in + raw_extra,
                noctx_input=actor_in + wm_in + noctx_extra, malformed_steps=sum(m for a,r,m in parsed))

def stats(values, weights=None):
    if weights and not all(math.isclose(w, weights[0]) for w in weights):
        ordered = sorted(zip(values, weights))
        population = sum(weights)
        mean = sum(v * w for v, w in ordered) / population
        positions, cumulative = [], 0.0
        for value, weight in ordered:
            positions.append((cumulative + weight / 2) / population)
            cumulative += weight
        def weighted_q(q):
            if q <= positions[0]: return ordered[0][0]
            for i in range(1, len(ordered)):
                if q <= positions[i]:
                    fraction = (q - positions[i-1]) / (positions[i] - positions[i-1])
                    return ordered[i-1][0] + fraction * (ordered[i][0] - ordered[i-1][0])
            return ordered[-1][0]
        return dict(n=len(values), mean=mean, median=weighted_q(.5),
                    sd=math.sqrt(sum(w * (v-mean)**2 for v,w in ordered) / population),
                    p10=weighted_q(.1), p90=weighted_q(.9), p95=weighted_q(.95), maximum=max(values))
    values = sorted(values)
    def q(p):
        at = (len(values) - 1) * p; lo = int(at)
        return values[lo] + (values[min(lo + 1, len(values) - 1)] - values[lo]) * (at - lo)
    return dict(n=len(values), mean=statistics.mean(values), median=statistics.median(values), sd=statistics.pstdev(values), p10=q(.1), p90=q(.9), p95=q(.95), maximum=max(values))

def aggregate(rows):
    result = {}
    for method in sorted({r['method'] for r in rows}):
        result[method] = {}
        for scenario in SCENARIOS:
            group = [r for r in rows if r['method'] == method and r['scenario'] == scenario]
            result[method][scenario] = {k: stats([r[k] for r in group], [r['population_weight'] for r in group]) for k in ('input','output','total','actor','wm','steps','wm_calls','fixed_prompt')}
            result[method][scenario]['by_subset'] = {s: stats([r['total'] for r in group if r['subset'] == s]) for s in EXPECTED}
    return result

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--officebench-root', type=Path, default=Path('/tmp/officebench-b1-code/officebench_data'))
    p.add_argument('--b1-records', type=Path, default=HERE / 'b1_run/named-outputs/output_dir/records.jsonl')
    p.add_argument('--ours-records', type=Path)
    p.add_argument('--update-summary', action='store_true')
    args = p.parse_args()
    cat_path = HERE / 'catalogue.json'
    if cat_path.exists():
        cat = json.loads(cat_path.read_text())
    else:
        cat = catalogue(args.officebench_root)
        cat_path.write_text(json.dumps(cat, indent=2) + '\n')
    inputs = {'b1': args.b1_records}
    if args.ours_records: inputs['ours'] = args.ours_records
    rows, sources = [], {}
    snap_cache, snapshot_hashes = {}, {}
    cache_dir = HERE / 'snapshots'; cache_dir.mkdir(exist_ok=True)
    for method, path in inputs.items():
        records = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
        records = [r for r in records if r['n_rollouts'] == 1 and r['method'] == method]
        observed = Counter(r['subset'] for r in records)
        assert all(0 < observed[s] <= EXPECTED[s] for s in EXPECTED)
        if method == 'b1': assert observed == EXPECTED
        assert len({r['task_id'] for r in records}) == len(records)
        assert all(r.get('llm_agent') == 'gpt-5.4' for r in records)
        assert all(not r.get('predict_all_actions') for r in records)
        sources[method] = dict(aml_jobs=sorted({r.get('source_aml_job', 'dw-officebench-b1-llm-wm-gpt54-n124-20260904' if method == 'b1' else 'dw-officebench-ours-gpt54-tokens-n1-20260908') for r in records}), path=str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path), sha256=sha(path), n_tasks=len(records),
                               subset_counts=dict(observed), population_subset_counts=EXPECTED, max_steps=sorted({r['solver_max_steps'] for r in records}),
                               source_token_fields=sorted({r['tokens'] for r in records}))
        for index, record in enumerate(records):
            cached = cache_dir / (record['task_id'].replace('/', '_') + '.json')
            if record['task_id'] not in snap_cache:
                if cached.exists():
                    raw_snapshot = cached.read_bytes()
                    snap = json.loads(raw_snapshot)
                else:
                    snap = snapshot(args.officebench_root, record['task_id'])
                    raw_snapshot = (json.dumps(snap, indent=2) + '\n').encode()
                    cached.write_bytes(raw_snapshot)
                snap_cache[record['task_id']] = snap
                snapshot_hashes[cached.name] = hashlib.sha256(raw_snapshot).hexdigest()
            snap = snap_cache[record['task_id']]
            for scenario in SCENARIOS:
                row = dict(method=method, task_id=record['task_id'], subset=record['subset'], scenario=scenario,
                           population_weight=EXPECTED[record['subset']] / observed[record['subset']],
                           **estimate(record, snap, cat, scenario))
                rows.append(row)
                if method == 'ours':
                    for variant, key in [('ours_raw_context_only','raw_only_input'), ('ours_no_context_management','noctx_input')]:
                        ablation = dict(row, method=variant, input=row[key], total=row[key] + row['output'], wm=row['wm'] + row[key] - row['input'])
                        rows.append(ablation)
            if (index + 1) % 50 == 0: print(method, index + 1, flush=True)
    summary = aggregate(rows)
    for method, block in summary.items():
        base_method = 'b1' if method == 'b1' else 'ours'
        block['status'] = 'estimated' if sources[base_method]['n_tasks'] == 300 else 'estimated_partial'
        block['population_n_tasks'] = 300
        block['observed_subset_counts'] = sources[base_method]['subset_counts']
        block['weighting'] = 'app_count_poststratification'
        block['sample_selection'] = 'all tasks' if sources[base_method]['n_tasks'] == 300 else 'completed trajectories at stop time; not random'
        block['provenance'] = 'experiments/officebench_gpt54_tokens_20260908/provenance.json'
        block['counterfactual'] = method.startswith('ours_')
    (HERE / 'gpt54_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    with (HERE / 'per_task.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    provenance = dict(status='trajectory_reconstruction_estimates_not_api_measurements', tokenizer='cl100k_base', scenarios=SCENARIOS,
                      assumptions='token_assumption.md', estimator_sha256=sha(Path(__file__)), sources=sources,
                      prompt_helper_sha256={Path(inspect.getsourcefile(f)).name: sha(Path(inspect.getsourcefile(f))) for f in (build_prompt, OfficeBenchVerifier, build_officebench_flat_system_prompt, _python_protocol)},
                      source_code_asset='328cf242-2c2e-43f9-867e-b99b9ff78638/1', catalogue_sha256=sha(cat_path), snapshots=dict(sorted(snapshot_hashes.items())),
                      units='cumulative input plus output tokens per task, N=1, including failures; means weighted to app-count population proportions',
                      weighting='Observed tasks in each subset receive weight population_count / observed_count. Weighted quantiles interpolate at cumulative-weight midpoints; complete runs retain ordinary linear sample quantiles.',
                      limitations=['Partial Ours coverage is completion-selected, not random; unfinished expensive tasks may be underrepresented even after app-count weighting.',
                                   'Reconstructed saved prompts and missing observations; not exact historical requests.',
                                   'Malformed/truncated action arguments are salvaged; missing content is not invented.',
                                   'Pristine documents use docx/openpyxl/PyMuPDF/Tesseract; shell output uses the pristine file listing.',
                                   'Missing read operands use the largest pristine source document as a conservative fallback.',
                                   'Context ablations hold actions, modeled calls and completions fixed.',
                                   'Full code/model runs use a 40-step limit; historical placeholders used different trajectories.'])
    (HERE / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    if args.update_summary:
        assert set(summary) == {'b1','ours','ours_raw_context_only','ours_no_context_management'}, 'Require B1 and collected Ours trajectories before replacing placeholders'
        path = ROOT / 'summary.jsonl'; data = json.loads(path.read_text()); data.update(summary)
        path.write_text(json.dumps(data, indent=2) + '\n')
    for method, block in summary.items(): print(method, block['central']['total']['mean'])

if __name__ == '__main__': main()
