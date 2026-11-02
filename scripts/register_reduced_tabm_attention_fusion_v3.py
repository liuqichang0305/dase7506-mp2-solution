"""One preregistered train-OOF mixture; prepare does not register or evaluate it.

The explicit stages preserve earlier experiments and require an idle scheduler
for shared-ledger mutations. No method, weight, split or seed CLI overrides.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

import psutil

ROOT = Path(__file__).resolve().parents[1]
BASE = 'results/research_v3/'
NAME = 'v3_fusion_reduced_tabm_attention_ema_s0'
ROUND = 'r12_reduced_tabm_attention_equal_fusion'
METHODS = ['v3:v3_tabm_k4_embedding_stats_s0', 'v3:v3_attention_ema_s0']
WEIGHTS = [.5, .5]
SELF = 'scripts/register_reduced_tabm_attention_fusion_v3.py'
DIAGNOSTIC = 'code/tools/diagnose_oof_v3.py'
GATE = BASE + 'diagnostic-r05-reduced-tabm-vs-attention/diagnostic.json'
COMPLETION = BASE + 'diagnostic-r05-reduced-tabm-vs-attention/COMPLETION.json'
PREPARED = 'docs/research_v3/rounds/' + ROUND + '_preregistration.json'
FORMAL = 'docs/research_v3/rounds/' + NAME + '.json'
REGISTRY = BASE + 'fusion_registry.jsonl'
OLD_ANALYSIS = BASE + 'fixed_fusion_analysis.json'
FINAL_ANALYSIS = BASE + 'fixed_fusion_analysis_final.json'
REGISTRATION_RECEIPT = BASE + 'fusion_r12_registration.json'
EXECUTION = BASE + 'fusion_r12_execution/'
LEDGER = BASE + 'round_decisions.jsonl'
LEDGER_SNAPSHOT = BASE + 'fusion_r12_round_decisions.snapshot.jsonl'
ROUND_SNAPSHOT = BASE + 'fusion_r12_round_record.json'
POLICY = {
    'matched_control': METHODS[1],
    'retain_for_final_comparison_only_if': 'hard_correct > control.hard_correct AND overall_correct >= control.overall_correct',
    'otherwise': 'Reject this fixed mixture; no post-result weight, temperature, seed, member or routing changes.',
    'single_positive': 'Disclose its exact change; it is not a post-hoc tuning criterion.',
    'final_selection': 'A local pass is not automatic deployment. The unchanged campaign finalist/resource rules still apply.',
    'historical_reference': 'research_v2:attention_regularized_s0',
    'historical_hard_correct': 1687,
    'continuation_reference': 'At registration lock the highest complete BASE957 common-three-fold hard mean across current V3, preserved V2 and prior registered fusions. Do not replace it after evaluating this mixture.',
    'campaign_threshold_pp_strictly_greater_than': .5,
}


def utc():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def lines(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write('\n')


def safe_path(root, name):
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()) or Path(name).is_absolute():
        raise ValueError('Evidence paths must remain repository relative')
    return path


def verify_map(root, mapping):
    if not mapping:
        raise ValueError('Evidence hash mapping must not be empty')
    for name, expected in mapping.items():
        if sha(safe_path(root, name)) != expected:
            raise ValueError('Evidence hash changed: ' + name)


def commit(root):
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()


def check_scope(spec):
    if (spec.get('experiment_id') != NAME or spec.get('round') != ROUND
            or spec.get('methods') != METHODS or spec.get('weights') != WEIGHTS
            or spec.get('representative') != 'first_model_first_member'
            or spec.get('decision_policy') != POLICY or spec.get('new_member_training_fits') != 0):
        raise ValueError('Fixed preregistered scope changed')


def gate_inputs(root):
    completion = read(root / COMPLETION)
    if (completion.get('status') != 'complete' or completion.get('only_reciprocal_gate') != METHODS
            or completion.get('fusion_run') is not False or completion.get('validation_test_ranking_read') is not False):
        raise ValueError('Prior train-only gate completion is missing')
    for field in ('source_sha256', 'input_sha256', 'output_sha256'):
        verify_map(root, completion[field])
    gate = read(root / GATE)
    matches = [row for row in gate['complementarity'] if row['methods'] == METHODS]
    if (gate.get('status') != 'complete' or gate.get('common_labels_groups_folds_verified') is not True
            or gate.get('fusion', {}).get('status') != 'not_run' or len(matches) != 1):
        raise ValueError('Exactly one prior specified gate is required')
    item = matches[0]
    difficult = item['hard']
    if (item.get('gate_passed') is not True or min(item['overall']['member_unique_correct']) < 10
            or difficult['n'] <= 0 or 100 * (difficult['oracle_correct'] - max(difficult['member_correct'])) < difficult['n']):
        raise ValueError('Reciprocal >=10 each and hard oracle >=1pp gate failed')
    members = {row['method']: row for row in gate['methods']}
    for key in METHODS:
        row = members[key]
        folder = BASE + key.split(':')[1] + '/'
        verify_map(root, {folder + name: digest for name, digest in row['input_sha256'].items()})
        if row['config']['seed'] != 3036900957 or row['metrics']['overall']['n'] != 4755:
            raise ValueError('Only complete fixed-seed OOF members are allowed')
    inputs = {**completion['input_sha256'], **completion['output_sha256'], COMPLETION: sha(root / COMPLETION)}
    return gate, item, members, inputs, completion['source_sha256']


def validate_old_analysis(root):
    old = read(root / OLD_ANALYSIS)
    registered = lines(root / REGISTRY)
    if (old.get('status') != 'completed' or old.get('configurations') != 2
            or len(old['results']) != 2 or sha(root / REGISTRY) != old['registry_sha256']
            or [r['experiment_id'] for r in registered] != [r['experiment_id'] for r in old['results']]):
        raise ValueError('The original immutable two-fusion analysis/registry is required')
    inputs = {REGISTRY: sha(root / REGISTRY), OLD_ANALYSIS: sha(root / OLD_ANALYSIS)}
    for row, result in zip(registered, old['results']):
        inputs[row['registration']] = row['registration_sha256']
        inputs[result['diagnostic_path']] = result['diagnostic_sha256']
    verify_map(root, inputs)
    return inputs


def prepare(root=ROOT):
    if (root / PREPARED).exists():
        raise FileExistsError('Prepared choices are immutable; do not overwrite')
    gate, evidence, members, inputs, sources = gate_inputs(root)
    inputs.update(validate_old_analysis(root))
    inputs[BASE + 'campaign_budget.json'] = sha(root / (BASE + 'campaign_budget.json'))
    sources = {**sources, SELF: sha(root / SELF)}
    record = {
        'schema': 'mp2-fixed-fusion-preregistration-v3', 'status': 'prepared_not_registered',
        'prepared_at_utc': utc(), 'experiment_id': NAME, 'round': ROUND,
        'purpose': 'single_fixed_equal_probability_oof',
        'hypothesis': 'Reduced TabM and Attention EMA have reciprocal train-OOF errors. Test one fixed equal answer-group probability mean; oracle potential does not predict actual mixture performance.',
        'methods': METHODS, 'weights': WEIGHTS, 'representative': 'first_model_first_member',
        'decision_policy': POLICY, 'new_member_training_fits': 0,
        'oof_sha256': {key: members[key]['input_sha256']['oof.npz'] for key in METHODS},
        'gate_evidence': {'path': GATE, 'sha256': sha(root / GATE)}, 'prior_gate': evidence,
        'folds_sha256': gate['folds_sha256'],
        'member_metrics': {key: members[key]['metrics'] for key in METHODS},
        'input_sha256': inputs, 'source_sha256': sources, 'source_commit': commit(root),
        'probability_rule': 'First average the four TabM member group probabilities, then average that model probability with Attention EMA at [0.5, 0.5]. Do not average logits or pool the five networks at equal weight.',
        'representative_and_ties': 'Choose the highest mean-probability answer group; exact group ties choose earliest original candidate. Within that group use the first model (reduced TabM), first internal member, highest raw score; exact ties use earliest candidate. All-None falls back to index 0.',
        'representation_capacity_caveat': '1040 inputs retain 1024 embedding coordinates plus 16 statistics. Removing 1024 absolute-deviation coordinates also shrinks first-layer capacity. The matched +62 hard-question gain is concentrated by fold (+45/+11/+6); it does not isolate information removal from capacity/regularization or prove seed stability.',
        'scope_limit': 'The upcoming reduced-feature k8 and EMA trials are not candidate replacements. Exactly these ordered seed-0 members; no weight grid, conditional routing, calibration, validation, test or ranking access.',
        'budget_note': 'One additional unique configuration, zero fitting runs. Preserve the initial 24h/96-config clock, pilot and all prior fusion configurations. Register/evaluate only after the scheduler queue is terminal and no trainer is active.',
    }
    verify_map(root, inputs); verify_map(root, sources)
    write_new(root / PREPARED, record)
    return record


def prepared(root):
    record = read(root / PREPARED)
    check_scope(record)
    if record.get('status') != 'prepared_not_registered':
        raise ValueError('Preparation status changed')
    # Registry is the only prior input legitimately extended by registration.
    verify_map(root, {k: v for k, v in record['input_sha256'].items() if k != REGISTRY})
    verify_map(root, record['source_sha256'])
    return record


def budget_end(root):
    budget = read(root / (BASE + 'campaign_budget.json'))
    stamp = lambda key: datetime.fromisoformat(budget[key].replace('Z', '+00:00')).timestamp()
    end = min(stamp('research_end_utc'), stamp('started_at_utc') + budget['maximum_wall_seconds'],
              stamp('deadline_utc') - 3600 * budget['deadline_buffer_hours'])
    if (root / (BASE + 'SEARCH_CLOSED.json')).exists() or time.time() >= end:
        raise RuntimeError('Absolute research budget is closed; no registration/evaluation')
    return budget, end


def incumbent_snapshot(root):
    """Saved train aggregates only; freeze the current global reference pre-result."""
    queue = read(root / (BASE + 'campaign_queue.json'))['jobs']
    paths = [(BASE + job['experiment_id'] + '/summary.json', 'research_v3:' + job['experiment_id'])
             for job in queue if job['status'] == 'completed']
    paths += [(p.relative_to(root).as_posix(), 'research_v2:' + p.parent.name)
              for p in sorted((root / 'results/research_v2').glob('*/summary.json'))]
    candidates = []
    for path, identity in paths:
        value = read(safe_path(root, path))
        if value.get('config', {}).get('seed') != 3036900957:
            continue
        folds = value.get('folds', [])
        if (value.get('status') != 'complete' or value.get('n') != 4755
                or sorted(row['fold'] for row in folds) != [0, 1, 2]
                or any(row.get('status') != 'complete' or row['n'] != 1585 for row in folds)
                or sorted((row['fold'], row['hard_n']) for row in folds) != [(0, 705), (1, 705), (2, 704)]
                or sum(row['hard_correct'] for row in folds) != value['hard_correct']):
            raise ValueError('Completed global-reference summary is incomplete')
        mean = sum(100 * row['hard_correct'] / row['hard_n'] for row in folds) / 3
        if not math.isclose(mean, value['hard_accuracy_fold_mean'], abs_tol=1e-10):
            raise ValueError('Incumbent hard fold mean differs from its actual counts')
        candidates.append({'experiment_id': identity, 'path': path, 'sha256': sha(root / path),
                           'hard_fold_mean': mean, 'hard_correct': value['hard_correct'], 'hard_n': value['hard_n']})
    for row in read(root / OLD_ANALYSIS)['results']:
        path = row['diagnostic_path']
        diagnostic = read(root / path)
        actual = diagnostic['fusion']
        if (actual['status'] != 'complete' or actual['metrics'] != row['metrics']
                or sha(root / path) != row['diagnostic_sha256']):
            raise ValueError('Prior fusion outcome changed')
        metric = actual['metrics']['hard']
        candidates.append({'experiment_id': 'research_v3:' + row['experiment_id'],
                           'path': path, 'sha256': sha(root / path),
                           'hard_fold_mean': metric['fold_mean'], 'hard_correct': metric['correct'], 'hard_n': metric['n']})
    if not candidates or any(not math.isfinite(row['hard_fold_mean']) or row['hard_n'] != 2114 for row in candidates):
        raise ValueError('A valid common-fold incumbent is required')
    chosen = sorted(candidates, key=lambda row: (-row['hard_fold_mean'], row['experiment_id']))[0]
    return {'captured_at_utc': utc(), 'seed': 3036900957, 'metric': 'hard_accuracy_fold_mean',
            'queue_sha256': sha(root / (BASE + 'campaign_queue.json')), 'chosen': chosen,
            'candidates': sorted(candidates, key=lambda row: row['experiment_id']),
            'tie_rule': 'Lexicographic experiment identity after identical hard fold mean; no seed selection.'}


def assert_idle(root):
    jobs = read(root / (BASE + 'campaign_queue.json'))['jobs']
    if any(row['status'] not in ('completed', 'failed', 'pruned') for row in jobs):
        raise RuntimeError('Campaign queue must be terminal before fusion execution')
    trainer = root / 'tmp/research_v3/trainer.lock'
    if trainer.exists():
        raise RuntimeError('Trainer lock exists; resolve its owner before continuing')
    for process in psutil.process_iter(['cmdline']):
        try:
            args = process.info['cmdline'] or []
            if any(str(arg).replace('\\', '/').endswith('code/train_v3.py') for arg in args):
                if Path(process.cwd()).resolve() == root.resolve():
                    raise RuntimeError('An active project trainer prevents OOF execution')
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return jobs


@contextmanager
def idle_lock(root):
    path = root / 'tmp/research_v3/scheduler.lock'
    path.parent.mkdir(parents=True, exist_ok=True)
    identity = {'pid': os.getpid(), 'create_time': psutil.Process().create_time(),
                'command_fragment': Path(SELF).name, 'utc': utc(), 'nonce': uuid.uuid4().hex}
    # Never rename another owner's lock; even a stale/initializing lock needs review.
    with path.open('x', encoding='utf-8') as stream:
        json.dump(identity, stream)
    try:
        assert_idle(root)
        yield
    finally:
        if path.exists() and read(path) == identity:
            path.unlink()  # Only this process's ephemeral lock, never evidence.


def atomic_append(path, before, record):
    if path.read_bytes() != before:
        raise ValueError('Shared ledger changed before append')
    if before and not before.endswith(b'\n'):
        raise ValueError('Existing ledger must end with a newline')
    encoded = json.dumps(record, ensure_ascii=False).encode('utf-8') + b'\n'
    pending = path.with_name(path.name + '.' + uuid.uuid4().hex + '.pending')
    with pending.open('xb') as stream:
        stream.write(before + encoded)
    if path.read_bytes() != before:
        raise ValueError('Shared ledger changed during append; pending journal preserved')
    pending.replace(path)


def registered(root):
    spec = read(root / FORMAL)
    check_scope(spec)
    if (spec.get('status') != 'registered' or spec.get('purpose') != 'single_fixed_equal_probability_oof'
            or spec.get('preregistration_sha256') != sha(root / PREPARED)):
        raise ValueError('Formal registration differs from independent preparation')
    prior = prepared(root)
    for key in ('oof_sha256', 'gate_evidence', 'input_sha256', 'source_sha256', 'member_metrics'):
        if spec[key] != prior[key]:
            raise ValueError('Formal registration changed frozen evidence')
    incumbent = spec['global_incumbent_before_evaluation']
    verify_map(root, {row['path']: row['sha256'] for row in incumbent['candidates']})
    expected_incumbent = sorted(incumbent['candidates'], key=lambda row: (-row['hard_fold_mean'], row['experiment_id']))[0]
    if incumbent['chosen'] != expected_incumbent:
        raise ValueError('Global incumbent does not maximize the preregistered fold mean')
    all_rows = lines(root / REGISTRY)
    match = [row for row in all_rows if row['experiment_id'] == NAME]
    expected = {'registration': FORMAL, 'registration_sha256': sha(root / FORMAL), **spec}
    if match != [expected] or len(all_rows) != 3:
        raise ValueError('Exactly the prior two plus this one fusion must be registered')
    raw = (root / REGISTRY).read_bytes()
    prefix = b''.join(raw.splitlines(keepends=True)[:-1])
    if hashlib.sha256(prefix).hexdigest() != prior['input_sha256'][REGISTRY]:
        raise ValueError('Original registry bytes changed')
    return spec


def register(root=ROOT):
    with idle_lock(root):
        prior = prepared(root)
        budget, _ = budget_end(root)
        queue = read(root / (BASE + 'campaign_queue.json'))['jobs']
        old = lines(root / REGISTRY)
        trained = {job['experiment_id'] for job in queue}
        external = {row['experiment_id'] for row in old}
        if len(old) != len(external) or trained & external or NAME in trained:
            raise ValueError('Unique configuration identities overlap')
        count = len(trained | external | {NAME}) + 1  # Actual timing pilot.
        if count > budget['maximum_unique_configurations']:
            raise RuntimeError('Unique configuration cap exceeded')
        receipt_path = root / REGISTRATION_RECEIPT
        if NAME in external:
            registered(root)
            if receipt_path.exists():
                receipt = read(receipt_path)
                verify_map(root, receipt['output_sha256'])
                return receipt
        else:
            if sha(root / REGISTRY) != prior['input_sha256'][REGISTRY]:
                raise ValueError('Prior two-entry registry changed')
            if not (root / FORMAL).exists():
                formal = {**prior, 'schema': 'mp2-fixed-fusion-registration-v3', 'status': 'registered',
                          'registered_at_utc': utc(), 'registration_source_commit': commit(root),
                          'preregistration_path': PREPARED, 'preregistration_sha256': sha(root / PREPARED),
                          'global_incumbent_before_evaluation': incumbent_snapshot(root)}
                write_new(root / FORMAL, formal)
            formal = read(root / FORMAL)
            check_scope(formal)
            if (formal.get('schema') != 'mp2-fixed-fusion-registration-v3'
                    or formal.get('status') != 'registered'
                    or formal.get('preregistration_sha256') != sha(root / PREPARED)
                    or formal.get('global_incumbent_before_evaluation', {}).get('queue_sha256') != sha(root / (BASE + 'campaign_queue.json'))):
                raise ValueError('Partial registration is invalid or its incumbent queue changed')
            if any(formal[k] != prior[k] for k in prior if k not in ('schema', 'status')):
                raise ValueError('Partial formal registration differs from prepared choices')
            row = {'registration': FORMAL, 'registration_sha256': sha(root / FORMAL), **formal}
            atomic_append(root / REGISTRY, (root / REGISTRY).read_bytes(), row)
            registered(root)
        receipt = {'schema': 'mp2-v3-one-fusion-registration-receipt-v1', 'status': 'registered',
                   'created_at_utc': utc(), 'experiment_id': NAME, 'unique_configurations_including_pilot': count,
                   'new_configurations': 1, 'new_training_fits': 0, 'preregistration_sha256': sha(root / PREPARED),
                   'queue_sha256_at_registration': sha(root / (BASE + 'campaign_queue.json')),
                   'prior_registry_sha256': prior['input_sha256'][REGISTRY],
                   'output_sha256': {FORMAL: sha(root / FORMAL), REGISTRY: sha(root / REGISTRY)}}
        write_new(receipt_path, receipt)
        return receipt


def validate_result(root, spec):
    folder = root / (BASE + NAME)
    diagnostic = read(folder / 'diagnostic.json')
    result = diagnostic['fusion']
    if (diagnostic.get('status') != 'complete' or diagnostic.get('validation_test_rank_read') is not False
            or diagnostic.get('rejected_reads') != [] or result.get('status') != 'complete'
            or result.get('methods') != METHODS or result.get('weights') != WEIGHTS
            or result.get('registration_file') != 'fusion-registration.json'
            or result.get('registration_sha256') != sha(root / FORMAL)
            or sha(folder / 'fusion-registration.json') != sha(root / FORMAL)
            or result.get('oof_sha256') != sha(folder / 'fusion-oof.npz')
            or diagnostic.get('source_sha256') != spec['source_sha256'][DIAGNOSTIC]
            or diagnostic.get('folds_sha256') != spec['folds_sha256']):
        raise ValueError('Actual fusion output does not match its registration/provenance')
    for subset, count in [('overall', 4755), ('hard', 2114), ('single_positive', 711)]:
        metric = result['metrics'][subset]
        if (metric['n'] != count or type(metric['correct']) is not int or not 0 <= metric['correct'] <= count
                or sum(row['correct'] for row in metric['folds']) != metric['correct']
                or sum(row['n'] for row in metric['folds']) != count
                or sorted(row['fold'] for row in metric['folds']) != [0, 1, 2]):
            raise ValueError('Invalid complete OOF outcome counts')
    return diagnostic


def execute(root=ROOT):
    with idle_lock(root):
        spec = registered(root)
        _, end = budget_end(root)
        folder = root / EXECUTION
        if folder.exists():
            result = read(folder / 'execution.json')  # Partial attempts are preserved, not replayed.
            if result.get('status') != 'complete':
                raise RuntimeError('Previous execution failed; retain evidence and obtain explicit repair review')
            verify_map(root, result['input_sha256']); verify_map(root, result['source_sha256'])
            verify_map(root, result['output_sha256']); validate_result(root, spec)
            return result
        if (root / (BASE + NAME)).exists():
            raise FileExistsError('Existing fusion output has no matching execution receipt')
        folder.mkdir(parents=True, exist_ok=False)
        incumbent = spec['global_incumbent_before_evaluation']
        if sha(root / (BASE + 'campaign_queue.json')) != incumbent['queue_sha256']:
            raise ValueError('Queue changed since incumbent registration; preserve choices and review before execution')
        direct_python = Path(getattr(sys, '_base_executable', sys.executable)).resolve()
        site_packages = root / '.venv/Lib/site-packages'
        if not direct_python.is_file() or not site_packages.is_dir():
            raise ValueError('Direct base Python and the fixed local environment are required')
        # Direct actual worker avoids the Windows venv launcher surviving timeout.
        # Existing diagnostic guard forbids spawned processes once initialized.
        command = [str(direct_python), '-S', DIAGNOSTIC,
                   '--method', METHODS[0], '--method', METHODS[1], '--pair', *METHODS,
                   '--output', BASE + NAME, '--fusion-registration', FORMAL]
        inputs = {**spec['input_sha256'], REGISTRY: sha(root / REGISTRY), FORMAL: sha(root / FORMAL),
                  PREPARED: sha(root / PREPARED), REGISTRATION_RECEIPT: sha(root / REGISTRATION_RECEIPT),
                  **{row['path']: row['sha256'] for row in incumbent['candidates']}}
        start = {'schema': 'mp2-v3-one-fusion-execution-start-v1', 'started_at_utc': utc(),
                 'command': command, 'cwd': str(root), 'input_sha256': inputs,
                 'source_sha256': spec['source_sha256'], 'new_training_fits': 0}
        write_new(folder / 'STARTED.json', start)
        environment = dict(os.environ)
        environment['PYTHONPATH'] = str(site_packages.resolve())
        environment['PYTHONNOUSERSITE'] = '1'
        for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
            environment[key] = '1'
        began = time.perf_counter()
        code, error = None, None
        try:
            with (folder / 'stdout.log').open('xb') as out, (folder / 'stderr.log').open('xb') as err:
                code = subprocess.run(command, cwd=root, env=environment, stdout=out, stderr=err,
                                      timeout=max(.1, min(120., end - time.time())), check=False).returncode
            child_wall = time.perf_counter() - began
            if code != 0:
                raise RuntimeError('Fixed OOF diagnostic returned nonzero exit code: ' + str(code))
            validate_result(root, spec)
            verify_map(root, inputs); verify_map(root, spec['source_sha256'])
        except BaseException as failure:
            child_wall = time.perf_counter() - began
            error = {'type': type(failure).__name__, 'message': str(failure)}
        outputs = {p.relative_to(root).as_posix(): sha(p) for directory in (folder, root / (BASE + NAME))
                   if directory.exists() for p in sorted(directory.rglob('*')) if p.is_file()}
        result = {**start, 'schema': 'mp2-v3-one-fusion-execution-v1',
                  'status': 'complete' if error is None else 'failed', 'completed_at_utc': utc(),
                  'child_exit_code': code, 'child_wall_seconds': child_wall,
                  'total_observed_wall_seconds': time.perf_counter() - began,
                  'wall_scope': 'Child wall includes Python startup, train-OOF verification/mixing and diagnostic output. Total also includes parent postchecks/hash receipt preparation. Neither is inference latency or CPU time.',
                  'output_sha256': outputs, 'error': error, 'validation_test_ranking_read': False}
        write_new(folder / 'execution.json', result)
        if error is not None:
            raise RuntimeError('Fusion failed; immutable failure receipt saved: ' + str(error))
        return result


def decision(metrics, control):
    keep = metrics['hard']['correct'] > control['hard']['correct'] and metrics['overall']['correct'] >= control['overall']['correct']
    return ('Retain only for the unchanged final comparison/resource rules; no further mixture tuning.' if keep else POLICY['otherwise']), keep


def summarize(root=ROOT):
    with idle_lock(root):
        spec = registered(root)
        execution_path = root / EXECUTION / 'execution.json'
        execution = read(execution_path)
        if execution.get('status') != 'complete' or execution.get('child_exit_code') != 0:
            raise ValueError('A successful measured actual fusion is required')
        for key in ('input_sha256', 'source_sha256', 'output_sha256'):
            verify_map(root, execution[key])
        actual = validate_result(root, spec)['fusion']
        metrics = actual['metrics']
        control = spec['member_metrics'][METHODS[1]]
        text, keep = decision(metrics, control)
        old = read(root / OLD_ANALYSIS)
        row = {'experiment_id': NAME, 'methods': METHODS, 'weights': WEIGHTS, 'metrics': metrics,
               'attention_control': METHODS[1],
               'hard_delta_questions_vs_attention_control': metrics['hard']['correct'] - control['hard']['correct'],
               'overall_delta_questions_vs_attention_control': metrics['overall']['correct'] - control['overall']['correct'],
               'single_positive_delta_questions_vs_attention_control': metrics['single_positive']['correct'] - control['single_positive']['correct'],
               'hard_delta_questions_vs_historical_best': metrics['hard']['correct'] - POLICY['historical_hard_correct'],
               'maximum_member_asset_bytes': max(x['total_member_uncompressed_bytes'] for x in actual['assets_per_fold']),
               'new_training_fits': 0, 'decision': text, 'local_retention_rule_passed': keep,
               'decision_policy': POLICY, 'diagnostic_path': BASE + NAME + '/diagnostic.json',
               'diagnostic_sha256': sha(root / (BASE + NAME + '/diagnostic.json'))}
        incumbent = spec['global_incumbent_before_evaluation']['chosen']
        improvement = metrics['hard']['fold_mean'] - incumbent['hard_fold_mean']
        new_round = {'schema': 'mp2-fixed-fusion-round-decision-v3', 'created_at_utc': utc(),
                     'round': ROUND, 'round_type': 'hypothesis', 'status': 'completed',
                     'configurations': 1, 'completed_outer_fits': 0, 'new_training_fits': 0,
                     'results': [row], 'exceeds_registered_half_point': improvement > .5,
                     'increment_above_global_reference_pp': improvement,
                     'global_reference': incumbent['experiment_id'], 'global_incumbent_before_evaluation': incumbent,
                     'decision': text,
                     'terminal_observed_wall_seconds': {NAME: execution['child_wall_seconds']},
                     'execution_path': EXECUTION + 'execution.json', 'execution_sha256': sha(execution_path)}
        aggregate = {**old, 'created_at_utc': utc(), 'round': 'all_three_registered_fixed_fusions',
                     'configurations': 3, 'results': [*old['results'], row],
                     'global_reference': incumbent['experiment_id'],
                     'exceeds_registered_half_point': old['exceeds_registered_half_point'] or improvement > .5,
                     'decision': 'Original two mixtures retain their recorded rejections. Third mixture: ' + text,
                     'terminal_observed_wall_seconds': {**old['terminal_observed_wall_seconds'], NAME: execution['child_wall_seconds']},
                     'wall_measurement_source': {'prior_two': old['wall_measurement_source'], 'third': execution['wall_scope']},
                     'registry_sha256': sha(root / REGISTRY),
                     'prior_analysis': {'path': OLD_ANALYSIS, 'sha256': sha(root / OLD_ANALYSIS)},
                     'new_round_record': new_round,
                     'input_sha256': {OLD_ANALYSIS: sha(root / OLD_ANALYSIS), REGISTRY: sha(root / REGISTRY),
                                      PREPARED: sha(root / PREPARED), FORMAL: sha(root / FORMAL),
                                      EXECUTION + 'execution.json': sha(execution_path)},
                     'source_sha256': spec['source_sha256']}
        target = root / FINAL_ANALYSIS
        if target.exists():
            stored = read(target)
            # Resume only the ledger append after a fully written immutable aggregate.
            for field in ('results', 'input_sha256', 'source_sha256', 'registry_sha256'):
                if stored[field] != aggregate[field]:
                    raise ValueError('Existing final aggregate differs; preserve it')
            aggregate, new_round = stored, stored['new_round_record']
        else:
            write_new(target, aggregate)
        prior = lines(root / LEDGER)
        matching = [item for item in prior if item['round'] == ROUND]
        if matching and matching != [new_round]:
            raise ValueError('Existing round decision differs')
        if not matching:
            atomic_append(root / LEDGER, (root / LEDGER).read_bytes(), new_round)
        # Later stability rounds may append to the live ledger. Preserve this
        # historical snapshot and the exact new record; never bind future bytes.
        snapshot = root / LEDGER_SNAPSHOT
        if not snapshot.exists():
            with snapshot.open('xb') as stream:
                stream.write((root / LEDGER).read_bytes())
        if [item for item in lines(snapshot) if item['round'] == ROUND] != [new_round]:
            raise ValueError('Historical ledger snapshot lacks the exact R12 record')
        if not (root / ROUND_SNAPSHOT).exists():
            write_new(root / ROUND_SNAPSHOT, new_round)
        if read(root / ROUND_SNAPSHOT) != new_round:
            raise ValueError('Immutable R12 record changed')
        receipt_path = root / (BASE + 'fusion_r12_final_ledger.json')
        if receipt_path.exists():
            receipt = read(receipt_path)
            verify_map(root, receipt['output_sha256'])
            return receipt
        receipt = {'schema': 'mp2-v3-one-fusion-final-ledger-v1', 'status': 'complete',
                   'created_at_utc': utc(), 'experiment_id': NAME, 'new_configurations': 1,
                   'new_training_fits': 0, 'completed_outer_fits': 0, 'decision_policy': POLICY,
                   'decision': text, 'child_wall_seconds': execution['child_wall_seconds'],
                   'total_observed_wall_seconds': execution['total_observed_wall_seconds'],
                   'started_at_utc': execution['started_at_utc'], 'completed_at_utc': execution['completed_at_utc'],
                   'input_sha256': aggregate['input_sha256'], 'source_sha256': spec['source_sha256'],
                   'output_sha256': {FINAL_ANALYSIS: sha(target), LEDGER_SNAPSHOT: sha(snapshot),
                                     ROUND_SNAPSHOT: sha(root / ROUND_SNAPSHOT)},
                   'validation_test_ranking_read': False}
        write_new(receipt_path, receipt)
        return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'register', 'evaluate', 'summarize'])
    args = parser.parse_args()
    result = {'prepare': prepare, 'register': register, 'evaluate': execute, 'summarize': summarize}[args.stage]()
    print(json.dumps({'stage': args.stage, 'status': result['status'], 'experiment_id': NAME}, indent=2))


if __name__ == '__main__':
    main()
