"""One-way confirmation of one or two immutable V3 validation finalists.

No test option exists. All locks/source/assets are verified before any validation
data access. Completed candidate evidence can be resumed by hash; partial or
changed evidence requires explicit internal review and is never overwritten.
"""
from __future__ import annotations

import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS',
             'OMP_THREAD_LIMIT'):
    os.environ[_key] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import platform
import re
import subprocess
import sys
import sysconfig
import time
import uuid

import psutil

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/research_v3'
PROTOCOL = 'dase7506-mp2-open-hard-verifier-v5'
SEED = 3036900957
LIMIT_BYTES = 16 * 1024 * 1024
LIMIT_SECONDS = 30.
THREAD_ENV = {name: '1' for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
    'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS', 'OMP_THREAD_LIMIT')}
OFFICIAL_SOURCE = {
    'code/common.py': 'aecd10f7ff2e9e82ac5673d34d576e122caac51d2caaa7c7a8aa05906da51f6c',
    'code/predict.py': 'e7a627fbdc9ef8d4ab3189a6fa5ac02d56ae703ccd1a81507aea0bd66e042162',
    'code/score_predictions.py': 'fe12b809b82117aea722e5dfef7a55a88950a7cf50c49ec2ea007ea847959960'}
TRAIN_PATHS = {'code/data/manifest.json', 'code/data/gsm8k_train.jsonl',
               'code/data/candidates/gsm8k_train.jsonl', 'code/data/candidates/gsm8k_train.npz'}
INFERENCE_SOURCE = set(OFFICIAL_SOURCE) | {
    'code/solution_v3.py', 'code/mp2/__init__.py', 'code/mp2/features.py',
    'code/mp2/models_v2.py', 'code/mp2/selection_v2.py',
    'code/mp2/research_v3/models.py',
    'code/mp2/research_v3/preprocess.py', 'code/mp2/research_v3/selection.py',
    'code/tools/audit_assets.py', 'code/tools/confirm_validation_v3.py'}
PHASE_WRAPPER = """import json,os,runpy,sys
from pathlib import Path
from threadpoolctl import threadpool_info
script,report=sys.argv[1:3]
sys.argv=[script]+sys.argv[3:]
sys.path.insert(0,str(Path(script).resolve().parent))
try:
    runpy.run_path(script,run_name='__main__')
finally:
    record={'script':script,'executable':sys.executable,'pythonpath':os.environ.get('PYTHONPATH'),
            'thread_pools':threadpool_info(),'numpy_version':getattr(sys.modules.get('numpy'),'__version__',None),
            'torch_imported':'torch' in sys.modules,
            'thread_environment':{k:os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','OMP_THREAD_LIMIT')}}
    with open(report,'x',encoding='utf-8') as f:
        json.dump(record,f,indent=2);f.write('\\n')
"""


def utc():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(part)
    return result.hexdigest()


def atomic(path, value, *, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, ensure_ascii=False) + '\n'
    if exclusive:
        with path.open('x', encoding='utf-8') as stream:
            stream.write(payload)
        return
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.pending')
    temporary.write_text(payload, encoding='utf-8')
    temporary.replace(path)


def safe_path(relative, *, prefix=None):
    if not isinstance(relative, str) or Path(relative).is_absolute():
        raise ValueError('Expected a repository-relative path')
    path = (ROOT / relative).resolve()
    if not path.is_relative_to(ROOT.resolve()) or prefix and not path.is_relative_to((ROOT / prefix).resolve()):
        raise ValueError('Locked path escapes its permitted repository scope')
    return path


def check_hashes(mapping, *, prefix=None):
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError('A nonempty SHA256 mapping is required')
    for name, expected in mapping.items():
        if not re.fullmatch('[0-9a-f]{64}', str(expected)):
            raise ValueError('Invalid SHA256 digest')
        path = safe_path(name, prefix=prefix)
        if not path.is_file() or sha(path) != expected:
            raise ValueError('Locked source or asset changed: ' + name)


def instant(value):
    result = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if result.tzinfo is None:
        raise ValueError('Lock timestamps must include a time zone')
    return result.timestamp()


def install_data_guard():
    """Initially deny all data except manifest/train; validation opens explicitly."""
    access = {'validation_open': False, 'rejected': []}
    data = (ROOT / 'code/data').resolve()
    allowed_train = {safe_path(name) for name in TRAIN_PATHS}
    allowed_validation = {data / 'gsm8k_validation.jsonl',
                          data / 'candidates/gsm8k_validation.jsonl',
                          data / 'candidates/gsm8k_validation.npz'}

    def hook(event, arguments):
        if event != 'open' or isinstance(arguments[0], int):
            return
        path = Path(os.fsdecode(arguments[0])).resolve()
        if path.is_relative_to(data):
            allowed = allowed_train | (allowed_validation if access['validation_open'] else set())
            if path not in allowed:
                access['rejected'].append(str(path))
                raise PermissionError('Validation is locked, or the requested data is not permitted')
            mode, flags = arguments[1], arguments[2]
            if (isinstance(mode, str) and any(c in mode for c in 'wax+')) or flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC):
                raise PermissionError('Official data cannot be modified')
    sys.addaudithook(hook)
    return access


def validate_lock():
    """No validation input/reference is touched by this function."""
    lock_path = OUT / 'FINALISTS_LOCK.json'
    selection_path, closed_path = OUT / 'FINALISTS_SELECTION.json', OUT / 'SEARCH_CLOSED.json'
    lock, selection, closed = read(lock_path), read(selection_path), read(closed_path)
    if lock.get('schema') != 'mp2-finalists-lock-v3' or lock.get('search_closed') is not True:
        raise ValueError('A sealed V3 finalist lock is required')
    if closed.get('search_closed') is not True or selection.get('schema') != 'mp2-finalists-selection-v3':
        raise ValueError('Train-only search closure and finalist selection are required')
    if sha(selection_path) != lock['selection_sha256'] or sha(closed_path) != lock['search_closed_sha256']:
        raise ValueError('Selection or search closure changed after locking')
    if not instant(closed['created_at_utc']) <= instant(selection['created_at_utc']) <= instant(lock['created_at_utc']) <= time.time():
        raise ValueError('Invalid search/selection/final-lock time order')
    if selection.get('deployment_seed') != SEED:
        raise ValueError('Deployment seed differs from the preregistered seed')
    finalists = lock.get('finalists', [])
    if not 1 <= len(finalists) <= 2:
        raise ValueError('Exactly one or two new validation finalists are permitted')
    names = [row.get('name') for row in finalists]
    if len(set(names)) != len(names) or any(not isinstance(n, str) or re.fullmatch('[a-z0-9_]+', n) is None for n in names):
        raise ValueError('Finalist names must be unique safe identifiers')
    proposed = selection.get('finalists', [])
    if len(proposed) != len(finalists) or {r['name'] for r in proposed} != set(names):
        raise ValueError('Finalist names differ from the train-only selection')
    allowed = selection.get('allowed_final_fit_experiments', [])
    if len(allowed) != len(set(allowed)):
        raise ValueError('Final-fit component identifiers must be unique')
    check_hashes(lock['inference_source_sha256'], prefix='code')
    if not INFERENCE_SOURCE.issubset(lock['inference_source_sha256']):
        raise ValueError('The full inference and confirmation source closure must be locked')
    for name, expected in OFFICIAL_SOURCE.items():
        if lock['inference_source_sha256'].get(name) != expected:
            raise ValueError('The official parser/predictor/scorer must be unchanged')
    check_hashes(lock['asset_sha256'], prefix='artifacts/research_v3')
    manifest = ROOT / 'code/data/manifest.json'
    if sha(manifest) != lock['data_manifest_sha256']:
        raise ValueError('Official manifest changed')
    manifest_value = read(manifest)
    if manifest_value['protocol'] != PROTOCOL or manifest_value['splits']['validation'] != 798:
        raise ValueError('Unexpected validation protocol/count')
    sys.path.insert(0, str(ROOT / 'code/tools'))
    from audit_assets import audit
    expected_assets, used_new_components, resource = set(), set(), {}
    for finalist in finalists:
        method = finalist.get('method')
        if method != 'code/solution_v3.py' or method not in lock['inference_source_sha256']:
            raise ValueError('The reviewed V3 inference entry is required')
        if not isinstance(finalist.get('selector'), dict) or not finalist['selector']:
            raise ValueError('A structured confidence selector identity must be locked')
        checkpoint = safe_path(finalist['checkpoint'], prefix='artifacts/research_v3')
        inventory = audit(checkpoint)
        resource[finalist['name']] = inventory
        actual = {str((checkpoint / row['path']).relative_to(ROOT)).replace('\\', '/'): row['sha256'] for row in inventory['files']}
        expected_assets.update(actual)
        if any(lock['asset_sha256'].get(name) != digest for name, digest in actual.items()):
            raise ValueError('Whole-bundle inventory differs from the asset lock')
        components = finalist.get('components', [])
        if not 1 <= len(components) <= 3:
            raise ValueError('Each finalist needs one to three provenance-bound components')
        if [c['experiment_id'] for c in components] != next(p['component_ids'] for p in proposed if p['name'] == finalist['name']):
            raise ValueError('Finalist component order differs from train-only selection')
        if finalist['selector'] != next(p['selector'] for p in proposed if p['name'] == finalist['name']):
            raise ValueError('Confidence selector differs from train-only selection')
        bundle_config = read(checkpoint / 'config.json')
        if bundle_config.get('schema') not in ('mp2-research-v2', 'mp2-research-v3', 'mp2-research-v3-ensemble'):
            raise ValueError('Unsupported frozen inference bundle schema')
        if bundle_config.get('schema') == 'mp2-research-v3-ensemble' and (
                bundle_config.get('aggregation') != 'equal_group_probability'
                or bundle_config.get('representative') != 'first_model_first_member'):
            raise ValueError('Only the reviewed fixed equal-probability rule is permitted')
        members = bundle_config.get('members') if bundle_config.get('schema') == 'mp2-research-v3-ensemble' else ['.']
        if members != [c.get('bundle_subdir', '.') for c in components]:
            raise ValueError('Actual inference manifest differs from provenance component order')
        for component in components:
            if component.get('kind', 'new') not in ('new', 'historical_v2'):
                raise ValueError('Unknown component provenance kind')
            if component.get('kind', 'new') == 'new' and component.get('seed') != SEED:
                raise ValueError('A component selected another deployment seed')
            subdir = component.get('bundle_subdir', '.')
            target = (checkpoint / subdir).resolve()
            if not target.is_relative_to(checkpoint):
                raise ValueError('Component assets escape the candidate bundle')
            if component.get('kind', 'new') == 'historical_v2':
                freeze_path = safe_path(component['freeze_path'], prefix='results/research_v2')
                if sha(freeze_path) != component['freeze_sha256']:
                    raise ValueError('Historical component freeze changed')
                freeze = read(freeze_path)
                if freeze['selected_experiment_id'] != component['experiment_id']:
                    raise ValueError('Historical component identity differs')
                expected = {str(Path(k).relative_to(freeze['checkpoint'])).replace('\\', '/'): v for k, v in freeze['assets'].items()}
            else:
                used_new_components.add(component['experiment_id'])
                report_path = safe_path(component['final_fit_path'], prefix='results/research_v3')
                if sha(report_path) != component['final_fit_sha256']:
                    raise ValueError('Final training report changed')
                report = read(report_path)
                if report['config']['experiment_id'] != component['experiment_id'] or report['config']['seed'] != SEED:
                    raise ValueError('Final training identity/seed differs')
                provenance = report['provenance']
                if set(provenance['data_files']) != TRAIN_PATHS:
                    raise ValueError('Final training provenance must contain only official train plus manifest')
                check_hashes(provenance['data_files'])
                check_hashes(provenance['source_files'], prefix='code')
                if sha(OUT / 'common_folds.json') != provenance['folds_sha256']:
                    raise ValueError('Final training fold provenance differs')
                if report['config']['family'] != 'linear':
                    info = report['info']
                    if info['context'].get('phase') != 'final-all-train':
                        raise ValueError('Final model was not fitted on complete train')
                    state = Path(info['checkpoint']).resolve()
                    if not state.is_relative_to((ROOT / 'tmp/research_v3/training_states').resolve()) or sha(state) != info['checkpoint_sha256']:
                        raise ValueError('Final training state provenance changed')
                expected = report['asset_sha256']
            if not expected or any(not (target / name).resolve().is_relative_to(target)
                                   or sha(target / name) != digest for name, digest in expected.items()):
                raise ValueError('Inference component differs from its final training/frozen assets')
    if expected_assets != set(lock['asset_sha256']) or used_new_components != set(allowed):
        raise ValueError('Locked assets or allowed training components differ from the exact finalist set')
    baseline_path = ROOT / 'results/research_v2/V2_FREEZE.json'
    if sha(baseline_path) != lock['baseline_freeze_sha256']:
        raise ValueError('Historical fallback freeze changed')
    baseline = read(baseline_path)
    check_hashes(baseline['source_files'], prefix='code')
    check_hashes(baseline['assets'], prefix='artifacts/research_v2')
    metric = baseline['validation']
    if (metric['correct'], metric['n'], metric['hard_correct'], metric['hard_n']) != (740, 798, 297, 355):
        raise ValueError('Historical fallback evidence differs from the preregistration')
    return {'lock': lock, 'lock_sha256': sha(lock_path), 'resources': resource,
            'baseline': baseline, 'manifest': manifest_value}


def alive(identity):
    try:
        process = psutil.Process(identity['pid'])
        return process if (abs(process.create_time() - identity['create_time']) < .01
                           and process.is_running() and process.status() != psutil.STATUS_ZOMBIE) else None
    except psutil.NoSuchProcess:
        return None


def active_training():
    names = {'train.py', 'train_v2.py', 'train_v3.py', 'run_research_v2.py', 'pilot_v3.py',
             'run_campaign_v3.py'}
    found = []
    for process in psutil.process_iter(['pid', 'cmdline', 'create_time']):
        try:
            if process.pid == os.getpid() or Path(process.cwd()).resolve() != ROOT:
                continue
            if any(Path(arg.replace('\\', '/')).name in names for arg in process.info['cmdline'] or []):
                found.append({'pid': process.pid, 'create_time': process.info['create_time'], 'command': process.info['cmdline']})
        except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
            continue
    return found


@contextmanager
def exclusive_evaluation():
    """Use the trainer's cooperative lock, so no new project trainer may enter."""
    if active_training():
        raise RuntimeError('Stop the active project scheduler/trainers before validation timing')
    path = ROOT / 'tmp/research_v3/trainer.lock'
    path.parent.mkdir(parents=True, exist_ok=True)
    process = psutil.Process()
    owner = {'pid': process.pid, 'create_time': process.create_time(), 'command_line': process.cmdline(),
             'nonce': uuid.uuid4().hex, 'created_at_utc': utc(), 'purpose': 'validation_confirmation'}
    if path.exists():
        previous = read(path)
        if alive(previous):
            raise RuntimeError('A trainer or validation coordinator still owns the resource lock')
        path.rename(path.with_name('trainer.stale-validation-' + uuid.uuid4().hex + '.json'))
    atomic(path, owner, exclusive=True)
    try:
        if active_training():
            raise RuntimeError('A project training process appeared before timing')
        yield
    finally:
        if path.exists() and read(path) == owner:
            path.unlink()


def stop_tree(identities):
    """Terminate only the captured PID/creation-time identities, children first."""
    processes = []
    for identity in reversed(list(identities.values())):
        process = alive(identity)
        if process:
            try:
                processes.append(process)
                process.terminate()
            except psutil.NoSuchProcess:
                pass
    _, remaining = psutil.wait_procs(processes, timeout=3)
    for process in remaining:
        identity = identities.get(process.pid)
        if identity and alive(identity):
            process.kill()
    psutil.wait_procs(remaining, timeout=3)


def run_process(command, folder, stem, deadline, *, environment=None):
    """Time and sample the complete Windows launcher/actual-child process tree."""
    stdout, stderr = folder / (stem + '.stdout.log'), folder / (stem + '.stderr.log')
    tick = time.perf_counter()
    identities, peak_threads, peak_total_threads, peak_rss, cpu = {}, 0, 0, 0, {}
    coordinator = psutil.Process()
    timed_out = False
    env = os.environ.copy()
    env.update(THREAD_ENV)
    env['PYTHONUTF8'] = '1'
    env.update(environment or {})
    with stdout.open('xb') as out, stderr.open('xb') as err:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=out, stderr=err)
    try:
        parent = psutil.Process(process.pid)
        identities[parent.pid] = {'pid': parent.pid, 'create_time': parent.create_time()}
    except psutil.NoSuchProcess:
        parent = None
    try:
        while True:
            for known in list(identities.values()):
                ancestor = alive(known)
                if ancestor is None:
                    continue
                try:
                    for child in ancestor.children(recursive=True):
                        identities.setdefault(child.pid, {'pid': child.pid, 'create_time': child.create_time()})
                except psutil.NoSuchProcess:
                    pass
            live, threads, rss = [], 0, 0
            for identity in list(identities.values()):
                child = alive(identity)
                if child:
                    live.append(child)
                    try:
                        threads += child.num_threads()
                        rss += child.memory_info().rss
                        times = child.cpu_times()
                        cpu[child.pid] = max(cpu.get(child.pid, 0.), times.user + times.system)
                    except psutil.NoSuchProcess:
                        pass
            peak_threads, peak_rss = max(peak_threads, threads), max(peak_rss, rss)
            peak_total_threads = max(peak_total_threads, threads + coordinator.num_threads())
            if process.poll() is not None and not live:
                break
            if time.perf_counter() >= deadline:
                timed_out = True
                stop_tree(identities)
                break
            time.sleep(.01)
        returncode = process.wait(timeout=5)
    except BaseException:
        stop_tree(identities)
        raise
    return {'command': command, 'returncode': returncode, 'timed_out': timed_out,
            'wall_seconds': time.perf_counter() - tick, 'process_tree': list(identities.values()),
            'sampled_peak_tree_threads': peak_threads, 'sampled_peak_tree_rss_bytes': peak_rss,
            'sampled_peak_tree_plus_coordinator_threads': peak_total_threads,
            'sampled_cpu_seconds': sum(cpu.values()), 'thread_environment': THREAD_ENV,
            'sampling_note': '10 ms samples include launcher and all observed actual children; CPU/RSS/threads are sampled, not exact integrals.'}


def official_command(script, arguments, resource_output):
    """Avoid the Windows venv launcher; use only explicit active-venv packages."""
    executable = str(Path(getattr(sys, '_base_executable', sys.executable)).resolve())
    packages = sorted({sysconfig.get_paths()[key] for key in ('purelib', 'platlib')})
    return [executable, '-S', '-c', PHASE_WRAPPER, script, str(resource_output), *arguments], {
        'PYTHONPATH': os.pathsep.join(packages), 'PYTHONNOUSERSITE': '1'}


def verify_completed(folder, lock_sha):
    record = read(folder / 'evaluation_complete.json')
    if record['lock_sha256'] != lock_sha or record['name'] != folder.name:
        raise ValueError('Candidate result belongs to another finalist lock')
    for name, expected in record['output_sha256'].items():
        target = (folder / name).resolve()
        if not target.is_relative_to(folder.resolve()) or sha(target) != expected:
            raise ValueError('Completed validation evidence changed')
    return record


def evaluate_candidate(finalist, verified):
    folder = OUT / 'validation' / finalist['name']
    if (folder / 'evaluation_complete.json').exists():
        return verify_completed(folder, verified['lock_sha256'])
    if folder.exists() and any(folder.iterdir()):
        raise RuntimeError('Incomplete validation evidence preserved; review it before resuming this same frozen candidate')
    folder.mkdir(parents=True, exist_ok=True)
    atomic(folder / 'evaluation_started.json', {'created_at_utc': utc(), 'lock_sha256': verified['lock_sha256'],
                                               'name': finalist['name']}, exclusive=True)
    inventory = verified['resources'][finalist['name']]
    processes, elapsed = [], 0.
    if inventory['within_limit']:
        tick = time.perf_counter()
        deadline = tick + LIMIT_SECONDS
        prediction = ['--method', finalist.get('method', 'code/solution_v3.py'),
                      '--checkpoint', finalist['checkpoint'], '--split', 'validation',
                      '--output', str((folder / 'validation-predictions.json').relative_to(ROOT))]
        scoring = ['--split', 'validation',
                   '--predictions', str((folder / 'validation-predictions.json').relative_to(ROOT)),
                   '--output', str((folder / 'validation-score.json').relative_to(ROOT))]
        for stem, script, arguments in [('prediction', 'code/predict.py', prediction),
                                       ('scoring', 'code/score_predictions.py', scoring)]:
            command, environment = official_command(script, arguments, folder / (stem + '-thread-pools.json'))
            result = run_process(command, folder, stem, deadline, environment=environment)
            processes.append(result)
            if result['returncode'] != 0 or result['timed_out']:
                break
        elapsed = time.perf_counter() - tick
    successful = len(processes) == 2 and all(p['returncode'] == 0 and not p['timed_out'] for p in processes)
    if successful:
        score = read(folder / 'validation-score.json')
        prediction = read(folder / 'validation-predictions.json')
        if score.get('split') != 'validation' or score.get('protocol') != PROTOCOL or score.get('n') != 798:
            raise ValueError('Official validation output has the wrong split/protocol/count')
        if prediction.get('split') != 'validation' or prediction.get('protocol') != PROTOCOL:
            raise ValueError('Prediction output has the wrong split/protocol')
        if type(score.get('correct')) is not int or not 0 <= score['correct'] <= 798 or not math.isclose(score['accuracy'], 100 * score['correct'] / 798, abs_tol=1e-12):
            raise ValueError('Official validation score is inconsistent')
    total_threads = max((p['sampled_peak_tree_threads'] for p in processes), default=0)
    monitored_threads = max((p['sampled_peak_tree_plus_coordinator_threads'] for p in processes), default=0)
    libraries = []
    if successful:
        libraries = [read(folder / (stem + '-thread-pools.json')) for stem in ('prediction', 'scoring')]
    library_cap = bool(successful and all(not row['torch_imported'] and all(
        pool.get('num_threads') == 1 for pool in row['thread_pools']) for row in libraries))
    eligible = bool(successful and inventory['within_limit'] and elapsed <= LIMIT_SECONDS
                    and total_threads <= 4 and library_cap)
    timing = {'status': 'passed' if successful else 'failed', 'name': finalist['name'],
              'lock_sha256': verified['lock_sha256'], 'completed_at_utc': utc(),
              'prediction_plus_scoring_seconds': elapsed, 'processes': processes,
              'local_resource_eligible': eligible, 'resource': inventory,
              'sampled_peak_task_process_tree_threads': total_threads,
              'sampled_peak_worker_tree_plus_coordinator_threads': monitored_threads,
              'scientific_library_observations': libraries, 'scientific_library_thread_cap_passed': library_cap,
              'process_schedule': 'Prediction exits completely before independent scoring begins',
              'coordinator_scope': 'External measurement monitor is disclosed separately; it is not submitted prediction/scoring code.',
              'hardware': {'platform': platform.platform(), 'processor': platform.processor(),
                           'logical_cpu_count': psutil.cpu_count(), 'physical_cpu_count': psutil.cpu_count(logical=False),
                           'memory_bytes': psutil.virtual_memory().total, 'python': sys.version},
              'limitation': 'Local measured eligibility only; not certification on the reference Xeon Platinum 8457C.'}
    atomic(folder / 'validation-timing.json', timing, exclusive=True)
    record = {'schema': 'mp2-validation-candidate-complete-v3', 'lock_sha256': verified['lock_sha256'],
              'name': finalist['name'], 'status': timing['status'], 'local_resource_eligible': eligible,
              'prediction_plus_scoring_seconds': elapsed, 'resource': inventory,
              'output_sha256': {p.name: sha(p) for p in sorted(folder.iterdir()) if p.is_file()}}
    atomic(folder / 'evaluation_complete.json', record, exclusive=True)
    return record


def diagnostic_events(finalist, verified, record):
    """Only called after every locked official prediction/scoring has finished."""
    folder = OUT / 'validation' / finalist['name']
    metadata = folder / 'validation-events.json'
    if metadata.exists():
        previous = read(metadata)
        if (previous['selection_lock']['sha256'] != verified['lock_sha256']
                or previous['selector'] != finalist['selector']
                or previous['source_sha256'] != verified['lock']['inference_source_sha256']
                or previous['prediction_sha256'] != sha(folder / 'validation-predictions.json')
                or sha(folder / 'validation-events.npz') != previous['predictions_npz']['sha256']):
            raise ValueError('Saved diagnostic events changed')
        return previous['metrics']
    import numpy as np
    sys.path.insert(0, str(ROOT / 'code'))
    from common import load_inputs, parse_answer, read_jsonl, validate_choices
    from mp2.selection_v2 import answer_groups
    bank, arrays = load_inputs('validation')
    references = read_jsonl(ROOT / 'code/data/gsm8k_validation.jsonl')
    prediction, score = read(folder / 'validation-predictions.json'), read(folder / 'validation-score.json')
    validate_choices(bank, prediction['ids'], prediction['selected'])
    if len(references) != len(bank) or any(r['id'] != b['id'] or r['question'] != b['question'] for r, b in zip(references, bank)):
        raise ValueError('Validation references are misaligned')
    labels = np.asarray([[parse_answer(c['text']) is not None and parse_answer(c['text']) == parse_answer(r['answer'])
                          for c in b['candidates']] for b, r in zip(bank, references)], dtype=bool)
    choices = np.asarray(prediction['selected'], dtype=np.int64)
    correct = labels[np.arange(len(bank)), choices]
    groups = answer_groups(bank)
    hard = np.asarray([len(set(row[row >= 0])) >= 2 for row in groups])
    single = labels.sum(axis=1) == 1
    if int(correct.sum()) != score['correct'] or len(bank) != score['n']:
        raise ValueError('Diagnostic selected-answer event differs from the independent official scorer')
    method_path = safe_path(finalist.get('method', 'code/solution_v3.py'), prefix='code')
    spec = importlib.util.spec_from_file_location('locked_validation_method', method_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    details = module.prediction_details(bank, arrays, safe_path(finalist['checkpoint']))
    if details['chosen'] != prediction['selected']:
        raise ValueError('Diagnostic inference changed the saved official choices')
    probability = np.asarray(details['group_probabilities'], dtype=np.float64)
    selected_group = groups[np.arange(len(bank)), choices]
    confidence = np.zeros(len(bank), dtype=np.float64)
    valid = selected_group >= 0
    confidence[valid] = probability[np.flatnonzero(valid), selected_group[valid]]
    if not np.array_equal(confidence, np.asarray(details['chosen_group_probability'])):
        raise ValueError('Confidence does not describe the actual selected-answer group')
    metrics = {}
    for prefix, mask in [('', np.ones(len(bank), bool)), ('hard_', hard), ('single_positive_', single)]:
        n = int(mask.sum())
        metrics.update({prefix + 'correct': int(correct[mask].sum()), prefix + 'n': n,
                        prefix + 'accuracy': float(100 * correct[mask].mean()) if n else None})
    pending = folder / 'validation-events.pending.npz'
    if (folder / 'validation-events.npz').exists():
        raise RuntimeError('Incomplete diagnostic evidence preserved')
    np.savez(pending, ids=np.asarray(prediction['ids'], dtype=str), choices=choices, selected=choices,
             correct=correct, hard=hard, single_positive=single, confidence=confidence,
             groups=groups, group_probabilities=probability)
    pending.replace(folder / 'validation-events.npz')
    report = {'schema': 'mp2-v3-confidence-events-v1', 'split': 'validation',
              'selector': finalist['selector'], 'metrics': metrics,
              'selection_lock': {'path': 'results/research_v3/FINALISTS_LOCK.json', 'sha256': verified['lock_sha256']},
              'source_sha256': verified['lock']['inference_source_sha256'],
              'prediction_sha256': sha(folder / 'validation-predictions.json'),
              'predictions_npz': {'path': 'validation-events.npz',
                                  'sha256': sha(folder / 'validation-events.npz')},
              'event': 'The locked deployed selector chose a candidate with the correct final numeric answer',
              'usage': 'Post-lock confirmation only; validation must not fit calibration, weights, preprocessing or fusion.'}
    atomic(metadata, report, exclusive=True)
    return metrics


def choose_finalist(results, baseline):
    eligible = [r for r in results if r['local_resource_eligible'] and r.get('validation')
                and r['validation']['hard_correct'] >= 297 and r['validation']['correct'] >= 740
                and (r['validation']['hard_correct'] > 297 or r['validation']['correct'] > 740)]
    if not eligible:
        return {'name': baseline['selected_experiment_id'], 'method': baseline['method'],
                'checkpoint': baseline['checkpoint'], 'historical_fallback': True}
    winner = min(eligible, key=lambda r: (-r['validation']['hard_correct'], -r['validation']['correct'],
                r['resource']['uncompressed_bytes'], r['prediction_plus_scoring_seconds'], r['name']))
    return {key: winner[key] for key in ('name', 'method', 'checkpoint')} | {'historical_fallback': False}


def confirm():
    access = install_data_guard()
    with exclusive_evaluation():
        verified = validate_lock()
        marker = OUT / 'VALIDATION_OPENED.json'
        if marker.exists():
            opened = read(marker)
            if (opened.get('lock_sha256') != verified['lock_sha256']
                    or opened.get('candidate_names') != [x['name'] for x in verified['lock']['finalists']]
                    or opened.get('test_enabled') is not False):
                raise ValueError('Validation was already opened under a different lock')
        else:
            atomic(marker, {'schema': 'mp2-validation-opened-v3', 'created_at_utc': utc(),
                            'lock_sha256': verified['lock_sha256'],
                            'candidate_names': [x['name'] for x in verified['lock']['finalists']],
                            'test_enabled': False}, exclusive=True)
        access['validation_open'] = True
        destination = OUT / 'validation_summary.json'
        if destination.exists():
            result = read(destination)
            if result['lock_sha256'] != verified['lock_sha256']:
                raise ValueError('Final validation summary belongs to another lock')
            for row in verified['lock']['finalists']:
                verify_completed(OUT / 'validation' / row['name'], verified['lock_sha256'])
            for relative, expected in result['evidence_sha256'].items():
                if sha(safe_path(relative, prefix='results/research_v3')) != expected:
                    raise ValueError('Final validation evidence changed')
            return result
        results = []
        for finalist in verified['lock']['finalists']:
            if active_training():
                raise RuntimeError('Unexpected training overlap; do not measure validation')
            record = evaluate_candidate(finalist, verified)
            results.append({**finalist, **record})
        # References are not loaded by this coordinator until all official runs end.
        for result, finalist in zip(results, verified['lock']['finalists']):
            if result['status'] == 'passed':
                result['validation'] = diagnostic_events(finalist, verified, result)
        # Sources/assets must remain the same across evaluation, including resumes.
        if validate_lock()['lock_sha256'] != verified['lock_sha256']:
            raise ValueError('The finalist lock changed during evaluation')
        selection = choose_finalist(results, verified['baseline'])
        evidence = {}
        for finalist in verified['lock']['finalists']:
            folder = OUT / 'validation' / finalist['name']
            for path in sorted(folder.iterdir()):
                if path.is_file():
                    evidence[str(path.relative_to(ROOT)).replace('\\', '/')] = sha(path)
        result = {'schema': 'mp2-validation-confirmation-v3', 'completed_at_utc': utc(),
                  'lock_sha256': verified['lock_sha256'], 'finalists': results, 'selected': selection,
                  'baseline_validation': verified['baseline']['validation'], 'evidence_sha256': evidence,
                  'selection_status': 'Development choice pending independent final contract/export and clean reproduction',
                  'selection_rule': 'Resource eligible; hard>=297 and full>=740 with at least one strict improvement; hard/full/bytes/latency; exact historical tie retains historical checkpoint',
                  'search_remains_closed': True, 'new_test_evaluated': False, 'test_enabled': False,
                  'rejected_data_reads': access['rejected'],
                  'limitation': 'Previously used development holdout; at most two locked new candidates. Local timing does not certify reference hardware.'}
        atomic(destination, result, exclusive=True)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()  # No split, candidate, checkpoint or output override.
    result = confirm()
    print(json.dumps({'selected': result['selected'], 'new_test_evaluated': False,
                      'summary': 'results/research_v3/validation_summary.json'}, indent=2))


if __name__ == '__main__':
    main()
