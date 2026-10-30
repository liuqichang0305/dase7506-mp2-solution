"""Once-only frozen V3 official full-test evaluation; no research feedback.

No candidate, seed, method, checkpoint, split or destination overrides exist.
Partial evidence requires explicit review and is never automatically rerun.
"""
from __future__ import annotations

import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS',
              'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS', 'OMP_THREAD_LIMIT'):
    os.environ[_name] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'

from datetime import datetime, timezone
import json
import math
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code/tools'))
import freeze_v3 as freezing
import confirm_validation_v3 as timing

DEADLINE = datetime(2026, 10, 31, 16, tzinfo=timezone.utc)
OUTPUT_NAMES = ('test-predictions.json', 'test-score.json', 'test-timing.json',
                'test-prediction.stdout.log', 'test-prediction.stderr.log', 'test-prediction-thread-pools.json',
                'test-scoring.stdout.log', 'test-scoring.stderr.log', 'test-scoring-thread-pools.json')


def utcnow():
    return datetime.now(timezone.utc)


def worker_guard(root, phase, output, resource):
    """Prepend a phase-specific data/network audit hook to the shared wrapper."""
    data = (Path(root) / 'code/data').resolve()
    allowed = [data / 'manifest.json', data / 'candidates/gsm8k_test.jsonl', data / 'candidates/gsm8k_test.npz']
    if phase == 'scoring':
        allowed.append(data / 'gsm8k_test.jsonl')
    specification = {'data': str(data), 'allowed': [str(p) for p in allowed],
        'result_roots': [str((Path(root) / part).resolve()) for part in ('results', 'submission')],
        'assets': str((Path(root) / freezing.CHECKPOINT).resolve()),
        'outputs': [str(Path(output).resolve()), str(Path(resource).resolve())],
        'prediction': str((Path(root) / 'submission/v3/test-predictions.json').resolve()), 'phase': phase}
    return """import os,sys,json
from pathlib import Path
_scope=json.loads(%r)
def _phase_guard(event,args):
    if event in ('socket.connect','socket.getaddrinfo'):
        raise PermissionError('Official evaluation is offline')
    if event!='open' or isinstance(args[0],int):return
    p=Path(os.fsdecode(args[0])).resolve()
    mode,flags=args[1],args[2]
    writing=(isinstance(mode,str) and any(c in mode for c in 'wax+')) or flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC)
    if p.is_relative_to(Path(_scope['data'])):
        if str(p) not in _scope['allowed'] or writing:
            raise PermissionError('Data access forbidden in this official phase')
    if p.is_relative_to(Path(_scope['assets'])) and writing:
        raise PermissionError('Frozen inference assets are read-only')
    if any(p.is_relative_to(Path(base)) for base in _scope['result_roots']):
        allowed_output=str(p) in _scope['outputs']
        allowed_prediction=_scope['phase']=='scoring' and str(p)==_scope['prediction'] and not writing
        if not allowed_output and not allowed_prediction:
            raise PermissionError('Prior outcomes/research evidence are unavailable to official workers')
sys.addaudithook(_phase_guard)
""" % json.dumps(specification)


def official_command(root, phase, arguments, resource):
    script = 'code/predict.py' if phase == 'prediction' else 'code/score_predictions.py'
    command, environment = timing.official_command(script, arguments, resource)
    output = Path(root) / 'submission/v3' / ('test-predictions.json' if phase == 'prediction' else 'test-score.json')
    command[3] = worker_guard(root, phase, output, resource) + command[3]
    return command, environment


def verify_complete(root, freeze):
    folder = Path(root) / 'submission/v3'
    record = freezing.read(folder / 'TEST_COMPLETE.json')
    marker = freezing.read(folder / 'TEST_OPENED.json')
    if record.get('schema') != 'mp2-v3-test-complete-v1' or record.get('status') != 'passed' or record.get('local_resource_eligible') is not True:
        raise ValueError('A passing immutable full-test completion is required for reuse')
    if record['freeze_sha256'] != freezing.sha(folder / 'FREEZE.json') or marker['freeze_sha256'] != record['freeze_sha256']:
        raise ValueError('Test evidence belongs to another freeze')
    if marker['method'] != freeze['method'] or marker['checkpoint'] != freeze['checkpoint'] or marker['split'] != 'test':
        raise ValueError('Test marker differs from the frozen method')
    freezing.check_hashes(root, record['output_sha256'])
    return record


def evaluate(root=ROOT, *, verify=freezing.verify_freeze, process_runner=timing.run_process,
             command_builder=official_command, now_fn=utcnow, clock=time.perf_counter):
    """Called under the shared exclusive_evaluation resource lock by main."""
    root, folder = Path(root).resolve(), Path(root).resolve() / 'submission/v3'
    freeze = verify(root)
    if (folder / 'TEST_COMPLETE.json').exists():
        return verify_complete(root, freeze)
    if (folder / 'TEST_OPENED.json').exists() or any((folder / name).exists() for name in OUTPUT_NAMES):
        raise RuntimeError('Partial official-test evidence is preserved; automatic re-evaluation is prohibited')
    if now_fn() >= DEADLINE:
        raise ValueError('Official score deadline passed; no new test run')
    if timing.active_training():
        raise RuntimeError('Project training/scheduler is active; test timing cannot begin')
    marker = {'schema': 'mp2-v3-test-opened-v1', 'opened_at_utc': now_fn().isoformat(),
              'freeze_sha256': freezing.sha(folder / 'FREEZE.json'), 'method': freeze['method'],
              'checkpoint': freeze['checkpoint'], 'split': 'test', 'n': 467,
              'purpose': 'One final official evaluation; results never feed research or tuning',
              'retry_policy': 'Partial run is not automatically repeated; completed evidence is hash-verified only.'}
    freezing.write_new(folder / 'TEST_OPENED.json', marker)
    started = clock()
    process_deadline = started + 30.
    processes = []
    try:
        prediction = ['--method', freeze['method'], '--checkpoint', freeze['checkpoint'], '--split', 'test',
                      '--output', 'submission/v3/test-predictions.json']
        scoring = ['--split', 'test', '--predictions', 'submission/v3/test-predictions.json',
                   '--output', 'submission/v3/test-score.json']
        for phase, arguments in [('prediction', prediction), ('scoring', scoring)]:
            if clock() >= process_deadline or now_fn() >= DEADLINE:
                raise RuntimeError('Shared timing/deadline budget exhausted before the next official phase')
            resource = folder / f'test-{phase}-thread-pools.json'
            command, environment = command_builder(root, phase, arguments, resource)
            result = process_runner(command, folder, 'test-' + phase, process_deadline, environment=environment)
            processes.append(result)
            if result['returncode'] != 0 or result['timed_out']:
                raise RuntimeError('Official process failed or exceeded the shared thirty-second budget')
        elapsed = clock() - started
        # Parse only to check the official output contract; never print IDs/choices.
        score = freezing.read(folder / 'test-score.json')
        prediction = freezing.read(folder / 'test-predictions.json')
        if any(value.get('split') != 'test' or value.get('protocol') != timing.PROTOCOL for value in (score, prediction)):
            raise ValueError('Official output has wrong split/protocol')
        if type(score.get('n')) is not int or score['n'] != 467 or type(score.get('correct')) is not int or not 0 <= score['correct'] <= 467:
            raise ValueError('Official score must cover all 467 questions')
        if not math.isfinite(score['accuracy']) or not math.isclose(score['accuracy'], 100 * score['correct'] / 467, rel_tol=0., abs_tol=1e-12):
            raise ValueError('Official aggregate count/accuracy mismatch')
        if len(prediction['ids']) != 467 or len(set(prediction['ids'])) != 467 or len(prediction['selected']) != 467 or any(type(i) is not int or not 0 <= i < 4 for i in prediction['selected']):
            raise ValueError('Official prediction indices/count are invalid')
        libraries = [freezing.read(folder / f'test-{phase}-thread-pools.json') for phase in ('prediction', 'scoring')]
        library_cap = all(not row['torch_imported'] and sum(pool.get('num_threads', 999) for pool in row['thread_pools']) <= 4
                          and all(pool.get('num_threads') == 1 for pool in row['thread_pools']) for row in libraries)
        peak = max(row['sampled_peak_tree_threads'] for row in processes)
        eligible = elapsed <= 30. and peak <= 4 and library_cap and freeze['resource']['within_limit']
        # Source/assets/gates are rechecked after the one official run; never retrain.
        verify(root)
        report = {'schema': 'mp2-v3-test-timing-v1', 'status': 'passed' if eligible else 'failed',
            'freeze_sha256': marker['freeze_sha256'], 'prediction_plus_scoring_seconds': elapsed,
            'processes': processes, 'sampled_peak_task_process_tree_threads': peak,
            'sampled_peak_worker_tree_plus_coordinator_threads': max(row['sampled_peak_tree_plus_coordinator_threads'] for row in processes),
            'scientific_library_observations': libraries, 'scientific_library_thread_cap_passed': library_cap,
            'local_resource_eligible': eligible, 'resource': freeze['resource'],
            'hardware': {'platform': platform.platform(), 'processor': platform.processor(), 'python': sys.version},
            'measurement': 'Sequential fresh official prediction and independent scoring, including startup/imports/input/assets/output and process wrappers. One shared 30-second deadline.',
            'scope': 'Task process tree has at most four OS threads; external measurement coordinator is separately disclosed.',
            'limitation': 'Local measurement only, not reference Xeon eligibility certification.',
            'data_integrity': 'Unchanged official predictor/scorer verify complete test input/reference file hashes against the frozen original manifest.'}
        freezing.write_new(folder / 'test-timing.json', report)
        if not eligible:
            raise RuntimeError('Official run completed but local resource constraints failed; no automatic retry')
        outputs = {f'submission/v3/{name}': freezing.sha(folder / name) for name in OUTPUT_NAMES}
        outputs['submission/v3/TEST_OPENED.json'] = freezing.sha(folder / 'TEST_OPENED.json')
        complete = {'schema': 'mp2-v3-test-complete-v1', 'status': 'passed', 'local_resource_eligible': True,
                    'completed_at_utc': now_fn().isoformat(), 'freeze_sha256': marker['freeze_sha256'],
                    'method': freeze['method'], 'checkpoint': freeze['checkpoint'], 'split': 'test', 'n': 467,
                    'output_sha256': outputs, 'new_test_evaluated': True,
                    'research_feedback': 'Prohibited; official outputs remain in submission/v3 for final reporting/submission only.'}
        freezing.write_new(folder / 'TEST_COMPLETE.json', complete)
        return complete
    except BaseException as error:
        failure = folder / 'TEST_INCOMPLETE.json'
        if not failure.exists():
            freezing.write_new(failure, {'created_at_utc': now_fn().isoformat(), 'freeze_sha256': marker['freeze_sha256'],
                'exception_type': type(error).__name__, 'processes': processes,
                'status': 'Test already opened. Preserve all outputs; automatic repeat is prohibited.'})
        raise


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()  # Intentionally no override flags.
    with timing.exclusive_evaluation():
        record = evaluate()
    print(json.dumps({'status': record['status'], 'local_resource_eligible': record['local_resource_eligible'],
                      'n': record['n'], 'score_file': 'submission/v3/test-score.json',
                      'new_test_evaluated': True, 'research_feedback': 'prohibited'}, indent=2))


if __name__ == '__main__':
    main()
