"""Check custom inference in a copied environment with no data/reference files."""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

THREAD_ENV = {key: '1' for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                                   'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS', 'OMP_THREAD_LIMIT')}
os.environ.update(THREAD_ENV)
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.dont_write_bytecode = True


def worker(sandbox):
    import numpy as np
    # Import the audit harness's own assertion helpers before restricting the method.
    # NumPy lazily imports testing support, which may inspect the host/OS null device.
    np.testing.assert_allclose([0.0], [0.0])
    try:
        from threadpoolctl import threadpool_info
    except ImportError:
        threadpool_info = None
    sandbox = sandbox.resolve()
    bank = json.loads((sandbox / 'inputs.json').read_text(encoding='utf-8'))
    embeddings = np.load(sandbox / 'inputs.npy', allow_pickle=False)
    checkpoint = sandbox / 'bundle'
    allowed_roots = [sandbox, Path(sys.prefix).resolve(), Path(sys.base_prefix).resolve()]
    rejected_events = []

    def guard(event, arguments):
        if event.startswith('socket.') or event in ('subprocess.Popen', 'os.system', 'os.spawn'):
            rejected_events.append(event)
            raise PermissionError(f'Forbidden inference operation: {event}')
        if event == 'open':
            filename, mode, flags = arguments
            if isinstance(filename, int):
                return
            path = Path(os.fsdecode(filename)).resolve()
            if not any(path.is_relative_to(root) for root in allowed_roots):
                rejected_events.append(f'open:{path}')
                raise PermissionError(f'Inference may not read outside copied bundle/runtime: {path}')
            if 'gsm8k_' in path.name or path.name.lower().startswith('reference'):
                rejected_events.append(f'reference:{path}')
                raise PermissionError('Reference/bank file access forbidden inside inference')
            if (isinstance(mode, str) and any(x in mode for x in 'wax+')) or (isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC)):
                rejected_events.append(f'write:{path}')
                raise PermissionError('Inference must be read-only')

    os.chdir(sandbox)
    sys.path.insert(0, str(sandbox))
    sys.addaudithook(guard)
    spec = importlib.util.spec_from_file_location('audited_solution', sandbox / 'solution.py')
    method = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = method
    spec.loader.exec_module(method)
    report = {'isolation': 'Copied method/package/parser/checkpoint only; no official data or references; Python audit hook rejects external file access, networking, subprocesses, writes.',
              'checks': {}, 'candidate_permutation': {}}

    def predict(rows, values):
        result = method.predict(rows, {'embeddings': values}, checkpoint)
        assert type(result) is list, 'Output is not a Python list'
        assert len(result) == len(rows), 'Wrong output length'
        assert all(type(x) is int and 0 <= x <= 3 for x in result), 'Output indices are not Python ints 0..3'
        return result

    baseline = predict(copy.deepcopy(bank), embeddings.copy())
    report['checks']['types_length_range'] = True
    for count in sorted({0, 1, 3, min(7, len(bank)), len(bank)}):
        choices = predict(copy.deepcopy(bank[:count]), embeddings[:count].copy())
        assert choices == baseline[:count], 'Prediction depends on batch size'
    report['checks']['different_n_and_batch_independence'] = True
    changed = copy.deepcopy(bank)
    for i, row in enumerate(changed):
        row['id'] = f'arbitrary-audit-{i * 17 + 13}'
    assert predict(changed, embeddings.copy()) == baseline, 'ID dependence'
    report['checks']['id_independence'] = True
    order = np.random.default_rng(2026).permutation(len(bank))
    shuffled = predict([copy.deepcopy(bank[int(i)]) for i in order], embeddings[order].copy())
    assert shuffled == [baseline[int(i)] for i in order], 'Question-order dependence'
    report['checks']['question_order'] = True
    assert predict(copy.deepcopy(bank), embeddings.copy()) == baseline, 'Nondeterministic inference'
    report['checks']['determinism'] = True

    permutation = np.array([2, 0, 3, 1])
    permuted = copy.deepcopy(bank)
    for row in permuted:
        row['candidates'] = [row['candidates'][int(j)] for j in permutation]
        for index, candidate in enumerate(row['candidates']):
            candidate['index'] = index
    moved = predict(permuted, embeddings[:, permutation].copy())
    mapped = [int(permutation[index]) for index in moved]
    mismatch = [i for i, (a, b) in enumerate(zip(baseline, mapped)) if a != b]
    report['candidate_permutation']['n'] = len(bank)
    report['candidate_permutation']['mapped_choice_mismatches'] = len(mismatch)
    if hasattr(method, 'score_candidates'):
        scores = np.asarray(method.score_candidates(bank, {'embeddings': embeddings}, checkpoint))
        moved_scores = np.asarray(method.score_candidates(permuted, {'embeddings': embeddings[:, permutation]}, checkpoint))
        assert scores.shape == (len(bank), 4) and np.isfinite(scores).all(), 'Invalid candidate scores'
        np.testing.assert_allclose(moved_scores, scores[:, permutation], rtol=2e-5, atol=2e-6)
        maxima = scores.max(axis=1, keepdims=True)
        ties = np.isclose(scores, maxima, rtol=2e-5, atol=2e-6).sum(axis=1) > 1
        assert all(ties[i] for i in mismatch), 'Permutation failure on a unique score maximum'
        report['candidate_permutation'].update({'score_equivariance': True, 'tie_rows': int(ties.sum()),
                                                'note': 'Lowest-index tie breaking can change selected content under a candidate permutation.'})
    else:
        report['candidate_permutation']['note'] = 'No public score API; choices checked but tie attribution requires source review.'
        assert not mismatch, 'Candidate permutation mismatch; inspect deterministic tie behavior'

    # Malformed/nonfinite inputs should fail clearly, not emit a silent prediction.
    invalid = embeddings[:1].copy()
    invalid[0, 0, 0] = np.nan
    for name, values in [('nonfinite', invalid), ('wrong_shape', embeddings[:1, :, :-1])]:
        try:
            method.predict(bank[:1], {'embeddings': values}, checkpoint)
        except (ValueError, TypeError, AssertionError, KeyError):
            report['checks'][f'reject_{name}'] = True
        else:
            raise AssertionError(f'No error on {name} input')
    try:
        method.predict(bank[:1], {'embeddings': embeddings[:1]}, sandbox / 'missing_bundle')
    except (FileNotFoundError, ValueError, NotADirectoryError):
        report['checks']['reject_missing_bundle'] = True
    else:
        raise AssertionError('Missing checkpoint silently accepted')
    report['checks']['no_forbidden_access'] = not rejected_events
    assert not rejected_events, f'Inference attempted forbidden access: {rejected_events}'
    if threadpool_info:
        pools = threadpool_info()
        report['threadpools'] = pools
        report['configured_pool_threads_total'] = sum(pool['num_threads'] for pool in pools)
        assert report['configured_pool_threads_total'] <= 4, 'Native thread pools exceed four'
    else:
        report['threadpools'] = 'threadpoolctl unavailable; native pool verification not performed'
    report['status'] = 'passed'
    print(json.dumps(report, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--input-source', choices=('synthetic', 'validation'), default='synthetic')
    p.add_argument('--output', type=Path)
    p.add_argument('--worker', type=Path, help=argparse.SUPPRESS)
    a = p.parse_args()
    if a.worker:
        worker(a.worker)
        return
    if a.checkpoint is None:
        p.error('--checkpoint is required')
    import numpy as np
    root = a.root.resolve()
    if a.input_source == 'validation':
        sys.path.insert(0, str(root / 'code'))
        from common import load_inputs
        bank, arrays = load_inputs('validation')
        bank, embeddings = bank[:37], arrays['embeddings'][:37]
    else:
        bank = [{'id': f'synthetic-{i}', 'question': f'A synthetic interface fixture {i}.',
                 'candidates': [{'index': j, 'text': f'Fixture with candidate {j}. #### {2 + (i + j) % 3}'} for j in range(4)]}
                for i in range(23)]
        embeddings = np.random.default_rng(7506).normal(size=(len(bank), 4, 1024)).astype('float32')
    with tempfile.TemporaryDirectory(prefix='mp2-inference-audit-') as temporary:
        sandbox = Path(temporary)
        for name in ('solution.py', 'common.py'):
            shutil.copy2(root / 'code' / name, sandbox / name)
        if (root / 'code/mp2').exists():
            shutil.copytree(root / 'code/mp2', sandbox / 'mp2', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        shutil.copytree((root / a.checkpoint).resolve(), sandbox / 'bundle')
        (sandbox / 'inputs.json').write_text(json.dumps(bank), encoding='utf-8')
        np.save(sandbox / 'inputs.npy', embeddings, allow_pickle=False)
        env = os.environ.copy()
        env.update(THREAD_ENV)
        env['PYTHONUTF8'] = '1'
        completed = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--worker', str(sandbox)],
                                   cwd=sandbox, env=env, capture_output=True, text=True, encoding='utf-8', errors='replace')
    if completed.returncode == 0:
        report = json.loads(completed.stdout)
    else:
        report = {'status': 'failed', 'returncode': completed.returncode, 'stdout': completed.stdout, 'stderr': completed.stderr}
    report['input_source'] = a.input_source
    report['reference_files_read'] = False
    payload = json.dumps(report, indent=2) + '\n'
    if a.output:
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(payload, encoding='utf-8')
    print(payload)
    raise SystemExit(completed.returncode)


if __name__ == '__main__':
    main()
