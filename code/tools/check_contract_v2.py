"""Audit v2 inference offline in a copy without official reference/data files.

Only synthetic or validation inputs may be selected by this audit. The isolated
worker receives in-memory inputs before the read/network/process/write guard.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
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


def admissible_tie_choices(scores, groups, decoder, *, rtol=2e-5, atol=2e-6):
    """Candidate indices consistent with numerically tied actual decisions.

    Group-level and within-winning-group ties are distinct. All-None rows are
    allowed to break equivariance because fallback index 0 is positional.
    """
    import numpy as np
    from mp2.selection_v2 import group_logits
    scores, groups = np.asarray(scores), np.asarray(groups)
    mode, mask = decoder.get('mode', 'candidate_max'), decoder.get('mask', True)
    logits = group_logits(scores, groups, decoder.get('alpha', 0), decoder.get('tau', 1)) if mode == 'group_lse' else None
    choices = []
    for i, (row, group) in enumerate(zip(scores, groups)):
        valid = group >= 0
        if not valid.any() and mask:
            choices.append(set(range(4)))
        elif mode in ('candidate_max', 'group_max'):
            eligible = valid if mask else np.ones(4, dtype=bool)
            maximum = row[eligible].max()
            choices.append(set(np.flatnonzero(eligible & np.isclose(row, maximum, rtol=rtol, atol=atol)).tolist()))
        elif mode == 'group_lse':
            tied_groups = np.flatnonzero(np.isclose(logits[i], logits[i].max(), rtol=rtol, atol=atol))
            allowed = set()
            for gid in tied_groups:
                member = group == gid
                maximum = row[member].max()
                allowed.update(np.flatnonzero(member & np.isclose(row, maximum, rtol=rtol, atol=atol)).tolist())
            choices.append(allowed)
        else:
            raise ValueError(f'Unknown decoder: {mode}')
    return choices


def worker(sandbox):
    import numpy as np
    np.testing.assert_allclose([0.0], [0.0])  # Lazy testing imports before guard.
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
                raise PermissionError(f'External inference read: {path}')
            if 'gsm8k_' in path.name or path.name.lower().startswith('reference'):
                rejected_events.append(f'reference:{path}')
                raise PermissionError('Reference-file read forbidden')
            if (isinstance(mode, str) and any(x in mode for x in 'wax+')) or (isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC)):
                rejected_events.append(f'write:{path}')
                raise PermissionError('Inference must be read-only')

    os.chdir(sandbox)
    sys.path.insert(0, str(sandbox))
    sys.addaudithook(guard)
    spec = importlib.util.spec_from_file_location('audited_solution_v2', sandbox / 'solution_v2.py')
    method = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = method
    spec.loader.exec_module(method)
    from mp2.models_v2 import load_bundle
    from mp2.selection_v2 import answer_groups, selection_details
    config, _ = load_bundle(checkpoint)
    decoder = config['decoder']
    report = {'isolation': 'Copied v2 method/package/parser/assets and input arrays only; no official data/references. Python audit hook rejects external reads, writes, network and subprocesses.',
              'checks': {}, 'candidate_permutation': {}, 'n': len(bank)}

    def predict(rows, values):
        result = method.predict(rows, {'embeddings': values}, checkpoint)
        assert type(result) is list and len(result) == len(rows)
        assert all(type(j) is int and 0 <= j < 4 for j in result)
        return result

    base = predict(copy.deepcopy(bank), embeddings.copy())
    report['checks']['python_int_length_range'] = True
    for n in sorted({0, 1, 3, min(7, len(bank)), len(bank)}):
        assert predict(copy.deepcopy(bank[:n]), embeddings[:n].copy()) == base[:n], 'Batch dependence'
    report['checks']['arbitrary_n_batch_independence'] = True
    changed = copy.deepcopy(bank)
    for i, row in enumerate(changed):
        row['id'] = f'unknown-audit-id-{17*i+13}'
    assert predict(changed, embeddings.copy()) == base, 'ID dependence'
    report['checks']['id_independence'] = True
    order = np.random.default_rng(2026).permutation(len(bank))
    assert predict([copy.deepcopy(bank[int(i)]) for i in order], embeddings[order].copy()) == [base[int(i)] for i in order]
    report['checks']['question_order'] = True
    assert predict(copy.deepcopy(bank), embeddings.copy()) == base
    report['checks']['determinism'] = True

    groups = answer_groups(bank)
    scores = np.asarray(method.score_candidates(bank, {'embeddings': embeddings}, checkpoint))
    assert scores.shape == (len(bank), 4) and np.isfinite(scores).all()
    details = selection_details(scores, groups, **decoder)
    assert details['chosen'] == base
    confidence = np.asarray(method.predict_confidence(bank, {'embeddings': embeddings}, checkpoint))
    np.testing.assert_allclose(confidence, details['chosen_group_probability'], rtol=0, atol=0)
    assert np.isfinite(confidence).all() and ((confidence >= 0) & (confidence <= 1)).all()
    report['checks']['confidence_actual_chosen_group'] = True
    report['all_unparseable_rows'] = details['all_unparseable_rows']
    for i in details['all_unparseable_rows']:
        assert base[i] == 0 and confidence[i] == 0
    valid = groups >= 0
    if decoder.get('mask', True):
        assert all(groups[i, base[i]] >= 0 for i in range(len(bank)) if valid[i].any())
    report['checks']['parseability_mask_and_all_none_fallback'] = True

    permutation = np.array([2, 0, 3, 1])
    moved_bank = copy.deepcopy(bank)
    for row in moved_bank:
        row['candidates'] = [row['candidates'][int(j)] for j in permutation]
        for j, candidate in enumerate(row['candidates']):
            candidate['index'] = j
    moved = predict(moved_bank, embeddings[:, permutation].copy())
    mapped = [int(permutation[j]) for j in moved]
    moved_scores = np.asarray(method.score_candidates(moved_bank, {'embeddings': embeddings[:, permutation]}, checkpoint))
    np.testing.assert_allclose(moved_scores, scores[:, permutation], rtol=2e-5, atol=2e-6)
    allowed = admissible_tie_choices(scores, groups, decoder)
    mismatch = [i for i, (a, b) in enumerate(zip(base, mapped)) if a != b]
    assert all(mapped[i] in allowed[i] for i in mismatch), 'Non-tie candidate permutation failure'
    report['candidate_permutation'] = {'score_equivariance': True, 'mapped_choice_mismatches': len(mismatch),
        'decision_tie_rows': sum(len(x) > 1 for x in allowed), 'all_mismatches_tie_explained': True,
        'note': 'Ties are checked using the actual group decoder and within-group maximum; fixed positional all-None fallback is explicitly exempt from content equivariance.',
        'rtol': 2e-5, 'atol': 2e-6}
    for name, values in [('nonfinite', embeddings[:1].copy()), ('wrong_shape', embeddings[:1, :, :-1])]:
        if name == 'nonfinite':
            values[0, 0, 0] = np.nan
        try:
            method.predict(bank[:1], {'embeddings': values}, checkpoint)
        except (ValueError, TypeError, AssertionError, KeyError):
            report['checks']['reject_' + name] = True
        else:
            raise AssertionError(f'Invalid {name} accepted')
    try:
        method.predict(bank[:1], {'embeddings': embeddings[:1]}, sandbox / 'missing')
    except (ValueError, FileNotFoundError, NotADirectoryError):
        report['checks']['missing_bundle_rejected'] = True
    else:
        raise AssertionError('Missing bundle accepted')
    assert not rejected_events, f'Forbidden operations: {rejected_events}'
    assert 'torch' not in sys.modules, 'Inference imported training-only torch'
    report['checks']['offline_no_references_or_forbidden_access'] = True
    report['checks']['numpy_only_no_torch_import'] = True
    if threadpool_info:
        report['threadpools'] = threadpool_info()
        report['configured_pool_threads_total'] = sum(x['num_threads'] for x in report['threadpools'])
        assert report['configured_pool_threads_total'] <= 4
    else:
        report['threadpools'] = 'threadpoolctl not installed; pool introspection not performed'
    report['status'] = 'passed'
    print(json.dumps(report, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--input-source', choices=('synthetic', 'validation'), default='synthetic')
    p.add_argument('--output', type=Path)
    p.add_argument('--worker', type=Path, help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.worker:
        worker(args.worker)
        return
    if args.checkpoint is None:
        p.error('--checkpoint is required')
    if args.output and args.output.exists():
        raise FileExistsError(f'Audit evidence already exists: {args.output}')
    import numpy as np
    root = args.root.resolve()
    if args.input_source == 'validation':
        sys.path.insert(0, str(root / 'code'))
        from common import load_inputs
        bank, arrays = load_inputs('validation')
        embeddings = arrays['embeddings']
    else:
        bank = []
        for i in range(23):
            texts = ['No final answer can be extracted.'] * 4 if i == 0 else [f'Answer: #### {2 + (i + j) % 3}' for j in range(4)]
            if i in (1, 2):
                texts[0] = 'Unfinished reasoning without a numeric answer.'
            if i == 2:
                texts[1:] = ['#### 2', '#### 4/2', '#### 2.0']
            bank.append({'id': f'synthetic-{i}', 'question': 'Synthetic interface fixture.',
                'candidates': [{'index': j, 'text': text} for j, text in enumerate(texts)]})
        embeddings = np.random.default_rng(7506).normal(size=(len(bank), 4, 1024)).astype('float32')
    checkpoint = (root / args.checkpoint).resolve()
    with tempfile.TemporaryDirectory(prefix='mp2-v2-inference-audit-') as temporary:
        sandbox = Path(temporary)
        for name in ('solution_v2.py', 'common.py'):
            shutil.copy2(root / 'code' / name, sandbox / name)
        shutil.copytree(root / 'code' / 'mp2', sandbox / 'mp2', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        shutil.copytree(checkpoint, sandbox / 'bundle')
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
    report.update({'input_source': args.input_source, 'reference_files_read': False,
        'checkpoint': str(args.checkpoint), 'python': sys.version, 'method_sha256': hashlib.sha256((root/'code/solution_v2.py').read_bytes()).hexdigest(),
        'command': subprocess.list2cmdline([sys.executable, *sys.argv])})
    payload = json.dumps(report, indent=2) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding='utf-8')
    print(payload)
    raise SystemExit(completed.returncode)


if __name__ == '__main__':
    main()
