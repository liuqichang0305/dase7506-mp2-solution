"""Run V3 inference in an isolated process with reads/network/writes guarded.

Synthetic inputs are the default. Train inputs are allowed for development;
validation requires the explicit frozen-finalist flag. Test is never supported.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
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


def synthetic_inputs(n=137):
    import numpy as np
    bank = []
    for i in range(n):
        texts = ['No final answer.'] * 4 if i == 0 else [f'#### {2 + (i + j) % 3}' for j in range(4)]
        if i in (1, 2):
            texts[0] = 'Unfinished reasoning.'
        if i == 2:
            texts[1:] = ['#### 2', '#### 4/2', '#### 2.0']
        if i == 3:
            texts = ['#### 2'] * 4
        bank.append({'id': f'synthetic-{i}', 'question': 'Synthetic interface fixture.',
                     'candidates': [{'index': j, 'text': text} for j, text in enumerate(texts)]})
    embeddings = np.random.default_rng(7506).normal(size=(n, 4, 1024)).astype('float32')
    if n > 3:
        embeddings[3] = embeddings[3, 0]  # Exact same-group representative tie.
    return bank, {'embeddings': embeddings}


def install_input_guard(source):
    """Forbid every official-data file except the selected candidate inputs."""
    from common import DATA
    data = DATA.resolve()
    allowed = set() if source == 'synthetic' else {
        data / 'manifest.json', data / 'candidates' / f'gsm8k_{source}.jsonl',
        data / 'candidates' / f'gsm8k_{source}.npz'}
    rejected = []

    def guard(event, arguments):
        if event.startswith('socket.') or event in ('subprocess.Popen', 'os.system', 'os.spawn'):
            # Parent contract audit needs to spawn its isolated worker; its
            # worker installs the stronger process/network guard below.
            if event.startswith('socket.'):
                rejected.append(event)
                raise PermissionError('Audit tools do not access the network')
        if event == 'open' and not isinstance(arguments[0], int):
            path = Path(os.fsdecode(arguments[0])).resolve()
            if path.is_relative_to(data) and path not in allowed:
                rejected.append(str(path))
                raise PermissionError('Only selected candidate inputs are allowed; references/test are forbidden')
    sys.addaudithook(guard)
    return rejected


def reference_decision(raw_scores, groups, member_configs, kind):
    """Independent explicit probability/representative decision for the audit."""
    import numpy as np
    from mp2.selection_v2 import group_probabilities, select_candidates
    probabilities = []
    for raw, config in zip(raw_scores, member_configs):
        decoder = config['decoder']
        members = [raw] if raw.ndim == 2 else [raw[:, j] for j in range(raw.shape[1])]
        probabilities.append(np.mean([group_probabilities(score, groups,
            alpha=decoder.get('alpha', 0), tau=decoder.get('tau', 1)) for score in members], axis=0))
    probability = np.mean(probabilities, axis=0)
    anchor = raw_scores[0] if raw_scores[0].ndim == 2 else raw_scores[0][:, 0]
    valid = groups >= 0
    if kind == 'single' and raw_scores[0].ndim == 2:
        chosen = select_candidates(anchor, groups, **member_configs[0]['decoder'])
    else:
        expanded = np.take_along_axis(probability, np.maximum(groups, 0), axis=1)
        first = np.where(valid, expanded, -np.inf).argmax(axis=1)
        winner = groups[np.arange(len(groups)), first]
        selected = np.where(valid & (groups == winner[:, None]), anchor, -np.inf).argmax(axis=1)
        chosen = [int(x) for x in np.where(valid.any(axis=1), selected, 0)]
    return chosen, probability, np.asarray(anchor, dtype=np.float64)


def admissible_ties(raw_scores, groups, configs, kind, rtol=2e-5, atol=2e-6):
    import numpy as np
    _, probability, anchor = reference_decision(raw_scores, groups, configs, kind)
    choices = []
    raw_max = kind == 'single' and raw_scores[0].ndim == 2 and configs[0]['decoder'].get('mode') in ('candidate_max', 'group_max')
    for row, group, p in zip(anchor, groups, probability):
        valid = group >= 0
        if not valid.any():
            choices.append(set(range(4)))
            continue
        if raw_max:
            eligible = valid if configs[0]['decoder'].get('mask', True) else np.ones(4, bool)
            choices.append(set(np.flatnonzero(eligible & np.isclose(row, row[eligible].max(), rtol=rtol, atol=atol))))
            continue
        active = np.unique(group[valid])
        tied_groups = active[np.isclose(p[active], p[active].max(), rtol=rtol, atol=atol)]
        allowed = set()
        for gid in tied_groups:
            member = group == gid
            allowed.update(np.flatnonzero(member & np.isclose(row, row[member].max(), rtol=rtol, atol=atol)))
        choices.append(allowed)
    return choices


def worker(sandbox):
    import numpy as np
    np.testing.assert_allclose([0.], [0.])
    try:
        from threadpoolctl import threadpool_info
    except ImportError:
        threadpool_info = None
    sandbox = sandbox.resolve()
    bank = json.loads((sandbox / 'inputs.json').read_text(encoding='utf-8'))
    embeddings = np.load(sandbox / 'inputs.npy', allow_pickle=False)
    checkpoint = sandbox / 'bundle'
    allowed_roots = [sandbox, Path(sys.prefix).resolve(), Path(sys.base_prefix).resolve()]
    rejected = []

    def guard(event, arguments):
        if event.startswith('socket.') or event in ('subprocess.Popen', 'os.system', 'os.spawn'):
            rejected.append(event)
            raise PermissionError(f'Forbidden inference operation: {event}')
        if event == 'open':
            filename, mode, flags = arguments
            if isinstance(filename, int):
                return
            path = Path(os.fsdecode(filename)).resolve()
            if not any(path.is_relative_to(root) for root in allowed_roots):
                rejected.append(f'external:{path}')
                raise PermissionError('External inference read forbidden')
            if 'gsm8k_' in path.name or path.name.lower().startswith('reference'):
                rejected.append(f'reference:{path}')
                raise PermissionError('Reference read forbidden')
            if (isinstance(mode, str) and any(x in mode for x in 'wax+')) or (isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC)):
                rejected.append(f'write:{path}')
                raise PermissionError('Inference must be read-only')

    os.chdir(sandbox)
    sys.path.insert(0, str(sandbox))
    sys.addaudithook(guard)
    spec = importlib.util.spec_from_file_location('audited_solution_v3', sandbox / 'solution_v3.py')
    method = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = method
    spec.loader.exec_module(method)
    from mp2.selection_v2 import answer_groups
    loaded = method.load_inference_checkpoint(checkpoint)
    configs = [item[1][0] for item in loaded['members']]
    report = {'n': len(bank), 'kind': loaded['kind'], 'checks': {},
              'isolation': 'Separate process containing copied code, frozen numeric assets and candidate inputs only; no official data/references. Audit hook denies external reads, writes, network and subprocesses.'}

    def predict(rows, values):
        choices = method.predict(rows, {'embeddings': values}, checkpoint)
        assert type(choices) is list and len(choices) == len(rows)
        assert all(type(x) is int and 0 <= x < 4 for x in choices)
        return choices

    base = predict(copy.deepcopy(bank), embeddings.copy())
    report['checks']['python_int_length_range'] = True
    for n in sorted({0, 1, 3, min(7, len(bank)), min(129, len(bank)), len(bank)}):
        assert predict(copy.deepcopy(bank[:n]), embeddings[:n].copy()) == base[:n], 'Batch dependence'
    report['checks']['arbitrary_n_batch_independence'] = True
    changed = copy.deepcopy(bank)
    for i, row in enumerate(changed):
        row['id'] = f'unseen-{9001-i}'
    assert predict(changed, embeddings.copy()) == base, 'ID dependence'
    report['checks']['id_independence'] = True
    order = np.random.default_rng(2026).permutation(len(bank))
    assert predict([copy.deepcopy(bank[int(i)]) for i in order], embeddings[order].copy()) == [base[int(i)] for i in order]
    report['checks']['question_order'] = True
    assert predict(copy.deepcopy(bank), embeddings.copy()) == base
    report['checks']['determinism'] = True

    groups = answer_groups(bank)
    raw = method.score_members(bank, {'embeddings': embeddings}, checkpoint)
    expected, probability, anchor = reference_decision(raw, groups, configs, loaded['kind'])
    assert base == expected, 'Inference differs from explicit member probability average'
    confidence = method.predict_confidence(bank, {'embeddings': embeddings}, checkpoint)
    selected_group = groups[np.arange(len(groups)), np.asarray(base, dtype=int)]
    expected_confidence = np.zeros(len(groups))
    good = selected_group >= 0
    expected_confidence[good] = probability[np.flatnonzero(good), selected_group[good]]
    np.testing.assert_allclose(confidence, expected_confidence, rtol=0, atol=0)
    assert np.isfinite(confidence).all() and ((confidence >= 0) & (confidence <= 1)).all()
    report['checks']['member_probability_average_and_actual_chosen_confidence'] = True
    all_none = np.flatnonzero((groups < 0).all(axis=1))
    masked = loaded['kind'] == 'ensemble' or configs[0]['decoder'].get('mask', True)
    if masked:
        assert all(base[int(i)] == 0 and confidence[int(i)] == 0 for i in all_none)
        assert all(groups[i, base[i]] >= 0 for i in range(len(bank)) if (groups[i] >= 0).any())
    report['checks']['parseability_mask_and_all_none_fallback'] = True
    report['all_unparseable_rows'] = all_none.tolist()

    permutation = np.array([2, 0, 3, 1])
    moved_bank = copy.deepcopy(bank)
    for row in moved_bank:
        row['candidates'] = [row['candidates'][int(j)] for j in permutation]
        for j, candidate in enumerate(row['candidates']):
            candidate['index'] = j
    moved = predict(moved_bank, embeddings[:, permutation].copy())
    mapped = [int(permutation[j]) for j in moved]
    moved_raw = method.score_members(moved_bank, {'embeddings': embeddings[:, permutation]}, checkpoint)
    for before, after in zip(raw, moved_raw):
        assert before.shape[0] == len(bank) and before.shape[-1] == 4 and np.isfinite(before).all()
        np.testing.assert_allclose(after, before[..., permutation], rtol=2e-5, atol=2e-6)
    allowed = admissible_ties(raw, groups, configs, loaded['kind'])
    mismatch = [i for i, (a, b) in enumerate(zip(base, mapped)) if a != b]
    assert all(mapped[i] in allowed[i] for i in mismatch), 'Non-tie candidate permutation failure'
    report['candidate_permutation'] = {'raw_score_equivariance': True, 'mapped_choice_mismatches': len(mismatch),
        'tie_rows': sum(len(x) > 1 for x in allowed), 'all_mismatches_tie_explained': True,
        'note': 'Ties use actual averaged group probabilities and first-model/first-member representative scores; positional all-None fallback is exempt.', 'rtol': 2e-5, 'atol': 2e-6}
    for name, values in [('nonfinite', embeddings[:1].copy()), ('wrong_shape', embeddings[:1, :, :-1]), ('misaligned', embeddings[:0])]:
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
    assert not rejected, f'Forbidden operations: {rejected}'
    assert 'torch' not in sys.modules, 'Inference imported training-only Torch'
    report['checks']['offline_no_references_or_forbidden_access'] = True
    report['checks']['numpy_only_no_torch_import'] = True
    report['rejected_operations'] = rejected
    if threadpool_info:
        report['threadpools'] = threadpool_info()
        report['configured_pool_threads_total'] = sum(x['num_threads'] for x in report['threadpools'])
        assert report['configured_pool_threads_total'] <= 4
    else:
        report['threadpools'] = 'Introspection unavailable; environment limits recorded.'
    report['thread_environment'] = {key: os.environ[key] for key in THREAD_ENV}
    report['status'] = 'passed'
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--input-source', choices=('synthetic', 'train', 'validation'), default='synthetic')
    parser.add_argument('--allow-frozen-validation', action='store_true')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--worker', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        worker(args.worker)
        return
    if args.checkpoint is None:
        parser.error('--checkpoint is required')
    if args.input_source == 'validation' and not args.allow_frozen_validation:
        parser.error('Validation requires a frozen finalist and --allow-frozen-validation')
    if args.output and args.output.exists():
        raise FileExistsError('Refusing to overwrite audit evidence')
    root = args.root.resolve()
    sys.path.insert(0, str(root / 'code'))
    import numpy as np
    from common import load_inputs
    from solution_v3 import checkpoint_asset_paths
    rejected_inputs = install_input_guard(args.input_source)
    bank, arrays = synthetic_inputs() if args.input_source == 'synthetic' else load_inputs(args.input_source)
    checkpoint = (root / args.checkpoint).resolve()
    assets = checkpoint_asset_paths(checkpoint)
    hashes = {str(path.relative_to(checkpoint)): hashlib.sha256(path.read_bytes()).hexdigest() for path in assets}
    with tempfile.TemporaryDirectory(prefix='mp2-v3-inference-audit-') as temporary:
        sandbox = Path(temporary)
        for name in ('solution_v3.py', 'common.py'):
            shutil.copy2(root / 'code' / name, sandbox / name)
        shutil.copytree(root / 'code' / 'mp2', sandbox / 'mp2', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        for path in assets:
            destination = sandbox / 'bundle' / path.relative_to(checkpoint)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
        (sandbox / 'inputs.json').write_text(json.dumps(bank), encoding='utf-8')
        np.save(sandbox / 'inputs.npy', arrays['embeddings'], allow_pickle=False)
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
        'timestamp_utc': datetime.now(timezone.utc).isoformat(), 'rejected_input_reads': rejected_inputs,
        'checkpoint': str(args.checkpoint), 'assets_sha256': hashes, 'python': sys.version,
        'method_sha256': hashlib.sha256((root / 'code/solution_v3.py').read_bytes()).hexdigest(),
        'command': subprocess.list2cmdline([sys.executable, *sys.argv])})
    payload = json.dumps(report, indent=2) + '\n'
    if args.output:
        if args.output.resolve().is_relative_to(checkpoint):
            raise ValueError('Write evidence outside frozen checkpoints')
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding='utf-8')
    print(payload)
    raise SystemExit(completed.returncode)


if __name__ == '__main__':
    main()
