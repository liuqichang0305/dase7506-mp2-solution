"""Read-only train-OOF coverage/provenance audit; no model/data fitting."""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code'))

def guard(event, args):
    if event != 'open' or not isinstance(args[0], (str, bytes, os.PathLike)):
        return
    value = str(Path(os.fsdecode(args[0])).resolve()).replace('\\', '/').lower()
    relative = value.removeprefix(str(ROOT).replace('\\', '/').lower() + '/')
    if 'leaderboard' in relative or 'ranking_background' in relative:
        raise PermissionError('Ranking access forbidden in current train-only review')
    if relative.startswith('code/data/') and Path(relative).name not in ('manifest.json', 'gsm8k_train.jsonl', 'gsm8k_train.npz'):
        raise PermissionError('Only training data may be hash checked')
    if relative.startswith(('results/research_v3/validation', 'results/research_v3/selected_confidence', 'submission/v3/test-')):
        raise PermissionError('Real holdout evidence is out of scope')

sys.addaudithook(guard)
import numpy as np
from mp2.selection_v2 import select_candidates
from mp2.research_v3.selection import select_raw, collapse_scores

CACHE = {}
def sha(path):
    path = Path(path).resolve()
    if path not in CACHE:
        h = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                h.update(chunk)
        CACHE[path] = h.hexdigest()
    return CACHE[path]

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def verify_mapping(mapping):
    for name, expected in mapping.items():
        assert sha(ROOT / name) == expected, name

def difference(first, second):
    return {key: {'first': first.get(key), 'second': second.get(key)}
            for key in sorted(set(first) | set(second))
            if key != 'experiment_id' and first.get(key) != second.get(key)}

def main():
    instant = datetime.now(timezone.utc)
    destination = ROOT / ('results/research_v3/audit_coverage_r05_' + instant.strftime('%Y%m%dT%H%M%SZ'))
    destination.mkdir(exist_ok=False)
    queue_bytes = (ROOT / 'results/research_v3/campaign_queue.json').read_bytes()
    ledger_bytes = (ROOT / 'results/research_v3/round_decisions.jsonl').read_bytes()
    (destination / 'campaign_queue.snapshot.json').write_bytes(queue_bytes)
    (destination / 'round_decisions.snapshot.jsonl').write_bytes(ledger_bytes)
    queue = json.loads(queue_bytes)
    ledger = [json.loads(line) for line in ledger_bytes.decode('utf-8-sig').splitlines() if line.strip()]
    jobs = queue['jobs']
    completed = [job for job in jobs if job['status'] == 'completed']
    current = []
    reference_arrays = None
    correctness = {}
    source_identity = None
    data_identity = None
    config_index = {job['experiment_id']: job['config'] for job in jobs}
    folds = read(ROOT / 'results/research_v3/common_folds.json')
    fold_list = folds['folds'] if isinstance(folds, dict) else folds
    expected_fold = np.full(4755, -1, dtype=np.int64)
    for fold in fold_list:
        expected_fold[fold['holdout']] = fold['fold']
    assert np.all(expected_fold >= 0)
    for job in completed:
        name = job['experiment_id']; folder = ROOT / 'results/research_v3' / name
        summary = read(folder / 'summary.json'); registration = read(folder / 'registration.json')
        assert summary['status'] == 'complete' and summary['config'] == job['config'] == registration['config']
        assert summary['n'] == 4755 and [f['fold'] for f in summary['folds']] == [0, 1, 2]
        provenance = summary['provenance']
        for key in ('source_files', 'data_files', 'folds_sha256'):
            assert provenance[key] == registration[key]
        verify_mapping(provenance['source_files']); verify_mapping(provenance['data_files'])
        assert sha(ROOT / 'results/research_v3/common_folds.json') == provenance['folds_sha256']
        if source_identity is None:
            source_identity = provenance['source_files']; data_identity = provenance['data_files']
        assert source_identity == provenance['source_files'] and data_identity == provenance['data_files']
        assert sha(folder / 'oof.npz') == summary['oof_sha256']
        with np.load(folder / 'oof.npz', allow_pickle=False) as archive:
            labels, groups, held = (archive[k].copy() for k in ('labels', 'groups', 'fold'))
            raw, scores = archive['raw_scores'], archive['scores']
            assert labels.shape == groups.shape == scores.shape == (4755, 4)
            assert np.array_equal(held, expected_fold)
            assert np.isin(labels, (0, 1)).all() and (labels.sum(1) > 0).all()
            assert np.isfinite(raw).all() and np.isfinite(scores).all()
            if reference_arrays is None:
                reference_arrays = (labels, groups, held)
            assert all(np.array_equal(a, b) for a, b in zip(reference_arrays, (labels, groups, held)))
            direct = np.asarray(select_raw(raw, groups, job['config']['decoder']))
            collapsed = collapse_scores(raw, groups, job['config']['decoder'])
            assert np.array_equal(collapsed, scores)
            saved = np.asarray(select_candidates(scores, groups, **summary['oof_decoder']))
            assert np.array_equal(direct, saved)
            correct = labels[np.arange(len(labels)), direct].astype(bool)
        hard = groups.max(1) >= 1; single = labels.sum(1) == 1
        for mask, prefix in ((np.ones(4755, bool), ''), (hard, 'hard_'), (single, 'single_positive_')):
            assert int(mask.sum()) == summary[prefix + 'n']
            assert int(correct[mask].sum()) == summary[prefix + 'correct']
        fold_counts = []
        for fold in summary['folds']:
            number = fold['fold']; mask = held == number
            assert fold['status'] == 'complete' and fold['n'] == 1585
            assert int(correct[mask].sum()) == fold['correct']
            assert int(correct[mask & hard].sum()) == fold['hard_correct']
            assert fold['fit']['best_epoch'] == fold['inner_stopping']['best_epoch']
            assert fold['fit']['torch_threads'] == fold['inner_stopping']['torch_threads'] == 1
            for field in ('source_files', 'data_files', 'folds_sha256'):
                assert fold['provenance'][field] == provenance[field]
            for file, expected in fold['output_sha256'].items():
                assert sha(folder / file) == expected
            for file, expected in fold['asset_sha256'].items():
                assert sha(ROOT / fold['checkpoint'] / file) == expected
            fold_counts.append({'fold': number, 'hard_correct': fold['hard_correct'], 'hard_n': fold['hard_n'],
                                'single_correct': int(correct[mask & single].sum()),
                                'best_epoch': fold['inner_stopping']['best_epoch']})
        correctness[name] = correct
        current.append({'experiment_id': name, 'round': job['round'], 'family': job['config']['family'],
                        'seed': job['config']['seed'], 'hard_correct': summary['hard_correct'],
                        'correct': summary['correct'], 'single_positive_correct': summary['single_positive_correct'],
                        'hard_fold_mean': summary['hard_accuracy_fold_mean'],
                        'hard_fold_sd': summary['hard_accuracy_fold_sd'], 'folds': fold_counts,
                        'max_parameters': max(f['parameters'] for f in summary['folds']),
                        'max_numeric_asset_bytes': max(f['numeric_asset_bytes'] for f in summary['folds']),
                        'summary_sha256': sha(folder / 'summary.json'), 'oof_sha256': summary['oof_sha256'],
                        'full_three_fold_identity_scores_and_asset_hashes_passed': True})
    diagnosis = ROOT / 'results/research_v3/diagnostic-r05-reduced-tabm-vs-attention'
    completion = read(diagnosis / 'COMPLETION.json')
    for key in ('source_sha256', 'input_sha256', 'output_sha256'):
        verify_mapping(completion[key])
    paired = read(diagnosis / 'paired-and-slices.json')
    names = ['v3_tabm_k4_embedding_stats_s0', 'v3_tabm_k4_s0', 'v3_attention_ema_s0']
    a, b, c = [correctness[name] for name in names]
    assert difference(config_index[names[0]], config_index[names[1]]) == {'input_features': {'first': 'embedding_stats', 'second': None}}
    expected_pair = paired['matched_feature_ablation']['paired_comparison']
    for subset, mask in (('overall', np.ones(4755, bool)), ('hard', hard)):
        observed = {'first_correct': int(a[mask].sum()), 'second_correct': int(b[mask].sum()),
                    'correct_difference': int(a[mask].sum() - b[mask].sum()),
                    'first_only_correct': int((a & ~b & mask).sum()), 'second_only_correct': int((b & ~a & mask).sum()),
                    'both_wrong': int((~a & ~b & mask).sum()), 'both_correct': int((a & b & mask).sum())}
        assert all(expected_pair[subset][k] == v for k, v in observed.items())
    gate = paired['only_reciprocal_gate']
    assert gate['hard']['member_unique_correct'] == [int((a & ~c & hard).sum()), int((c & ~a & hard).sum())]
    assert gate['hard']['oracle_correct'] == int(((a | c) & hard).sum())
    assert completion['fusion_run'] is False and paired['fusion']['status'] == 'not_run'
    rounds = defaultdict(Counter)
    for job in jobs:
        rounds[job['round']][job['status']] += 1
    registered_followups = []
    for path in (ROOT / 'docs/research_v3/rounds').glob('r*.json'):
        item = read(path)
        registered_followups.append({'path': path.relative_to(ROOT).as_posix(), 'sha256': sha(path),
                                    'round': item['round'], 'configurations': len(item['configs']),
                                    'not_in_queue': [c['experiment_id'] for c in item['configs'] if c['experiment_id'] not in config_index]})
    matches = {}
    control = config_index['v3_attention_adamw_s0']
    for job in jobs:
        if job['round'] == 'r06_attention_objectives':
            matches[job['experiment_id']] = difference(job['config'], control)
    record = {'scope': 'Read-only source/saved train-OOF audit. No training, model inference, new bootstrap, validation/test/leaderboard access.',
        'created_at_utc': instant.isoformat(), 'queue_updated_at_utc': queue['updated_at_utc'],
        'queue_snapshot_sha256': sha(destination / 'campaign_queue.snapshot.json'),
        'ledger_snapshot_sha256': sha(destination / 'round_decisions.snapshot.jsonl'),
        'round_status': {k: dict(v) for k, v in rounds.items()},
        'completed_configurations': len(current), 'completed_rows': current,
        'shared_source_files': source_identity, 'shared_data_files': data_identity,
        'all_completed_oof_label_group_fold_arrays_identical': True,
        'r05_diagnostic_source_input_output_hash_counts': {key: len(completion[key]) for key in ('source_sha256', 'input_sha256', 'output_sha256')},
        'r05_diagnostic_completion_sha256': sha(diagnosis / 'COMPLETION.json'),
        'r05_verified_counts': {'reduced_correct': int(a.sum()), 'full_correct': int(b.sum()), 'attention_correct': int(c.sum()),
            'reduced_only_vs_full': int((a & ~b).sum()), 'full_only_vs_reduced': int((b & ~a).sum()),
            'fold_differences': [int(a[held == i].sum() - b[held == i].sum()) for i in range(3)],
            'single_fold_differences': [int(a[(held == i) & single].sum() - b[(held == i) & single].sum()) for i in range(3)],
            'reciprocal_gate': gate},
        'attention_objective_configuration_differences': matches,
        'registered_round_documents': registered_followups,
        'live_queue_changed_since_snapshot': (ROOT / 'results/research_v3/campaign_queue.json').read_bytes() != queue_bytes,
        'holdout_markers_absent': not any((ROOT / name).exists() for name in ('results/research_v3/VALIDATION_OPENED.json', 'submission/v3/TEST_OPENED.json')),
        'passed': True}
    (destination / 'review.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'path': destination.relative_to(ROOT).as_posix(), 'completed': len(current),
                      'round_status': record['round_status'], 'passed': True,
                      'objective_results': [r for r in current if r['round'] == 'r06_attention_objectives']}, indent=2))

if __name__ == '__main__':
    main()
