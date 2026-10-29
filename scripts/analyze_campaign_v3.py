"""Recompute a transparent common-fold TRAIN-OOF table. No held-out data access."""
from datetime import datetime, timezone
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code'))
from mp2.selection_v2 import select_candidates


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def bootstrap(delta, seed=3036901059):
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(50):
        means.extend(delta[rng.integers(0, len(delta), (100, len(delta)))].mean(axis=1) * 100)
    return np.quantile(means, [.025, .975]).tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', required=True)
    args = parser.parse_args()
    destination = ROOT / 'results/research_v3/analysis' / args.name
    destination.mkdir(parents=True, exist_ok=False)
    fold_path = ROOT / 'results/research_v3/common_folds.json'
    fold_def = read(fold_path)
    fold_vector = np.empty(fold_def['questions'], dtype=np.int32)
    for split in fold_def['folds']:
        fold_vector[split['holdout']] = split['fold']
    if sha(fold_path) != sha(ROOT / 'results/research_v2/common_folds.json'):
        raise ValueError('V2/V3 question folds differ')
    queue_path = ROOT / 'results/research_v3/campaign_queue.json'
    jobs = read(queue_path)['jobs'] if queue_path.exists() else []
    rows, decisions, standard_labels, standard_groups = [], {}, None, None
    for version in ('research_v2', 'research_v3'):
        for path in sorted((ROOT / 'results' / version).glob('*/summary.json')):
            summary = read(path)
            if 'config' not in summary or len(summary.get('folds', [])) != 3 or summary.get('n') != 4755:
                continue
            config = summary['config']
            name = summary['experiment_id']
            registration = read(path.with_name('registration.json'))
            if version == 'research_v2':
                if registration['folds_sha256'] != sha(fold_path):
                    raise ValueError('Historical fold hash mismatch')
                for relative, expected in {**registration['source_files'], **registration['data_files']}.items():
                    if sha(ROOT / relative) != expected:
                        raise ValueError(f'Historical input changed: {relative}')
            data = np.load(path.with_name('oof.npz'), allow_pickle=False)
            scores, labels, groups, folds = [data[k] for k in ('scores', 'labels', 'groups', 'fold')]
            if not np.array_equal(folds, fold_vector):
                raise ValueError(f'Non-common OOF folds: {name}')
            if standard_labels is None:
                standard_labels, standard_groups = labels.copy(), groups.copy()
            if not np.array_equal(labels, standard_labels) or not np.array_equal(groups, standard_groups):
                raise ValueError('OOF labels/groups disagree between methods')
            decoder = summary.get('oof_decoder', config['decoder'])
            choices = np.asarray(select_candidates(scores, groups, **decoder))
            correct = labels[np.arange(len(labels)), choices].astype(bool)
            hard = groups.max(1) >= 1
            one = labels.sum(1) == 1
            if int(correct.sum()) != summary['correct'] or int(correct[hard].sum()) != summary['hard_correct']:
                raise ValueError(f'Summary does not reproduce from OOF: {name}')
            fold_acc = [100 * correct[hard & (folds == f)].mean() for f in range(3)]
            numeric_bytes = [f.get('numeric_asset_bytes', f.get('uncompressed_bytes', 0)) for f in summary['folds']]
            row = {'version': version, 'experiment_id': name, 'family': config['family'],
                'seed': config['seed'], 'status': 'completed', 'correct': int(correct.sum()), 'n': len(labels),
                'accuracy': 100 * correct.mean(), 'hard_correct': int(correct[hard].sum()), 'hard_n': int(hard.sum()),
                'hard_accuracy': 100 * correct[hard].mean(), 'hard_fold_mean': float(np.mean(fold_acc)),
                'hard_fold_sd': float(np.std(fold_acc, ddof=1)), 'hard_fold_accuracies': fold_acc,
                'single_correct': int(correct[one].sum()), 'single_n': int(one.sum()),
                'training_seconds': summary.get('training_seconds'), 'numeric_asset_bytes': max(numeric_bytes),
                'parameters': max(f.get('parameters', 0) for f in summary['folds']),
                'summary_sha256': sha(path), 'oof_sha256': sha(path.with_name('oof.npz')),
                'config': config, 'decoder': decoder}
            rows.append(row)
            decisions[name] = correct
    rows.sort(key=lambda r: (-r['hard_fold_mean'], -r['accuracy'], r['numeric_asset_bytes'], r['experiment_id']))
    reference = 'attention_regularized_s0'
    if reference in decisions:
        for row in rows:
            if row['version'] == 'research_v3':
                delta = decisions[row['experiment_id']].astype(float) - decisions[reference].astype(float)
                row['paired_vs_v2_attention'] = {'hard_delta_pp': 100 * delta[hard].mean(),
                    'hard_bootstrap_95pct': bootstrap(delta[hard]), 'full_delta_pp': 100 * delta.mean(),
                    'full_bootstrap_95pct': bootstrap(delta)}
    completed = {r['experiment_id'] for r in rows if r['version'] == 'research_v3'}
    pending = [{k: job.get(k) for k in ('experiment_id', 'round', 'status', 'repair_retries')}
               for job in jobs if job['experiment_id'] not in completed]
    report = {'utc': datetime.now(timezone.utc).isoformat(), 'source_sha256': sha(__file__),
        'folds_sha256': sha(fold_path), 'rows': rows, 'noncomplete_jobs': pending,
        'limitations': 'Train OOF is used for selection. Intervals are paired descriptive uncertainty, not selection-unbiased claims. No validation or test inputs read.'}
    (destination / 'analysis.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    lines = ['# Common three-fold train OOF', '', report['limitations'], '',
        '| Version / configuration | Seed | Hard correct | Hard fold mean ± SD (%) | Full correct | Single positive | Params | Bytes | Fit seconds |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        lines.append(f"| {row['version']} / {row['experiment_id']} | {row['seed']} | {row['hard_correct']}/{row['hard_n']} | {row['hard_fold_mean']:.3f} ± {row['hard_fold_sd']:.3f} | {row['correct']}/{row['n']} | {row['single_correct']}/{row['single_n']} | {row['parameters']} | {row['numeric_asset_bytes']} | {row['training_seconds'] if row['training_seconds'] is not None else 'NA'} |")
    lines.extend(['', 'Incomplete jobs are excluded from rankings:', '', '```json', json.dumps(pending, indent=2), '```', ''])
    (destination / 'EXPERIMENT_TABLE.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps({'output': str(destination), 'complete_rows': len(rows), 'v3_rows': len(completed),
        'top': [{k: r[k] for k in ('experiment_id', 'hard_correct', 'hard_n', 'hard_fold_mean', 'correct')} for r in rows[:8]]}, indent=2))


if __name__ == '__main__':
    main()
