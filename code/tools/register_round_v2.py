"""Register the bounded, train-OOF-driven follow-up rounds before fitting."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / 'results/research_v2'


def read_results():
    return [json.loads(p.read_text()) for p in RESULTS.glob('*/summary.json')
            if 'config' in json.loads(p.read_text())]


def rank(row):
    return row['hard_accuracy_fold_mean'], row['accuracy'], -row['folds'][0]['parameters']


def best_families(rows):
    best = {}
    for row in rows:
        family = row['config']['family']
        if family == 'linear' or row['config']['seed'] != 3036900957:
            continue
        if family not in best or rank(row) > rank(best[family]):
            best[family] = row
    return sorted(best.values(), key=rank, reverse=True)[:2]


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('round', choices=('regularization', 'stability', 'ablations'))
    a = p.parse_args()
    if (RESULTS / 'FINALISTS_LOCK.json').exists():
        raise RuntimeError('Selection locked; no new registration')
    output = RESULTS / (a.round + '-registration.json')
    config_list = RESULTS / (a.round + '-configs.json')
    if output.exists() or config_list.exists():
        raise FileExistsError('Round already registered')
    rows = read_results()
    configs = []
    if a.round == 'regularization':
        selected = best_families(rows)
        for row in selected:
            cfg = row['config'].copy()
            cfg.update(experiment_id=cfg['family'] + '_regularized_s0', dropout=.3, lr=.0003, weight_decay=.01)
            configs.append(cfg)
    elif a.round == 'stability':
        selected = best_families(rows)
        for row in selected:
            for seed_index in (1, 2):
                cfg = row['config'].copy()
                cfg.update(experiment_id=cfg['experiment_id'].removesuffix('_s0') + f'_s{seed_index}', seed=3036900957 + seed_index)
                configs.append(cfg)
    else:
        previous = json.loads((RESULTS / 'stability-registration.json').read_text())
        selected = [next(row for row in rows if row['experiment_id'] == item['experiment_id']) for item in previous['selected']]
        def stable_score(row):
            stem = row['experiment_id'].removesuffix('_s0')
            matches = [r for r in rows if r['experiment_id'] in [stem + f'_s{i}' for i in range(3)]]
            if len(matches) != 3:
                raise ValueError('Complete all three fixed seeds before ablations')
            return sum(r['hard_accuracy_fold_mean'] for r in matches) / 3
        selected.sort(key=stable_score, reverse=True)
        cfg = selected[0]['config'].copy()
        cfg.update(experiment_id=cfg['family'] + '_hard_ablation', reduction='hard', scale_regularization=True)
        configs.append(cfg)
        deep = next((row for row in selected if row['config']['family'] == 'deepsets'), None)
        if deep:
            cfg = deep['config'].copy()
            cfg.update(experiment_id='deepsets_no_context', context=False)
            configs.append(cfg)
    record = {'registered_at_utc': datetime.now(timezone.utc).isoformat(), 'round': a.round,
              'selection_data': 'train common OOF only', 'validation_opened_for_new_models': False,
              'selected': [{'experiment_id': r['experiment_id'], 'hard_fold_mean': r['hard_accuracy_fold_mean'],
                            'overall_accuracy': r['accuracy']} for r in selected], 'configs': configs,
              'ablation_finalist_eligibility': False if a.round == 'ablations' else None}
    output.write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
    config_list.write_text(json.dumps(configs, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(record))
