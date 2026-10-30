"""Registerable targeted follow-up to the actual reduced-feature TabM gain."""
import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    path = ROOT / 'code/configs/research_v3/v3_tabm_k4_embedding_stats_s0.json'
    base = json.loads(path.read_text(encoding='utf-8'))
    configs = []
    for suffix, updates in [('k8', {'k': 8}), ('ema', {'ema_decay': .99})]:
        item = copy.deepcopy(base)
        item.update(experiment_id=f'v3_tabm_embedding_stats_{suffix}_s0', **updates)
        configs.append(item)
    spec = {'round': 'r11_tabm_reduced_features_variance',
        'hypothesis': 'R05 reduced-feature TabM k4 reached1668/2114 hard, versus1606 for its matched full-relation control, a62-question gain. The representation still retains16 relation/statistic features; this removes only the1024 absolute-deviation coordinates. Independently transfer the two previously measured variance treatments: k8 members, or EMA0.99. Each changes one setting from the new reduced-feature control. No learning-rate/weight sweep or validation feedback.',
        'evidence': ['results/research_v3/v3_tabm_k4_embedding_stats_s0/summary.json',
                     'results/research_v3/v3_tabm_k4_s0/summary.json',
                     'results/research_v3/v3_tabm_k8_s0/summary.json',
                     'results/research_v3/v3_tabm_k4_ema_s0/summary.json'],
        'configs': configs}
    destination = ROOT / 'docs/research_v3/rounds/r11_tabm_reduced_features_variance.json'
    if destination.exists():
        if json.loads(destination.read_text(encoding='utf-8')) != spec:
            raise ValueError('Preserve the preregistered follow-up')
    else:
        destination.write_text(json.dumps(spec, indent=2) + '\n', encoding='utf-8')
    print(destination.relative_to(ROOT))


if __name__ == '__main__':
    main()
