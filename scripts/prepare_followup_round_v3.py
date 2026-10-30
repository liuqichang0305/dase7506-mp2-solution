"""Write immutable evidence-led follow-up registrations, without training."""
import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    base = json.loads((ROOT / 'code/configs/research_v3/v3_tabm_k4_s0.json').read_text(encoding='utf-8'))
    variants = [
        ('v3_tabm_k4_lr0003_s0', {'lr': .0003}),
        ('v3_tabm_k4_ema_s0', {'ema_decay': .99}),
        ('v3_tabm_k4_embedding_stats_s0', {'input_features': 'embedding_stats'}),
    ]
    configs = []
    for name, updates in variants:
        config = copy.deepcopy(base)
        config.update(experiment_id=name, **updates)
        configs.append(config)
    spec = {
        'round': 'r05_tabm_optimization_features',
        'hypothesis': 'R02 TabM k4 selected epoch2 in all three inner folds and reached1606/2114 hard; k8 reached1619. Before rejecting the family, independently test a lower learning rate for optimization resolution, same-trajectory EMA for variance, and removal of abs(candidate-minus-question-mean) features for input complexity. Each variant changes only one setting from the existing k4 control; the common three outer folds and fixed deployment seed remain unchanged.',
        'evidence': ['results/research_v3/analysis/r02_complete/analysis.json',
                     'results/research_v3/v3_tabm_k4_s0/summary.json'],
        'configs': configs,
    }
    target = ROOT / 'docs/research_v3/rounds' / (spec['round'] + '.json')
    if target.exists():
        if json.loads(target.read_text(encoding='utf-8')) != spec:
            raise ValueError('Refuse to replace registered hypothesis')
    else:
        target.write_text(json.dumps(spec, indent=2) + '\n', encoding='utf-8')
    print(target.relative_to(ROOT))


if __name__ == '__main__':
    main()
