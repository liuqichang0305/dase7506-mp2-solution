"""Write immutable evidence-led follow-up registrations, without training."""
import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def save(spec):
    target = ROOT / 'docs/research_v3/rounds' / (spec['round'] + '.json')
    if target.exists():
        if json.loads(target.read_text(encoding='utf-8')) != spec:
            raise ValueError('Refuse to replace registered hypothesis')
    else:
        target.write_text(json.dumps(spec, indent=2) + '\n', encoding='utf-8')
    print(target.relative_to(ROOT))


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
    save(spec)
    attention = json.loads((ROOT / 'code/configs/research_v3/v3_attention_adamw_s0.json').read_text(encoding='utf-8'))
    objectives = []
    for objective in ('listwise', 'bce', 'pairwise'):
        item = copy.deepcopy(attention)
        item.update(experiment_id=f'v3_attention_{objective}_s0', objective=objective)
        objectives.append(item)
    item = copy.deepcopy(attention)
    item.update(experiment_id='v3_attention_group_alpha05_s0', alpha=.5)
    item['decoder']['alpha'] = .5
    objectives.append(item)
    save({'round': 'r06_attention_objectives',
          'hypothesis': 'V3 Attention AdamW reached1682/2114 hard, well above new feed-forward families. Hold that architecture and optimizer fixed while testing original unmasked multi-positive listwise, candidate BCE, question-averaged positive-negative pairwise, and group loss plus matching decoder alpha0.5. These objectives retain every positive. R01 diagnostic single-positive weakness motivates direct discrimination; decoder alignment is changed only in the explicit alpha ablation.',
          'evidence': ['results/research_v3/v3_attention_adamw_s0/summary.json',
                       'docs/research_v3/OOF_DIAGNOSTIC_R1.md'],
          'configs': objectives})
    interactions = []
    for family in ('deepsets', 'resmlp'):
        for ema in (False, True):
            item = copy.deepcopy(attention)
            item.pop('heads', None)
            item.update(experiment_id=f'v3_{family}_' + ('ema_s0' if ema else 'adamw_s0'), family=family)
            if family == 'resmlp':
                item['blocks'] = 1
            if ema:
                item['ema_decay'] = .99
            interactions.append(item)
    save({'round': 'r07_regularized_interactions',
          'hypothesis': 'Historical Deep Sets is the selected validation fallback while the old ResMLP used a different less-regularized recipe. Establish both with the fixed small-width regularized V3 AdamW recipe and independently add same-trajectory EMA within each family. This tests a transferable regularized recipe and averaging, not a pure capacity-matched claim about pooling; actual parameter/asset counts remain explicit.',
          'evidence': ['results/research_v3/analysis/r03_complete/analysis.json',
                       'code/configs/research_v2/deepsets_regularized_s0.json',
                       'code/configs/research_v2/resmlp_group_s0.json'],
          'configs': interactions})
    regularization = []
    for suffix, change in [('wd01', {'weight_decay': .1}),
                           ('drop05', {'dropout': .5}),
                           ('width256', {'width': 256})]:
        item = copy.deepcopy(attention)
        item.update(experiment_id=f'v3_attention_{suffix}_s0', **change)
        regularization.append(item)
    save({'round': 'r09_attention_regularization_capacity',
          'hypothesis': 'Attention remains the strongest family. Historical recipe changes were confounded, and V3 inner curves show training loss continuing to fall after the inner stopping optimum. Resolve regularization versus capacity with three separate one-setting changes from the V3 AdamW control: weight decay0.1, dropout0.5, or width256. Do not infer that any historical gain was caused by weight decay alone.',
          'evidence': ['results/research_v3/v3_attention_adamw_s0/summary.json',
                       'results/research_v3/v3_attention_adamw_s0/fold0-inner-epochs.json'],
          'configs': regularization})
    realmlp = json.loads((ROOT / 'code/configs/research_v3/v3_realmlp_robust_all_s0.json').read_text(encoding='utf-8'))
    resolution = []
    for suffix, change in [('lr0003', {'lr': .003}),
                           ('width256', {'width': 256}),
                           ('embedding_stats', {'input_features': 'embedding_stats'})]:
        item = copy.deepcopy(realmlp)
        item.update(experiment_id=f'v3_realmlp_robust_{suffix}_s0', **change)
        resolution.append(item)
    save({'round': 'r10_realmlp_resolution',
          'hypothesis': 'Robust-all RealMLP improved by14 hard questions over its standard control (1611 versus1597), although it remains behind Attention. Test whether step size, capacity or relation-feature complexity limits the adapted recipe: independently lower LR0.01 to0.003, increase width128 to256, or remove absolute-deviation features. Keep robust preprocessing, NTP/SELU, optimizer multipliers, folds and seed fixed.',
          'evidence': ['results/research_v3/analysis/r03_complete/analysis.json'],
          'configs': resolution})


if __name__ == '__main__':
    main()
