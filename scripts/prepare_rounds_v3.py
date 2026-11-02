"""Materialize the initial preregistered rounds; never executes training."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = {'family': 'mlp', 'width': 128, 'hidden': 64, 'dropout': .3,
    'objective': 'group', 'alpha': 0, 'reduction': 'full', 'optimizer': 'adamw',
    'lr': .0003, 'weight_decay': .001, 'batch_size': 128, 'max_epochs': 60,
    'patience': 10, 'seed': 3036900957, 'preprocess': 'standard',
    'decoder': {'mode': 'group_lse', 'alpha': 0, 'tau': 1, 'mask': True}}


def config(name, **updates):
    return dict(BASE, experiment_id=name, **updates)


ROUNDS = [
    {'round': 'r01_mlp_optimizer',
     'hypothesis': 'Hold MLP architecture, objective and preprocessing fixed; compare AdamW, same-trajectory EMA, and two preregistered Muon hidden-matrix learning rates. This V3 AdamW excludes bias/norm from decay, so the V2 historical recipe is a comparator, not a bitwise control.',
     'configs': [config('v3_mlp_adamw_s0'), config('v3_mlp_ema_s0', ema_decay=.99),
        config('v3_mlp_muon001_s0', optimizer='muon', muon_lr=.01),
        config('v3_mlp_muon002_s0', optimizer='muon', muon_lr=.02)]},
    {'round': 'r02_tabm',
     'hypothesis': 'A parameter-efficient independent-member small TabM may reduce MLP variance. Compare the fixed k4/dropout0.1 baseline with dropout0.3 and k8 separately; average member objectives and group probabilities, never logits.',
     'configs': [config('v3_tabm_k4_s0', family='tabm', k=4, depth=3, dropout=.1, lr=.001),
        config('v3_tabm_k4_drop03_s0', family='tabm', k=4, depth=3, dropout=.3, lr=.001),
        config('v3_tabm_k8_s0', family='tabm', k=8, depth=3, dropout=.1, lr=.001)]},
    {'round': 'r03_realmlp_preprocess',
     'hypothesis': 'RealMLP-inspired NTP/SELU and learned input scaling may benefit this small task. With its appropriate fixed Adam beta2/LR multipliers, compare standard scaling, robust clipping for auxiliary statistics only, and robust clipping for all dimensions. All statistics fit within the fitting partition.',
     'configs': [config('v3_realmlp_standard_s0', family='realmlp', depth=3, dropout=0, lr=.01,
                        beta2=.95, input_scale_lr_multiplier=6, bias_lr_multiplier=.1),
        config('v3_realmlp_robust_stats_s0', family='realmlp', depth=3, dropout=0, lr=.01,
               beta2=.95, input_scale_lr_multiplier=6, bias_lr_multiplier=.1, preprocess='robust_stats'),
        config('v3_realmlp_robust_all_s0', family='realmlp', depth=3, dropout=0, lr=.01,
               beta2=.95, input_scale_lr_multiplier=6, bias_lr_multiplier=.1, preprocess='robust_all')]},
    {'round': 'r04_attention_optimizer_capacity',
     'hypothesis': 'Re-establish the strongest historical interaction family with V3 AdamW, then independently test EMA, Muon at hidden LR0.01, and width64. The fixed control prevents attributing trainer/decay differences to the new optimizer.',
     'configs': [config('v3_attention_adamw_s0', family='attention', heads=4, weight_decay=.01),
        config('v3_attention_ema_s0', family='attention', heads=4, weight_decay=.01, ema_decay=.99),
        config('v3_attention_muon_s0', family='attention', heads=4, weight_decay=.01, optimizer='muon', muon_lr=.01),
        config('v3_attention_small_s0', family='attention', width=64, heads=4, weight_decay=.01)]},
]


if __name__ == '__main__':
    destination = ROOT / 'docs/research_v3/rounds'
    destination.mkdir(parents=True, exist_ok=True)
    for specification in ROUNDS:
        path = destination / (specification['round'] + '.json')
        if path.exists():
            if json.loads(path.read_text(encoding='utf-8')) != specification:
                raise ValueError('Refuse to alter a preregistered round')
        else:
            path.write_text(json.dumps(specification, indent=2) + '\n', encoding='utf-8')
        print(path.relative_to(ROOT))
