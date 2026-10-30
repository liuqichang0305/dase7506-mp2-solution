"""Consolidate recorded OOF, limited confirmation and the rejected mixture."""
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/research_v2'
sys.path.insert(0, str(ROOT / 'code'))
from mp2.models_v2 import parameter_count

analysis = json.loads((OUT / 'analysis-final/analysis.json').read_text(encoding='utf-8'))
confirmed = json.loads((OUT / 'validation_summary.json').read_text(encoding='utf-8'))
finalists = {r['experiment_id']: r for r in confirmed['finalists']}


def metric(value):
    return f"{value['fold_mean']:.3f} ± {value['fold_sd']:.3f}; {value['correct']}/{value['n']}"


lines = ['# Actual performance-v2 experiment table', '',
    'Every fitted row uses the same three outer question folds; ± is sample SD across folds. '
    'All seeds and negative results are retained. Decoder-only rows reuse fits and cost. '
    'Hyperparameter-selected OOF is not an unbiased final generalization estimate.', '',
    '| Actual experiment / decoder | Full CV %; correct/n | Hard CV %; correct/n | Structural parameter count | Measured weight parameters | Audited MiB | Fit wall seconds | Validation full; hard; single-positive | Fresh prediction + scoring s |',
    '|---|---|---|---:|---:|---:|---:|---|---:|']
for row in analysis['rows']:
    cfg = json.loads((OUT / row['experiment_id'] / 'registration.json').read_text(encoding='utf-8'))['config']
    validation, timing = '按协议未评估', '—'
    if row['row_id'] in finalists:
        value = finalists[row['row_id']]
        v = value['validation']
        validation = f"{v['correct']}/{v['n']}; {v['hard_correct']}/{v['hard_n']}; {v['single_positive_correct']}/{v['single_positive_n']}"
        timing = f"{value['prediction_plus_scoring_seconds']:.4f}"
    elif row['row_id'] == 'linear_original':
        validation = '724/798; 282/355; 59/117 (v1 A0 replay)'
        replay = json.loads((OUT / 'baseline_reproduction/validation/validation-timing.json').read_text(encoding='utf-8'))
        timing = f"{replay['total_wall_seconds']:.4f}"
    lines.append(f"| {row['row_id']} | {metric(row['metrics']['overall'])} | {metric(row['metrics']['hard'])} | {parameter_count(cfg):,} | {max(row['parameters_per_fold']):,} | {row['audited_uncompressed_mib_max']:.4f} | {row['training_seconds']:.3f} | {validation} | {timing} |")
mixture = json.loads((OUT / 'ensemble-selected/summary.json').read_text(encoding='utf-8'))
members = [next(r for r in analysis['rows'] if r['row_id'] == name) for name in (mixture['first'], mixture['second'])]
params = sum(max(r['parameters_per_fold']) for r in members)
size = sum(r['audited_uncompressed_mib_max'] for r in members)
cost = sum(r['training_seconds'] for r in members)
lines.append(f"| Equal group-probability mixture (rejected) | {metric(mixture['metrics']['overall'])} | {metric(mixture['metrics']['hard'])} | {params:,} | {params:,} | {size:.4f} members only | {cost:.3f} reused | 按协议未评估 | — |")
lines.append('| TabM | 未运行 | 未运行 | — | — | — | — | 按协议未评估 | — |')
lines.extend(['', 'Structural counts are architecture arithmetic, not estimated accuracy; measured parameters come from saved numeric state arrays. '
    'MiB includes configuration and uncompressed NPZ accounting for one fold bundle; the mixture is the actual sum of its two members and was never packaged/deployed. '
    'Final selected bundle is independently audited at963,670bytes. All per-fold raw bytes/costs are in analysis-final/analysis.json.', '',
    'The common-fold fit total is ' + f"{analysis['total_actual_training_seconds_distinct_experiments']:.3f}" +
    ' wall seconds, including the documented long interval in attention_regularized_s1. It is not CPU-only time; see docs/research_v2/BUDGET_DEVIATION.md.', '',
    'The selected Deep Sets validation gain over frozen v1 is16questions (+2.005percentage points), '
    'question-paired bootstrap95% interval[0.376,3.512]points. Hard gain is15questions (+4.225points), interval[0.845,7.887]. '
    'These intervals do not remove model-selection bias. Attention has the higher seed-averaged train OOF result; Deep Sets wins this confirmation by only2questions. '
    'Neither result establishes a new test score or rank.', '',
    'Negative controls: group-training did not beat merely decoding old linear scores by group; MLP group loss did not beat original listwise; '
    'hard normalization did not improve Attention; removing Deep Sets learned pooling changed hard correctness by only+1; '
    'the single equal mixture lost13hard questions versus Attention. No search was reopened after validation.', ''])
(OUT / 'EXPERIMENT_TABLE.md').write_text('\n'.join(lines), encoding='utf-8')
print(json.dumps({'rows': len(analysis['rows']) + 2, 'path': 'results/research_v2/EXPERIMENT_TABLE.md'}))
