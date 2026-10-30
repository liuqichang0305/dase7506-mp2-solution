"""Compare actual train-only V2 OOF runs and prespecified decoder ablations.

Never opens validation/test data or invokes a scorer. Each analysis destination
is immutable: choose a new output directory when more experiments have finished.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code'))
from mp2.selection_v2 import select_candidates, selection_details
from audit_assets import audit


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def measure(correct, labels, groups, fold):
    hard = np.array([len(set(row[row >= 0])) >= 2 for row in groups])
    single = labels.sum(axis=1) == 1
    result = {}
    for name, subset in (('overall', np.ones(len(labels), bool)), ('hard', hard), ('single_positive', single)):
        by_fold = []
        for number in sorted(set(fold.tolist())):
            chosen = subset & (fold == number)
            by_fold.append({'fold': int(number), 'correct': int(correct[chosen].sum()),
                            'n': int(chosen.sum()), 'accuracy': float(100 * correct[chosen].mean()) if chosen.any() else None})
        values = [r['accuracy'] for r in by_fold if r['accuracy'] is not None]
        result[name] = {'correct': int(correct[subset].sum()), 'n': int(subset.sum()),
                        'pooled_accuracy': float(100 * correct[subset].mean()) if subset.any() else None,
                        'fold_mean': float(np.mean(values)) if values else None,
                        'fold_sd': float(np.std(values, ddof=1)) if len(values) > 1 else None,
                        'folds': by_fold}
    return result


def paired_comparison(first, second, hard, repetitions, seed):
    def subset_result(mask, offset):
        left, right = first[mask].astype(np.int8), second[mask].astype(np.int8)
        delta = left - right
        if not len(delta):
            return {'n': 0}
        rng = np.random.default_rng(seed + offset)
        values = []
        # Bound temporary memory; the bootstrap unit is a whole question.
        for start in range(0, repetitions, 100):
            samples = rng.integers(0, len(delta), size=(min(100, repetitions-start), len(delta)))
            values.extend((100 * delta[samples].mean(axis=1)).tolist())
        return {'n': len(delta), 'first_correct': int(left.sum()), 'second_correct': int(right.sum()),
                'correct_difference': int(delta.sum()), 'difference_percentage_points': float(100 * delta.mean()),
                'paired_bootstrap_95_interval': np.quantile(values, [.025, .975]).tolist(),
                'first_only_correct': int(((left == 1) & (right == 0)).sum()),
                'second_only_correct': int(((left == 0) & (right == 1)).sum()),
                'both_wrong': int(((left == 0) & (right == 0)).sum()),
                'both_correct': int(((left == 1) & (right == 1)).sum()),
                'bootstrap_seed': seed + offset}
    errors_left, errors_right = ~first, ~second
    union = int((errors_left | errors_right).sum())
    intersection = int((errors_left & errors_right).sum())
    return {'overall': subset_result(np.ones(len(first), bool), 0), 'hard': subset_result(hard, 1),
            'error_overlap': {'intersection': intersection, 'union': union,
                              'jaccard': intersection / union if union else 1.,
                              'oracle_correct_upper_bound': int((first | second).sum()),
                              'note': 'Oracle is diagnostic only, not a realizable ensemble result.'},
            'bootstrap_repetitions': repetitions,
            'interval_caveat': 'Paired question bootstrap conditional on completed OOF predictions; model-selection optimism remains.'}


def decoder_variants(experiment, config):
    variants = [('registered', config['decoder'])]
    if experiment == 'linear_original':
        variants.extend([
            ('mask_max', {'mode': 'candidate_max', 'mask': True}),
            ('group_max', {'mode': 'group_max', 'mask': True}),
            *[(f'group_lse_a{alpha:g}', {'mode': 'group_lse', 'alpha': alpha, 'tau': 1., 'mask': True})
              for alpha in (0., .5, 1.)]])
    elif config['family'] != 'linear':
        variants.append(('candidate_max', {'mode': 'candidate_max', 'mask': True}))
    return variants


def reliability(correct, confidence):
    """Descriptive OOF calibration of the actual selected-answer event."""
    bins, ece = [], 0.
    for number in range(10):
        low, high = number / 10, (number + 1) / 10
        mask = (confidence >= low) & ((confidence < high) if number < 9 else (confidence <= high))
        count = int(mask.sum())
        average = float(confidence[mask].mean()) if count else None
        accuracy = float(correct[mask].mean()) if count else None
        bins.append({'low': low, 'high': high, 'n': count, 'mean_confidence': average, 'accuracy_fraction': accuracy})
        if count:
            ece += count / len(correct) * abs(average - accuracy)
    return {'n': len(correct), 'brier': float(np.mean((confidence - correct.astype(float)) ** 2)),
            'ece_10_equal_width': float(ece), 'bins': bins,
            'confidence_event': 'Actual selected numeric answer group is correct; not maximum candidate softmax.',
            'calibration': 'Uncalibrated group probability; OOF descriptive diagnostic only.'}


def regularization_equivalence(results_root, full_name='linear_group_a0', hard_name='linear_group_hard'):
    """Audit the prespecified group-loss denominator/L2 scaling control."""
    results_root = Path(results_root)
    full_summary = json.loads((results_root / full_name / 'summary.json').read_text(encoding='utf-8'))
    hard_summary = json.loads((results_root / hard_name / 'summary.json').read_text(encoding='utf-8'))
    with np.load(results_root / full_name / 'oof.npz', allow_pickle=False) as data:
        full_scores, groups, fold, labels = data['scores'].copy(), data['groups'].copy(), data['fold'].copy(), data['labels'].copy()
    with np.load(results_root / hard_name / 'oof.npz', allow_pickle=False) as data:
        hard_scores = data['scores'].copy()
        if not np.array_equal(groups, data['groups']) or not np.array_equal(fold, data['fold']):
            raise ValueError('Normalization comparison does not use the same OOF questions')
    first = np.asarray(select_candidates(full_scores, groups, **full_summary['config']['decoder']))
    second = np.asarray(select_candidates(hard_scores, groups, **hard_summary['config']['decoder']))
    delta = np.abs(full_scores - hard_scores)
    indices = np.arange(len(first))
    different_groups = groups[indices, first] != groups[indices, second]
    different_events = labels[indices, first] != labels[indices, second]
    rows = []
    for number in (0, 1, 2):
        full = next(r for r in full_summary['folds'] if r['fold'] == number)
        hard = next(r for r in hard_summary['folds'] if r['fold'] == number)
        fraction = full['fit']['hard_fraction']
        expected = full['fit']['effective_l2'] / fraction
        rows.append({'fold': number, 'full_effective_l2': full['fit']['effective_l2'],
                     'hard_fraction': fraction, 'expected_hard_effective_l2': expected,
                     'actual_hard_effective_l2': hard['fit']['effective_l2'],
                     'regularization_scale_matches': bool(np.isclose(expected, hard['fit']['effective_l2'], rtol=1e-12, atol=0)),
                     'max_abs_score_difference': float(delta[fold == number].max()),
                     'different_selected_indices': int((first[fold == number] != second[fold == number]).sum()),
                     'different_selected_answer_groups': int(different_groups[fold == number].sum()),
                     'different_correctness_events': int(different_events[fold == number].sum())})
    return {'full_experiment': full_name, 'hard_experiment': hard_name,
            'max_abs_score_difference': float(delta.max()),
            'mean_abs_score_difference': float(delta.mean()),
            'different_selected_indices': int((first != second).sum()),
            'different_selected_answer_groups': int(different_groups.sum()),
            'different_correctness_events': int(different_events.sum()), 'n': len(first), 'folds': rows,
            'interpretation': 'If L_hard=L_full/hard_fraction and L2 is scaled identically, exact optima coincide; finite optimizer tolerances can leave small numeric score differences.'}


def fmt_metrics(metrics, name):
    value = metrics[name]
    if value['fold_mean'] is None:
        return 'N/A'
    sd = 'N/A' if value['fold_sd'] is None else f"{value['fold_sd']:.3f}"
    return f"{value['fold_mean']:.3f} ± {sd}; {value['correct']}/{value['n']}"


def analyze(results_root, output, experiments=None, pairs=None, repetitions=5000, seed=3036901059):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite prior analysis: {output}')
    results_root = Path(results_root).resolve()
    directories = [results_root / name for name in experiments] if experiments else sorted(
        p for p in results_root.iterdir() if p.is_dir() and (p / 'registration.json').exists())
    rows, pending, records, audits, seed_groups = [], [], {}, {}, {}
    shared = None
    input_files = {}
    for directory in directories:
        registration_path, summary_path, oof_path = (directory / name for name in ('registration.json', 'summary.json', 'oof.npz'))
        if not registration_path.is_file():
            raise FileNotFoundError(f'Experiment lacks registration: {directory}')
        registration = json.loads(registration_path.read_text(encoding='utf-8'))
        if 'config' not in registration:
            # The separately gated mixture has its own registration schema and
            # report; it is not an independently fitted single-model CV run.
            if experiments is not None:
                raise ValueError(f'Not a single-model experiment registration: {directory}')
            continue
        config = registration['config']
        experiment = config['experiment_id']
        input_files[str(registration_path.relative_to(ROOT))] = digest(registration_path)
        if not summary_path.is_file():
            pending.append({'experiment_id': experiment, 'status': 'not_complete',
                            'complete_folds': len(list(directory.glob('fold[0-2].json')))})
            continue
        summary = json.loads(summary_path.read_text(encoding='utf-8'))
        if summary['status'] != 'complete' or summary['config'] != config:
            raise ValueError(f'Inconsistent completed summary: {experiment}')
        with np.load(oof_path, allow_pickle=False) as data:
            scores, labels, groups, fold = (data[k].copy() for k in ('scores', 'labels', 'groups', 'fold'))
        if scores.shape != labels.shape or groups.shape != labels.shape or labels.shape[1:] != (4,) or fold.shape != (len(labels),):
            raise ValueError(f'Malformed OOF arrays: {experiment}')
        if not np.isfinite(scores).all() or set(fold.tolist()) != {0, 1, 2}:
            raise ValueError(f'Incomplete/nonfinite common OOF: {experiment}')
        if shared is None:
            shared = (labels, groups, fold, registration['folds_sha256'])
        elif not all(np.array_equal(a, b) for a, b in zip(shared[:3], (labels, groups, fold))) or shared[3] != registration['folds_sha256']:
            raise ValueError(f'Experiments do not share the same question-level OOF protocol: {experiment}')
        input_files[str(summary_path.relative_to(ROOT))] = digest(summary_path)
        input_files[str(oof_path.relative_to(ROOT))] = digest(oof_path)
        fold_reports = sorted(summary['folds'], key=lambda r: r['fold'])
        if [r['fold'] for r in fold_reports] != [0, 1, 2]:
            raise ValueError(f'Expected exactly three fold reports: {experiment}')
        bundle_bytes = []
        for report in fold_reports:
            checkpoint = report['checkpoint']
            audited = audit(ROOT / checkpoint)
            audits[checkpoint] = audited
            bundle_bytes.append(audited['uncompressed_bytes'])
            saved = directory / f"fold{report['fold']}-scores.npy"
            if digest(saved) != report['scores_sha256'] or not np.array_equal(np.load(saved, allow_pickle=False), scores[fold == report['fold']]):
                raise ValueError(f'OOF score hash/order mismatch: {experiment} fold {report["fold"]}')
        numeric_bytes = [r['numeric_asset_bytes'] for r in fold_reports]
        parameters = [r['parameters'] for r in fold_reports]
        for variant, decoder in decoder_variants(experiment, config):
            choices = np.asarray(select_candidates(scores, groups, **decoder), dtype=np.int64)
            correct = labels[np.arange(len(labels)), choices].astype(bool)
            row_id = experiment if variant == 'registered' else f'{experiment}::{variant}'
            confidence = selection_details(scores, groups, **decoder)['chosen_group_probability']
            row = {'row_id': row_id, 'experiment_id': experiment, 'variant': variant,
                   'family': config['family'], 'seed': config['seed'], 'decoder': decoder,
                   'status': 'complete', 'metrics': measure(correct, labels, groups, fold),
                   'reliability': reliability(correct, confidence),
                   'parameters_per_fold': parameters, 'numeric_asset_bytes_per_fold': numeric_bytes,
                   'audited_uncompressed_bytes_per_fold': bundle_bytes,
                   'audited_uncompressed_mib_max': max(bundle_bytes) / 1024**2,
                   'training_seconds': summary['training_seconds'],
                   'cost_note': 'Reuses the same fitted OOF models; no additional training.' if variant != 'registered' else 'Actual three outer fits plus inner stopping fits when present.',
                   'validation': '按协议未评估（此工具只读取 train OOF）',
                   'folds_sha256': registration['folds_sha256'],
                   'oof_sha256': digest(oof_path)}
            rows.append(row)
            records[row_id] = {'correct': correct, 'choices': choices, 'row': row}
            if variant == 'registered':
                seed_config = {k: v for k, v in config.items() if k not in ('seed', 'experiment_id')}
                signature = json.dumps(seed_config, sort_keys=True)
                seed_groups.setdefault(signature, []).append(row)
        if experiment == 'linear_original':
            if not np.array_equal(records[experiment + '::mask_max']['choices'], records[experiment + '::group_max']['choices']):
                raise ValueError('group-max implementation does not match masked candidate-max')
            if np.any(records[experiment]['correct'] & ~records[experiment + '::mask_max']['correct']):
                raise ValueError('Mask harmed a formerly correct prediction, contradicting the task contract')
    if not rows:
        raise ValueError('No complete three-fold OOF experiment to analyze')
    comparisons = []
    hard = np.array([len(set(row[row >= 0])) >= 2 for row in shared[1]])
    for first, second in pairs or []:
        if first not in records or second not in records:
            raise ValueError(f'Unknown comparison row: {first}, {second}')
        comparisons.append({'first': first, 'second': second, **paired_comparison(
            records[first]['correct'], records[second]['correct'], hard, repetitions, seed)})
    stability = []
    for signature, members in seed_groups.items():
        if len(members) >= 2:
            values = [r['metrics']['hard']['fold_mean'] for r in members]
            stability.append({'configuration_without_seed': json.loads(signature),
                              'experiments': [r['experiment_id'] for r in members], 'seeds': [r['seed'] for r in members],
                              'hard_fold_mean_across_seeds': float(np.mean(values)),
                              'hard_fold_mean_sd_across_seeds': float(np.std(values, ddof=1)),
                              'note': 'All registered seeds retained; no best-seed selection.'})
    value = {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'split': 'train OOF only',
             'rows': rows, 'pending': pending, 'paired_comparisons': comparisons, 'seed_stability': stability,
             'input_sha256': input_files, 'audits': audits,
             'total_actual_training_seconds_distinct_experiments': sum(r['training_seconds'] for r in rows if r['variant'] == 'registered'),
             'notes': ['Every row uses shared question-level outer folds; fold SD is sample SD, not a confidence interval.',
                       'Training cost repeats on decoder-only rows because they reuse the same trained model.',
                       'Asset bytes cover one deployable fold bundle, not all research archives. Final selected bundle must be audited separately.',
                       'No new validation/test score is computed here. OOF hyperparameter selection still causes optimistic bias.']}
    lines = ['# Actual train OOF research results', '', f'Generated UTC: {value["created_at_utc"]}', '',
             'Common outer folds; percentages are fold mean ± sample SD followed by pooled correct/N. '
             'No official validation/test inputs or references were opened by this tool.', '',
             '| Experiment / decoder | Status | Overall CV (%) | Hard CV (%) | Single-positive CV (%) | Parameters | Audited MiB (max fold) | Training seconds | Official validation |',
             '|---|---|---|---|---|---:|---:|---:|---|']
    for row in rows:
        lines.append(f"| {row['row_id']} | complete | {fmt_metrics(row['metrics'], 'overall')} | {fmt_metrics(row['metrics'], 'hard')} | {fmt_metrics(row['metrics'], 'single_positive')} | {max(row['parameters_per_fold'])} | {row['audited_uncompressed_mib_max']:.4f} | {row['training_seconds']:.3f} | 按协议未评估 |")
    lines.extend(['', 'Decoder-only rows reuse fitted models and cost; do not sum them as separate training jobs. '
                  'Numeric payload and full audited bundle bytes are separately retained in analysis.json. '
                  'Finalist validation and final artifact timing belong to the separately authorized validation report.', ''])
    for comparison in comparisons:
        lines.extend([f"## {comparison['first']} versus {comparison['second']}", '',
                      '| Subset | Additional correct | Difference (pp) | Paired 95% bootstrap interval | First-only correct | Second-only correct | Both wrong |',
                      '|---|---:|---:|---|---:|---:|---:|'])
        for subset in ('overall', 'hard'):
            item = comparison[subset]
            if item['n']:
                low, high = item['paired_bootstrap_95_interval']
                lines.append(f"| {subset} | {item['correct_difference']} | {item['difference_percentage_points']:.3f} | [{low:.3f}, {high:.3f}] | {item['first_only_correct']} | {item['second_only_correct']} | {item['both_wrong']} |")
        lines.extend(['', f"Error Jaccard: {comparison['error_overlap']['jaccard']:.4f}. "
                      'Complementarity is necessary evidence for considering an ensemble, not proof that one will improve.', ''])
    if pending:
        lines.extend(['## Incomplete runs', ''])
        lines.extend(f"- {item['experiment_id']}: {item['complete_folds']}/3 completed folds; no fabricated full-CV result." for item in pending)
    output.mkdir(parents=True)
    (output / 'analysis.json').write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    (output / 'RESULTS.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    np.savez(output / 'selection_events.npz', **{key + '__correct': item['correct'] for key, item in records.items()},
             **{key + '__choices': item['choices'] for key, item in records.items()})
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-root', type=Path, default=ROOT / 'results/research_v2')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--experiments', nargs='+')
    parser.add_argument('--pair', nargs=2, action='append', metavar=('FIRST', 'SECOND'))
    parser.add_argument('--bootstrap', type=int, default=5000)
    parser.add_argument('--seed', type=int, default=3036901059)
    args = parser.parse_args()
    if args.bootstrap < 1:
        parser.error('--bootstrap must be positive')
    value = analyze(args.results_root, args.output, args.experiments, args.pair, args.bootstrap, args.seed)
    print(json.dumps({'output': str(args.output), 'rows': len(value['rows']), 'pending': value['pending'],
                      'training_seconds_distinct_experiments': value['total_actual_training_seconds_distinct_experiments']}))


if __name__ == '__main__':
    main()
