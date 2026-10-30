"""One preregistered equal-probability mixture, gated by train-OOF complementarity.

This analysis tool never opens official validation/test banks or references.
The first run locks the pair and weights; a new output directory cannot silently
be used to search additional pairs. Existing experiment assets remain untouched.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code'))
from mp2.selection_v2 import group_probabilities, select_candidates
from analyze_oof_v2 import digest, measure, paired_comparison, reliability
from audit_assets import audit

DEPLOYMENT_SEED = 3036900957
BOOTSTRAP_SEED = 3036901059
BOOTSTRAP_REPETITIONS = 5000


def write_new(path, value):
    """Write exactly once, including when two processes contend for a path."""
    with Path(path).open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def complementarity_gate(first_correct, second_correct, hard):
    first_correct, second_correct, hard = (np.asarray(x, dtype=bool) for x in (first_correct, second_correct, hard))
    if first_correct.ndim != 1 or first_correct.shape != second_correct.shape or hard.shape != first_correct.shape:
        raise ValueError('Gate inputs must be matching question-level vectors')
    first_only = int((first_correct & ~second_correct).sum())
    second_only = int((second_correct & ~first_correct).sum())
    hard_n = int(hard.sum())
    stronger = max(int(first_correct[hard].sum()), int(second_correct[hard].sum()))
    oracle = int((first_correct[hard] | second_correct[hard]).sum())
    gain = 100 * (oracle - stronger) / hard_n if hard_n else None
    # Integer comparison avoids floating-point ambiguity at exactly one pp.
    passed = first_only >= 10 and second_only >= 10 and hard_n > 0 and 100 * (oracle - stronger) >= hard_n
    return {'passed': bool(passed), 'first_only_correct': first_only, 'second_only_correct': second_only,
            'hard_n': hard_n, 'stronger_member_hard_correct': stronger, 'hard_oracle_correct': oracle,
            'hard_oracle_gain_over_stronger_percentage_points': gain,
            'requirements': {'minimum_each_corrects_other': 10, 'minimum_hard_oracle_gain_percentage_points': 1.0},
            'interpretation': 'Passing permits exactly one equal-weight probability mixture; oracle is not an achievable prediction.'}


def choose_from_group_probabilities(probability, groups, first_scores):
    """Select group, then highest-scoring original candidate of member one.

    Group ties go to the group with the earliest original candidate, then score
    ties within the group go to the earliest original candidate. All-None rows
    return index zero with selected probability zero. No labels are accepted.
    """
    probability = np.asarray(probability, dtype=np.float64)
    groups = np.asarray(groups)
    first_scores = np.asarray(first_scores, dtype=np.float64)
    if probability.ndim != 2 or probability.shape[1:] != (4,) or groups.shape != probability.shape or first_scores.shape != probability.shape:
        raise ValueError('Probability, group IDs and scores must have shape [N,4]')
    if not np.issubdtype(groups.dtype, np.integer) or np.any((groups < -1) | (groups > 3)):
        raise ValueError('Invalid group IDs')
    if not np.isfinite(probability).all() or np.any(probability < 0) or not np.isfinite(first_scores).all():
        raise ValueError('Probabilities/scores must be finite and probabilities nonnegative')
    membership = groups[:, :, None] == np.arange(4)[None, None, :]
    active = membership.any(axis=1)
    if np.any(probability[~active] != 0) or not np.allclose(probability.sum(axis=1), active.any(axis=1).astype(float), atol=1e-10, rtol=1e-10):
        raise ValueError('Probability mass must be normalized over valid groups only')
    valid = groups >= 0
    candidate_probability = np.take_along_axis(probability, np.maximum(groups, 0), axis=1)
    candidate_probability = np.where(valid, candidate_probability, -np.inf)
    first_candidate = candidate_probability.argmax(axis=1)
    rows = np.arange(len(groups))
    chosen_group = groups[rows, first_candidate]
    chosen = np.where(valid & (groups == chosen_group[:, None]), first_scores, -np.inf).argmax(axis=1)
    chosen = np.where(valid.any(axis=1), chosen, 0)
    selected_probability = np.zeros(len(groups), dtype=np.float64)
    has_group = chosen_group >= 0
    selected_probability[has_group] = probability[rows[has_group], chosen_group[has_group]]
    return [int(j) for j in chosen], selected_probability


def equal_probability_mixture(first_scores, second_scores, groups, first_decoder, second_decoder):
    """The sole evaluated mixing rule: 0.5*p(first group)+0.5*p(second group)."""
    first_probability = group_probabilities(first_scores, groups, alpha=first_decoder.get('alpha', 0.), tau=first_decoder.get('tau', 1.))
    second_probability = group_probabilities(second_scores, groups, alpha=second_decoder.get('alpha', 0.), tau=second_decoder.get('tau', 1.))
    probability = .5 * first_probability + .5 * second_probability
    chosen, selected_probability = choose_from_group_probabilities(probability, groups, first_scores)
    return {'first_probability': first_probability, 'second_probability': second_probability,
            'probability': probability, 'choices': chosen, 'selected_probability': selected_probability}


def load_experiment(results_root, experiment):
    results_root = Path(results_root).resolve()
    if not experiment or Path(experiment).name != experiment or '/' in experiment or '\\' in experiment:
        raise ValueError('Experiment must be a single registered directory name')
    folder = results_root / experiment
    registration_path, summary_path, oof_path = [folder / name for name in ('registration.json', 'summary.json', 'oof.npz')]
    registration = json.loads(registration_path.read_text(encoding='utf-8'))
    summary = json.loads(summary_path.read_text(encoding='utf-8'))
    config = registration['config']
    if config['experiment_id'] != experiment or config != summary['config'] or summary['status'] != 'complete':
        raise ValueError('Experiment registration/summary is inconsistent or incomplete')
    if config['seed'] != DEPLOYMENT_SEED:
        raise ValueError('The registered ensemble uses fixed deployment seed 3036900957; no lucky-seed selection')
    if not registration['data_files'] or any('train' not in Path(name).name or any(split in Path(name).name for split in ('validation', 'test', 'audit')) for name in registration['data_files']):
        raise ValueError('Only registered official train inputs are permitted')
    folds_path = results_root / 'common_folds.json'
    if digest(folds_path) != registration['folds_sha256']:
        raise ValueError('Common fold manifest changed after fitting')
    manifest = json.loads(folds_path.read_text(encoding='utf-8'))
    with np.load(oof_path, allow_pickle=False) as data:
        scores, labels, groups, fold = [data[k].copy() for k in ('scores', 'labels', 'groups', 'fold')]
    if scores.ndim != 2 or scores.shape[1:] != (4,) or labels.shape != scores.shape or groups.shape != scores.shape or fold.shape != (len(scores),):
        raise ValueError('Malformed OOF arrays')
    if not np.isfinite(scores).all() or not np.isin(labels, (0, 1)).all() or np.any(labels.sum(axis=1) < 1):
        raise ValueError('Incomplete/nonfinite OOF predictions or invalid training labels')
    if not np.issubdtype(groups.dtype, np.integer) or np.any((groups < -1) | (groups > 3)) or np.any((groups < 0).all(axis=1)):
        raise ValueError('Official train OOF requires valid parsed answer groups')
    if set(fold.tolist()) != {0, 1, 2} or len(scores) != manifest['questions']:
        raise ValueError('All three common OOF folds are required')
    if sorted(r['fold'] for r in summary['folds']) != [0, 1, 2]:
        raise ValueError('Three completed fold reports are required')
    for split in manifest['folds']:
        number, holdout = split['fold'], np.asarray(split['holdout'], dtype=int)
        if not np.array_equal(np.flatnonzero(fold == number), np.sort(holdout)):
            raise ValueError('OOF memberships do not match the common fold manifest')
        report = next(r for r in summary['folds'] if r['fold'] == number)
        saved = folder / f'fold{number}-scores.npy'
        if digest(saved) != report['scores_sha256'] or not np.array_equal(np.load(saved, allow_pickle=False), scores[holdout]):
            raise ValueError('OOF predictions do not match immutable fold scores')
    choices = np.asarray(select_candidates(scores, groups, **config['decoder']), dtype=np.int64)
    correct = labels[np.arange(len(labels)), choices].astype(bool)
    return {'experiment_id': experiment, 'registration': registration, 'summary': summary, 'config': config,
            'scores': scores, 'labels': labels, 'groups': groups, 'fold': fold, 'choices': choices, 'correct': correct,
            'input_sha256': {'registration.json': digest(registration_path), 'summary.json': digest(summary_path), 'oof.npz': digest(oof_path)}}


def evaluate_pair(first_name, second_name, output, results_root=ROOT / 'results/research_v2'):
    start = time.perf_counter()
    output, results_root = Path(output), Path(results_root).resolve()
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite ensemble evidence: {output}')
    if first_name == second_name:
        raise ValueError('Two distinct registered experiments are required')
    first, second = (load_experiment(results_root, name) for name in (first_name, second_name))
    for key in ('labels', 'groups', 'fold'):
        if not np.array_equal(first[key], second[key]):
            raise ValueError(f'Ensemble members do not share aligned train OOF {key}')
    for key in ('folds_sha256', 'data_files'):
        if first['registration'][key] != second['registration'][key]:
            raise ValueError(f'Ensemble members do not share {key}')
    specification = {'first': first_name, 'second': second_name, 'weights': [.5, .5],
                     'deployment_seed': DEPLOYMENT_SEED, 'first_input_sha256': first['input_sha256'],
                     'second_input_sha256': second['input_sha256'],
                     'first_decoder': first['config']['decoder'], 'second_decoder': second['config']['decoder'],
                     'folds_sha256': first['registration']['folds_sha256']}
    lock = results_root / 'ENSEMBLE_PAIR_LOCK.json'
    repeated = lock.exists()
    if repeated:
        if json.loads(lock.read_text(encoding='utf-8'))['specification'] != specification:
            raise ValueError('A different ensemble pair/input is already locked; no implicit pair/weight search permitted')
    else:
        write_new(lock, {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'specification': specification,
                         'rule': 'One equal-weight pair only; subsequent same-pair runs are explicit reproductions.'})
    output.mkdir(parents=True)
    write_new(output / 'registration.json', {'created_at_utc': datetime.now(timezone.utc).isoformat(),
              'specification': specification, 'same_pair_reproduction': repeated,
              'analysis_source_sha256': digest(__file__), 'split': 'train OOF only'})
    groups, labels, fold = first['groups'], first['labels'], first['fold']
    hard = np.array([len(set(row[row >= 0])) >= 2 for row in groups])
    gate = complementarity_gate(first['correct'], second['correct'], hard)
    # This file is durably closed before the ensemble probabilities are formed.
    write_new(output / 'complementarity_gate.json', gate)
    if not gate['passed']:
        summary = {'status': 'not_run_complementarity_gate_failed', 'ensemble_evaluated': False,
                   'first': first_name, 'second': second_name, 'gate': gate,
                   'analysis_wall_seconds': time.perf_counter() - start,
                   'validation': 'Not evaluated', 'test': 'Disabled'}
        write_new(output / 'summary.json', summary)
        (output / 'RESULTS.md').write_text('# Ensemble not run\n\nThe preregistered complementarity gate failed. '
                                         'No mixture score or accuracy was computed. See complementarity_gate.json.\n', encoding='utf-8')
        return summary
    probability_start = time.perf_counter()
    mixture = equal_probability_mixture(first['scores'], second['scores'], groups, first['config']['decoder'], second['config']['decoder'])
    probability_seconds = time.perf_counter() - probability_start
    choices = np.asarray(mixture['choices'], dtype=np.int64)
    correct = labels[np.arange(len(labels)), choices].astype(bool)
    comparisons = [{'first': 'equal_probability_ensemble', 'second': member['experiment_id'],
                    **paired_comparison(correct, member['correct'], hard, BOOTSTRAP_REPETITIONS, BOOTSTRAP_SEED)}
                   for member in (first, second)]
    assets = []
    for number in (0, 1, 2):
        member_audits, numeric, parameters = [], 0, 0
        for member in (first, second):
            report = next(r for r in member['summary']['folds'] if r['fold'] == number)
            member_audits.append(audit(ROOT / report['checkpoint']))
            numeric += report['numeric_asset_bytes']
            parameters += report['parameters']
        assets.append({'fold': number, 'parameters': parameters, 'numeric_asset_bytes': numeric,
                       'member_audited_uncompressed_bytes_sum': sum(a['uncompressed_bytes'] for a in member_audits),
                       'member_audits': member_audits})
    np.savez(output / 'oof-mixture.npz', groups=groups, labels=labels, fold=fold,
             first_probability=mixture['first_probability'], second_probability=mixture['second_probability'],
             probability=mixture['probability'], choices=choices, correct=correct,
             selected_probability=mixture['selected_probability'],
             first_choices=first['choices'], second_choices=second['choices'])
    summary = {'status': 'complete', 'ensemble_evaluated': True, 'first': first_name, 'second': second_name,
               'weights': [.5, .5], 'gate': gate, 'metrics': measure(correct, labels, groups, fold),
               'reliability': reliability(correct, mixture['selected_probability']),
               'member_metrics': {member['experiment_id']: measure(member['correct'], labels, groups, fold) for member in (first, second)},
               'paired_comparisons': comparisons, 'assets_per_fold': assets,
               'asset_note': 'Sum of actual member bundles; final deployment wrapper/config must also be counted and audited if selected.',
               'existing_training_seconds_reused': sum(member['summary']['training_seconds'] for member in (first, second)),
               'additional_training_seconds': 0., 'group_probability_combine_seconds': probability_seconds,
               'analysis_wall_seconds': time.perf_counter() - start,
               'cost_note': 'Analysis timings are not official fresh-process prediction/scoring measurements.',
               'oof_mixture_sha256': digest(output / 'oof-mixture.npz'),
               'validation': 'Not evaluated by this train-only tool', 'test': 'Disabled',
               'caveat': 'Pair selection used OOF results; bootstrap intervals do not remove model-selection optimism.'}
    write_new(output / 'summary.json', summary)
    lines = ['# Equal group-probability ensemble: actual train OOF', '',
             f'Members: `{first_name}` and `{second_name}`; weights 0.5 / 0.5. No weight grid.', '',
             '| Method | Overall correct/N | Overall fold mean ± SD (%) | Hard correct/N | Hard fold mean ± SD (%) |',
             '|---|---:|---:|---:|---:|']
    for name, metrics in [*summary['member_metrics'].items(), ('equal_probability_ensemble', summary['metrics'])]:
        a, h = metrics['overall'], metrics['hard']
        lines.append(f"| {name} | {a['correct']}/{a['n']} | {a['fold_mean']:.3f} ± {a['fold_sd']:.3f} | {h['correct']}/{h['n']} | {h['fold_mean']:.3f} ± {h['fold_sd']:.3f} |")
    lines.extend(['', 'Complementarity was recorded before forming the mixture. Correctness uses the selected '
                  'numeric answer group, with the highest-scoring candidate from member one used only to return an original index.', '',
                  f"Reused measured member training: {summary['existing_training_seconds_reused']:.3f} seconds; additional training: zero. "
                  f"Maximum member asset sum: {max(r['member_audited_uncompressed_bytes_sum'] for r in assets)/1024**2:.4f} MiB. "
                  'A final deployment wrapper is not included until one is created.', '',
                  'Validation/test were not evaluated. Paired bootstrap comparisons and raw group probabilities are saved alongside this table.'])
    (output / 'RESULTS.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--first', required=True)
    parser.add_argument('--second', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--results-root', type=Path, default=ROOT / 'results/research_v2')
    args = parser.parse_args()
    summary = evaluate_pair(args.first, args.second, args.output, args.results_root)
    print(json.dumps({'status': summary['status'], 'output': str(args.output),
                      'gate': summary['gate'], 'metrics': summary.get('metrics')}))


if __name__ == '__main__':
    main()
