"""Prepare ten report pages from completed, hash-bound saved evidence only.

No NumPy/Torch, model imports, official-bank reads, fitting, evaluation, network,
or PDF rendering. This entry point is intentionally unusable before the one
frozen test completes. All numbers come from existing aggregate records.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[2]
SEED = 3036900957
BASE = 'results/research_v3/'
SUB = 'submission/v3/'
REQUIRED = (
    BASE + 'SEARCH_CLOSED.json', BASE + 'FINALISTS_LOCK.json',
    BASE + 'validation_summary.json', SUB + 'SELECTED_MODEL.json',
    SUB + 'FREEZE.json', SUB + 'REVIEW_RESULT.json', SUB + 'REPRODUCTION_RESULT.json',
    SUB + 'TEST_COMPLETE.json', SUB + 'test-score.json', SUB + 'test-timing.json',
)
TEST_OUTPUTS = {SUB + name for name in (
    'test-predictions.json', 'test-score.json', 'test-timing.json',
    'test-prediction.stdout.log', 'test-prediction.stderr.log', 'test-prediction-thread-pools.json',
    'test-scoring.stdout.log', 'test-scoring.stderr.log', 'test-scoring-thread-pools.json', 'TEST_OPENED.json')}
DEFAULTS = {
    'analysis': BASE + 'analysis/final_search/analysis.json',
    'v2_analysis': 'results/research_v2/analysis-final/analysis.json',
    'fusion_analysis': BASE + 'fixed_fusion_analysis.json',
    'confidence_fit': BASE + 'confidence_calibration/report.json',
    'confidence_validation': BASE + 'confidence_validation/report.json',
    'figures': 'reports/research_v3/figures/figure_manifest.json',
}


class Evidence:
    def __init__(self, root):
        self.root, self.hashes = Path(root).resolve(), {}

    def path(self, relative):
        relative = str(relative).replace('\\', '/')
        path = (self.root / relative).resolve()
        if not path.is_relative_to(self.root) or Path(relative).is_absolute():
            raise ValueError('Evidence must be a repository-relative path')
        canonical = path.relative_to(self.root).as_posix()
        if canonical != relative or canonical.startswith('code/data/') or 'RANKING_BACKGROUND' in canonical.upper() or 'leaderboard' in canonical.lower():
            raise ValueError('Reports cannot open official banks or ranking/background inputs')
        return path, canonical

    def bind(self, relative, expected=None):
        path, key = self.path(relative)
        if path.is_symlink() or not path.is_file():
            raise ValueError('Required completed evidence is missing: ' + key)
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if expected is not None and actual != expected:
            raise ValueError('Saved evidence hash mismatch: ' + key)
        self.hashes[key] = actual
        return path

    def read(self, relative, expected=None):
        path = self.bind(relative, expected)
        if path.suffix != '.json':
            raise ValueError('Aggregate readers accept JSON only')
        # Official predictions and event arrays are only hash-bound, never parsed.
        if path.name.endswith('predictions.json') or path.name.endswith('events.json'):
            raise ValueError('Report content cannot inspect per-question predictions')
        return json.loads(path.read_text(encoding='utf-8-sig'))

    def lines(self, relative):
        return [json.loads(line) for line in self.bind(relative).read_text(encoding='utf-8-sig').splitlines() if line.strip()]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite(value):
    require(type(value) in (float, int) and math.isfinite(value), 'A finite measured number is required')
    return value


def count(metric, prefix=''):
    n, correct = metric[prefix + 'n'], metric[prefix + 'correct']
    require(type(n) is int and type(correct) is int and 0 <= correct <= n and n > 0, 'Invalid aggregate count')
    return f'{correct}/{n}'


def pct(metric, prefix=''):
    count(metric, prefix)
    return f"{100 * metric[prefix + 'correct'] / metric[prefix + 'n']:.2f}"


def f(value, digits=3):
    return f'{finite(value):.{digits}f}'


def block(kind, text):
    return {'kind': kind, 'text': text}


def table(headers, rows, widths):
    require(len(headers) == len(widths) and all(len(row) == len(headers) for row in rows), 'Table columns differ')
    return {'kind': 'table', 'headers': headers, 'rows': [[str(v) for v in row] for row in rows], 'widths': widths}


def metric_cell(metric):
    return count(metric) + ' (' + pct(metric) + '%)'


def fold_cell(metric):
    return f(metric['fold_mean']) + ' +/- ' + f(metric['fold_sd'])


def recipe_label(row):
    # Registered names are immutable identifiers, never an encoding of the LR.
    rate = row.get('config', {}).get('lr')
    return row['experiment_id'] + (f' (lr={finite(rate):g})' if rate is not None else '')


def validate_summary(row, summary):
    """Cross-check secondary table values against first-level aggregate JSON."""
    require(summary.get('status') == 'complete' and summary['experiment_id'] == row['experiment_id']
            and summary['config'] == row['config'] and row['seed'] == summary['config']['seed']
            and row['family'] == summary['config']['family'], 'Analysis identity/config differs from the completed summary')
    for key in ('n', 'correct', 'hard_n', 'hard_correct'):
        require(row[key] == summary[key], 'Analysis count differs from the completed summary: ' + key)
    folds = sorted(summary['folds'], key=lambda item: item['fold'])
    require([item['fold'] for item in folds] == [0, 1, 2] and sum(item['n'] for item in folds) == row['n']
            and sum(item['correct'] for item in folds) == row['correct']
            and sum(item['hard_n'] for item in folds) == row['hard_n']
            and sum(item['hard_correct'] for item in folds) == row['hard_correct'], 'Summary folds do not pool to its counts')
    values = [100 * item['hard_correct'] / item['hard_n'] for item in folds]
    expected = {'hard_fold_mean': statistics.mean(values), 'hard_fold_sd': statistics.stdev(values),
                'accuracy': 100 * row['correct'] / row['n'], 'training_seconds': summary['training_seconds']}
    for key, value in expected.items():
        require(math.isclose(finite(row[key]), finite(value), rel_tol=1e-12, abs_tol=1e-10), 'Analysis statistic differs from the completed summary: ' + key)
    require(all(math.isclose(finite(a), b, abs_tol=1e-10) for a, b in zip(row['hard_fold_accuracies'], values)), 'Analysis fold scores differ from saved fold counts')
    require(row['parameters'] == max(item.get('parameters', 0) for item in folds)
            and row['numeric_asset_bytes'] == max(item.get('numeric_asset_bytes', item.get('uncompressed_bytes', 0)) for item in folds), 'Analysis parameter/byte count differs from saved folds')


def stability_groups(rows):
    """Descriptive summaries of already saved seed-level scores, never selection."""
    grouped = defaultdict(list)
    for row in rows:
        config = {k: v for k, v in row['config'].items() if k not in ('seed', 'id', 'experiment_id')}
        grouped[(row['version'], json.dumps(config, sort_keys=True, separators=(',', ':')))].append(row)
    result = []
    for (version, _), members in grouped.items():
        if len(members) < 2:
            continue
        seeds = [row['seed'] for row in members]
        require(len(set(seeds)) == len(seeds), 'Duplicate seeds within one identical recipe')
        require(set(seeds).issubset({SEED, SEED + 1, SEED + 2}), 'Unexpected unregistered stability seed')
        values = [finite(row['hard_fold_mean']) for row in members]
        primary = next((row for row in members if row['seed'] == SEED), None)
        require(primary is not None, 'Seed stability group lacks the fixed deployment seed')
        result.append({'version': version, 'experiment_id': primary['experiment_id'],
            'label': recipe_label(primary),
            'experiments': [row['experiment_id'] for row in sorted(members, key=lambda r: r['seed'])],
            'seeds': sorted(seeds), 'mean': statistics.mean(values), 'sample_sd': statistics.stdev(values),
            'values': [row['hard_fold_mean'] for row in sorted(members, key=lambda r: r['seed'])]})
    return sorted(result, key=lambda r: (r['version'], r['experiment_id']))


def _validate_confidence(report, split, selector, n, correct):
    require(report.get('schema') == 'mp2-v3-confidence-report-v1' and report.get('split') == split,
            'Missing completed confidence report for ' + split)
    require(report.get('selector') == selector and report.get('choices_changed') == 0,
            'Confidence analysis changed the selected method/choices')
    for kind in ('raw', 'calibrated'):
        metric = report[kind]
        require(metric['n'] == n and metric['correct'] == correct, 'Confidence event counts differ from saved choices')
        for key in ('brier', 'ece_10_equal_width', 'aurc_discrete_mean'):
            require(0 <= finite(metric[key]) <= 1, 'Invalid reliability metric')
        require(len(metric['bins']) == 10 and sum(row['n'] for row in metric['bins']) == n, 'All ten calibration bins must retain their counts')
        for i, item in enumerate(metric['bins']):
            require(item['bin'] == i and item['lower'] == i / 10 and item['upper'] == (i + 1) / 10, 'ECE bins differ from fixed equal-width definition')
        curve = metric['risk_coverage']
        require(len(curve['retained']) == len(curve['risk']) == n and curve['retained'] == list(range(1, n + 1)), 'Complete saved risk-coverage curve is required')


def prepare(root=ROOT, **paths):
    paths = {**DEFAULTS, **paths}
    e = Evidence(root)
    paths = {key: e.path(value)[1] for key, value in paths.items()}
    # Presence is checked before even reading validation/test aggregate JSON.
    for name in REQUIRED:
        require(e.path(name)[0].is_file(), 'Final report is not ready; missing ' + name)
    closed, freeze, complete = (e.read(name) for name in (REQUIRED[0], SUB + 'FREEZE.json', SUB + 'TEST_COMPLETE.json'))
    require(closed.get('search_closed') is True and freeze.get('frozen') is True
            and freeze.get('test_evaluated_before_freeze') is False, 'Search closure and genuine pre-test freeze are required')
    require(complete.get('schema') == 'mp2-v3-test-complete-v1' and complete.get('status') == 'passed'
            and complete.get('local_resource_eligible') is True and complete.get('split') == 'test' and complete.get('n') == 467,
            'A completed once-only full official test is required')
    require(complete['freeze_sha256'] == e.hashes[SUB + 'FREEZE.json'] and set(complete['output_sha256']) == TEST_OUTPUTS,
            'Test completion must bind this freeze and the entire output inventory')
    for name, digest in complete['output_sha256'].items():
        e.bind(name, digest)
    selected = e.read(SUB + 'SELECTED_MODEL.json', freeze['selected_model_sha256'])
    validation = e.read(BASE + 'validation_summary.json', freeze['validation_summary_sha256'])
    e.read(BASE + 'FINALISTS_LOCK.json', freeze['finalists_lock_sha256'])
    e.bind(BASE + 'SEARCH_CLOSED.json', freeze['search_closed_sha256'])
    require(selected.get('schema') == 'mp2-selected-model-v3' and selected['name'] == freeze['selected_name'] == validation['selected']['name'], 'Unique selected method differs')
    require(selected['validation_summary_sha256'] == e.hashes[BASE + 'validation_summary.json']
            and selected['selector'] == freeze['selector'] and selected['deployment_seed'] == SEED, 'Selected selector/seed/validation binding differs')
    require(selected['method'] == freeze['method'] == complete['method'] == 'code/solution_v3.py'
            and selected['checkpoint'] == freeze['checkpoint'] == complete['checkpoint'] == 'artifacts/research_v3/final_selected', 'Frozen method/checkpoint differs')
    require(validation.get('search_remains_closed') is True and validation.get('new_test_evaluated') is False
            and len(validation['finalists']) <= 2, 'Locked confirmation protocol differs')
    for name, digest in validation['evidence_sha256'].items():
        e.bind(name, digest)
    for kind, key in (('REVIEW_RESULT.json', 'review_result_sha256'), ('REPRODUCTION_RESULT.json', 'reproduction_result_sha256')):
        record = e.read(SUB + kind, freeze[key])
        require(record.get('passed') is True and record.get('selected_model_sha256') == e.hashes[SUB + 'SELECTED_MODEL.json'], 'Final review/reproduction is incomplete or bound to another method')
        for name, digest in record['evidence_files'].items():
            e.bind(name, digest)
        if kind.startswith('REPRODUCTION'):
            reproduction = record
    require(reproduction['source_commit'] == freeze['source_commit'] and reproduction.get('fresh_environment') is True
            and reproduction.get('choices_match') is True and reproduction.get('official_score_match') is True, 'Fresh frozen reproduction did not pass')
    fresh_score = e.read(reproduction['score_result']['path'], reproduction['score_result']['sha256'])
    fresh_timing = e.read(reproduction['timing_result']['path'], reproduction['timing_result']['sha256'])
    fresh_environment = e.read(reproduction['environment_result']['path'], reproduction['environment_result']['sha256'])
    require(fresh_score.get('split') == 'validation' and fresh_score.get('n') == 798
            and fresh_timing.get('status') == 'passed' and fresh_timing.get('local_resource_eligible') is True
            and 0 <= finite(fresh_timing['prediction_plus_scoring_seconds']) <= 30
            and fresh_environment.get('schema') == 'mp2-v3-clean-environment-v1', 'Typed clean reproduction results are incomplete')
    score = e.read(SUB + 'test-score.json')
    timing = e.read(SUB + 'test-timing.json')
    require(score.get('split') == 'test' and score.get('protocol') == freeze['protocol'] and score.get('n') == 467, 'Official score identity differs')
    count(score)
    require(abs(score['accuracy'] - 100 * score['correct'] / score['n']) < 1e-10, 'Official percentage and count disagree')
    require(timing.get('status') == 'passed' and timing.get('local_resource_eligible') is True
            and timing.get('freeze_sha256') == complete['freeze_sha256'] and timing['resource'] == freeze['resource'], 'Final resource evidence differs')
    require(0 <= finite(timing['prediction_plus_scoring_seconds']) <= 30
            and 0 <= timing['sampled_peak_task_process_tree_threads'] <= 4
            and timing['scientific_library_thread_cap_passed'] is True
            and freeze['resource']['within_limit'] is True and freeze['resource']['uncompressed_bytes'] <= 16 * 1024**2, 'Final measured resource cap failed')
    # Existing arrays/source/assets are hash checked only; never imported or decoded.
    for name, digest in {**freeze['source_files'], **freeze['assets']}.items():
        e.bind(name, digest)
    for recipe in freeze['training_recipes']:
        e.bind(recipe['final_fit_path'], recipe['final_fit_sha256'])

    analysis = e.read(paths['analysis'])
    e.bind('scripts/analyze_campaign_v3.py', analysis['source_sha256'])
    e.bind(str(Path(paths['analysis']).with_name('EXPERIMENT_TABLE.md')))
    v2 = e.read(paths['v2_analysis'])
    require(bool(v2.get('input_sha256')), 'Preserved V2 analysis must bind its original input inventory')
    for name, digest in v2['input_sha256'].items():
        e.bind(name, digest)
    fusion = e.read(paths['fusion_analysis'])
    queue = e.read(BASE + 'campaign_queue.json')
    ledger = e.lines(BASE + 'round_decisions.jsonl')
    e.bind(BASE + 'common_folds.json', analysis['folds_sha256'])
    require(ledger and len({r['round'] for r in ledger}) == len(ledger), 'Complete unique round ledger is required')
    require(all(j['status'] in ('completed', 'failed', 'pruned') for j in queue['jobs']), 'Campaign queue is not terminal')
    require({j['round'] for j in queue['jobs']}.issubset({r['round'] for r in ledger}), 'A registered training round is missing from the final ledger')
    rows = analysis['rows']
    require(rows and len({(r['version'], r['experiment_id']) for r in rows}) == len(rows), 'Complete aggregate table contains duplicates or no rows')
    index = {(r['version'], r['experiment_id']): r for r in rows}
    for row in rows:
        require(row['status'] == 'completed' and row['n'] == 4755 and len(row['hard_fold_accuracies']) == 3, 'Only full three-fold train OOF may appear in the main table')
        count(row); count(row, 'hard_')
        finite(row['training_seconds'])
        directory = f"results/{row['version']}/{row['experiment_id']}/"
        summary = e.read(directory + 'summary.json', row['summary_sha256'])
        validate_summary(row, summary)
        e.bind(directory + 'oof.npz', row['oof_sha256'])
    completed = {j['experiment_id'] for j in queue['jobs'] if j['status'] == 'completed'}
    require(completed == {r['experiment_id'] for r in rows if r['version'] == 'research_v3'}, 'Final analysis does not contain exactly the completed campaign jobs')
    references = e.read(BASE + 'deterministic_train_references/summary.json')
    reference_complete = e.read(BASE + 'deterministic_train_references/COMPLETION.json')
    require(reference_complete.get('schema') == 'mp2-v3-deterministic-train-references-complete-v1'
            and reference_complete.get('status') == 'complete'
            and reference_complete.get('new_fitting_runs') == 0
            and reference_complete.get('validation_test_ranking_read') is False
            and reference_complete.get('source_sha256') == references['source_sha256']
            and reference_complete.get('input_sha256') == references['input_sha256'],
            'Completed deterministic reference receipt differs from the aggregate provenance')
    require(set(reference_complete['output_sha256']) == {
        BASE + 'deterministic_train_references/summary.json',
        BASE + 'deterministic_train_references/selections.npz',
        BASE + 'deterministic_train_references/deterministic_train_references_v3.source.py'},
        'Deterministic reference completion must cover its whole output inventory')
    for name, digest in reference_complete['output_sha256'].items():
        e.bind(name, digest)
    require(references.get('schema') == 'mp2-v3-deterministic-train-references-v1'
            and references.get('status') == 'complete' and references.get('split') == 'train'
            and references.get('n') == 4755 and references.get('new_fitting_runs') == 0
            and references.get('validation_test_ranking_read') is False
            and references.get('majority_independent_group_count_choices_equal') is True
            and references.get('folds_sha256') == analysis['folds_sha256']
            and references['official_training_data_sha256']['code/data/manifest.json'] == freeze['data_manifest_sha256'],
            'Deterministic train references must use the same complete train/fold protocol without fitting')
    require(set(references['results']) == {'fixed0', 'valid_answer_majority'}
            and references.get('source_sha256') and references.get('input_sha256'),
            'Both predefined, source-bound deterministic train references are required')
    for name, digest in {**references['source_sha256'], **references['input_sha256']}.items():
        if name.startswith('code/data/'):
            require(references['official_training_data_sha256'].get(name) == digest,
                    'Reference data provenance differs; reporting does not open raw banks')
        else:
            e.bind(name, digest)
    e.bind(references['selections']['path'], references['selections']['sha256'])
    for result in references['results'].values():
        require(result['overall']['n'] == 4755 and result['hard']['n'] == rows[0]['hard_n']
                and result['single_positive']['n'] == rows[0]['single_n'],
                'Reference question populations differ from the common-fold table')
        for subset in ('overall', 'hard', 'single_positive'):
            metric = result[subset]
            count(metric)
            require(len(metric['folds']) == 3 and sorted(v['fold'] for v in metric['folds']) == [0, 1, 2]
                    and sum(v['n'] for v in metric['folds']) == metric['n']
                    and sum(v['correct'] for v in metric['folds']) == metric['correct'],
                    'Reference fold counts do not reproduce the aggregate')
            accuracies = [100 * v['correct'] / v['n'] for v in metric['folds']]
            require(abs(statistics.mean(accuracies) - finite(metric['fold_mean'])) < 1e-10
                    and abs(statistics.stdev(accuracies) - finite(metric['fold_sd'])) < 1e-10,
                    'Printed reference fold mean/SD differs from actual saved counts')
    historical_bindings = []
    for round_ in ledger:
        if 'analysis' in round_:
            prior_analysis = e.read(round_['analysis'], round_.get('analysis_sha256'))
            previous = {item['experiment_id']: item for item in prior_analysis['rows'] if item['version'] == 'research_v3'}
            for result in round_['results']:
                name = result.get('experiment_id', result.get('id'))
                current = index['research_v3', name]
                aliases = {'correct': 'full_correct', 'n': 'full_n'}
                for key in ('seed', 'correct', 'n', 'hard_correct', 'hard_n', 'hard_fold_mean', 'hard_fold_sd', 'training_seconds'):
                    require(previous[name][key] == current[key], 'Historical analysis disagrees with the final completed aggregates')
                    ledger_key = key if key in result else aliases.get(key)
                    if ledger_key is not None and ledger_key in result:
                        require(result[ledger_key] == current[key], 'Round ledger disagrees with saved/current completed aggregates')
            if 'analysis_sha256' not in round_:
                key = e.path(round_['analysis'])[1]
                historical_bindings.append({'round': round_['round'], 'analysis': key, 'current_sha256': e.hashes[key],
                    'original_ledger_declared_sha256': False,
                    'limitation': 'Historical ledger omitted the analysis hash; this report binds the presently verified file and cross-checks its recorded counts against completed summaries, without rewriting the ledger or claiming a past hash existed.'})
    require(fusion.get('schema') == 'mp2-fixed-fusion-analysis-v3' and fusion.get('status') == 'completed'
            and fusion.get('new_training_fits') == 0, 'Actual fixed-fusion analysis is required')
    e.bind(BASE + 'fusion_registry.jsonl', fusion['registry_sha256'])
    fusion_details = []
    for item in fusion['results']:
        diagnostic = e.read(item['diagnostic_path'], item['diagnostic_sha256'])
        actual = diagnostic['fusion']
        require(actual['status'] == 'complete' and actual['metrics'] == item['metrics']
                and actual['methods'] == item['methods'] and actual['weights'] == item['weights'], 'Fixed fusion aggregate differs from its saved diagnostic')
        directory = str(Path(item['diagnostic_path']).parent).replace('\\', '/') + '/'
        e.bind(directory + 'fusion-oof.npz', actual['oof_sha256'])
        e.bind(directory + actual['registration_file'], actual['registration_sha256'])
        fusion_details.append((item, actual))

    chosen_validation = validation['baseline_validation'] if selected['historical_fallback'] else next(
        row['validation'] for row in validation['finalists'] if row['name'] == selected['name'])
    count(chosen_validation); count(chosen_validation, 'hard_')
    require(fresh_score['correct'] == chosen_validation['correct'], 'Fresh reproduction aggregate differs from the selected validation outcome')
    validation_comparison = e.read(BASE + 'selected_validation_comparison.json')
    require(validation_comparison.get('schema') == 'mp2-v3-selected-validation-comparison-v1'
            and validation_comparison['selected_model_sha256'] == e.hashes[SUB + 'SELECTED_MODEL.json']
            and validation_comparison['selected_name'] == selected['name'] and validation_comparison['split'] == 'validation'
            and validation_comparison['new_official_score_runs'] == 0 and validation_comparison['model_selection_changed'] is False
            and validation_comparison['test_opened'] is False
            and validation_comparison['data_manifest_sha256'] == freeze['data_manifest_sha256'],
            'Post-selection paired validation evidence differs from the unique selected method')
    for name, digest in validation_comparison['evidence_sha256'].items():
        e.bind(name, digest)
    e.bind('code/tools/analyze_selected_validation_v3.py', validation_comparison['comparison_source_sha256'])
    e.bind('code/tools/analyze_oof_v2.py', validation_comparison['bootstrap_source_sha256'])
    paired_validation_rows = []
    for baseline in ('frozen_v1', 'historical_v2_selected'):
        comparison = validation_comparison['comparisons'][baseline]
        require(comparison['bootstrap_repetitions'] == 5000 and 'post-selection validation' in comparison['interval_caveat'],
                'Saved validation intervals must disclose the fixed bootstrap and selection limitation')
        for subset, prefix, offset in (('overall', '', 0), ('hard', 'hard_', 1)):
            metric = comparison[subset]
            require(metric['n'] == chosen_validation[prefix + 'n']
                    and metric['first_correct'] == chosen_validation[prefix + 'correct']
                    and metric['correct_difference'] == metric['first_correct'] - metric['second_correct']
                    and metric['bootstrap_seed'] == 3036901059 + offset,
                    'Paired validation counts or registered seeds differ')
            interval = metric['paired_bootstrap_95_interval']
            require(len(interval) == 2 and finite(interval[0]) <= finite(interval[1]), 'Invalid saved paired interval')
            paired_validation_rows.append([baseline + ' / ' + subset, str(metric['correct_difference']),
                f(metric['difference_percentage_points']), '[' + f(interval[0]) + ', ' + f(interval[1]) + ']'])
    fit = e.read(paths['confidence_fit'])
    conf_val = e.read(paths['confidence_validation'])
    calibration = e.read(str(Path(paths['confidence_fit']).with_name('calibration.json')))
    calibration_key = e.path(str(Path(paths['confidence_fit']).with_name('calibration.json')))[1]
    require(conf_val['source_sha256']['calibration_json'] == e.hashes[calibration_key],
            'Validation confidence evaluated a different saved calibration map')
    for path, record in ((paths['confidence_fit'], fit), (paths['confidence_validation'], conf_val)):
        e.bind(str(Path(path).with_name('confidence-events.npz')), record['events_sha256'])
    _validate_confidence(fit, 'train_oof', selected['selector'], 4755, fit['raw']['correct'])
    _validate_confidence(conf_val, 'validation', selected['selector'], 798, chosen_validation['correct'])
    require('IN-SAMPLE' in fit['interpretation'] and calibration['source_split'] == 'train_oof'
            and calibration['selector'] == selected['selector'] and calibration['C'] == 1.0
            and calibration['seed'] == SEED and calibration['fit_n'] == 4755
            and calibration['fit_correct'] == fit['raw']['correct']
            and fit['source_sha256']['selected_model_json'] == e.hashes[SUB + 'SELECTED_MODEL.json'],
            'Only the fixed post-selection train-OOF confidence map is reportable')
    prep = e.read(BASE + 'selected_confidence/preparation.json')
    require(prep['selected_model_sha256'] == e.hashes[SUB + 'SELECTED_MODEL.json']
            and prep['new_official_score_runs'] == 0 and prep['candidate_selection_changed'] is False,
            'Confidence preparation must preserve the saved official choices')
    # Bind event metadata without parsing its per-question event reference.
    e.bind(prep['events'])
    require(conf_val['source_sha256']['prediction_events_json'] == e.hashes[e.path(prep['events'])[1]],
            'Validation confidence evaluated different selected-answer events')
    for name, digest in prep['original_official_output_sha256'].items():
        e.bind(name, digest)
    for relative in ('docs/research_v3/SEARCH_PLAN_V3.md', 'docs/research_v3/MODELS_SOURCES.md',
                     'docs/research_v3/OPTIMIZER_SOURCES.md', 'results/research_v3/model-source-evidence.json'):
        e.bind(relative)

    figures = e.read(paths['figures'])
    require(figures.get('schema') == 'mp2-v3-report-figures-v1' and figures.get('scope') == 'saved_final_evidence_only'
            and len(figures.get('figures', [])) == 2,
            'Two completed saved-evidence figures are required')
    figure_inputs = {paths['analysis'], BASE + 'campaign_queue.json', paths['confidence_fit'], paths['confidence_validation']}
    figure_generator = 'reports/research_v3/prepare_figures.py'
    require(figure_inputs | {figure_generator} == set(figures['evidence_sha256']), 'Figures do not bind the same complete input set')
    for name, digest in figures['evidence_sha256'].items():
        if name in figure_inputs:
            require(digest == e.hashes[name], 'Figures used different report inputs')
        e.bind(name, digest)
    expected_figures = ['reports/research_v3/figures/cv_fit_cost_vs_hard_accuracy.png',
                        'reports/research_v3/figures/chosen_event_confidence.png']
    require([item['path'] for item in figures['figures']] == expected_figures, 'Final figure identities/order differ')
    for number, item in enumerate(figures['figures']):
        require(item['height'] == 235, 'Figure layout differs from the compact tenth page')
        e.bind(item['path'], item['sha256'])
        require(item['generation_record']['path'] == Path(item['path']).with_suffix('.generation.json').as_posix(), 'Figure generation receipt path differs')
        generation = e.read(item['generation_record']['path'], item['generation_record']['sha256'])
        subset_paths = ({paths['analysis'], BASE + 'campaign_queue.json'} if number == 0
                        else {paths['confidence_fit'], paths['confidence_validation']}) | {figure_generator}
        subset = {name: figures['evidence_sha256'][name] for name in subset_paths}
        require(generation.get('schema') == 'mp2-v3-figure-generation-v1'
                and generation.get('scope') == 'saved_final_evidence_only'
                and generation.get('path') == item['path'] and generation.get('sha256') == item['sha256']
                and generation.get('source_sha256') == subset
                and generation.get('generator_sha256') == figures['evidence_sha256'][figure_generator]
                and generation.get('format') == 'PNG' and generation.get('report_height_points') == item['height']
                and generation.get('caption') == item['caption'] and generation.get('raw_bank_or_test_read') is False,
                'Figure generation receipt differs from the final manifest/current generator')

    stable = stability_groups(rows)
    require(stable, 'Actual fixed-seed stability evidence is required')
    v3_rows = [r for r in rows if r['version'] == 'research_v3']
    v2_rows = [r for r in rows if r['version'] == 'research_v2']
    train_seconds = sum(r['training_seconds'] for r in v3_rows)
    v2_seconds = sum(r['training_seconds'] for r in v2_rows)
    attempt_seconds = sum(a.get('observed_wall_seconds', 0) for j in queue['jobs'] for a in j.get('attempts', []))
    failed = [j for j in queue['jobs'] if j['status'] != 'completed']
    rounds = []
    for record in ledger:
        rr = [index['research_v3', j['experiment_id']] for j in queue['jobs']
              if j['round'] == record['round'] and j['status'] == 'completed'
              and index['research_v3', j['experiment_id']]['seed'] == SEED]
        best = max(rr, key=lambda r: r['hard_fold_mean']) if rr else None
        increment = record.get('increment_above_global_reference_pp')
        threshold = record.get('exceeds_registered_half_point', increment is not None and increment > .5)
        rounds.append([record['round'], record['configurations'], record.get('completed_outer_fits', 0),
                       f(best['hard_fold_mean']) if best else 'See fixed-seed/fusion table',
                       'yes' if threshold else 'no'])
    require(len(rounds) <= 16 and len(stable) <= 12, 'Aggregate scope exceeds the compact layout; revise pages before generating the final report')

    # V2 decoder-only controls use identical saved scores; refitted objectives are separate rows.
    v2_index = {r['row_id']: r for r in v2['rows']}
    controls = ['linear_original', 'linear_original::mask_max', 'linear_original::group_max',
        'linear_original::group_lse_a0', 'linear_original::group_lse_a0.5', 'linear_original::group_lse_a1',
        'linear_group_a0', 'linear_group_a0p5', 'linear_group_a1', 'linear_group_hard',
        'mlp_group_s0', 'resmlp_group_s0', 'deepsets_group_s0', 'attention_group_s0']
    require(set(controls).issubset(v2_index), 'The preserved V2 mechanism analysis is incomplete')
    mechanism = [[name + ' (no fitting)', fold_cell(result['hard']), count(result['hard']), count(result['overall'])]
                 for name, result in references['results'].items()]
    mechanism += [[name, fold_cell(v2_index[name]['metrics']['hard']), count(v2_index[name]['metrics']['hard']),
                  count(v2_index[name]['metrics']['overall'])] for name in controls]
    # One primary-seed representative per registered round, with saved bootstrap only.
    bootstrap_rows = []
    for record in ledger:
        candidates = [index['research_v3', j['experiment_id']] for j in queue['jobs']
                      if j['round'] == record['round'] and j['status'] == 'completed'
                      and index['research_v3', j['experiment_id']]['seed'] == SEED]
        if not candidates:
            continue
        representative = max(candidates, key=lambda r: r['hard_fold_mean'])
        pair = representative['paired_vs_v2_attention']
        interval = pair['hard_bootstrap_95pct']
        bootstrap_rows.append([recipe_label(representative), count(representative, 'hard_'),
            f(representative['hard_fold_mean']) + ' +/- ' + f(representative['hard_fold_sd']),
            f(pair['hard_delta_pp']), '[' + f(interval[0]) + ', ' + f(interval[1]) + ']'])
    require(bootstrap_rows, 'Saved paired-bootstrap comparisons are missing')
    recipes = freeze['training_recipes']
    recipe_table = [[r['experiment_id'], r['configuration']['family'], r['seed'], r['epochs'], r['preprocess'], r['evaluation_weights']] for r in recipes]
    recipe_details = '; '.join(r['experiment_id'] + ': ' + ', '.join(
        key + '=' + json.dumps(r['configuration'][key], separators=(',', ':'))
        for key in ('family', 'hidden', 'width', 'depth', 'k', 'dropout', 'optimizer', 'lr', 'muon_lr', 'weight_decay', 'objective')
        if key in r['configuration']) + '; decoder=' + json.dumps(r.get('decoder', r['configuration'].get('decoder', {})), separators=(',', ':')) for r in recipes)
    selected_kind = 'preserved historical V2 fallback' if selected['historical_fallback'] else 'new V3 finalist'
    prior = freeze['known_old_test_aggregate']
    require(prior['known_before_v3'] is True, 'Prior test exposure must be disclosed')

    def page(*blocks):
        return {'blocks': list(blocks)}

    pages = [page(
        block('title', 'Learning to choose the correct answer\nDASE7506 MP2: bounded performance research'),
        block('subtitle', 'Student 3036900957 | Reproducible V3 study | Final frozen outcome'),
        block('paragraph', f"The final method is {selected['name']} ({selected_kind}). A train-only campaign evaluated {len(v3_rows)} complete V3 configurations on the same three question folds, alongside {len(v2_rows)} preserved V2 configurations. The locked confirmation rule selected one checkpoint before a single new full test run. The official final test result is {metric_cell(score)}; this is an observed result, not a promised ranking."),
        table(['Stage', 'Population / outcome', 'Permitted role'], [
            ['Train OOF', '4,755 questions; common 3 folds', 'Architecture, objective and optimizer research'],
            ['Locked validation', metric_cell(chosen_validation), 'One bounded final confirmation / fallback'],
            ['Final official test', metric_cell(score), 'One frozen result; reporting only']], [1, 1.6, 2.2]),
        block('heading', 'What the study establishes'),
        block('paragraph', 'The report distinguishes decoder-only changes from refitted objectives, compares candidate-independent and candidate-context networks, tests optimizer and preprocessing hypotheses, and retains failed or negative results. The strongest CV score alone does not override the registered validation rule. Full train OOF, hard questions and single-positive questions are different populations; their accuracies are not interchangeable.'),
        block('paragraph', f"Historical exposure is explicit: the earlier V1 aggregate {count(prior)} was known before V3, and historical V2 validation scores were already known. This is not an untouched-holdout study. No new test result was used to choose an architecture, seed, epoch, decoder or calibration map."),
        block('small', 'All tables summarize saved evidence. Evidence hashes travel with report_inputs.json; the complete experiment table is results/research_v3/analysis/final_search/EXPERIMENT_TABLE.md. No leaderboard/ranking conclusion is inferred from CV or validation.'),
    ), page(
        block('heading', '1. Representation, selector and architectures'),
        block('paragraph', 'Each question supplies four candidate texts and four 1,024-dimensional embeddings. Relation features concatenate the embedding, its absolute deviation from the four-candidate mean (another 1,024 coordinates), and 16 fixed statistics: peer cosine similarities, answer parseability/votes, text length and simple formatting counts. Features never use question IDs or reference answers. Standard or robust scalers are fitted inside the relevant fitting partition.'),
        block('paragraph', 'The official answer parser defines equivalent-answer groups. Unparseable candidates receive no probability mass. Group logit A_g = log(sum_{j in g} exp(s_j/tau)) - alpha * log(|g|); group cross-entropy targets the correct answer group. Bounded alpha controls are 0, 0.5 and 1, at tau=1. Candidate maximum and group maximum are equivalent under the same mask. Stable ties use the first group occurrence and an original candidate index; an all-unparseable row returns the deterministic fallback with confidence zero.'),
        table(['Family', 'Task adaptation'], [
            ['Linear / MLP / ResMLP', 'Candidate scores from shared features; learned nonlinear transformations or residual blocks'],
            ['Deep Sets / Attention', 'Permutation-compatible candidate context; no candidate-position embedding'],
            ['TabM', 'Shared matrices plus member adapters; average member losses, then average answer-group probabilities'],
            ['RealMLP-inspired', 'Learnable input scale, NTP / SELU, zero head, optional robust coordinate scaling'],
            ['Muon / EMA', 'Muon on hidden 2D weights; head/bias/norm/adapters use auxiliary AdamW. EMA updates after optimizer steps.']], [1.2, 3.8]),
        block('heading', 'Final deployed recipe'),
        table(['Component', 'Family', 'Seed', 'Epochs', 'Preprocess', 'Weights'], recipe_table, [2, 1, 1.2, .6, 1.2, .7]),
        block('small', recipe_details),
        block('small', 'Exact layer widths, dropout, optimizer rates, decay, objective and decoder are preserved in FREEZE.training_recipes[].configuration and each bound final_fit.json. TabM averages probabilities, not logits. These compact task adaptations are not claims to reproduce the source papers\' benchmark configurations.'),
    ), page(
        block('heading', '2. Shared folds, finite budget and all V3 rounds'),
        block('paragraph', 'All main comparisons use the saved question-level three-fold split. Inner stopping chooses epochs without outer labels; preprocessing fits inner training only. A fresh outer fit uses all outer training rows and the chosen epoch. Final fitting uses the predeclared aggregation of inner epochs and the fixed deployment seed. A logged fold seed is the base experiment seed plus its deterministic fold offset; it is not an extra selected seed. EMA, when registered, is used consistently for inner selection, outer prediction and final export.'),
        table(['Registered round', 'Configs', 'Outer fits', 'Best primary-seed hard mean (%)', '>0.5 pp threshold'], rounds, [2.4, .55, .65, 1.6, .7]),
        block('paragraph', f"Completed V3 optimization took {f(train_seconds, 1)} seconds summed over recorded inner/outer fits; preserved V2 runs contributed {f(v2_seconds, 1)} seconds historically. Recorded V3 process-attempt wall time is {f(attempt_seconds, 1)} seconds and includes startup/overhead where captured. These clocks have different scopes and are not added. {len(failed)} failed or pruned jobs remain visible; incomplete jobs never enter full-CV rankings."),
        block('small', 'Research was bounded by the original 24-hour clock, at most 96 configurations and at most two justified repair retries, with no paid compute. One CPU training process used one compute thread. Research closure, interrupted-epoch recovery and per-round decisions are retained in SEARCH_CLOSED.json, campaign_queue.json and round_decisions.jsonl. Missing crash-time measurement is not invented. Full per-round decisions and all configurations are in the bound ledger/table.'),
    ), page(
        block('heading', '3. Deterministic references, V2 mechanisms and objectives'),
        table(['Control / experiment', 'Hard fold mean +/- SD (%)', 'Hard correct', 'All correct'], mechanism, [2.3, 1.5, 1, 1]),
        block('small', 'Deterministic references use all 4,755 train questions with no fitted parameters; common-fold statistics are descriptive partitions, not three training runs. Fixed0 always returns original index0. Majority excludes None, uses equal official parsed values, breaks ties by earliest original index, and returns0 if all are unparseable. Single-positive counts: ' + '; '.join(name + ' ' + count(result['single_positive']) for name, result in references['results'].items()) + '.'),
        block('paragraph', 'The audited original model learns one linear weight vector after scaling, using LSE(all candidates) minus LSE(all correct candidates), then raw candidate argmax. Its engineered relation features are already nonlinear; correct alternatives are not mislabeled negative. The six linear_original rows reuse exactly the same scores to isolate masking/group decoding. With valid candidates and alpha=0, tau=1, group CE equals the original multi-positive objective. Hard/full mean changes require corresponding regularization accounting; they cannot by themselves establish an architecture gain.'),
        block('paragraph', 'V3 objective rounds compare group cross-entropy, multi-positive listwise loss, binary cross-entropy and question-averaged positive-negative softplus pairs, as registered. Correct candidates are never relabeled negative. The deployed selector masks unparseable candidates; loss controls retain their declared semantics, including the original unmasked listwise denominator. Pairwise loss uses valid positive-negative pairs; questions without a pair contribute zero under the declared reduction. Cross-family maxima do not identify a causal effect.'),
        block('small', 'Full V2 decoder/ablation table and seed records: results/research_v2/analysis-final/analysis.json. V3 objective-specific configurations and all pooled counts: results/research_v3/analysis/final_search/EXPERIMENT_TABLE.md. These are training OOF results, not test scores.'),
    ), page(
        block('heading', '4. Stability and paired uncertainty'),
        table(['Identical recipe (version / primary ID)', 'Fixed seeds (last digits)', 'Mean of hard fold means (%)', 'Across-seed sample SD'],
              [[r['version'].replace('research_', '') + '/' + r['label'], ','.join(str(s)[-3:] for s in r['seeds']), f(r['mean']), f(r['sample_sd'])] for r in stable], [2.5, 1, 1.4, 1]),
        block('paragraph', 'Seeds 3036900957, 3036900958 and 3036900959 assess an identical recipe; the deployment seed is always 3036900957. The table includes every available registered repeat rather than selecting the luckiest seed. Across-seed SD describes seed sensitivity; within-run fold SD describes fold variation. Neither is a confidence interval.'),
        table(['Primary-seed round representative', 'Hard count', 'Fold mean +/- SD (%)', 'Delta pp', 'Paired 95% interval'], bootstrap_rows, [2.1, .8, 1.1, .6, 1.1]),
        block('small', 'Saved intervals use 5,000 question-paired bootstrap resamples, seed 3036901059, against the preserved V2 attention_regularized_s0 predictions. For compactness the table uses the highest hard-mean primary-seed configuration within each registered round; every configuration remains in the full table. Model and representative selection used these OOF outcomes, so intervals are descriptive and selection-optimistic. A CI crossing zero is compatible with sampling uncertainty; it is not evidence of equality.'),
    ), page(
        block('heading', '5. Fixed fusion and negative evidence'),
        block('paragraph', 'Fusion was conditional on actual complementary errors: each member must correct at least ten errors of the other and the hard-question oracle must improve at least one percentage point over the stronger member. The oracle is an upper bound, not an attainable fusion result. Only the registered equal answer-group probability mixture was measured; independent weights and logits were not averaged.'),
        table(['Fixed mixture', 'Hard fold mean +/- SD (%)', 'Hard / all correct', 'Delta hard questions vs Attention', 'Member assets MiB'],
              [[item['experiment_id'], fold_cell(item['metrics']['hard']), count(item['metrics']['hard']) + ' / ' + count(item['metrics']['overall']),
                item['hard_delta_questions_vs_attention_control'], f(item['maximum_member_asset_bytes'] / 1024**2)] for item, actual in fusion_details], [2.3, 1.3, 1.2, .8, .8]),
        *[block('small', item['experiment_id'] + ': ' + item['decision']) for item, actual in fusion_details],
        block('paragraph', fusion['decision']),
        block('paragraph', fusion['cost_note']),
        block('small', 'Gate measurements, reciprocal errors and oracle counts are retained in each hash-bound diagnostic.json; complete mixture rows, fixed registrations and recorded mixing costs are in ' + paths['fusion_analysis'] + '. Any rejected mixture has no claimed formal deployment latency. Existing member training is counted once; fusion introduces zero new fitting runs.'),
    ), page(
        block('heading', '6. Locked validation and chosen-answer confidence'),
        table(['Method', 'Full validation', 'Hard validation', 'Role'],
              [['Historical V2 fallback', metric_cell(validation['baseline_validation']), count(validation['baseline_validation'], 'hard_'), 'Previously known comparator']] +
              [[row['name'], metric_cell(row['validation']), count(row['validation'], 'hard_'), 'Locked new finalist'] for row in validation['finalists']], [2, 1.2, 1, 1.5]),
        block('paragraph', f"The unique selected method is {selected['name']}. The registered replacement rule requires both hard and overall validation counts to meet the historical comparator and at least one to improve, subject to resource/contract checks; otherwise the historical checkpoint is retained. This confirmation does not reopen research."),
        table(['Confidence event set / map', 'Brier', 'ECE (10 bins)', 'AURC'],
              [[label, f(record[kind]['brier'], 5), f(record[kind]['ece_10_equal_width'], 5), f(record[kind]['aurc_discrete_mean'], 5)]
               for label, record, kind in [('Train OOF raw', fit, 'raw'), ('Train OOF calibrated (IN-SAMPLE)', fit, 'calibrated'),
                                         ('Locked validation raw', conf_val, 'raw'), ('Locked validation calibrated', conf_val, 'calibrated')]], [2.4, 1, 1, 1]),
        block('paragraph', f"Confidence is the probability of the answer group actually chosen by the final selector. A single fixed C=1 logistic map of clipped logit-confidence was fitted to train OOF correctness (coefficient {f(calibration['coefficient'], 5)}, intercept {f(calibration['intercept'], 5)}). Its training-set metrics are in-sample and optimistic. Validation assesses the saved mapping without fitting it; validation also informed final model selection, so this remains development evidence."),
        table(['Equal-width confidence bin', 'Raw val n', 'Calibrated val n'],
              [[f'{i/10:.1f}-{(i+1)/10:.1f}', conf_val['raw']['bins'][i]['n'], conf_val['calibrated']['bins'][i]['n']] for i in range(10)], [2, 1, 1]),
        block('small', 'All bins, including empty bins, and complete risk-coverage arrays are saved in the confidence reports. AURC is the arithmetic mean of cumulative risk at each retained count, with stable tie ordering. Calibration changes zero choices, is not loaded by official predict, and is not part of the inference bundle.'),
    ), page(
        block('heading', '7. Paired validation and final frozen evaluation'),
        table(['Selected minus preserved baseline / subset', 'Question delta', 'Delta pp', 'Paired 95% interval'],
              paired_validation_rows, [2.5, .8, .8, 1.4]),
        block('small', 'Validation intervals reuse saved selected/baseline choices with 5,000 question-paired draws (base seed 3036901059, hard subset offset +1). This validation set was already used for selection; the descriptive intervals do not remove adaptive bias or constitute an untouched confirmatory test. Fallback versus the same preserved V2 model naturally has zero difference.'),
        table(['Final observation', 'Measured value'], [
            ['Official full test', metric_cell(score)],
            ['Prediction + independent scoring', f(timing['prediction_plus_scoring_seconds'], 6) + ' seconds'],
            ['Peak sampled task-tree OS threads', timing['sampled_peak_task_process_tree_threads']],
            ['Frozen uncompressed inference assets', str(freeze['resource']['uncompressed_bytes']) + ' bytes (' + f(freeze['resource']['uncompressed_bytes'] / 1024**2) + ' MiB)'],
            ['Fresh validation reproduction', '798/798 saved choices match; official aggregate matches'],
            ['Fresh-environment prediction + scoring', f(fresh_timing['prediction_plus_scoring_seconds'], 6) + ' seconds'],
            ['Source commit', freeze['source_commit']]], [2, 3.3]),
        block('paragraph', 'The measured interval includes separate fresh official prediction and scoring processes, interpreter startup/imports, input/assets loading and output writing, under one 30-second deadline. Task-tree OS threads are sampled; compute pools are separately constrained. The measurement coordinator is disclosed separately. This is a local measurement, not certification on the course reference Xeon.'),
        block('paragraph', f"The historical V1 test aggregate was {count(prior)} and the new frozen aggregate is {count(score)}. The arithmetic difference is {score['correct'] - prior['correct']:+d} questions; it is descriptive because prior test exposure and subsequent development prevent an untouched-test claim. No accuracy target, leaderboard position or comparative significance is inferred."),
        block('paragraph', 'The final method passed full-input Torch/NumPy export and inference-contract audits, then reproduced the locked validation choices and official score in a fresh checkout and environment. All parameters, preprocessing arrays and member weights are in the audited bundle. Optimizer/EMA history and the optional confidence map are outside inference assets and are never loaded by prediction.'),
        block('small', 'TEST_OPENED.json was persisted before the official test process; a partial run cannot silently restart. TEST_COMPLETE.json binds the complete original output/log inventory. One successful test may be verified by hashes on resume but is not re-evaluated. Repository source, bundle bytes, lock, selected record and original data-manifest hash are bound by FREEZE.json.'),
    ), page(
        block('heading', '8. Sources, limitations, AI disclosure and reproduction'),
        block('paragraph', 'AI assistance was substantive: Codex agents implemented methods and tooling, researched primary sources, ran experiments under user authorization, reviewed each other\'s code and prepared this report. Automated tests and independent agent review are internal AI verification, not teacher approval or independent human certification. The student remains responsible for the submitted work.'),
        block('small', 'Primary method sources: Gorishniy et al., TabM, ICLR 2025, arXiv:2410.24210v3; Holzmuller et al., Better by Default / RealMLP, NeurIPS 2024, arXiv:2407.04491v3; Gorishniy et al., Benchmarking Optimizers for MLPs in Tabular Deep Learning, arXiv:2604.15297v1 (2026 preprint); native PyTorch 2.15 Muon documentation/source. Author repository commits, inspected implementation sections and licenses are recorded in MODELS_SOURCES.md and OPTIMIZER_SOURCES.md. TabM and the optimizer research repositories use Apache-2.0; the inspected RealMLP implementation uses MIT. No pretrained weights, external training examples or copied author implementation were introduced.'),
        block('paragraph', 'Limitations: OOF model selection is optimistic, only three question folds and a few fixed seeds are available, architectures are compact task adaptations, and uncertainty does not include the adaptive research process. Historical validation and the old test aggregate were known. The frozen test has only 467 questions. Resource observations are local and sampled. Calibration is optional descriptive analysis. Neither CV nor validation justifies a public-rank claim.'),
        *([block('small', 'Historical logging limitation: an early round omitted its analysis SHA256. The original ledger is preserved; this report hashes the present analysis and reconciles its counts/settings with the completed summaries. This does not recreate a missing historical hash.')] if historical_bindings else []),
        block('heading', 'Reproduction without reopening test'),
        block('small', f"Use immutable source commit {freeze['source_commit']}. Install the pinned dependencies in a clean environment. The recorded fresh-environment command and package inventory are linked from REPRODUCTION_RESULT.json. For inference: python code/predict.py --method code/solution_v3.py --checkpoint artifacts/research_v3/final_selected --split validation --output <new-path>; then python code/score_predictions.py --predictions <new-path> --split validation --output <new-score-path>. Follow the direct-base resource wrapper in scripts/reproduce_selected_v3.py for measured reproduction. Do not invoke the once-only official test runner again."),
        block('small', 'Training reconstruction: registered exact configurations are in docs/research_v3/rounds and campaign_queue.json; code/train_v3.py --config <registered-json> records the fitting recipe. The closed submission workspace deliberately refuses further search. A separate disclosed research reproduction may rebuild those training results, but does not replace the frozen submission. Report reproduction: python reports/research_v3/prepare_report_inputs.py, then python reports/research_v3/build_report.py --inputs reports/research_v3/report_inputs.json. Render and inspect every PDF page before publication.'),
        block('small', 'Evidence index: analysis/final_search/EXPERIMENT_TABLE.md; round_decisions.jsonl; fixed_fusion_analysis.json; confidence_calibration/report.json; confidence_validation/report.json; submission/v3/{SELECTED_MODEL,FREEZE,REVIEW_RESULT,REPRODUCTION_RESULT,TEST_COMPLETE}.json. Peer review is a separate later stage and remains pending until the instructor actually opens it.'),
    ), page(
        block('heading', '9. Saved-evidence visual diagnostics'),
        *[{'kind': 'image', 'path': item['path'], 'height': item['height'], 'caption': item['caption']} for item in figures['figures']],
        block('small', 'Figures summarize the same hash-bound aggregate inputs as the tables. Training OOF is used for model selection; fitted-OOF calibration is in-sample. Validation was used for final confirmation. No new model fit, question-level diagnosis or test visualization is performed.'),
    )]
    return {'schema': 'mp2-v3-final-report-inputs-v1', 'title': 'DASE7506 MP2: bounded performance research and frozen evaluation',
        'created_at_utc': datetime.now(timezone.utc).isoformat(), 'student_id': '3036900957',
        'generator_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'pages': pages,
        'evidence_sha256': dict(sorted(e.hashes.items())),
        'derived_aggregates': {'seed_stability': stable, 'v3_completed_configurations': len(v3_rows),
            'deterministic_train_references': references['results'],
            'historical_analysis_bindings': historical_bindings,
            'v3_fit_seconds': train_seconds, 'v2_historical_fit_seconds': v2_seconds,
            'v3_recorded_attempt_seconds': attempt_seconds, 'failed_or_pruned': len(failed)},
        'final_facts': {'selected_name': selected['name'], 'historical_fallback': selected['historical_fallback'],
            'validation': chosen_validation, 'test': score, 'asset_bytes': freeze['resource']['uncompressed_bytes'],
            'test_seconds': timing['prediction_plus_scoring_seconds'], 'source_commit': freeze['source_commit']}}


def chinese_summary(record):
    value = record['final_facts']
    return ('# V3 最终结果说明\n\n'
        f"最终方法：{value['selected_name']}。" + ('按预注册规则保留历史 V2 模型。' if value['historical_fallback'] else '由锁定后的 V3 候选确认产生。') + '\n\n'
        f"验证集：{count(value['validation'])}；其中难题 {count(value['validation'], 'hard_')}。唯一冻结测试：{count(value['test'])}（{pct(value['test'])}%）。\n\n"
        f"完整预测加独立评分耗时 {f(value['test_seconds'], 6)} 秒，推理资产 {value['asset_bytes']} 字节。源提交为 {value['source_commit']}。这些是保存证据中的实测值，不表示排行榜名次。\n\n"
        '训练内共同三折用于研究；额外固定种子用于稳定性分析，不挑最好种子。验证集仅用于已经锁定候选的有限确认。测试结果只进入最终报告和提交，不反馈训练或调参。旧 V1 测试汇总及历史 V2 验证结果已知，因此不能声称所有留出数据从未被看过。\n\n'
        '置信度描述最终实际选中答案组的正确概率。固定 C=1 校准器只拟合训练 OOF；其拟合数据上的指标明确属于样本内。验证结果用于描述泛化，校准器不改变任何答案、不被正式预测加载。\n\n'
        '英文报告记录全部轮次、旧机制对照、固定种子、配对 bootstrap、负面融合、资源、复现和 AI 辅助。完整表及逐文件 SHA256 位于 report_inputs.json 所引用的证据中。同行评审仍以教师实际开放时间为准。\n')


def write_outputs(record, output, chinese_output):
    output, chinese_output = Path(output), Path(chinese_output)
    require(output != chinese_output, 'Output destinations must differ')
    if output.exists() or chinese_output.exists():
        raise FileExistsError('Preserve existing report evidence; choose new output paths')
    for path in (output, chinese_output):
        path.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(record, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
    with chinese_output.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(chinese_summary(record))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key, default in DEFAULTS.items():
        parser.add_argument('--' + key.replace('_', '-'), default=default)
    parser.add_argument('--output', type=Path, default=ROOT / 'reports/research_v3/report_inputs.json')
    parser.add_argument('--chinese-output', type=Path, default=ROOT / 'reports/research_v3/understand_final_zh.md')
    args = vars(parser.parse_args())
    output, chinese = args.pop('output'), args.pop('chinese_output')
    record = prepare(ROOT, **args)
    write_outputs(record, output, chinese)
    print(json.dumps({'status': 'prepared_from_completed_evidence', 'pages': len(record['pages']),
                      'evidence_files': len(record['evidence_sha256']), 'output': str(output)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
