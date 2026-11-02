"""Prepare an honest post-deadline research archive, never a new course score.

Standard-library aggregate reader only. No model/data loading, fitting, scoring,
network, or PDF creation. The separate TEST_COMPLETE-gated workflow is unchanged.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[2]
HELPER = ROOT / 'reports/research_v3/prepare_report_inputs.py'
spec = importlib.util.spec_from_file_location('preserved_report_aggregate_helpers', HELPER)
shared = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shared)
Evidence, require, finite = shared.Evidence, shared.require, shared.finite
block, table, count, f = shared.block, shared.table, shared.count, shared.f
BASE, SUB, SEED = shared.BASE, shared.SUB, shared.SEED
REQUIRED = (BASE + 'SEARCH_CLOSED.json', BASE + 'FINALISTS_LOCK.json', BASE + 'validation_summary.json',
    SUB + 'SELECTED_MODEL.json', SUB + 'FREEZE.json', SUB + 'REVIEW_RESULT.json', SUB + 'REPRODUCTION_RESULT.json',
    SUB + 'FORMAL_SUBMISSION_BLOCKED.json', SUB + 'RESEARCH_CLOSEOUT.json', 'submission/SUBMISSION_VERIFICATION.json')
FORBIDDEN = (SUB + 'TEST_OPENED.json', SUB + 'TEST_COMPLETE.json', SUB + 'test-score.json',
             SUB + 'SUBMISSION_SUCCESS.json')
DEFAULTS = {'analysis': BASE + 'analysis/final_search/analysis.json',
            'v2_analysis': 'results/research_v2/analysis-final/analysis.json',
            'fusion_analysis': BASE + 'fixed_fusion_analysis_complete.json'}
DEADLINE = datetime(2026, 10, 31, 16, tzinfo=timezone.utc)
NOTICE = 'POST-DEADLINE RESEARCH ARCHIVE: V3 FORMAL TEST AND SCORE UPDATE NOT PERFORMED.'


def aware(value):
    parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
    require(parsed.tzinfo is not None, 'An explicit UTC offset is required')
    return parsed


def bind_map(e, mapping):
    for path, digest in mapping.items():
        e.bind(path, digest)


def final_evidence(e):
    for name in REQUIRED:
        require(e.path(name)[0].is_file(), 'Research closeout is not ready; missing ' + name)
    require(not any((e.root / name).exists() for name in FORBIDDEN),
            'Research-only no-test/no-submission claim conflicts with a V3 execution artifact')
    close = e.read(SUB + 'RESEARCH_CLOSEOUT.json')
    require(close.get('schema') == 'mp2-v3-post-deadline-closeout-v1'
            and all(close.get(key) is True for key in ('research_only', 'missed_deadline', 'no_new_test', 'no_new_score_update'))
            and aware(close['created_at_utc']) > DEADLINE, 'An actual post-deadline research-only closeout is required')
    blocked = e.read(SUB + 'FORMAL_SUBMISSION_BLOCKED.json', close['formal_submission_blocked_sha256'])
    require(blocked.get('missed_deadline') is True and blocked.get('reason')
            and (blocked.get('no_new_test_or_score_update') is True
                 or (blocked.get('no_new_test') is True and blocked.get('no_new_score_update') is True))
            and DEADLINE < aware(blocked['created_at_utc']) <= aware(close['created_at_utc']),
            'Formal-submission block does not establish the missed deadline and no new result')
    e.bind(close['rules_audit_path'], close['rules_audit_sha256'])
    freeze = e.read(SUB + 'FREEZE.json', close['freeze_sha256'])
    selected = e.read(SUB + 'SELECTED_MODEL.json', close['selected_model_sha256'])
    require(freeze.get('frozen') is True and freeze.get('test_evaluated_before_freeze') is False
            and freeze['selected_model_sha256'] == close['selected_model_sha256']
            and freeze.get('research_only') is True and freeze.get('formal_test_allowed') is False
            and freeze.get('formal_submission_blocked_sha256') == close['formal_submission_blocked_sha256']
            and aware(blocked['created_at_utc']) <= aware(freeze['created_at_utc']) <= aware(close['created_at_utc']),
            'Research archive freeze is incomplete or incorrectly allows formal testing')
    closed = e.read(BASE + 'SEARCH_CLOSED.json', freeze['search_closed_sha256'])
    require(closed.get('search_closed') is True, 'Search must remain closed')
    lock = e.read(BASE + 'FINALISTS_LOCK.json', freeze['finalists_lock_sha256'])
    validation = e.read(BASE + 'validation_summary.json', freeze['validation_summary_sha256'])
    require(selected.get('schema') == 'mp2-selected-model-v3'
            and selected['name'] == freeze['selected_name'] == validation['selected']['name']
            and selected['historical_fallback'] == validation['selected']['historical_fallback']
            and selected['validation_summary_sha256'] == e.hashes[BASE + 'validation_summary.json']
            and selected['selector'] == freeze['selector'] and selected['deployment_seed'] == SEED
            and selected['method'] == freeze['method'] == 'code/solution_v3.py'
            and selected['checkpoint'] == freeze['checkpoint'] == 'artifacts/research_v3/final_selected',
            'Unique research selection and frozen method differ')
    require(validation.get('search_remains_closed') is True and validation.get('new_test_evaluated') is False
            and 1 <= len(validation['finalists']) <= 2, 'Research confirmation must retain the locked bounded protocol')
    bind_map(e, validation['evidence_sha256'])
    chosen = validation['baseline_validation'] if selected['historical_fallback'] else next(
        row['validation'] for row in validation['finalists'] if row['name'] == selected['name'])
    count(chosen); count(chosen, 'hard_')
    require(chosen['n'] == 798, 'Full saved validation aggregate is required')
    for file, key in (('REVIEW_RESULT.json', 'review_result_sha256'), ('REPRODUCTION_RESULT.json', 'reproduction_result_sha256')):
        record = e.read(SUB + file, freeze[key])
        require(record.get('passed') is True and record['selected_model_sha256'] == close['selected_model_sha256'],
                'Research review/reproduction is incomplete or belongs to another selection')
        bind_map(e, record['evidence_files'])
        if file.startswith('REPRODUCTION'):
            reproduction = record
    require(reproduction['source_commit'] == freeze['source_commit']
            and all(reproduction.get(key) is True for key in ('fresh_environment', 'choices_match', 'official_score_match')),
            'Fresh research reproduction did not pass')
    fresh = {}
    for key in ('score', 'timing', 'environment'):
        reference = reproduction[key + '_result']
        fresh[key] = e.read(reference['path'], reference['sha256'])
    require(fresh['score'].get('split') == 'validation' and fresh['score']['n'] == chosen['n']
            and fresh['score']['correct'] == chosen['correct']
            and fresh['environment'].get('schema') == 'mp2-v3-clean-environment-v1'
            and fresh['timing'].get('status') == 'passed' and fresh['timing'].get('local_resource_eligible') is True
            and 0 <= finite(fresh['timing']['prediction_plus_scoring_seconds']) <= 30,
            'Typed clean validation reproduction differs from the archived outcome')
    require(freeze['resource']['within_limit'] is True
            and 0 < freeze['resource']['uncompressed_bytes'] <= 16 * 1024**2, 'Frozen inference assets exceed the cap')
    bind_map(e, freeze['source_files']); bind_map(e, freeze['assets'])
    for recipe in freeze['training_recipes']:
        e.bind(recipe['final_fit_path'], recipe['final_fit_sha256'])
        require(recipe['seed'] == SEED, 'Deployment seed cannot be replaced by a better repeat')
    prior = e.read('submission/SUBMISSION_VERIFICATION.json')
    require(prior['issue_number'] == close['prior_issue_number'] == 294
            and prior['issue_url'] == close['prior_issue_url'] == 'https://github.com/xudongwu-0/xudongwu-0.github.io/issues/294'
            and prior['student_id'] == '3036900957'
            and prior['actual_score']['correct'] == 348 and prior['actual_score']['n'] == 467,
            'The preserved V1 submission must not be presented as a new V3 score')
    require(close.get('monitor_gap_disclosure') and isinstance(close['monitor_gap_disclosure'], str),
            'The actual monitoring gaps and missed deadline must be disclosed')
    return {'closeout': close, 'freeze': freeze, 'selected': selected, 'validation': validation,
            'chosen': chosen, 'fresh': fresh, 'prior': prior}


def train_evidence(e, paths, freeze, close):
    analysis = e.read(paths['analysis'])
    e.bind('scripts/analyze_campaign_v3.py', analysis['source_sha256'])
    e.bind(str(Path(paths['analysis']).with_name('EXPERIMENT_TABLE.md')))
    e.bind(BASE + 'common_folds.json', analysis['folds_sha256'])
    rows = analysis['rows']
    index = {(row['version'], row['experiment_id']): row for row in rows}
    require(len(index) == len(rows) and rows, 'The complete train table must have unique rows')
    summaries = {}
    for key, row in index.items():
        require(row['status'] == 'completed' and row['n'] == 4755 and len(row['hard_fold_accuracies']) == 3,
                'Partial experiments cannot appear in the complete OOF table')
        folder = f'results/{key[0]}/{key[1]}/'
        summaries[key] = e.read(folder + 'summary.json', row['summary_sha256'])
        shared.validate_summary(row, summaries[key])
        e.bind(folder + 'oof.npz', row['oof_sha256'])
    queue = e.read(BASE + 'campaign_queue.json')
    ledger = e.lines(BASE + 'round_decisions.jsonl')
    require(all(job['status'] in ('completed', 'failed', 'pruned') for job in queue['jobs'])
            and len({job['experiment_id'] for job in queue['jobs']}) == len(queue['jobs']), 'Campaign queue is not terminal/unique')
    completed = {job['experiment_id'] for job in queue['jobs'] if job['status'] == 'completed'}
    require(completed == {row['experiment_id'] for row in rows if row['version'] == 'research_v3'},
            'Train analysis must include every completed campaign job exactly once')
    require(ledger and len({row['round'] for row in ledger}) == len(ledger)
            and {job['round'] for job in queue['jobs']}.issubset({row['round'] for row in ledger}), 'Round ledger is incomplete')
    omitted_hashes = []
    for decision in ledger:
        if 'analysis' in decision:
            old = e.read(decision['analysis'], decision.get('analysis_sha256'))
            old_index = {(row['version'], row['experiment_id']): row for row in old['rows']}
            for item in decision['results']:
                name = item.get('experiment_id', item.get('id'))
                for field in ('correct', 'hard_correct', 'hard_n', 'hard_fold_mean', 'training_seconds'):
                    require(old_index['research_v3', name][field] == index['research_v3', name][field],
                            'Historical and final completed results disagree')
            if 'analysis_sha256' not in decision:
                omitted_hashes.append(decision['round'])
    incomplete = [job for job in queue['jobs'] if job['status'] != 'completed']
    stability_ids = sorted(job['experiment_id'] for job in incomplete if job.get('config', {}).get('seed', SEED) != SEED)
    require(sorted(close['incomplete_stability_ids']) == stability_ids,
            'Closeout incomplete-seed disclosure differs from the actual terminal queue')
    for job in queue['jobs']:
        for attempt in job.get('attempts', []):
            if attempt.get('log'):
                e.bind(attempt['log'])
            if 'observed_wall_seconds' in attempt:
                require(finite(attempt['observed_wall_seconds']) >= 0, 'Recorded attempt cost cannot be negative')
    r05 = shared.reduced_tabm_evidence(e, index, summaries, analysis['folds_sha256'], freeze['data_manifest_sha256'])
    ref_dir = BASE + 'deterministic_train_references/'
    references, receipt = e.read(ref_dir + 'summary.json'), e.read(ref_dir + 'COMPLETION.json')
    require(references.get('schema') == 'mp2-v3-deterministic-train-references-v1'
            and receipt.get('schema') == 'mp2-v3-deterministic-train-references-complete-v1'
            and references.get('status') == receipt.get('status') == 'complete'
            and references.get('split') == 'train' and references.get('n') == 4755
            and references.get('new_fitting_runs') == receipt.get('new_fitting_runs') == 0
            and references.get('validation_test_ranking_read') is False and receipt.get('validation_test_ranking_read') is False
            and references.get('folds_sha256') == analysis['folds_sha256']
            and references['official_training_data_sha256']['code/data/manifest.json'] == freeze['data_manifest_sha256']
            and references['source_sha256'] == receipt['source_sha256'] and references['input_sha256'] == receipt['input_sha256'],
            'Completed no-fit train references are inconsistent')
    require(set(receipt['output_sha256']) == {ref_dir + leaf for leaf in
        ('summary.json', 'selections.npz', 'deterministic_train_references_v3.source.py')}, 'Reference completion inventory differs')
    bind_map(e, receipt['output_sha256'])
    for name, digest in {**references['source_sha256'], **references['input_sha256']}.items():
        if name.startswith('code/data/'):
            require(references['official_training_data_sha256'].get(name) == digest, 'Reference declared data hash differs')
        else:
            e.bind(name, digest)
    require(set(references['results']) == {'fixed0', 'valid_answer_majority'}, 'Both deterministic references are required')
    for result in references['results'].values():
        for subset, n in (('overall', 4755), ('hard', rows[0]['hard_n']), ('single_positive', rows[0]['single_n'])):
            metric = result[subset]; count(metric)
            folds = metric['folds']
            require(metric['n'] == n and sorted(v['fold'] for v in folds) == [0, 1, 2]
                    and sum(v['n'] for v in folds) == n and sum(v['correct'] for v in folds) == metric['correct'],
                    'Reference fold counts differ')
            values = [100 * v['correct'] / v['n'] for v in folds]
            require(abs(statistics.mean(values) - finite(metric['fold_mean'])) < 1e-10
                    and abs(statistics.stdev(values) - finite(metric['fold_sd'])) < 1e-10, 'Reference fold statistics differ')
    v2 = e.read(paths['v2_analysis']); require(v2.get('input_sha256'), 'Preserved V2 input receipts are missing')
    bind_map(e, v2['input_sha256'])
    fusion = e.read(paths['fusion_analysis'])
    require(fusion.get('schema') == 'mp2-fixed-fusion-analysis-v3' and fusion.get('status') == 'completed'
            and fusion.get('new_training_fits') == 0 and len(fusion['results']) == 4, 'All four completed fixed fusions are required')
    e.bind(BASE + 'fusion_registry.jsonl', fusion['registry_sha256'])
    for item in fusion['results']:
        raw = e.read(item['diagnostic_path'], item['diagnostic_sha256'])['fusion']
        require(raw['status'] == 'complete' and raw['metrics'] == item['metrics'] and raw['methods'] == item['methods']
                and raw['weights'] == item['weights'] and item['decision'].lower().startswith('reject'),
                'Rejected fixed-fusion summary differs from actual saved diagnostics')
        directory = Path(item['diagnostic_path']).parent
        e.bind((directory / 'fusion-oof.npz').as_posix(), raw['oof_sha256'])
        e.bind((directory / raw['registration_file']).as_posix(), raw['registration_sha256'])
    return {'rows': rows, 'index': index, 'queue': queue, 'ledger': ledger, 'incomplete': incomplete,
            'references': references, 'v2': v2, 'fusion': fusion, 'r05': r05,
            'stability': shared.stability_groups(rows), 'omitted_historical_analysis_hashes': omitted_hashes}


def confidence_evidence(e, final, index):
    """Optional complete saved diagnostic; never fit or alter any choices."""
    fit_path = BASE + 'confidence_calibration/report.json'
    val_path = BASE + 'confidence_validation/report.json'
    calibration_path = BASE + 'confidence_calibration/calibration.json'
    prep_path = BASE + 'selected_confidence/preparation.json'
    paths = (fit_path, val_path, calibration_path, prep_path)
    present = [(e.root / path).is_file() for path in paths]
    if not any(present):
        return {'status': 'not_included', 'rows': [],
                'text': 'No completed optional confidence diagnostic is included in this archive.'}
    require(all(present), 'Optional confidence evidence is partial; preserve it and complete the records')
    fit, validation, calibration, prep = [e.read(path) for path in paths]
    selected, chosen = final['selected'], final['chosen']
    version = 'research_v2' if selected['historical_fallback'] else 'research_v3'
    fit_correct = index[version, selected['name']]['correct']
    shared._validate_confidence(fit, 'train_oof', selected['selector'], 4755, fit_correct)
    shared._validate_confidence(validation, 'validation', selected['selector'], 798, chosen['correct'])
    selected_hash = e.hashes[SUB + 'SELECTED_MODEL.json']
    require('IN-SAMPLE' in fit['interpretation'] and calibration['source_split'] == 'train_oof'
            and calibration['selector'] == selected['selector'] and calibration['C'] == 1
            and calibration['seed'] == SEED and calibration['fit_n'] == 4755 and calibration['fit_correct'] == fit_correct
            and fit['source_sha256']['selected_model_json'] == selected_hash
            and validation['source_sha256']['calibration_json'] == e.hashes[calibration_path]
            and prep['selected_model_sha256'] == selected_hash and prep['new_official_score_runs'] == 0
            and prep['candidate_selection_changed'] is False, 'Saved confidence analysis does not match the unique archive selection')
    for path, record in ((fit_path, fit), (val_path, validation)):
        e.bind(Path(path).with_name('confidence-events.npz').as_posix(), record['events_sha256'])
    e.bind(prep['events'])
    require(validation['source_sha256']['prediction_events_json'] == e.hashes[prep['events']],
            'Confidence confirmation uses different selected-answer events')
    bind_map(e, prep['original_official_output_sha256'])
    rows = [[name, f(record[kind]['brier'], 4), f(record[kind]['ece_10_equal_width'], 4),
             f(record[kind]['aurc_discrete_mean'], 4)]
            for name, record, kind in (('Train OOF raw', fit, 'raw'), ('Train OOF mapped (in-sample)', fit, 'calibrated'),
                                       ('Used validation raw', validation, 'raw'), ('Used validation mapped', validation, 'calibrated'))]
    return {'status': 'complete', 'rows': rows,
        'text': 'Confidence is the probability of the actually chosen answer group. The fixed C=1 logistic map fits train OOF only; '
                'its fitted-OOF metrics are IN-SAMPLE, and validation was already used for confirmation. Ten equal-width ECE bins '
                'retain counts; AURC summarizes saved risk-coverage. The diagnostic changes no choices and is not loaded by formal prediction.'}


def prepare(root=ROOT, **paths):
    paths = {**DEFAULTS, **paths}; e = Evidence(root)
    final = final_evidence(e)
    close, selected, freeze = final['closeout'], final['selected'], final['freeze']
    train = train_evidence(e, paths, freeze, close)
    confidence = confidence_evidence(e, final, train['index'])
    rows, queue, ledger = train['rows'], train['queue'], train['ledger']
    v3 = [row for row in rows if row['version'] == 'research_v3']
    v2 = [row for row in rows if row['version'] == 'research_v2']
    require(1 <= len(v3) <= 36 and 1 <= len(v2) <= 20 and len(ledger) <= 16,
            'Complete evidence exceeds this ten-page layout; revise before rendering')
    for path in ('reports/research_v3/prepare_report_inputs.py', 'reports/research_v2/build_report.py',
                 'docs/research_v3/SEARCH_PLAN_V3.md', 'docs/research_v3/MODELS_SOURCES.md',
                 'docs/research_v3/OPTIMIZER_SOURCES.md', 'results/research_v3/model-source-evidence.json'):
        e.bind(path)
    fit_seconds = sum(row['training_seconds'] for row in v3)
    historical_seconds = sum(row['training_seconds'] for row in v2)
    attempts = [attempt for job in queue['jobs'] for attempt in job.get('attempts', [])]
    wall_seconds = sum(attempt.get('observed_wall_seconds', 0) for attempt in attempts)
    missing_wall = sum('observed_wall_seconds' not in attempt for attempt in attempts)
    scope = ('All values below are training OOF results on common question folds. Hard fold mean and sample SD are percentages; '
             'fit seconds sum the recorded inner/outer optimization cost. Full configuration IDs are immutable identifiers, not encodings of learning rates.')
    def page(*blocks): return {'blocks': list(blocks)}
    def experiment_table(part):
        return table(['Experiment ID', 'All correct', 'Hard correct', 'Hard mean +/- SD', 'Fit s'],
            [[row['experiment_id'], count(row), count(row, 'hard_'), f(row['hard_fold_mean'], 2) + ' +/- ' + f(row['hard_fold_sd'], 2),
              f(row['training_seconds'], 1)] for row in part], [2.8, .85, .85, 1.2, .65])
    refs = train['references']['results']
    ref_table = [[name, count(value['overall']), count(value['hard']), shared.fold_cell(value['hard'])]
                 for name, value in refs.items()]
    controls = ['linear_original', 'linear_original::mask_max', 'linear_original::group_max',
        'linear_original::group_lse_a0', 'linear_original::group_lse_a0.5', 'linear_original::group_lse_a1',
        'linear_group_a0', 'linear_group_a0p5', 'linear_group_a1', 'linear_group_hard',
        'mlp_group_s0', 'resmlp_group_s0', 'deepsets_group_s0', 'attention_group_s0']
    mechanism = {row['row_id']: row for row in train['v2']['rows']}
    require(set(controls).issubset(mechanism), 'Preserved mechanism controls are incomplete')
    mechanism_rows = [[name, count(mechanism[name]['metrics']['hard']), shared.fold_cell(mechanism[name]['metrics']['hard'])] for name in controls]
    round_rows, paired_rows = [], []
    for decision in ledger:
        jobs = [job for job in queue['jobs'] if job['round'] == decision['round']]
        complete = [train['index']['research_v3', job['experiment_id']] for job in jobs if job['status'] == 'completed']
        primary = [row for row in complete if row['seed'] == SEED]
        best = max(primary, key=lambda row: row['hard_fold_mean']) if primary else None
        round_rows.append([decision['round'], len(jobs), len(complete), f(best['hard_fold_mean'], 2) if best else 'No complete primary run'])
        if best and best.get('paired_vs_v2_attention'):
            pair = best['paired_vs_v2_attention']; interval = pair['hard_bootstrap_95pct']
            require(len(interval) == 2 and finite(interval[0]) <= finite(interval[1]), 'Saved train paired interval is invalid')
            paired_rows.append([decision['round'], f(pair['hard_delta_pp'], 2), '[' + f(interval[0], 2) + ', ' + f(interval[1], 2) + ']'])
    fusion_rows = [[item['experiment_id'], count(item['metrics']['hard']),
                   str(item['hard_delta_questions_vs_attention_control']), 'Rejected'] for item in train['fusion']['results']]
    stable_rows = [[item['version'] + '/' + item['experiment_id'], ','.join(str(seed)[-3:] for seed in item['seeds']),
                   f(item['mean'], 2), f(item['sample_sd'], 2)] for item in train['stability']]
    val = final['validation']
    val_rows = [['Historical V2 fallback', count(val['baseline_validation']), count(val['baseline_validation'], 'hard_')]]
    val_rows += [[item['name'], count(item['validation']), count(item['validation'], 'hard_')] for item in val['finalists']]
    recipe_rows = [[recipe['experiment_id'], recipe['configuration']['family'], recipe['seed'], recipe['epochs'],
                    recipe['preprocess'], recipe['evaluation_weights']] for recipe in freeze['training_recipes']]
    prior_score = final['prior']['actual_score']
    if selected['historical_fallback']:
        base = val['baseline_validation']
        no_gain = all(item['validation'][key] <= base[key] for item in val['finalists'] for key in ('correct', 'hard_correct'))
        confirmation_text = ('No locked V3 candidate improved either overall or hard validation counts over the preserved V2 fallback. '
                             if no_gain else 'No locked V3 candidate satisfied the fixed replacement rule. ')
        confirmation_text += 'The historical V2 checkpoint is retained byte-for-byte; no accepted replacement is claimed.'
    else:
        confirmation_text = 'The locked new candidate satisfied the saved confirmation rule. This offline outcome is not a formal score update.'
    status_text = ('The formal deadline was missed. No V3 official test was run, no new score Issue was posted, and no V3 score update is claimed. '
        f"The earlier V1 Issue #{final['prior']['issue_number']} remains the submitted result: {count(prior_score)} "
        f"({100 * prior_score['correct'] / prior_score['n']:.4f}%). This historical test score does not evaluate the V3 archive model.")
    pages = [page(block('title', 'DASE7506 MP2\nPost-deadline research closeout'),
        block('subtitle', 'Student 3036900957 | Research archive only | Not a new course submission'),
        block('heading', NOTICE), block('paragraph', status_text),
        block('paragraph', confirmation_text),
        block('paragraph', f"The retained research model is {selected['name']}. Its saved offline validation outcome is "
              f"{count(final['chosen'])}, hard {count(final['chosen'], 'hard_')}. Independent review and a fresh-environment validation reproduction "
              'passed before archive freeze. These checks establish reproducibility; they neither reopen submission nor establish a test score or rank.'),
        table(['Evidence', 'Actual scope'], [['Complete V3 runs', str(len(v3)) + ' common-three-fold configurations'],
            ['Preserved V2 runs', str(len(v2)) + ' historical configurations'], ['Fixed fusions', str(len(fusion_rows)) + ' rejected comparisons'],
            ['Incomplete campaign jobs', str(len(train['incomplete'])) + ' excluded from complete-CV tables'],
            ['V3 formal test / update', 'NOT PERFORMED']], [1.4, 3.8]),
        block('paragraph', close['monitor_gap_disclosure']),
        block('small', 'Deadline: 2026-10-31 16:00 UTC (2026-11-01 00:00 UTC+8). Research closure after expiry is documented, not a claim that every planned experiment finished within the original budget.')),
      page(block('heading', '1. Representation and controlled comparisons'),
        block('paragraph', 'Four candidate answers share an official answer parser and one question-level decision. Relation inputs contain 1,024 embedding coordinates, their 1,024 absolute deviations from the candidate mean, and 16 parsing/vote/text statistics. Reduced inputs omit the deviation block. Features exclude IDs and references. Scalers fit only the fitting partition.'),
        block('paragraph', 'The selector masks unparseable answers, groups equal parsed values and uses group log-sum-exp minus alpha times log group size, at tau=1. Deterministic ties return an original candidate index; all-unparseable rows return index0 with confidence zero. TabM averages member answer-group probabilities, not logits.'),
        table(['Architecture / control', 'Implementation tested'], [['MLP / ResMLP', 'Shared candidate network / residual transformations'],
            ['Deep Sets / Attention', 'Permutation-compatible candidate context, without positional IDs'],
            ['TabM', 'Shared matrices with member adapters; average member losses'],
            ['RealMLP-inspired', 'Input scale, NTP, SELU, zero head and robust preprocessing controls'],
            ['Muon / EMA', 'Hidden-matrix Muon plus auxiliary AdamW; same-trajectory post-step EMA']], [1.4, 3.8]),
        block('small', train['r05']['paragraph']),
        table(['No-fit train reference', 'All correct', 'Hard correct', 'Hard mean +/- SD'], ref_table, [1.7, 1, 1, 1.5]),
        block('small', 'Fixed0 always returns original index0. Majority ignores None, groups equal official parsed values, breaks ties by earliest original index and returns0 if all are unparseable. Their fold statistics are descriptive partitions, not three fits.')),
      page(block('heading', '2. Complete V3 training experiments (1/2)'), block('small', scope), experiment_table(v3[:18]),
        block('small', 'All rows, including losses, rates, preprocessing, decoder, source hashes and numeric asset bytes, remain in the bound final_search/EXPERIMENT_TABLE.md and original per-experiment summaries.')),
      page(block('heading', '3. Complete V3 training experiments (2/2)'), block('small', scope), experiment_table(v3[18:]),
        block('paragraph', f"Recorded completed-V3 fit time totals {f(fit_seconds, 1)} s. Recorded process-attempt wall time totals {f(wall_seconds, 1)} s; "
              f"{missing_wall} attempts lack that field. These clocks have different scopes and are not added. Wall time includes elapsed gaps and startup; it does not establish continuous CPU execution or continuous monitoring.")),
      page(block('heading', '4. Preserved historical V2 experiments'), block('small', scope), experiment_table(v2),
        block('small', f"Historical V2 recorded fit time: {f(historical_seconds, 1)} s. Those earlier runs and assets are retained unchanged. They are not additional V3 configurations or newly obtained test results.")),
      page(block('heading', '5. Decoder, objective and stability evidence'),
        table(['Saved V2 control', 'Hard correct', 'Hard mean +/- SD'], mechanism_rows, [2.6, 1, 1.6]),
        block('small', 'The six linear_original rows reuse identical scores to isolate masking/group decoding. Original listwise training includes all candidates in its denominator and all positives in its numerator; it does not relabel alternate correct answers negative. Group CE, listwise, BCE and valid positive-negative softplus pairs are distinct registered objectives. Hard/full reduction requires corresponding regularization accounting.'),
        table(['Completed identical recipe', 'Seed suffixes', 'Mean hard %', 'Across-seed SD'], stable_rows, [2.7, .8, .85, .85]),
        block('small', 'Only genuinely completed identical recipes enter this descriptive seed table. Across-seed sample SD is not a confidence interval or fold SD. The deployment seed remains3036900957. Incomplete V3 repeat seeds cannot establish V3 seed stability.')),
      page(block('heading', '6. Research rounds, interruptions and limits'),
        table(['Registered round', 'Jobs', 'Complete', 'Best base-seed hard mean %'], round_rows, [2.4, .6, .7, 1.5]),
        table(['Incomplete job', 'Terminal status', 'Last recorded reason'], [[job['experiment_id'], job['status'],
             (job.get('attempts') or [{}])[-1].get('stop_reason', 'No reason in the saved attempt')] for job in train['incomplete']], [2.5, .7, 2]),
        block('small', 'Incomplete jobs and partial folds remain preserved and are excluded from whole-CV rankings. A logged fold seed is the fixed base seed plus the fold offset, not an independently selected repeat. Epoch checkpoints preserve model/optimizer/EMA/RNG state; a resumed attempt is not counted as another completed configuration.'),
        block('small', 'Original planning used a 24-hour absolute research clock and per-attempt budgets. Actual monitoring gaps, late closure and deadline failure are disclosed on page1; no claim of uninterrupted 24-hour CPU work or successful budget enforcement is made. Missing timing is not invented. Historical ledger rows without an analysis hash: ' + (', '.join(train['omitted_historical_analysis_hashes']) or 'none') + '. Their currently verified summaries are bound without rewriting prior records.')),
      page(block('heading', '7. Negative fixed fusions and paired uncertainty'),
        table(['Fixed probability mixture', 'Hard correct', 'Delta questions vs control', 'Decision'], fusion_rows, [2.8, .8, .9, .7]),
        block('small', train['fusion']['cost_note']),
        block('paragraph', 'Each fusion followed saved reciprocal-error/oracle evidence and one registered equal-probability mixture. All four were rejected. No result is rescued by a post-hoc weight, temperature, member, seed or routing sweep. Passing a complementarity gate alone did not guarantee an improvement.'),
        table(['Base-seed round representative', 'Hard delta pp', 'Saved paired 95% interval'], paired_rows, [2.6, 1, 1.6]),
        block('small', 'Round representatives are selected from completed train OOF results and compared with preserved V2 Attention predictions. Saved paired intervals are conditional on those predictions and retain search-selection optimism. They are not test intervals, proof of a causal architecture effect, or a ranking claim.')),
      page(block('heading', '8. Offline confirmation and clean reproduction'),
        block('paragraph', 'After search closure, at most two fixed-seed finalists were locked for validation. The predeclared confirmation/fallback rule selected one archive checkpoint. Validation had already informed earlier work; this is not an untouched-holdout estimate. No subsequent test was opened.'),
        table(['Locked candidate / fallback', 'All validation', 'Hard validation'], val_rows, [2.8, 1.2, 1.2]),
        block('small', confirmation_text),
        table(['Component', 'Family', 'Seed', 'Epochs', 'Preprocess', 'Weights'], recipe_rows, [1.8, 1, 1.1, .5, 1, .8]),
        block('paragraph', f"Fresh-environment reproduction retained identical saved choices and official validation score: {count(final['fresh']['score'])}. "
              f"Prediction plus scoring took {f(final['fresh']['timing']['prediction_plus_scoring_seconds'], 6)} s; frozen inference assets total "
              f"{freeze['resource']['uncompressed_bytes']} uncompressed bytes. These are validation reproduction measurements, not V3 test latency or acceptance by the course."),
        block('small', 'Exact configuration, preprocessing, decoder, seed/epoch aggregation and EMA/live weight choice are bound in FREEZE.training_recipes and the original final_fit records. Independent contract/export checks and clean-environment receipts must pass before this report can be generated.'),
        *([table(['Chosen-event diagnostic', 'Brier', 'ECE10', 'AURC'], confidence['rows'], [2.5, .9, .9, .9])] if confidence['rows'] else []),
        block('small', confidence['text'])),
      page(block('heading', '9. Reproduction, provenance and honest limits'),
        block('paragraph', 'Archive source commit: ' + freeze['source_commit'] + '. Model: code/solution_v3.py; assets: artifacts/research_v3/final_selected. Follow the retained source/asset hashes and environment package versions, then reproduce validation predictions and official scoring using scripts/reproduce_selected_v3.py. The post-deadline block remains active; do not invoke a new formal test or score Issue.'),
        block('paragraph', 'Training regeneration uses the committed configurations and common_folds.json through code/train_v3.py, including fold-local scaling, nested stopping and fresh refits. Existing closed-search guards remain active. Reproduction of archived predictions is separate from reopening model research.'),
        block('paragraph', 'AI assistance was substantive: source inspection, implementation, experiment orchestration, diagnostics, report preparation and internal AI review. Actual logs and hashes support numerical claims; AI review is not instructor approval or a claim of independent human understanding. Primary model/optimizer references, author repository commits and licenses are retained in MODELS_SOURCES.md, OPTIMIZER_SOURCES.md and model-source-evidence.json.'),
        block('paragraph', 'Limitations: missed formal deadline; monitoring gaps; four incomplete V3 seed repeats; selection-optimistic CV/bootstrap; validation reuse; known historical test aggregate; compact task adaptations rather than full source-paper benchmarks; negative fusions; and no new V3 test result. No training or validation statistic establishes a leaderboard position.'),
        block('small', 'Peer review is handled through the instructor\'s actual released workflow as a separate task. This report neither marks it complete nor postpones it by assumption. Consult its independent current evidence.'),
        block('small', 'Evidence index: final_search/EXPERIMENT_TABLE.md; round_decisions.jsonl; campaign_queue.json; fixed_fusion_analysis_complete.json; deterministic_train_references; R05 paired-and-slices; SELECTED_MODEL/FREEZE/REVIEW_RESULT/REPRODUCTION_RESULT; FORMAL_SUBMISSION_BLOCKED/RESEARCH_CLOSEOUT. All are hash-bound in closeout_inputs.json.'),
        block('heading', NOTICE))]
    return {'schema': 'mp2-v3-post-deadline-report-inputs-v1', 'title': 'DASE7506 MP2: post-deadline research closeout',
        'created_at_utc': datetime.now(timezone.utc).isoformat(), 'student_id': '3036900957',
        'research_only': True, 'no_new_test': True, 'no_new_score_update': True,
        'generator_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'pages': pages,
        'evidence_sha256': dict(sorted(e.hashes.items())), 'paths': paths,
        'derived_aggregates': {'completed_v3': len(v3), 'historical_v2': len(v2), 'fixed_fusions': len(fusion_rows),
            'confidence': confidence, 'confirmation_outcome': confirmation_text,
            'incomplete_stability_ids': close['incomplete_stability_ids'], 'seed_stability': train['stability'],
            'v3_fit_seconds': fit_seconds, 'historical_fit_seconds': historical_seconds, 'recorded_attempt_seconds': wall_seconds,
            'attempts_without_wall_seconds': missing_wall, 'r05': train['r05']},
        'archive_facts': {'selected_name': selected['name'], 'validation': final['chosen'], 'prior_issue_url': close['prior_issue_url'],
            'prior_score': prior_score, 'source_commit': freeze['source_commit'], 'asset_bytes': freeze['resource']['uncompressed_bytes'],
            'validation_reproduction_seconds': final['fresh']['timing']['prediction_plus_scoring_seconds'],
            'monitor_gap_disclosure': close['monitor_gap_disclosure']}}


def chinese_guide(record):
    facts, stats = record['archive_facts'], record['derived_aggregates']
    return ('# V3 截止后研究归档说明\n\n'
        '**已错过正式截止时间。本轮 V3 没有进行正式 test，也没有创建新成绩 Issue 或更新课程成绩。**\n\n'
        f"保留的原 V1 提交是 {facts['prior_issue_url']}，历史成绩 {count(facts['prior_score'])}。它不是 V3 新模型的测试成绩。\n\n"
        f"研究保留模型：{facts['selected_name']}。实际离线验证为 {count(facts['validation'])}，难题 {count(facts['validation'], 'hard_')}。"
        f"新环境复现验证预测及评分一致，耗时 {f(facts['validation_reproduction_seconds'], 6)} 秒；推理资产 {facts['asset_bytes']} 字节。\n\n"
        f"完整训练内三折实验：V3 {stats['completed_v3']} 项，历史 V2 {stats['historical_v2']} 项；固定融合 {stats['fixed_fusions']} 项均拒绝。"
        f"未完成的 V3 稳定性实验：{', '.join(stats['incomplete_stability_ids'])}。不能用部分折推断完整成绩或宣称完成多种子验证。\n\n"
        + facts['monitor_gap_disclosure'] + '\n\n'
        '报告区分优化耗时、进程经过时间和监控间断，不声称连续运行24小时。训练 OOF 和已使用的 validation 只能支持有限研究结论，不能推断榜单名次。'
        '缩减输入同时改变容量与初始化；配对区间保留选模偏差；历史测试汇总已知。\n\n'
        f"归档提交：{facts['source_commit']}。按 FREEZE 与复现记录校验源码和资产，复现已归档验证预测；不得因此重新打开正式测试或成绩更新。"
        '英文报告逐页证据和完整表见 closeout_inputs.json 及其 evidence_sha256。正式报告旧流程仍要求 TEST_COMPLETE，本流程没有伪造该记录。\n\n'
        'AI 实质参与了代码、实验、检查和写作；内部 AI 审查不等于教师认可。同行评审按教师实际开放流程单独处理，本说明不宣称已完成。\n')


def write_outputs(record, output, chinese_output):
    paths = [Path(output), Path(chinese_output)]
    require(paths[0].resolve() != paths[1].resolve(), 'Output paths must differ')
    require(not any(path.exists() for path in paths), 'Preserve existing closeout evidence; choose new output paths')
    for path in paths: path.parent.mkdir(parents=True, exist_ok=True)
    with paths[0].open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2, allow_nan=False); stream.write('\n')
    with paths[1].open('x', encoding='utf-8', newline='\n') as stream: stream.write(chinese_guide(record))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key, value in DEFAULTS.items(): parser.add_argument('--' + key.replace('_', '-'), default=value)
    parser.add_argument('--output', type=Path, default=ROOT / 'reports/research_v3/closeout_inputs.json')
    parser.add_argument('--chinese-output', type=Path, default=ROOT / 'reports/research_v3/closeout_zh.md')
    args = parser.parse_args()
    record = prepare(**{key: getattr(args, key) for key in DEFAULTS})
    write_outputs(record, args.output, args.chinese_output)
    print(json.dumps({'output': str(args.output), 'intended_pages': len(record['pages']), 'research_only': True}))


if __name__ == '__main__': main()
