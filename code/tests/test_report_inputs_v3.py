"""Generated aggregate fixtures only; no official banks, models or final PDF."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('report_inputs_v3', ROOT / 'reports/research_v3/prepare_report_inputs.py')
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)


class ReportInputTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=ROOT / 'tmp', prefix='report-v3-synthetic-')
        self.root = Path(self.tmp.name)
        self.fixture()

    def tearDown(self):
        self.tmp.cleanup()

    def put(self, name, value):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value) if not isinstance(value, str) else value, encoding='utf-8')
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def read(self, name):
        return json.loads((self.root / name).read_text(encoding='utf-8'))

    def fixture(self):
        self.metrics = {'correct': 740, 'n': 798, 'accuracy': 100 * 740 / 798,
                        'hard_correct': 297, 'hard_n': 355, 'hard_accuracy': 100 * 297 / 355}
        selector = {'kind': 'single_model', 'experiment_id': 'historical_s0', 'family': 'deepsets', 'decoder': {'mode': 'group_lse'}}
        closed_hash = self.put(r.BASE + 'SEARCH_CLOSED.json', {'search_closed': True})
        lock_hash = self.put(r.BASE + 'FINALISTS_LOCK.json', {'synthetic': True})
        validation_hash = self.put(r.BASE + 'validation_summary.json', {'schema': 'mp2-validation-confirmation-v3',
            'selected': {'name': 'historical_s0', 'historical_fallback': True}, 'finalists': [],
            'baseline_validation': self.metrics, 'search_remains_closed': True,
            'new_test_evaluated': False, 'evidence_sha256': {}})
        selected_hash = self.put(r.SUB + 'SELECTED_MODEL.json', {'schema': 'mp2-selected-model-v3',
            'name': 'historical_s0', 'historical_fallback': True, 'method': 'code/solution_v3.py',
            'checkpoint': 'artifacts/research_v3/final_selected', 'deployment_seed': r.SEED,
            'selector': selector, 'validation_summary_sha256': validation_hash})
        fresh_score_path = r.BASE + 'fresh/score.json'
        fresh_timing_path = r.BASE + 'fresh/timing.json'
        fresh_environment_path = r.BASE + 'fresh/environment.json'
        fresh = {fresh_score_path: self.put(fresh_score_path, {'split': 'validation', 'correct': 740, 'n': 798}),
                 fresh_timing_path: self.put(fresh_timing_path, {'status': 'passed', 'local_resource_eligible': True, 'prediction_plus_scoring_seconds': 5.5}),
                 fresh_environment_path: self.put(fresh_environment_path, {'schema': 'mp2-v3-clean-environment-v1'})}
        review_hash = self.put(r.SUB + 'REVIEW_RESULT.json', {'passed': True, 'selected_model_sha256': selected_hash, 'evidence_files': {}})
        repro_hash = self.put(r.SUB + 'REPRODUCTION_RESULT.json', {'passed': True, 'selected_model_sha256': selected_hash,
            'source_commit': 'a' * 40, 'fresh_environment': True, 'choices_match': True, 'official_score_match': True,
            'evidence_files': fresh, 'score_result': {'path': fresh_score_path, 'sha256': fresh[fresh_score_path]},
            'timing_result': {'path': fresh_timing_path, 'sha256': fresh[fresh_timing_path]},
            'environment_result': {'path': fresh_environment_path, 'sha256': fresh[fresh_environment_path]}})
        final_fit_path = 'results/research_v2/historical_s0/final_fit.json'
        final_fit_hash = self.put(final_fit_path, {'synthetic': True})
        resource = {'within_limit': True, 'uncompressed_bytes': 123456}
        freeze = {'frozen': True, 'test_evaluated_before_freeze': False, 'source_commit': 'a' * 40,
            'selected_model_sha256': selected_hash, 'validation_summary_sha256': validation_hash,
            'finalists_lock_sha256': lock_hash, 'search_closed_sha256': closed_hash, 'selected_name': 'historical_s0',
            'selector': selector, 'method': 'code/solution_v3.py', 'checkpoint': 'artifacts/research_v3/final_selected',
            'review_result_sha256': review_hash, 'reproduction_result_sha256': repro_hash,
            'protocol': 'synthetic-score-protocol', 'resource': resource, 'source_files': {}, 'assets': {},
            'data_manifest_sha256': '1' * 64,
            'known_old_test_aggregate': {'correct': 348, 'n': 467, 'known_before_v3': True},
            'training_recipes': [{'experiment_id': 'historical_s0', 'configuration': {'family': 'deepsets'},
                'seed': r.SEED, 'epochs': 9, 'preprocess': 'standard', 'evaluation_weights': 'live',
                'final_fit_path': final_fit_path, 'final_fit_sha256': final_fit_hash}]}
        freeze_hash = self.put(r.SUB + 'FREEZE.json', freeze)
        score = {'split': 'test', 'protocol': 'synthetic-score-protocol', 'n': 467, 'correct': 350, 'accuracy': 350 / 467 * 100}
        timing = {'status': 'passed', 'local_resource_eligible': True, 'freeze_sha256': freeze_hash,
                  'resource': resource, 'prediction_plus_scoring_seconds': 4.0, 'sampled_peak_task_process_tree_threads': 2,
                  'scientific_library_thread_cap_passed': True}
        outputs = {name: self.put(name, score if name.endswith('test-score.json') else timing if name.endswith('test-timing.json') else 'synthetic bytes, deliberately not JSON') for name in r.TEST_OUTPUTS}
        self.put(r.SUB + 'TEST_COMPLETE.json', {'schema': 'mp2-v3-test-complete-v1', 'status': 'passed',
            'local_resource_eligible': True, 'split': 'test', 'n': 467, 'freeze_sha256': freeze_hash,
            'method': 'code/solution_v3.py', 'checkpoint': 'artifacts/research_v3/final_selected', 'output_sha256': outputs})
        folds = self.put(r.BASE + 'common_folds.json', {'synthetic': True})
        rows = []
        for name, seed, version in [('primary_s0', r.SEED, 'research_v3'), ('primary_s1', r.SEED + 1, 'research_v3'),
                                     ('historical_s0', r.SEED, 'research_v2')]:
            directory = f'results/{version}/{name}/'
            config = {'family': 'mlp', 'seed': seed, 'experiment_id': name}
            summary = {'experiment_id': name, 'status': 'complete', 'config': config, 'n': 4755, 'correct': 4200,
                'hard_n': 2100, 'hard_correct': 1575, 'training_seconds': 2.,
                'folds': [{'fold': i, 'n': 1585, 'correct': 1400, 'hard_n': 700, 'hard_correct': 518 + 7 * i,
                           'parameters': 1000, 'numeric_asset_bytes': 4000} for i in range(3)]}
            rows.append({'version': version, 'experiment_id': name, 'seed': seed, 'family': 'mlp', 'status': 'completed',
                'n': 4755, 'correct': 4200, 'hard_n': 2100, 'hard_correct': 1575, 'accuracy': 4200 / 4755 * 100,
                'hard_fold_accuracies': [74., 75., 76.], 'hard_fold_mean': 75., 'hard_fold_sd': 1., 'training_seconds': 2.0,
                'parameters': 1000, 'numeric_asset_bytes': 4000, 'single_n': 711,
                'summary_sha256': self.put(directory + 'summary.json', summary),
                'oof_sha256': self.put(directory + 'oof.npz', 'synthetic opaque bytes'),
                'config': config,
                'paired_vs_v2_attention': {'hard_delta_pp': -.5, 'hard_bootstrap_95pct': [-1.0, .1]}})
        analysis = {'rows': rows, 'folds_sha256': folds, 'noncomplete_jobs': [],
                    'source_sha256': self.put('scripts/analyze_campaign_v3.py', '# synthetic source')}
        analysis_hash = self.put(r.DEFAULTS['analysis'], analysis)
        def reference_metric(correct, n):
            return {'correct': correct, 'n': n, 'fold_mean': 100 * correct / n, 'fold_sd': 0.,
                    'folds': [{'fold': i, 'correct': correct // 3, 'n': n // 3} for i in range(3)]}
        reference_source = 'code/tools/deterministic_train_references_v3.py'
        reference_npz = r.BASE + 'deterministic_train_references/selections.npz'
        reference_summary_path = r.BASE + 'deterministic_train_references/summary.json'
        reference_summary_hash = self.put(reference_summary_path, {
            'schema': 'mp2-v3-deterministic-train-references-v1', 'status': 'complete',
            'split': 'train', 'n': 4755, 'new_fitting_runs': 0, 'validation_test_ranking_read': False,
            'majority_independent_group_count_choices_equal': True, 'folds_sha256': folds,
            'official_training_data_sha256': {'code/data/manifest.json': '1' * 64},
            'source_sha256': {reference_source: self.put(reference_source, '# synthetic source')},
            'input_sha256': {r.BASE + 'common_folds.json': folds},
            'selections': {'path': reference_npz, 'sha256': self.put(reference_npz, 'synthetic bytes')},
            'results': {name: {'overall': reference_metric(3600, 4755),
                              'hard': reference_metric(1200, 2100),
                              'single_positive': reference_metric(150, 711)}
                        for name in ('fixed0', 'valid_answer_majority')}})
        reference_summary = self.read(reference_summary_path)
        reference_snapshot = r.BASE + 'deterministic_train_references/deterministic_train_references_v3.source.py'
        self.put(r.BASE + 'deterministic_train_references/COMPLETION.json', {
            'schema': 'mp2-v3-deterministic-train-references-complete-v1', 'status': 'complete',
            'new_fitting_runs': 0, 'validation_test_ranking_read': False,
            'source_sha256': reference_summary['source_sha256'], 'input_sha256': reference_summary['input_sha256'],
            'output_sha256': {reference_summary_path: reference_summary_hash,
                             reference_npz: reference_summary['selections']['sha256'],
                             reference_snapshot: self.put(reference_snapshot, '# synthetic source')}})
        self.put(str(Path(r.DEFAULTS['analysis']).with_name('EXPERIMENT_TABLE.md')), '# Synthetic full table')
        jobs = [{'experiment_id': row['experiment_id'], 'round': 'r01_optimizer', 'status': 'completed',
                 'attempts': [{'observed_wall_seconds': 3}]} for row in rows if row['version'] == 'research_v3']
        self.put(r.BASE + 'campaign_queue.json', {'jobs': jobs})
        round_ = {'round': 'r01_optimizer', 'configurations': 2, 'completed_outer_fits': 6,
                  'results': rows[:2], 'exceeds_registered_half_point': False,
                  'analysis': r.DEFAULTS['analysis'], 'analysis_sha256': analysis_hash}
        self.put(r.BASE + 'round_decisions.jsonl', json.dumps(round_) + '\n')
        metric = {'correct': 1500, 'n': 2000, 'fold_mean': 75., 'fold_sd': 1.}
        self.put(r.DEFAULTS['v2_analysis'], {'input_sha256': {'results/research_v2/historical_s0/summary.json': rows[-1]['summary_sha256']},
            'rows': [{'row_id': name, 'metrics': {'hard': metric, 'overall': {'correct': 4200, 'n': 4755}}} for name in (
            'linear_original', 'linear_original::mask_max', 'linear_original::group_max', 'linear_original::group_lse_a0',
            'linear_original::group_lse_a0.5', 'linear_original::group_lse_a1', 'linear_group_a0', 'linear_group_a0p5',
            'linear_group_a1', 'linear_group_hard', 'mlp_group_s0', 'resmlp_group_s0', 'deepsets_group_s0', 'attention_group_s0')]})
        registry = self.put(r.BASE + 'fusion_registry.jsonl', 'synthetic registry bytes')
        diagnostic_path = r.BASE + 'fusion/diagnostic.json'
        actual = {'status': 'complete', 'metrics': {'hard': metric, 'overall': {'correct': 4200, 'n': 4755}},
                  'methods': ['v2:historical_s0', 'v3:primary_s0'], 'weights': [.5, .5],
                  'oof_sha256': self.put(r.BASE + 'fusion/fusion-oof.npz', 'synthetic opaque bytes'),
                  'registration_file': 'fusion-registration.json',
                  'registration_sha256': self.put(r.BASE + 'fusion/fusion-registration.json', {'synthetic': True})}
        item = {'experiment_id': 'synthetic_fusion', 'metrics': actual['metrics'], 'methods': actual['methods'],
            'weights': actual['weights'], 'diagnostic_path': diagnostic_path,
            'diagnostic_sha256': self.put(diagnostic_path, {'fusion': actual}),
            'hard_delta_questions_vs_attention_control': -2, 'maximum_member_asset_bytes': 20000,
            'decision': 'Synthetic negative result; reject.'}
        self.put(r.DEFAULTS['fusion_analysis'], {'schema': 'mp2-fixed-fusion-analysis-v3', 'status': 'completed',
            'new_training_fits': 0, 'registry_sha256': registry, 'results': [item],
            'decision': 'Synthetic negative result; no further sweep.', 'cost_note': 'Reuses trained members.'})
        for key, split, n, correct in [('confidence_fit', 'train_oof', 4755, 4200), ('confidence_validation', 'validation', 798, 740)]:
            metric = {'n': n, 'correct': correct, 'brier': .1, 'ece_10_equal_width': .05, 'aurc_discrete_mean': .03,
                'bins': [{'bin': i, 'lower': i / 10, 'upper': (i + 1) / 10, 'n': n if i == 9 else 0} for i in range(10)],
                'risk_coverage': {'retained': list(range(1, n + 1)), 'risk': [.05] * n}}
            path = r.DEFAULTS[key]
            self.put(path, {'schema': 'mp2-v3-confidence-report-v1', 'split': split, 'selector': selector, 'choices_changed': 0,
                'interpretation': 'IN-SAMPLE fit only' if split == 'train_oof' else 'Locked validation',
                'source_sha256': {'selected_model_json': selected_hash}, 'raw': metric, 'calibrated': metric,
                'events_sha256': self.put(str(Path(path).with_name('confidence-events.npz')), 'synthetic opaque bytes')})
        self.put(str(Path(r.DEFAULTS['confidence_fit']).with_name('calibration.json')), {'source_split': 'train_oof',
            'selector': selector, 'C': 1., 'seed': r.SEED, 'fit_n': 4755, 'fit_correct': 4200, 'coefficient': .8, 'intercept': .2})
        events = r.BASE + 'selected_confidence/events.json'
        self.put(events, 'synthetic event metadata not parsed')
        self.put(r.BASE + 'selected_confidence/preparation.json', {'selected_model_sha256': selected_hash,
            'new_official_score_runs': 0, 'candidate_selection_changed': False, 'events': events, 'original_official_output_sha256': {}})
        pair = {'bootstrap_repetitions': 5000, 'interval_caveat': 'Used post-selection validation; descriptive only.'}
        for subset, prefix, offset in (('overall', '', 0), ('hard', 'hard_', 1)):
            pair[subset] = {'n': self.metrics[prefix + 'n'], 'first_correct': self.metrics[prefix + 'correct'],
                'second_correct': self.metrics[prefix + 'correct'], 'correct_difference': 0,
                'difference_percentage_points': 0., 'paired_bootstrap_95_interval': [0., 0.], 'bootstrap_seed': 3036901059 + offset}
        self.put(r.BASE + 'selected_validation_comparison.json', {'schema': 'mp2-v3-selected-validation-comparison-v1',
            'selected_model_sha256': selected_hash, 'selected_name': 'historical_s0', 'split': 'validation',
            'new_official_score_runs': 0, 'model_selection_changed': False, 'test_opened': False,
            'data_manifest_sha256': '1' * 64, 'evidence_sha256': {},
            'comparison_source_sha256': self.put('code/tools/analyze_selected_validation_v3.py', '# synthetic source'),
            'bootstrap_source_sha256': self.put('code/tools/analyze_oof_v2.py', '# synthetic bootstrap source'),
            'comparisons': {'frozen_v1': pair, 'historical_v2_selected': pair}})
        conf_validation = self.read(r.DEFAULTS['confidence_validation'])
        conf_validation['source_sha256'].update({
            'calibration_json': hashlib.sha256((self.root / Path(r.DEFAULTS['confidence_fit']).with_name('calibration.json')).read_bytes()).hexdigest(),
            'prediction_events_json': hashlib.sha256((self.root / events).read_bytes()).hexdigest()})
        self.put(r.DEFAULTS['confidence_validation'], conf_validation)
        for name in ('docs/research_v3/SEARCH_PLAN_V3.md', 'docs/research_v3/MODELS_SOURCES.md',
                     'docs/research_v3/OPTIMIZER_SOURCES.md', 'results/research_v3/model-source-evidence.json'):
            self.put(name, 'Synthetic source evidence')
        figures = []
        figure_generator = 'reports/research_v3/prepare_figures.py'
        figure_source_hash = self.put(figure_generator, '# synthetic figure generator')
        for number, name in enumerate(('cv_fit_cost_vs_hard_accuracy', 'chosen_event_confidence')):
            path = f'reports/research_v3/figures/{name}.png'
            receipt = f'reports/research_v3/figures/{name}.generation.json'
            image_hash = self.put(path, 'synthetic image placeholder')
            subset = ([r.DEFAULTS['analysis'], r.BASE + 'campaign_queue.json'] if number == 0
                      else [r.DEFAULTS['confidence_fit'], r.DEFAULTS['confidence_validation']]) + [figure_generator]
            generation = {'schema': 'mp2-v3-figure-generation-v1', 'scope': 'saved_final_evidence_only',
                'path': path, 'sha256': image_hash, 'generator_sha256': figure_source_hash,
                'format': 'PNG', 'report_height_points': 235, 'caption': 'Synthetic aggregate-only figure',
                'raw_bank_or_test_read': False,
                'source_sha256': {key: hashlib.sha256((self.root / key).read_bytes()).hexdigest() for key in subset}}
            figures.append({'path': path, 'sha256': image_hash,
                'height': 235, 'caption': 'Synthetic aggregate-only figure',
                'generation_record': {'path': receipt, 'sha256': self.put(receipt, generation)}})
        self.put(r.DEFAULTS['figures'], {'schema': 'mp2-v3-report-figures-v1', 'scope': 'saved_final_evidence_only', 'figures': figures,
            'evidence_sha256': {path: hashlib.sha256((self.root / path).read_bytes()).hexdigest()
                for path in (r.DEFAULTS['analysis'], r.BASE + 'campaign_queue.json', r.DEFAULTS['confidence_fit'], r.DEFAULTS['confidence_validation'], figure_generator)}})

    def test_ten_pages_real_aggregate_contract_and_fallback(self):
        record = r.prepare(self.root)
        self.assertEqual(record['schema'], 'mp2-v3-final-report-inputs-v1')
        self.assertEqual(len(record['pages']), 10)
        self.assertTrue(set(r.REQUIRED).issubset(record['evidence_sha256']))
        self.assertEqual(record['derived_aggregates']['v3_completed_configurations'], 2)
        self.assertEqual(record['derived_aggregates']['v3_fit_seconds'], 4.)
        self.assertEqual(record['derived_aggregates']['v3_recorded_attempt_seconds'], 6.)
        self.assertTrue(record['final_facts']['historical_fallback'])
        text = json.dumps(record, ensure_ascii=False)
        self.assertIn('IN-SAMPLE', text)
        self.assertIn('No leaderboard/ranking conclusion', text)
        self.assertIn('350/467', r.chinese_summary(record))

    def test_missing_final_evidence_fails_before_aggregate_read(self):
        (self.root / (r.SUB + 'TEST_COMPLETE.json')).unlink()
        # Deliberately invalid validation JSON must not be parsed before readiness.
        self.put(r.BASE + 'validation_summary.json', 'not JSON')
        with self.assertRaisesRegex(ValueError, 'not ready'):
            r.prepare(self.root)

    def test_deterministic_references_reject_wrong_scope_and_inconsistent_counts(self):
        path = r.BASE + 'deterministic_train_references/summary.json'
        original = self.read(path)
        completion_path = r.BASE + 'deterministic_train_references/COMPLETION.json'
        def bind_changed_summary(value):
            digest = self.put(path, value)
            completion = self.read(completion_path)
            completion['output_sha256'][path] = digest
            self.put(completion_path, completion)
        for key, value in [('split', 'validation'), ('new_fitting_runs', 1), ('folds_sha256', '0' * 64)]:
            changed = dict(original, **{key: value})
            bind_changed_summary(changed)
            with self.assertRaises(ValueError):
                r.prepare(self.root)
        original['results']['fixed0']['hard']['folds'][0]['correct'] += 1
        bind_changed_summary(original)
        with self.assertRaisesRegex(ValueError, 'fold counts'):
            r.prepare(self.root)
        original['results']['fixed0']['hard']['folds'][0]['correct'] -= 1
        original['results']['fixed0']['hard']['fold_mean'] += 1
        bind_changed_summary(original)
        with self.assertRaisesRegex(ValueError, 'fold mean/SD'):
            r.prepare(self.root)
        original['results']['fixed0']['hard']['fold_mean'] -= 1
        bind_changed_summary(original)
        (self.root / completion_path).unlink()
        with self.assertRaises(ValueError):
            r.prepare(self.root)

    def test_unfinished_test_and_changed_output_rejected(self):
        complete = self.read(r.SUB + 'TEST_COMPLETE.json')
        complete['status'] = 'incomplete'
        self.put(r.SUB + 'TEST_COMPLETE.json', complete)
        with self.assertRaisesRegex(ValueError, 'completed once-only'):
            r.prepare(self.root)
        complete['status'] = 'passed'
        self.put(r.SUB + 'TEST_COMPLETE.json', complete)
        self.put(r.SUB + 'test-score.json', {'n': 467, 'correct': 467})
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            r.prepare(self.root)

    def test_pending_queue_and_missing_round_rejected(self):
        queue = self.read(r.BASE + 'campaign_queue.json')
        queue['jobs'][0]['status'] = 'running'
        self.put(r.BASE + 'campaign_queue.json', queue)
        with self.assertRaisesRegex(ValueError, 'not terminal'):
            r.prepare(self.root)
        queue['jobs'][0]['status'] = 'completed'
        queue['jobs'][0]['round'] = 'absent_round'
        self.put(r.BASE + 'campaign_queue.json', queue)
        with self.assertRaisesRegex(ValueError, 'missing from the final ledger'):
            r.prepare(self.root)

    def test_confidence_selector_changed_or_claimed_out_of_sample_rejected(self):
        value = self.read(r.DEFAULTS['confidence_fit'])
        value['selector']['experiment_id'] = 'another_model'
        self.put(r.DEFAULTS['confidence_fit'], value)
        with self.assertRaisesRegex(ValueError, 'selected method'):
            r.prepare(self.root)
        value['selector']['experiment_id'] = 'historical_s0'
        value['interpretation'] = 'Unbiased generalization estimate'
        self.put(r.DEFAULTS['confidence_fit'], value)
        with self.assertRaisesRegex(ValueError, 'fixed post-selection'):
            r.prepare(self.root)

    def test_no_raw_bank_prediction_or_background_json_read(self):
        evidence = r.Evidence(self.root)
        for name in ('code/data/manifest.json', 'code/data/test.gold.jsonl', 'docs/RANKING_BACKGROUND.json', '../outside.json', 'results/../code/data/manifest.json'):
            with self.subTest(path=name), self.assertRaises(ValueError):
                evidence.read(name)
        with self.assertRaisesRegex(ValueError, 'per-question'):
            evidence.read(r.SUB + 'test-predictions.json')

    def test_seed_summary_is_descriptive_and_never_selects_better_seed(self):
        rows = self.read(r.DEFAULTS['analysis'])['rows'][:2]
        rows[0]['hard_fold_mean'], rows[1]['hard_fold_mean'] = 74., 78.
        group = r.stability_groups(rows)[0]
        self.assertEqual(group['experiment_id'], 'primary_s0')
        self.assertEqual(group['mean'], 76.)
        self.assertAlmostEqual(group['sample_sd'], 2**.5 * 2)
        rows[1]['seed'] = rows[0]['seed']
        with self.assertRaisesRegex(ValueError, 'Duplicate seeds'):
            r.stability_groups(rows)

    def test_outputs_refuse_overwrite_and_do_not_create_pdf(self):
        record = r.prepare(self.root)
        output = self.root / 'reports/generated.json'
        chinese = self.root / 'reports/understand.md'
        r.write_outputs(record, output, chinese)
        original = output.read_bytes()
        with self.assertRaises(FileExistsError):
            r.write_outputs(record, output, chinese)
        self.assertEqual(output.read_bytes(), original)
        self.assertEqual(list(self.root.rglob('*.pdf')), [])

    def test_figures_must_bind_same_inputs_and_saved_image_bytes(self):
        manifest = self.read(r.DEFAULTS['figures'])
        manifest['scope'] = 'synthetic_fixture_only'
        self.put(r.DEFAULTS['figures'], manifest)
        with self.assertRaisesRegex(ValueError, 'completed saved-evidence figures'):
            r.prepare(self.root)
        manifest['scope'] = 'saved_final_evidence_only'
        manifest['evidence_sha256'][r.DEFAULTS['analysis']] = '0' * 64
        self.put(r.DEFAULTS['figures'], manifest)
        with self.assertRaisesRegex(ValueError, 'different report inputs'):
            r.prepare(self.root)
        manifest['evidence_sha256'][r.DEFAULTS['analysis']] = hashlib.sha256((self.root / r.DEFAULTS['analysis']).read_bytes()).hexdigest()
        self.put(r.DEFAULTS['figures'], manifest)
        self.put(manifest['figures'][0]['path'], 'changed synthetic image')
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            r.prepare(self.root)

    def test_changed_calibration_after_validation_is_rejected(self):
        path = str(Path(r.DEFAULTS['confidence_fit']).with_name('calibration.json'))
        calibration = self.read(path)
        calibration['coefficient'] += .01
        self.put(path, calibration)
        with self.assertRaisesRegex(ValueError, 'different saved calibration'):
            r.prepare(self.root)

    def test_paired_validation_must_describe_same_selected_counts(self):
        path = r.BASE + 'selected_validation_comparison.json'
        value = self.read(path)
        value['comparisons']['frozen_v1']['overall']['first_correct'] += 1
        self.put(path, value)
        with self.assertRaisesRegex(ValueError, 'Paired validation counts'):
            r.prepare(self.root)

    def test_legacy_round_omitted_hash_and_aliased_fields_are_disclosed_without_rewrite(self):
        path = r.BASE + 'round_decisions.jsonl'
        record = json.loads((self.root / path).read_text())
        del record['analysis_sha256']
        record['results'] = [{'id': item['experiment_id'], 'hard_correct': item['hard_correct'], 'hard_n': item['hard_n'],
            'full_correct': item['correct'], 'full_n': item['n'], 'training_seconds': item['training_seconds']} for item in record['results']]
        self.put(path, json.dumps(record) + '\n')
        original = (self.root / path).read_bytes()
        generated = r.prepare(self.root)
        binding = generated['derived_aggregates']['historical_analysis_bindings'][0]
        self.assertFalse(binding['original_ledger_declared_sha256'])
        self.assertEqual((self.root / path).read_bytes(), original)
        self.assertEqual(generated['pages'][2]['blocks'][2]['rows'][0][3], '75.000')

    def test_secondary_analysis_cost_must_match_original_summary(self):
        path = r.DEFAULTS['analysis']
        value = self.read(path)
        value['rows'][0]['training_seconds'] += 1
        self.put(path, value)
        with self.assertRaisesRegex(ValueError, 'statistic differs.*training_seconds'):
            r.prepare(self.root)

    def test_typed_figure_receipt_cannot_claim_another_scope_or_source(self):
        manifest = self.read(r.DEFAULTS['figures'])
        reference = manifest['figures'][0]['generation_record']
        original = self.read(reference['path'])
        for key, value in [('scope', 'synthetic_fixture_only'), ('generator_sha256', '0' * 64),
                           ('source_sha256', {}), ('path', 'another.png')]:
            receipt = {**original, key: value}
            reference['sha256'] = self.put(reference['path'], receipt)
            self.put(r.DEFAULTS['figures'], manifest)
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'receipt differs'):
                r.prepare(self.root)


if __name__ == '__main__':
    unittest.main(verbosity=2)
