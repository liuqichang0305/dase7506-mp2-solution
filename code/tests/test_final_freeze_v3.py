"""Synthetic freeze and one-shot runner checks; never use official test data."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code/tools'))
import freeze_v3 as f
spec = importlib.util.spec_from_file_location('evaluate_final_v3', ROOT / 'submission/v3/evaluate_final.py')
e = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e)


class FinalFreezeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=ROOT / 'tmp', prefix='final-freeze-synthetic-')
        self.root = Path(self.temporary.name)
        self.folder = self.root / 'submission/v3'
        self.folder.mkdir(parents=True)
        self.now = datetime(2026, 10, 30, 12, tzinfo=timezone.utc)
        self.commit = 'c' * 40
        source_names = ['code/common.py', 'code/predict.py', 'code/score_predictions.py', f.METHOD]
        for name in source_names:
            self.put(name, b'# synthetic fixture, not executable coursework\n')
        self.names = source_names
        self.source_provider = lambda root: {name: f.sha(self.root / name) for name in self.names}
        self.official_patch = patch.object(f.timing, 'OFFICIAL_SOURCE', {name: f.sha(self.root / name) for name in source_names[:3]})
        self.official_patch.start()
        self.active_patch = patch.object(f.timing, 'active_training', return_value=[])
        self.active_patch.start()
        self.config = {'experiment_id': 'synthetic', 'family': 'mlp', 'seed': f.SEED,
                       'preprocess': 'standard', 'decoder': {'mode': 'group_lse', 'alpha': 0., 'tau': 1., 'mask': True}}
        self.origin = 'artifacts/research_v3/synthetic_final'
        for bundle in [self.origin, f.CHECKPOINT]:
            self.put(bundle + '/config.json', json.dumps({'schema': 'mp2-research-v3', 'architecture': self.config}).encode())
            self.put(bundle + '/numeric.bin', b'synthetic numeric placeholder')
        self.save('code/data/manifest.json', {'protocol': f.timing.PROTOCOL, 'splits': {'train': 12, 'validation': 798, 'test': 467},
            'candidates_per_question': 4, 'sha256': {name: '1' * 64 for name in
                ('gsm8k_test.jsonl', 'candidates/gsm8k_test.jsonl', 'candidates/gsm8k_test.npz')}})
        self.save('results/research_v3/SEARCH_CLOSED.json', {'search_closed': True})
        self.save('results/research_v3/validation/synthetic/evidence.json', {'synthetic': True})
        final_fit = 'results/research_v3/synthetic/final_fit.json'
        self.save(final_fit, {'config': self.config, 'info': {'epochs_run': 2, 'evaluation_weights': 'ema'}})
        component = {'experiment_id': 'synthetic', 'kind': 'new', 'seed': f.SEED, 'bundle_subdir': '.',
                     'final_fit_path': final_fit, 'final_fit_sha256': f.sha(self.root / final_fit)}
        selector = {'kind': 'single_model', 'experiment_id': 'synthetic', 'family': 'mlp',
                    'decoder': self.config['decoder'], 'member_probability_average': False}
        self.lock = {'finalists': [{'name': 'synthetic', 'method': f.METHOD, 'checkpoint': self.origin,
                     'components': [component], 'selector': selector}],
                     'data_manifest_sha256': f.sha(self.root / 'code/data/manifest.json')}
        self.save('results/research_v3/FINALISTS_LOCK.json', self.lock)
        self.verified = {'lock': self.lock, 'lock_sha256': f.sha(self.root / 'results/research_v3/FINALISTS_LOCK.json'),
                         'baseline': {'selected_experiment_id': 'historical', 'method': 'code/solution_v2.py',
                                      'checkpoint': 'artifacts/research_v2/historical/final'}}
        row = {**self.lock['finalists'][0], 'local_resource_eligible': True,
               'validation': {'correct': 741, 'hard_correct': 298},
               'resource': {'uncompressed_bytes': 100}, 'prediction_plus_scoring_seconds': 1.}
        expected = f.timing.choose_finalist([row], self.verified['baseline'])
        self.summary = {'schema': 'mp2-validation-confirmation-v3', 'lock_sha256': self.verified['lock_sha256'],
            'finalists': [row], 'selected': expected, 'new_test_evaluated': False, 'search_remains_closed': True,
            'completed_at_utc': (self.now - timedelta(minutes=2)).isoformat(),
            'evidence_sha256': {'results/research_v3/validation/synthetic/evidence.json':
                               f.sha(self.root / 'results/research_v3/validation/synthetic/evidence.json')}}
        self.save('results/research_v3/validation_summary.json', self.summary)
        self.selected = {'schema': 'mp2-selected-model-v3', 'name': 'synthetic', 'historical_fallback': False,
            'method': f.METHOD, 'checkpoint': f.CHECKPOINT, 'origin_checkpoint': self.origin, 'deployment_seed': f.SEED,
            'selector': selector, 'oof_directory': 'results/research_v3/synthetic', 'components': [component],
            'validation_summary_sha256': f.sha(self.root / 'results/research_v3/validation_summary.json'),
            'created_at_utc': (self.now - timedelta(minutes=1)).isoformat()}
        self.save('submission/v3/SELECTED_MODEL.json', self.selected)
        self.put('docs/synthetic-audit.txt', b'synthetic review and reproduction evidence')
        assets, _ = f.asset_hashes(self.root)
        common = {'passed': True, 'selected_model_sha256': f.sha(self.folder / 'SELECTED_MODEL.json'),
            'method': f.METHOD, 'checkpoint': f.CHECKPOINT, 'source_files': self.source_provider(self.root), 'assets': assets,
            'evidence_files': {'docs/synthetic-audit.txt': f.sha(self.root / 'docs/synthetic-audit.txt')}}
        self.review = {**common, 'schema': 'mp2-v3-review-result-v1', 'dependency_audit': {'passed': True,
            'no_hidden_parameters': True, 'external_runtime_reads': 'bundle_and_inputs_only', 'inference_dependencies': ['numpy synthetic']}}
        self.reproduction = {**common, 'schema': 'mp2-v3-reproduction-result-v1', 'split': 'validation',
            'fresh_environment': True, 'choices_match': True, 'official_score_match': True, 'source_commit': self.commit}
        self.save('submission/v3/REVIEW_RESULT.json', self.review)
        self.save('submission/v3/REPRODUCTION_RESULT.json', self.reproduction)
        self.typed_gates()

    def tearDown(self):
        self.active_patch.stop()
        self.official_patch.stop()
        self.temporary.cleanup()

    def put(self, relative, value):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value)

    def save(self, relative, value):
        self.put(relative, json.dumps(value).encode())

    def freeze(self):
        return f.freeze(self.root, verified=self.verified, source_provider=self.source_provider,
                        git_check=lambda *args: None, commit=self.commit, now_fn=lambda: self.now)

    def typed_gates(self):
        checkpoint = str(self.root / f.CHECKPOINT)
        relative_assets = {name.removeprefix(f.CHECKPOINT + '/'): digest for name, digest in self.review['assets'].items()}
        stamp = (self.now - timedelta(seconds=30)).isoformat()
        checks = ('python_int_length_range arbitrary_n_batch_independence id_independence question_order determinism '
            'member_probability_average_and_actual_chosen_confidence parseability_mask_and_all_none_fallback '
            'reject_nonfinite reject_wrong_shape reject_misaligned missing_bundle_rejected '
            'offline_no_references_or_forbidden_access numpy_only_no_torch_import').split()
        contract = {'status': 'passed', 'input_source': 'validation', 'n': 798, 'checkpoint': checkpoint,
            'assets_sha256': relative_assets, 'method_sha256': self.source_provider(self.root)[f.METHOD],
            'reference_files_read': False, 'checks': {key: True for key in checks},
            'candidate_permutation': {'all_mismatches_tie_explained': True}, 'configured_pool_threads_total': 1,
            'timestamp_utc': stamp}
        export = {'status': 'passed', 'input_source': 'validation', 'questions': 798, 'checkpoint': checkpoint,
            'assets_sha256': relative_assets, 'reference_files_read': False, 'choice_agreement': {'equal': 798, 'n': 798},
            'group_probabilities_within_tolerance': True, 'differing_row_indices': [],
            'members': [{'status': 'passed', 'scores_outside_tolerance': 0}], 'timestamp_utc': stamp}
        generation = 'research_v2' if self.selected['historical_fallback'] else 'research_v3'
        origin = f"results/{generation}/validation/{self.selected['name']}"
        prediction = {'protocol': f.timing.PROTOCOL, 'split': 'validation',
                      'ids': ['synthetic-val-' + str(i) for i in range(798)], 'selected': [0] * 798}
        score = {'protocol': f.timing.PROTOCOL, 'split': 'validation', 'n': 798, 'correct': 740, 'accuracy': 100 * 740 / 798}
        self.save(origin + '/validation-predictions.json', prediction)
        self.save(origin + '/validation-score.json', score)
        prefix, checkout = self.root / 'tmp/clean-env', self.root / 'tmp/clean-checkout'
        self.put('tmp/clean-env/Scripts/python.exe', b'synthetic interpreter placeholder; never executed')
        for name in {**self.review['source_files'], **self.review['assets']}:
            self.put('tmp/clean-checkout/' + name, (self.root / name).read_bytes())
        data_names = ['candidates/gsm8k_validation.jsonl', 'candidates/gsm8k_validation.npz', 'gsm8k_validation.jsonl', 'manifest.json']
        for name in data_names:
            self.put('tmp/clean-checkout/code/data/' + name, (self.root / 'code/data/manifest.json').read_bytes() if name == 'manifest.json' else b'synthetic validation placeholder')
        environment = {'schema': 'mp2-v3-clean-environment-v1', 'source_commit': self.commit,
            'python_prefix': str(prefix), 'python_executable': str(prefix / 'Scripts/python.exe'), 'checkout': str(checkout),
            'packages': ['numpy==synthetic'], 'source_files': self.review['source_files'], 'assets': self.review['assets'],
            'official_data_files': data_names, 'captured_at_utc': stamp,
            'data_manifest_sha256': f.sha(self.root / 'code/data/manifest.json')}
        resource = {'status': 'passed', 'local_resource_eligible': True, 'prediction_plus_scoring_seconds': .5,
            'sampled_peak_task_process_tree_threads': 2, 'scientific_library_thread_cap_passed': True,
            'completed_at_utc': stamp,
            'processes': [{'returncode': 0, 'timed_out': False, 'started_at_utc': stamp, 'completed_at_utc': stamp,
                           'command': ['synthetic-python', '-S', '-c', 'synthetic-wrapper', script, 'pool.json', '--split', 'validation']}
                          for script in ('code/predict.py', 'code/score_predictions.py')],
            'scientific_library_observations': [{'torch_imported': False, 'thread_pools': [{'num_threads': 1}],
                                                'pythonpath': str(prefix / 'Lib/site-packages')}] * 2}
        for owner, values in [(self.review, {'contract_result': contract, 'export_result': export}),
                              (self.reproduction, {'predictions_result': prediction, 'score_result': score,
                                                  'timing_result': resource, 'environment_result': environment})]:
            owner['created_at_utc'] = (self.now - timedelta(seconds=10)).isoformat()
            for key, value in values.items():
                relative = 'results/research_v3/synthetic-gates/' + key + '.json'
                self.save(relative, value)
                owner[key] = {'path': relative, 'sha256': f.sha(self.root / relative)}
                owner['evidence_files'][relative] = f.sha(self.root / relative)
        for part in ('predictions', 'score'):
            name = origin + '/validation-' + part + '.json'
            self.reproduction['evidence_files'][name] = f.sha(self.root / name)
        self.save('submission/v3/REVIEW_RESULT.json', self.review)
        self.save('submission/v3/REPRODUCTION_RESULT.json', self.reproduction)

    def verify(self, root):
        return f.verify_freeze(root, source_provider=self.source_provider, git_check=lambda *args: None)

    def run_pipeline(self, runner, **kwargs):
        counter = iter([index / 10 for index in range(50)])
        return e.evaluate(self.root, verify=self.verify, process_runner=runner,
                          command_builder=lambda root, phase, arguments, resource: ([phase, *arguments], {}),
                          now_fn=lambda: datetime(2026, 10, 31, 12, tzinfo=timezone.utc),
                          clock=lambda: next(counter), **kwargs)

    def fake_runner(self, command, folder, stem, deadline, environment):
        (folder / (stem + '.stdout.log')).write_text('synthetic process only\n')
        (folder / (stem + '.stderr.log')).write_text('')
        self.save('submission/v3/' + stem + '-thread-pools.json', {'torch_imported': False,
            'thread_pools': [{'num_threads': 1}], 'numpy_version': 'synthetic'})
        if command[0] == 'prediction':
            self.save('submission/v3/test-predictions.json', {'protocol': f.timing.PROTOCOL, 'split': 'test',
                'ids': ['synthetic-' + str(i) for i in range(467)], 'selected': [0] * 467})
        else:
            self.save('submission/v3/test-score.json', {'protocol': f.timing.PROTOCOL, 'split': 'test',
                'correct': 321, 'n': 467, 'accuracy': 100 * 321 / 467})
        return {'command': command, 'returncode': 0, 'timed_out': False, 'wall_seconds': .1,
                'sampled_peak_tree_threads': 2, 'sampled_peak_tree_plus_coordinator_threads': 3}

    def test_freeze_records_recipe_and_verifies_reuse_without_test_access(self):
        record = self.freeze()
        self.assertTrue(record['frozen'])
        self.assertFalse(record['test_evaluated_before_freeze'])
        self.assertEqual(record['training_recipes'][0]['epochs'], 2)
        self.assertEqual(record['training_recipes'][0]['evaluation_weights'], 'ema')
        self.assertEqual(len(record['expected_test_file_sha256']), 3)
        self.assertFalse((self.root / 'code/data/gsm8k_test.jsonl').exists())
        before = f.sha(self.folder / 'FREEZE.json')
        self.assertEqual(self.freeze(), record)
        self.assertEqual(f.sha(self.folder / 'FREEZE.json'), before)

    def test_changed_selection_copy_or_gate_cannot_freeze(self):
        self.put(f.CHECKPOINT + '/numeric.bin', b'changed copy')
        with self.assertRaisesRegex(ValueError, 'Deployment copy differs'):
            self.freeze()
        self.put(f.CHECKPOINT + '/numeric.bin', b'synthetic numeric placeholder')
        self.reproduction['fresh_environment'] = False
        self.save('submission/v3/REPRODUCTION_RESULT.json', self.reproduction)
        with self.assertRaisesRegex(ValueError, 'Fresh-environment'):
            self.freeze()
        self.assertFalse((self.folder / 'FREEZE.json').exists())

    def test_historical_fallback_uses_original_recipe_and_unchanged_assets(self):
        historical = {**self.config, 'experiment_id': 'historical', 'family': 'deepsets'}
        origin = self.verified['baseline']['checkpoint']
        for bundle in (origin, f.CHECKPOINT):
            self.put(bundle + '/config.json', json.dumps({'schema': 'mp2-research-v2', 'architecture': historical}).encode())
            self.put(bundle + '/numeric.bin', b'historical immutable synthetic asset')
        self.save('results/research_v2/historical/final_fit.json', {'config': historical, 'info': {'epochs_run': 5}})
        old_assets = {origin + '/' + row['path']: row['sha256'] for row in f.audit(self.root / origin)['files']}
        self.save('results/research_v2/V2_FREEZE.json', {'selected_experiment_id': 'historical', 'checkpoint': origin, 'assets': old_assets})
        self.summary['finalists'][0]['validation'] = {'correct': 740, 'hard_correct': 297}
        self.summary['selected'] = f.timing.choose_finalist(self.summary['finalists'], self.verified['baseline'])
        self.save('results/research_v3/validation_summary.json', self.summary)
        self.selected.update({'name': 'historical', 'historical_fallback': True, 'origin_checkpoint': origin,
            'oof_directory': 'results/research_v2/historical',
            'selector': {'kind': 'single_model', 'experiment_id': 'historical', 'family': 'deepsets',
                         'decoder': historical['decoder'], 'member_probability_average': False},
            'components': [{'kind': 'historical_v2', 'experiment_id': 'historical', 'bundle_subdir': '.',
                            'freeze_path': 'results/research_v2/V2_FREEZE.json',
                            'freeze_sha256': f.sha(self.root / 'results/research_v2/V2_FREEZE.json')}],
            'validation_summary_sha256': f.sha(self.root / 'results/research_v3/validation_summary.json')})
        self.save('submission/v3/SELECTED_MODEL.json', self.selected)
        assets, _ = f.asset_hashes(self.root)
        for record, name in [(self.review, 'REVIEW_RESULT.json'), (self.reproduction, 'REPRODUCTION_RESULT.json')]:
            record['assets'] = assets
            record['selected_model_sha256'] = f.sha(self.folder / 'SELECTED_MODEL.json')
            self.save('submission/v3/' + name, record)
        self.typed_gates()
        result = self.freeze()
        self.assertEqual(result['selected_name'], 'historical')
        self.assertEqual(result['training_recipes'][0]['epochs'], 5)
        self.assertEqual(result['training_recipes'][0]['kind'], 'historical_v2')

    def test_completed_run_reuse_never_invokes_official_processes_again(self):
        self.freeze()
        calls = []
        def runner(*args, **kwargs):
            calls.append(args[0][0])
            self.assertTrue((self.folder / 'TEST_OPENED.json').exists())
            return self.fake_runner(*args, **kwargs)
        complete = self.run_pipeline(runner)
        self.assertEqual(calls, ['prediction', 'scoring'])
        self.assertTrue(complete['local_resource_eligible'])
        self.assertEqual(complete['n'], 467)
        self.assertNotIn('correct', complete)
        def prohibited(*args, **kwargs):
            self.fail('Completed test was evaluated again')
        self.assertEqual(self.run_pipeline(prohibited), complete)
        self.put('submission/v3/test-score.json', b'changed')
        with self.assertRaisesRegex(ValueError, 'bytes changed'):
            self.run_pipeline(prohibited)

    def test_partial_run_marker_blocks_automatic_retry(self):
        self.freeze()
        def failed(command, *args, **kwargs):
            return {'returncode': 1, 'timed_out': False}
        with self.assertRaisesRegex(RuntimeError, 'Official process failed'):
            self.run_pipeline(failed)
        self.assertTrue((self.folder / 'TEST_INCOMPLETE.json').exists())
        with self.assertRaisesRegex(RuntimeError, 'automatic re-evaluation'):
            self.run_pipeline(self.fake_runner)

    def test_resource_failure_is_not_a_second_chance(self):
        self.freeze()
        def excessive(*args, **kwargs):
            result = self.fake_runner(*args, **kwargs)
            result['sampled_peak_tree_threads'] = 5
            return result
        with self.assertRaisesRegex(RuntimeError, 'resource constraints failed'):
            self.run_pipeline(excessive)
        self.assertFalse((self.folder / 'TEST_COMPLETE.json').exists())
        self.assertTrue((self.folder / 'test-timing.json').exists())
        with self.assertRaisesRegex(RuntimeError, 'automatic re-evaluation'):
            self.run_pipeline(self.fake_runner)

    def test_deadline_or_active_training_blocks_before_open_marker(self):
        self.freeze()
        with self.assertRaisesRegex(ValueError, 'deadline passed'):
            e.evaluate(self.root, verify=self.verify, now_fn=lambda: e.DEADLINE)
        self.assertFalse((self.folder / 'TEST_OPENED.json').exists())
        with patch.object(e.timing, 'active_training', return_value=[{'pid': 123}]):
            with self.assertRaisesRegex(RuntimeError, 'training/scheduler'):
                self.run_pipeline(self.fake_runner)
        self.assertFalse((self.folder / 'TEST_OPENED.json').exists())

    def test_worker_guard_blocks_gold_for_prediction_but_allows_scoring(self):
        self.put('code/data/gsm8k_test.jsonl', b'synthetic gold fixture')
        self.put('code/data/candidates/gsm8k_test.jsonl', b'synthetic candidate fixture')
        output, resource = self.folder / 'test-predictions.json', self.folder / 'pool.json'
        gold = self.root / 'code/data/gsm8k_test.jsonl'
        code = e.worker_guard(self.root, 'prediction', output, resource) + '\nopen(' + repr(str(gold)) + ",'rb').read()"
        result = subprocess.run([sys.executable, '-S', '-c', code], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Data access forbidden', result.stderr)
        code = e.worker_guard(self.root, 'scoring', output, resource) + '\nopen(' + repr(str(gold)) + ",'rb').read()"
        result = subprocess.run([sys.executable, '-S', '-c', code], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_worker_guard_blocks_subprocess_and_all_socket_events(self):
        prefix = e.worker_guard(self.root, 'prediction', self.folder / 'test-predictions.json', self.folder / 'pool.json')
        for event in ('subprocess.Popen', 'os.system', 'os.posix_spawn', 'socket.sendto', 'socket.connect', 'socket.__new__'):
            with self.subTest(event=event):
                # Invoke only the audit event: no process is launched and no packet is sent.
                code = prefix + '\nsys.audit(' + repr(event) + ',None)'
                result = subprocess.run([sys.executable, '-S', '-c', code], capture_output=True, text=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('cannot spawn processes', result.stderr)

    def test_wrong_oof_directory_and_text_only_gate_are_rejected(self):
        self.selected['oof_directory'] = 'results/research_v3/unselected'
        self.save('submission/v3/SELECTED_MODEL.json', self.selected)
        with self.assertRaisesRegex(ValueError, 'actual fitted component'):
            self.freeze()
        self.selected['oof_directory'] = 'results/research_v3/synthetic'
        self.save('submission/v3/SELECTED_MODEL.json', self.selected)
        del self.review['contract_result']
        self.save('submission/v3/REVIEW_RESULT.json', self.review)
        with self.assertRaisesRegex(ValueError, 'typed evidence reference'):
            self.freeze()

    def test_completion_subset_or_wrong_identity_and_time_cannot_resume(self):
        self.freeze()
        complete = self.run_pipeline(self.fake_runner)
        for modified, pattern in [
            ({**complete, 'output_sha256': {'submission/v3/test-score.json': complete['output_sha256']['submission/v3/test-score.json']}}, 'entire fixed output'),
            ({**complete, 'method': 'code/wrong.py', 'split': 'validation', 'n': 798}, 'identity/split/count'),
            ({**complete, 'completed_at_utc': '2020-01-01T00:00:00Z'}, 'timestamps')]:
            self.save('submission/v3/TEST_COMPLETE.json', modified)
            with self.assertRaisesRegex(ValueError, pattern):
                self.run_pipeline(self.fake_runner)

    def test_actual_typed_export_or_reproduction_choice_mismatch_blocks_freeze(self):
        reference = self.review['export_result']
        report = f.read(self.root / reference['path'])
        report['choice_agreement']['equal'] = 797
        self.save(reference['path'], report)
        reference['sha256'] = f.sha(self.root / reference['path'])
        self.review['evidence_files'][reference['path']] = reference['sha256']
        self.save('submission/v3/REVIEW_RESULT.json', self.review)
        self.save('submission/v3/REPRODUCTION_RESULT.json', self.reproduction)
        with self.assertRaisesRegex(ValueError, 'full-input export'):
            self.freeze()
        self.typed_gates()
        reference = self.reproduction['predictions_result']
        report = f.read(self.root / reference['path'])
        report['selected'][0] = 1
        self.save(reference['path'], report)
        reference['sha256'] = f.sha(self.root / reference['path'])
        self.reproduction['evidence_files'][reference['path']] = reference['sha256']
        self.save('submission/v3/REVIEW_RESULT.json', self.review)
        self.save('submission/v3/REPRODUCTION_RESULT.json', self.reproduction)
        with self.assertRaisesRegex(ValueError, '798 locked original choices'):
            self.freeze()


if __name__ == '__main__':
    unittest.main()
