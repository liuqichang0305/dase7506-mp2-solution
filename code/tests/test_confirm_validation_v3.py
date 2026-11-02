"""Confirmation gate tests on isolated synthetic files; never official data."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'code/tools'))
import audit_assets  # Load the real data-free helper before temporary ROOT fixtures.
spec = importlib.util.spec_from_file_location('confirm_validation_v3_tested', REPO / 'code/tools/confirm_validation_v3.py')
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)
sys.path.insert(0, str(REPO / 'code'))
import numpy as np
from mp2.research_v3.models import save_bundle


class ConfirmationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='mp2-v3-validation-synthetic-')
        self.root = Path(self.tmp.name).resolve()
        self.out = self.root / 'results/research_v3'
        self.out.mkdir(parents=True)
        self.patchers = [patch.object(gate, 'ROOT', self.root), patch.object(gate, 'OUT', self.out)]
        for item in self.patchers: item.start()
        for name in gate.INFERENCE_SOURCE | gate.TRAIN_SOURCE:
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes((REPO / name).read_bytes())
        for name in gate.TRAIN_PATHS:
            self.text(name, 'synthetic train only\n')
        self.write('code/data/manifest.json', {'protocol': gate.PROTOCOL, 'splits': {'validation': 798}})
        self.write('results/research_v3/common_folds.json', {'synthetic': True})
        self.source = {name: gate.sha(self.root / name) for name in gate.INFERENCE_SOURCE}
        official = {name: self.source[name] for name in gate.OFFICIAL_SOURCE}
        self.patchers.append(patch.object(gate, 'OFFICIAL_SOURCE', official))
        self.patchers[-1].start()
        self.selector = {'kind': 'single_model', 'experiment_id': 'synthetic_s0', 'family': 'linear',
                         'decoder': {'mode': 'group_lse', 'mask': True, 'alpha': 0., 'tau': 1.},
                         'member_probability_average': False}
        save_bundle(self.root / 'artifacts/research_v3/final_synthetic', {'family': 'linear'},
                    {'weight': np.zeros(2064), 'bias': np.asarray(0.)},
                    np.zeros(2064), np.ones(2064), self.selector['decoder'])
        self.assets = {name: gate.sha(self.root / name) for name in (
            'artifacts/research_v3/final_synthetic/config.json', 'artifacts/research_v3/final_synthetic/weights.npz')}
        final_report = {'config': {'experiment_id': 'synthetic_s0', 'family': 'linear', 'seed': gate.SEED,
                                  'decoder': self.selector['decoder']},
                        'provenance': {'source_files': {name: gate.sha(self.root / name) for name in gate.TRAIN_SOURCE},
                          'data_files': {name: gate.sha(self.root / name) for name in gate.TRAIN_PATHS},
                          'folds_sha256': gate.sha(self.out / 'common_folds.json')},
                        'asset_sha256': {Path(name).name: digest for name, digest in self.assets.items()}}
        self.write('results/research_v3/synthetic_s0/final_fit.json', final_report)
        self.write('results/research_v3/synthetic_s0/summary.json', {
            'status': 'complete', 'config': final_report['config'], 'provenance': final_report['provenance'],
            'folds': [{'fold': index, 'n': 1585, 'status': 'complete'} for index in range(3)]})
        self.text('artifacts/research_v2/final_deepsets/weights.bin', 'frozen baseline\n')
        self.baseline = {'selected_experiment_id': 'deepsets_regularized_s0', 'method': 'code/solution_v2.py',
            'checkpoint': 'artifacts/research_v2/final_deepsets', 'source_files': self.source,
            'assets': {'artifacts/research_v2/final_deepsets/weights.bin':
                       gate.sha(self.root / 'artifacts/research_v2/final_deepsets/weights.bin')},
            'validation': {'correct': 740, 'n': 798, 'hard_correct': 297, 'hard_n': 355}}
        self.write('results/research_v2/V2_FREEZE.json', self.baseline)
        self.write('results/research_v3/SEARCH_CLOSED.json',
                   {'search_closed': True, 'created_at_utc': '2026-01-01T00:00:00+00:00'})
        self.selection = {'schema': 'mp2-finalists-selection-v3', 'created_at_utc': '2026-01-01T00:01:00+00:00',
            'deployment_seed': gate.SEED, 'allowed_final_fit_experiments': ['synthetic_s0'],
            'finalists': [{'name': 'candidate_a', 'component_ids': ['synthetic_s0'], 'selector': self.selector}]}
        self.write('results/research_v3/FINALISTS_SELECTION.json', self.selection)
        self.finalist = {'name': 'candidate_a', 'method': 'code/solution_v3.py',
            'checkpoint': 'artifacts/research_v3/final_synthetic', 'selector': self.selector,
            'components': [{'experiment_id': 'synthetic_s0', 'seed': gate.SEED, 'bundle_subdir': '.',
                 'final_fit_path': 'results/research_v3/synthetic_s0/final_fit.json',
                 'cv_summary_sha256': gate.sha(self.out / 'synthetic_s0/summary.json'),
                 'final_fit_sha256': gate.sha(self.out / 'synthetic_s0/final_fit.json')}]}
        self.lock = {'schema': 'mp2-finalists-lock-v3', 'created_at_utc': '2026-01-01T00:02:00+00:00',
            'search_closed': True, 'finalists': [self.finalist], 'inference_source_sha256': self.source,
            'asset_sha256': self.assets, 'data_manifest_sha256': gate.sha(self.root / 'code/data/manifest.json'),
            'baseline_freeze_sha256': gate.sha(self.root / 'results/research_v2/V2_FREEZE.json'),
            'selection_sha256': gate.sha(self.out / 'FINALISTS_SELECTION.json'),
            'search_closed_sha256': gate.sha(self.out / 'SEARCH_CLOSED.json')}
        self.save_lock()

    def tearDown(self):
        for item in reversed(self.patchers): item.stop()
        self.tmp.cleanup()

    def text(self, relative, text):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding='utf-8')
        return path

    def write(self, relative, value):
        return self.text(relative, json.dumps(value, indent=2) + '\n')

    def save_lock(self):
        self.write('results/research_v3/FINALISTS_LOCK.json', self.lock)

    def test_lock_positive_never_needs_validation_or_test_files(self):
        result = gate.validate_lock()
        self.assertEqual(result['lock_sha256'], gate.sha(self.out / 'FINALISTS_LOCK.json'))
        self.assertTrue(result['resources']['candidate_a']['within_limit'])
        self.assertFalse((self.root / 'code/data/gsm8k_validation.jsonl').exists())
        self.assertFalse((self.out / 'VALIDATION_OPENED.json').exists())

    def test_actual_source_inventory_exists(self):
        # research_v3 is a namespace package and has no __init__.py to hash.
        self.assertTrue(all((REPO / name).is_file() for name in gate.INFERENCE_SOURCE))

    def test_prelock_integrity_rejects_mutations_and_third_candidate(self):
        original = deepcopy(self.lock)
        cases = [
            lambda: self.lock.update(search_closed=False),
            lambda: self.lock.update(finalists=self.lock['finalists'] * 3),
            lambda: self.lock['inference_source_sha256'].pop('code/solution_v3.py'),
            lambda: self.lock['asset_sha256'].update({'../escape': '0' * 64}),
            lambda: self.lock.update(selection_sha256='0' * 64),
            lambda: self.lock.update(created_at_utc='2025-01-01T00:00:00+00:00'),
            lambda: self.lock['finalists'][0]['components'][0].update(seed=gate.SEED + 1),
            lambda: self.lock['finalists'][0].update(selector='candidate_a'),
            lambda: self.lock['finalists'][0]['components'][0].update(bundle_subdir='../outside'),
            lambda: self.lock['finalists'][0]['components'][0].update(final_fit_sha256='0' * 64),
        ]
        for index, change in enumerate(cases):
            with self.subTest(index=index):
                self.lock = deepcopy(original)
                change()
                self.save_lock()
                with self.assertRaises((ValueError, KeyError)):
                    gate.validate_lock()
        self.assertFalse((self.out / 'VALIDATION_OPENED.json').exists())

    def test_asset_source_and_fallback_changes_rejected(self):
        for name in ('code/solution_v3.py', 'artifacts/research_v3/final_synthetic/weights.npz',
                     'artifacts/research_v2/final_deepsets/weights.bin'):
            with self.subTest(name=name):
                path = self.root / name
                previous = path.read_bytes()
                path.write_bytes(b'tamper')
                with self.assertRaises(ValueError): gate.validate_lock()
                path.write_bytes(previous)

    def test_independent_preopen_counterexamples_are_rejected(self):
        report_path = self.out / 'synthetic_s0/final_fit.json'
        summary_path = self.out / 'synthetic_s0/summary.json'
        report, summary, selection, lock = (deepcopy(value) for value in (
            gate.read(report_path), gate.read(summary_path), self.selection, self.lock))
        for case in ('selector', 'source_subset', 'asset_subset', 'incomplete_cv', 'invalid_numeric_bundle'):
            with self.subTest(case=case):
                current_report, current_summary = deepcopy(report), deepcopy(summary)
                self.selection, self.lock = deepcopy(selection), deepcopy(lock)
                component = self.lock['finalists'][0]['components'][0]
                if case == 'selector':
                    false = {**self.selector, 'family': 'tabm', 'member_probability_average': True,
                             'decoder': {**self.selector['decoder'], 'alpha': 99.}}
                    self.selection['finalists'][0]['selector'] = false
                    self.lock['finalists'][0]['selector'] = false
                elif case == 'source_subset':
                    current_report['provenance']['source_files'] = {
                        'code/common.py': current_report['provenance']['source_files']['code/common.py']}
                    current_summary['provenance'] = deepcopy(current_report['provenance'])
                elif case == 'asset_subset':
                    current_report['asset_sha256'].pop('weights.npz')
                elif case == 'incomplete_cv':
                    current_summary['folds'][2]['status'] = 'running'
                else:
                    weights = self.root / 'artifacts/research_v3/final_synthetic/weights.npz'
                    saved = weights.read_bytes()
                    weights.write_bytes(b'invalid numeric container')
                    self.lock['asset_sha256']['artifacts/research_v3/final_synthetic/weights.npz'] = gate.sha(weights)
                    current_report['asset_sha256']['weights.npz'] = gate.sha(weights)
                self.write('results/research_v3/FINALISTS_SELECTION.json', self.selection)
                self.write('results/research_v3/synthetic_s0/final_fit.json', current_report)
                self.write('results/research_v3/synthetic_s0/summary.json', current_summary)
                self.lock['selection_sha256'] = gate.sha(self.out / 'FINALISTS_SELECTION.json')
                component['final_fit_sha256'] = gate.sha(report_path)
                component['cv_summary_sha256'] = gate.sha(summary_path)
                self.save_lock()
                with self.assertRaises(ValueError):
                    gate.validate_lock()
                self.assertFalse((self.out / 'VALIDATION_OPENED.json').exists())
                if case == 'invalid_numeric_bundle':
                    weights.write_bytes(saved)

    def test_improvement_rule_pareto_baseline_and_resource_gate(self):
        def row(name, hard, full, size=100, latency=1., resource=True):
            return {'name': name, 'method': 'code/solution_v3.py', 'checkpoint': name,
                    'local_resource_eligible': resource, 'validation': {'hard_correct': hard, 'correct': full},
                    'resource': {'uncompressed_bytes': size}, 'prediction_plus_scoring_seconds': latency}
        for hard, full, resource in [(297, 740, True), (298, 739, True), (296, 742, True), (300, 750, False)]:
            self.assertTrue(gate.choose_finalist([row('a', hard, full, resource=resource)], self.baseline)['historical_fallback'])
        self.assertEqual(gate.choose_finalist([row('a', 298, 740), row('b', 297, 745)], self.baseline)['name'], 'a')
        self.assertEqual(gate.choose_finalist([row('a', 298, 740), row('b', 298, 741)], self.baseline)['name'], 'b')
        self.assertEqual(gate.choose_finalist([row('a', 298, 741, 200), row('b', 298, 741, 100)], self.baseline)['name'], 'b')
        self.assertEqual(gate.choose_finalist([row('a', 298, 741, latency=2), row('b', 298, 741)], self.baseline)['name'], 'b')

    def test_fusion_identity_requires_actual_registered_complete_ordered_mixture(self):
        components = [{'kind': 'new', 'experiment_id': 'new_a'},
                      {'kind': 'historical_v2', 'experiment_id': 'old_b'}]
        keys = ['v3:new_a', 'v2:old_b']
        self.text('results/research_v3/new_a/oof.npz', 'synthetic hash-only OOF A')
        self.text('results/research_v2/old_b/oof.npz', 'synthetic hash-only OOF B')
        prior_path = self.write('results/research_v3/prior/diagnostic.json',
            {'created_at_utc': '2026-01-01T00:00:00Z',
             'methods': [{'method': key, 'input_sha256': {'oof.npz': gate.sha(self.root / path)}}
                         for key, path in zip(keys, ['results/research_v3/new_a/oof.npz',
                                                     'results/research_v2/old_b/oof.npz'])],
             'complementarity': [{'methods': keys, 'gate_passed': True}]})
        registration = {'status': 'registered', 'purpose': 'single_fixed_equal_probability_oof',
            'methods': keys, 'weights': [.5, .5], 'registered_at_utc': '2026-01-01T00:00:01Z',
            'oof_sha256': {keys[0]: gate.sha(self.root / 'results/research_v3/new_a/oof.npz'),
                           keys[1]: gate.sha(self.root / 'results/research_v2/old_b/oof.npz')},
            'gate_evidence': {'path': 'results/research_v3/prior/diagnostic.json', 'sha256': gate.sha(prior_path)}}
        registration_path = self.write('results/research_v3/fusion/fusion-registration.json', registration)
        fusion_oof = self.text('results/research_v3/fusion/fusion-oof.npz', 'synthetic immutable mixture bytes')
        result = {'status': 'complete', 'created_at_utc': '2026-01-01T00:00:02Z',
                  'fusion': {'status': 'complete', 'methods': keys, 'weights': [.5, .5],
                             'registration_file': 'fusion-registration.json',
                             'registration_sha256': gate.sha(registration_path),
                             'oof_sha256': gate.sha(fusion_oof),
                             'gate': {'gate_passed': True}}}
        result_path = self.write('results/research_v3/fusion/diagnostic.json', result)
        finalist = {'selector': {'kind': 'fixed_equal_probability', 'specification': registration},
                    'fusion_registration_path': 'results/research_v3/fusion/fusion-registration.json',
                    'fusion_registration_sha256': gate.sha(registration_path),
                    'fusion_result_path': 'results/research_v3/fusion/diagnostic.json',
                    'fusion_result_sha256': gate.sha(result_path)}
        gate.validate_fusion_identity(finalist, components)
        with self.assertRaises(ValueError):
            gate.validate_fusion_identity(finalist, list(reversed(components)))
        bad = deepcopy(finalist)
        bad['selector']['specification']['weights'] = [.7, .3]
        with self.assertRaises(ValueError): gate.validate_fusion_identity(bad, components)
        result['fusion']['status'] = 'not_run'
        self.write('results/research_v3/fusion/diagnostic.json', result)
        finalist['fusion_result_sha256'] = gate.sha(result_path)
        with self.assertRaises(ValueError): gate.validate_fusion_identity(finalist, components)

    def test_exclusive_resource_lock_and_pid_creation_identity(self):
        with patch.object(gate, 'active_training', return_value=[]):
            with gate.exclusive_evaluation():
                saved = gate.read(self.root / 'tmp/research_v3/trainer.lock')
                self.assertIsNotNone(gate.alive(saved))
                self.assertIsNone(gate.alive({**saved, 'create_time': saved['create_time'] - 1}))
                with self.assertRaises(RuntimeError):
                    with gate.exclusive_evaluation(): pass
            self.assertFalse((self.root / 'tmp/research_v3/trainer.lock').exists())
        with patch.object(gate, 'active_training', return_value=[{'pid': 1}]):
            with self.assertRaises(RuntimeError):
                with gate.exclusive_evaluation(): pass

    def test_completed_result_hash_resume_and_partial_preservation(self):
        folder = self.out / 'validation/candidate_a'
        folder.mkdir(parents=True)
        path = folder / 'official.json'
        path.write_text('synthetic result')
        record = {'name': 'candidate_a', 'lock_sha256': 'a' * 64, 'output_sha256': {'official.json': gate.sha(path)}}
        gate.atomic(folder / 'evaluation_complete.json', record, exclusive=True)
        self.assertEqual(gate.verify_completed(folder, 'a' * 64), record)
        with self.assertRaises(ValueError): gate.verify_completed(folder, 'b' * 64)
        path.write_text('changed')
        with self.assertRaises(ValueError): gate.verify_completed(folder, 'a' * 64)
        partial = self.out / 'validation/partial'
        partial.mkdir()
        (partial / 'partial.stdout.log').write_text('keep exact prior output')
        with self.assertRaises(RuntimeError):
            gate.evaluate_candidate({'name': 'partial'}, {})
        self.assertEqual((partial / 'partial.stdout.log').read_text(), 'keep exact prior output')

    def test_confirmation_resume_no_second_evaluation_and_no_changed_lock(self):
        verified = gate.validate_lock()
        metrics = {'correct': 741, 'n': 798, 'hard_correct': 298, 'hard_n': 355}
        def complete(finalist, verified):
            folder = self.out / 'validation' / finalist['name']
            folder.mkdir(parents=True)
            (folder / 'synthetic-evidence.json').write_text('{}\n')
            record = {'name': finalist['name'], 'lock_sha256': verified['lock_sha256'], 'status': 'passed',
                'local_resource_eligible': True, 'prediction_plus_scoring_seconds': .1,
                'resource': {'uncompressed_bytes': 100},
                'output_sha256': {'synthetic-evidence.json': gate.sha(folder / 'synthetic-evidence.json')}}
            gate.atomic(folder / 'evaluation_complete.json', record, exclusive=True)
            return record
        with patch.object(gate, 'install_data_guard', return_value={'validation_open': False, 'rejected': []}), \
             patch.object(gate, 'active_training', return_value=[]), \
             patch.object(gate, 'evaluate_candidate', side_effect=complete) as evaluate, \
             patch.object(gate, 'diagnostic_events', return_value=metrics):
            first = gate.confirm()
            second = gate.confirm()
            self.assertEqual(first, second)
            self.assertEqual(evaluate.call_count, 1)
            self.assertEqual(first['selected']['name'], 'candidate_a')
            self.lock['created_at_utc'] = '2026-01-01T00:02:01+00:00'
            self.save_lock()
            with self.assertRaises(ValueError): gate.confirm()

    def test_read_guard_in_fresh_process(self):
        for filename in ('gsm8k_validation.jsonl', 'gsm8k_test.jsonl'):
            self.text('code/data/' + filename, 'synthetic forbidden fixture\n')
        helper = self.text('read_guard_test.py', f"""
import importlib.util
from pathlib import Path
spec=importlib.util.spec_from_file_location('gate', {str(REPO / 'code/tools/confirm_validation_v3.py')!r})
gate=importlib.util.module_from_spec(spec); spec.loader.exec_module(gate)
gate.ROOT=Path({str(self.root)!r})
access=gate.install_data_guard()
val=gate.ROOT/'code/data/gsm8k_validation.jsonl'
test=gate.ROOT/'code/data/gsm8k_test.jsonl'
def denied(operation):
    try: operation()
    except PermissionError: return
    raise AssertionError('read/write guard failed')
denied(val.read_text); denied(test.read_text)
access['validation_open']=True
assert 'synthetic' in val.read_text()
denied(test.read_text)
denied(lambda: val.write_text('overwrite'))
assert 'synthetic' in (gate.ROOT/'code/data/gsm8k_train.jsonl').read_text()
print('guard passed')
""")
        result = subprocess.run([sys.executable, str(helper)], capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('guard passed', result.stdout)

    def test_prediction_then_independent_scoring_complete_timing(self):
        verified = gate.validate_lock()
        self.text('code/predict.py', """
import json, pathlib, sys
assert sys.argv[sys.argv.index('--split')+1] == 'validation'
output=pathlib.Path(sys.argv[sys.argv.index('--output')+1])
output.write_text(json.dumps({'protocol':'dase7506-mp2-open-hard-verifier-v5','split':'validation',
                            'ids':list(range(798)),'selected':[0]*798}))
""")
        self.text('code/score_predictions.py', """
import json, pathlib, sys
assert sys.argv[sys.argv.index('--split')+1] == 'validation'
assert pathlib.Path(sys.argv[sys.argv.index('--predictions')+1]).is_file()
output=pathlib.Path(sys.argv[sys.argv.index('--output')+1])
output.write_text(json.dumps({'protocol':'dase7506-mp2-open-hard-verifier-v5','split':'validation',
                            'n':798,'correct':741,'accuracy':100*741/798}))
""")
        result = gate.evaluate_candidate(self.finalist, verified)
        self.assertEqual(result['status'], 'passed')
        folder = self.out / 'validation/candidate_a'
        timing = gate.read(folder / 'validation-timing.json')
        self.assertEqual(len(timing['processes']), 2)
        self.assertTrue(all(item['returncode'] == 0 for item in timing['processes']))
        first, second = timing['processes']
        self.assertLessEqual(gate.instant(first['started_at_utc']), gate.instant(first['completed_at_utc']))
        self.assertLessEqual(gate.instant(first['completed_at_utc']), gate.instant(second['started_at_utc']))
        self.assertLessEqual(gate.instant(second['started_at_utc']), gate.instant(second['completed_at_utc']))
        self.assertGreater(timing['prediction_plus_scoring_seconds'],
                           sum(item['wall_seconds'] for item in timing['processes']))
        # Windows security/runtime threads can make even an empty interpreter
        # exceed four observed OS threads; preserve, rather than hide, that fact.
        self.assertEqual(result['local_resource_eligible'],
                         timing['sampled_peak_task_process_tree_threads'] <= 4)
        self.assertTrue(timing['scientific_library_thread_cap_passed'])
        with patch.object(gate, 'run_process', side_effect=AssertionError('must not rerun')):
            self.assertEqual(gate.evaluate_candidate(self.finalist, verified), result)

    def test_real_tree_wait_and_timeout_no_orphan(self):
        helper = self.text('sleep_tree.py', """
import subprocess, sys, time
child=subprocess.Popen([sys._base_executable, '-S', '-c', 'import time; time.sleep(float(__import__(\"sys\").argv[1]))', sys.argv[1]])
time.sleep(.12)
print(child.pid, flush=True)
""")
        folder = self.out / 'process-tree'
        folder.mkdir()
        start = time.perf_counter()
        # Match the actual official runner: direct base Python, no venv launcher.
        result = gate.run_process([sys._base_executable, '-S', str(helper), '.5'], folder, 'complete', start + 5)
        self.assertFalse(result['timed_out'])
        self.assertEqual(result['returncode'], 0)
        self.assertGreaterEqual(result['wall_seconds'], .45)
        self.assertGreaterEqual(len(result['process_tree']), 2)
        self.assertGreaterEqual(result['sampled_peak_tree_threads'], 2)
        result = gate.run_process([sys._base_executable, '-S', str(helper), '20'], folder, 'timeout', time.perf_counter() + .4)
        self.assertTrue(result['timed_out'])
        self.assertLess(result['wall_seconds'], 8)
        self.assertGreaterEqual(len(result['process_tree']), 2)
        self.assertTrue(all(gate.alive(item) is None for item in result['process_tree']))


if __name__ == '__main__':
    unittest.main()
