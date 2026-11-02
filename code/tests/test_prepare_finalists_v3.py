"""Temporary synthetic train records/bundles; no real holdout or training."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'code'), str(ROOT / 'code/tools')]
import numpy as np
import audit_assets
from mp2 import models_v2
from mp2.research_v3 import models
spec = importlib.util.spec_from_file_location('prepare_finalists_tested', ROOT / 'scripts/prepare_finalists_v3.py')
h = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h)


class FinalistPreparationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=ROOT / 'tmp', prefix='finalist-preparation-synthetic-')
        self.root = Path(self.tmp.name).resolve()
        import confirm_validation_v3 as protocol
        for name in protocol.INFERENCE_SOURCE | protocol.TRAIN_SOURCE | {h.PRODUCER, 'scripts/analyze_campaign_v3.py', 'code/tools/freeze_v3.py'}:
            target = self.root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((ROOT / name).read_bytes())
        self.gate = h.load_gate(self.root)
        self.freezer = h.load_freezer(self.root)
        self.name, self.analysis = 'synthetic_s0', h.BASE + 'analysis/final_search/analysis.json'
        self.reason = 'Synthetic only: explicitly retained recipe for a common-fold control.'
        self.decoder = {'mode': 'group_lse', 'alpha': 0., 'tau': 1., 'mask': True}
        self.config = {'experiment_id': self.name, 'seed': h.SEED, 'family': 'mlp', 'width': 2,
                       'hidden': 2, 'dropout': 0., 'decoder': self.decoder, 'preprocess': 'standard'}
        self.state = {'fc1.weight': np.zeros((2, 2064), np.float32), 'fc1.bias': np.zeros(2, np.float32),
                      'fc2.weight': np.zeros((2, 2), np.float32), 'fc2.bias': np.zeros(2, np.float32),
                      'head.weight': np.zeros((1, 2), np.float32), 'head.bias': np.zeros(1, np.float32)}
        for name in self.gate.TRAIN_PATHS:
            self.text(name, 'synthetic train hash fixture only')
        self.put('code/data/manifest.json', {'protocol': self.gate.PROTOCOL, 'splits': {'train': 4755, 'validation': 798, 'test': 467}})
        universe = set(range(4755))
        folds = []
        for number in range(3):
            held = list(range(1585 * number, 1585 * (number + 1)))
            outer = sorted(universe - set(held))
            folds.append({'fold': number, 'holdout': held, 'outer_fit': outer, 'inner_fit': outer[:2500], 'inner_stop': outer[2500:]})
        fold_record = {'questions': 4755, 'seed': h.SEED, 'folds': folds}
        self.put(h.BASE + 'common_folds.json', fold_record)
        self.put('results/research_v2/common_folds.json', fold_record)
        self.prov = {'source_files': {name: h.sha(self.root / name) for name in self.gate.TRAIN_SOURCE},
                     'data_files': {name: h.sha(self.root / name) for name in self.gate.TRAIN_PATHS},
                     'folds_sha256': h.sha(self.root / h.BASE / 'common_folds.json')}
        config_path = 'code/configs/research_v3/' + self.name + '.json'
        self.put(config_path, self.config)
        self.job = {'experiment_id': self.name, 'status': 'completed', 'round': 'synthetic', 'config': self.config,
            'config_path': config_path, 'config_sha256': h.canonical_sha(self.config),
            'registered_provenance': {'source_files': self.prov['source_files'],
                'train_and_fold_files': {**self.prov['data_files'], h.BASE + 'common_folds.json': self.prov['folds_sha256']}}}
        self.put(h.BASE + 'campaign_queue.json', {'schema': 'mp2-campaign-queue-v3', 'jobs': [self.job]})
        self.put(h.BASE + 'SEARCH_CLOSED.json', {'search_closed': True, 'created_at_utc': '2026-01-01T00:00:00Z'})
        directory = h.BASE + self.name + '/'
        self.put(directory + 'registration.json', {'config': self.config, 'config_sha256': h.canonical_sha(self.config), **self.prov})
        reports = []
        for number in range(3):
            checkpoint = f'artifacts/research_v3/{self.name}/fold{number}'
            self.bundle(checkpoint)
            inventory = audit_assets.audit(self.root / checkpoint)
            files = {}
            for suffix in ('scores', 'raw-scores'):
                filename = f'fold{number}-{suffix}.npy'
                self.text(directory + filename, 'opaque synthetic prediction bytes')
                files[filename] = h.sha(self.root / directory / filename)
            report = {'experiment_id': self.name, 'fold': number, 'status': 'complete', 'n': 1585, 'correct': 1400, 'hard_n': 700,
                'hard_correct': 518 + 7 * number, 'seed': h.SEED + number, 'provenance': self.prov,
                'checkpoint': checkpoint, 'output_sha256': files,
                'asset_sha256': {item['path']: item['sha256'] for item in inventory['files']},
                'parameters': 4139, 'numeric_asset_bytes': 53708,
                'inner_stopping': {'best_epoch': number + 2}, 'fit': {'seconds': 1.}}
            reports.append(report)
            self.put(directory + f'fold{number}.json', report)
        self.text(directory + 'oof.npz', 'opaque synthetic OOF bytes; never decoded')
        self.summary = {'experiment_id': self.name, 'status': 'complete', 'config': self.config, 'provenance': self.prov,
            'n': 4755, 'correct': 4200, 'hard_n': 2100, 'hard_correct': 1575, 'training_seconds': 3.,
            'accuracy': 100 * 4200 / 4755, 'hard_accuracy': 75.,
            'folds': reports, 'oof_sha256': h.sha(self.root / directory / 'oof.npz')}
        self.put(directory + 'summary.json', self.summary)
        self.job['result'] = {'summary_path': directory + 'summary.json', 'summary_sha256': h.sha(self.root / directory / 'summary.json'),
            'oof_sha256': self.summary['oof_sha256'], 'accuracy': self.summary['accuracy'], 'hard_accuracy': self.summary['hard_accuracy']}
        self.put(h.BASE + 'campaign_queue.json', {'schema': 'mp2-campaign-queue-v3', 'jobs': [self.job]})
        row = {'version': 'research_v3', 'experiment_id': self.name, 'status': 'completed', 'config': self.config,
            'seed': h.SEED, 'summary_sha256': h.sha(self.root / directory / 'summary.json'),
            'oof_sha256': self.summary['oof_sha256'], 'hard_fold_mean': 75., 'hard_fold_sd': 1.,
            'parameters': 4139, 'numeric_asset_bytes': 53708,
            **{key: self.summary[key] for key in ('n', 'correct', 'hard_n', 'hard_correct', 'training_seconds')}}
        self.put(self.analysis, {'folds_sha256': self.prov['folds_sha256'],
            'source_sha256': h.sha(self.root / 'scripts/analyze_campaign_v3.py'), 'rows': [row]})
        baseline_checkpoint = 'artifacts/research_v2/final_historical'
        baseline_config = {'experiment_id': 'historical_s0', 'family': 'linear', 'seed': h.SEED, 'decoder': self.decoder}
        models_v2.save_bundle(self.root / baseline_checkpoint, baseline_config,
            {'weight': np.zeros(2064), 'bias': np.asarray(0.)}, np.zeros(2064), np.ones(2064), self.decoder)
        self.put('results/research_v2/historical_s0/final_fit.json', {'config': baseline_config})
        inventory = audit_assets.audit(self.root / baseline_checkpoint)
        self.baseline = {'selected_experiment_id': 'historical_s0', 'method': 'code/solution_v2.py', 'checkpoint': baseline_checkpoint,
            'source_files': {name: h.sha(self.root / name) for name in self.gate.INFERENCE_SOURCE},
            'assets': {baseline_checkpoint + '/' + item['path']: item['sha256'] for item in inventory['files']},
            'validation': {'n': 798, 'correct': 740, 'hard_n': 355, 'hard_correct': 297}}
        self.put('results/research_v2/V2_FREEZE.json', self.baseline)

    def tearDown(self):
        self.tmp.cleanup()

    def put(self, relative, value):
        target = self.root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(h.payload(value))
        return target

    def text(self, relative, value):
        target = self.root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(value, encoding='utf-8')
        return target

    def bundle(self, relative):
        models.save_bundle(self.root / relative, self.config, self.state, np.zeros(2064), np.ones(2064), self.decoder)

    def select(self):
        return h.select(self.root, [self.name], self.analysis, self.reason, self.gate)

    def final_fit(self):
        checkpoint = 'artifacts/research_v3/final_' + self.name
        self.bundle(checkpoint)
        inventory = audit_assets.audit(self.root / checkpoint)
        state = self.text(f'tmp/research_v3/training_states/{self.name}/final/current.pt', 'opaque synthetic trusted training state')
        report = {'checkpoint': checkpoint, 'config': self.config, 'provenance': self.prov,
            'asset_sha256': {item['path']: item['sha256'] for item in inventory['files']},
            'info': {'context': {**self.prov, 'phase': 'final-all-train'}, 'epochs_run': 3, 'evaluation_weights': 'live',
                     'checkpoint': str(state.resolve()), 'checkpoint_sha256': h.sha(state)}}
        self.put(h.BASE + self.name + '/final_fit.json', report)
        return report

    def completed_validation(self, improved=True):
        verified = self.gate.validate_lock()
        row = {**verified['lock']['finalists'][0], 'local_resource_eligible': True,
            'validation': {'n': 798, 'correct': 741 if improved else 740, 'hard_n': 355, 'hard_correct': 298 if improved else 297},
            'resource': verified['resources'][self.name], 'prediction_plus_scoring_seconds': 1.}
        receipt = self.put(h.BASE + f'validation/{self.name}/synthetic-aggregate.json', {'synthetic': True})
        self.put(h.BASE + 'VALIDATION_OPENED.json', {'synthetic': True})
        summary = {'schema': 'mp2-validation-confirmation-v3', 'completed_at_utc': h.now(),
            'lock_sha256': verified['lock_sha256'], 'search_remains_closed': True, 'new_test_evaluated': False,
            'finalists': [row], 'baseline_validation': self.baseline['validation'],
            'selected': self.gate.choose_finalist([row], self.baseline),
            'evidence_sha256': {receipt.relative_to(self.root).as_posix(): h.sha(receipt)}}
        self.put(h.BASE + 'validation_summary.json', summary)
        return summary

    def test_select_then_actual_strict_lock_without_holdout_files(self):
        selected = self.select()
        self.assertEqual(selected['allowed_final_fit_experiments'], [self.name])
        self.assertEqual(selected['final_fit_commands'][0][-1], 'artifacts/research_v3/final_' + self.name)
        original = (self.root / h.SELECTION).read_bytes()
        self.assertEqual(self.select(), selected)
        self.assertEqual((self.root / h.SELECTION).read_bytes(), original)
        self.final_fit()
        locked = h.lock(self.root, self.gate)
        self.assertEqual(self.gate.validate_lock()['lock'], locked)
        locked_bytes = (self.root / h.LOCK).read_bytes()
        self.assertEqual(h.lock(self.root, self.gate), locked)
        self.assertEqual((self.root / h.LOCK).read_bytes(), locked_bytes)
        self.assertFalse((self.root / h.BASE / 'VALIDATION_OPENED.json').exists())
        self.assertFalse((self.root / 'code/data/gsm8k_validation.jsonl').exists())

    def test_missing_fold_rejected_before_selection_write(self):
        self.summary['folds'].pop()
        self.put(h.BASE + self.name + '/summary.json', self.summary)
        self.job['result']['summary_sha256'] = h.sha(self.root / h.BASE / self.name / 'summary.json')
        self.put(h.BASE + 'campaign_queue.json', {'schema': 'mp2-campaign-queue-v3', 'jobs': [self.job]})
        with self.assertRaisesRegex(ValueError, 'Missing or incomplete outer fold'):
            self.select()
        self.assertFalse((self.root / h.SELECTION).exists())

    def test_wrong_seed_and_more_than_two_and_duplicate_ids_rejected(self):
        for ids in ([self.name, 'b', 'c'], [self.name, self.name]):
            with self.assertRaisesRegex(ValueError, 'one or two distinct'):
                h.select(self.root, ids, self.analysis, self.reason, self.gate)
        self.config['seed'] += 1
        self.put(self.job['config_path'], self.config)
        self.job['config'] = self.config
        self.job['config_sha256'] = h.canonical_sha(self.config)
        self.put(h.BASE + 'campaign_queue.json', {'schema': 'mp2-campaign-queue-v3', 'jobs': [self.job]})
        with self.assertRaisesRegex(ValueError, 'fixed base deployment seed'):
            self.select()

    def test_changed_source_or_cv_scores_rejected(self):
        source = self.root / 'code/mp2/features.py'
        original = source.read_bytes()
        source.write_bytes(original + b'\n# synthetic mutation\n')
        with self.assertRaisesRegex(ValueError, 'Evidence changed'):
            self.select()
        source.write_bytes(original)
        self.text(h.BASE + self.name + '/fold1-scores.npy', 'changed opaque score bytes')
        with self.assertRaisesRegex(ValueError, 'Evidence changed'):
            self.select()

    def test_changing_existing_selection_or_lock_never_overwrites(self):
        self.select()
        saved = (self.root / h.SELECTION).read_bytes()
        with self.assertRaisesRegex(ValueError, 'Immutable finalist selection differs'):
            h.select(self.root, [self.name], self.analysis, 'Different selection reason', self.gate)
        self.assertEqual(saved, (self.root / h.SELECTION).read_bytes())
        self.final_fit()
        locked = h.lock(self.root, self.gate)
        locked['producer']['sha256'] = '0' * 64
        self.put(h.LOCK, locked)
        tampered = (self.root / h.LOCK).read_bytes()
        with self.assertRaisesRegex(ValueError, 'Immutable finalist lock changed'):
            h.lock(self.root, self.gate)
        self.assertEqual(tampered, (self.root / h.LOCK).read_bytes())
        with self.assertRaisesRegex(ValueError, 'already locked'):
            self.select()

    def test_final_fit_required_and_state_epoch_recipe_checked(self):
        self.select()
        with self.assertRaises(FileNotFoundError):
            h.lock(self.root, self.gate)
        self.assertFalse((self.root / h.LOCK).exists())
        report = self.final_fit()
        report['info']['epochs_run'] = 99
        self.put(h.BASE + self.name + '/final_fit.json', report)
        with self.assertRaisesRegex(ValueError, 'epoch/context'):
            h.lock(self.root, self.gate)
        report['info']['epochs_run'] = 3
        self.put(h.BASE + self.name + '/final_fit.json', report)
        Path(report['info']['checkpoint']).write_bytes(b'changed synthetic state')
        with self.assertRaisesRegex(ValueError, 'trusted full-train state'):
            h.lock(self.root, self.gate)
        self.assertFalse((self.root / h.LOCK).exists())

    def test_strict_prevalidation_failure_does_not_publish_lock_or_patch_validator(self):
        self.select()
        self.final_fit()
        original_read, original_sha = self.gate.read, self.gate.sha
        with patch.object(self.gate, 'validate_lock', side_effect=ValueError('Synthetic strict-validator rejection')):
            with self.assertRaisesRegex(ValueError, 'strict-validator'):
                h.lock(self.root, self.gate)
        self.assertIs(self.gate.read, original_read)
        self.assertIs(self.gate.sha, original_sha)
        self.assertFalse((self.root / h.LOCK).exists())

    def test_asset_addition_or_preexisting_final_fit_rejected(self):
        self.final_fit()
        with self.assertRaisesRegex(ValueError, 'precede all selected full-train'):
            self.select()
        self.assertFalse((self.root / h.SELECTION).exists())

    def test_extra_unrecorded_asset_is_rejected_before_lock_publication(self):
        self.select()
        self.final_fit()
        extra = self.text('artifacts/research_v3/final_' + self.name + '/unrecorded.bin', 'synthetic extra file')
        with self.assertRaisesRegex(ValueError, 'asset inventory differs'):
            h.lock(self.root, self.gate)
        self.assertTrue(extra.exists())
        self.assertFalse((self.root / h.LOCK).exists())

    def test_exactly_two_explicit_candidates_lock_with_exact_union(self):
        first = self.name
        second = 'synthetic_alternative_s0'
        config = {**self.config, 'experiment_id': second}
        directory = h.BASE + second + '/'
        job = deepcopy(self.job)
        job.update(experiment_id=second, config=config, config_sha256=h.canonical_sha(config),
                   config_path='code/configs/research_v3/' + second + '.json')
        self.put(job['config_path'], config)
        self.put(h.BASE + 'campaign_queue.json', {'schema': 'mp2-campaign-queue-v3', 'jobs': [self.job, job]})
        self.put(directory + 'registration.json', {'config': config, 'config_sha256': h.canonical_sha(config), **self.prov})
        summary = deepcopy(self.summary)
        summary.update(experiment_id=second, config=config)
        for report in summary['folds']:
            report['experiment_id'] = second
            report['checkpoint'] = f"artifacts/research_v3/{second}/fold{report['fold']}"
            self.bundle(report['checkpoint'])
            inventory = audit_assets.audit(self.root / report['checkpoint'])
            report['asset_sha256'] = {item['path']: item['sha256'] for item in inventory['files']}
            for filename, digest in report['output_sha256'].items():
                target = self.root / directory / filename
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes((self.root / h.BASE / first / filename).read_bytes())
            self.put(directory + f"fold{report['fold']}.json", report)
        self.text(directory + 'oof.npz', 'opaque synthetic OOF bytes; never decoded')
        self.put(directory + 'summary.json', summary)
        job['result']['summary_path'] = directory + 'summary.json'
        job['result']['summary_sha256'] = h.sha(self.root / directory / 'summary.json')
        self.put(h.BASE + 'campaign_queue.json', {'schema': 'mp2-campaign-queue-v3', 'jobs': [self.job, job]})
        analysis = h.read(self.root / self.analysis)
        row = {**analysis['rows'][0], 'experiment_id': second, 'config': config,
               'summary_sha256': h.sha(self.root / directory / 'summary.json')}
        analysis['rows'].append(row)
        self.put(self.analysis, analysis)
        selected = h.select(self.root, [first, second], self.analysis, self.reason, self.gate)
        self.assertEqual(selected['allowed_final_fit_experiments'], [first, second])
        self.final_fit()
        self.name, self.config = second, config
        self.final_fit()
        locked = h.lock(self.root, self.gate)
        self.assertEqual([item['name'] for item in locked['finalists']], [first, second])
        self.assertEqual(self.gate.validate_lock()['lock'], locked)

    def test_stale_completed_scheduler_receipt_is_rejected(self):
        self.job['result']['summary_sha256'] = '0' * 64
        self.put(h.BASE + 'campaign_queue.json', {'schema': 'mp2-campaign-queue-v3', 'jobs': [self.job]})
        with self.assertRaisesRegex(ValueError, 'Completed scheduler receipt'):
            self.select()
        self.assertFalse((self.root / h.SELECTION).exists())

    def test_new_selection_copy_is_exact_and_idempotent(self):
        self.select(); self.final_fit(); h.lock(self.root, self.gate)
        self.completed_validation(improved=True)
        selected = h.copy_selected(self.root, self.gate, self.freezer)
        self.assertFalse(selected['historical_fallback'])
        self.assertEqual(selected['oof_directory'], h.BASE + self.name)
        result = self.freezer.validate_selected(self.root, self.gate.validate_lock())
        self.assertEqual(result[0], selected)
        self.assertEqual(h.copy_selected(self.root, self.gate, self.freezer), selected)
        original = audit_assets.audit(self.root / selected['origin_checkpoint'])
        copied = audit_assets.audit(self.root / selected['checkpoint'])
        self.assertEqual([(r['path'], r['sha256']) for r in original['files']], [(r['path'], r['sha256']) for r in copied['files']])

    def test_historical_fallback_copy_uses_original_selector_and_oof(self):
        self.select(); self.final_fit(); h.lock(self.root, self.gate)
        self.completed_validation(improved=False)
        selected = h.copy_selected(self.root, self.gate, self.freezer)
        self.assertTrue(selected['historical_fallback'])
        self.assertEqual(selected['components'][0]['kind'], 'historical_v2')
        self.assertEqual(selected['oof_directory'], 'results/research_v2/historical_s0')
        self.assertEqual(selected['selector']['family'], 'linear')
        self.freezer.validate_selected(self.root, self.gate.validate_lock())

    def test_partial_copy_and_inconsistent_saved_selection_are_preserved(self):
        self.select(); self.final_fit(); h.lock(self.root, self.gate)
        summary = self.completed_validation(improved=True)
        summary['selected']['name'] = 'another_model'
        self.put(h.BASE + 'validation_summary.json', summary)
        with self.assertRaisesRegex(ValueError, 'Saved unique selection differs'):
            h.copy_selected(self.root, self.gate, self.freezer)
        self.completed_validation(improved=True)
        partial = self.text('artifacts/research_v3/final_selected/partial.bin', 'preserve synthetic partial')
        with self.assertRaisesRegex(ValueError, 'Partial/existing'):
            h.copy_selected(self.root, self.gate, self.freezer)
        self.assertEqual(partial.read_text(), 'preserve synthetic partial')
        self.assertFalse((self.root / 'submission/v3/SELECTED_MODEL.json').exists())

    def test_scope_closure_and_holdout_guards(self):
        for name in ('code/data/gsm8k_validation.jsonl', 'code/data/candidates/gsm8k_test.npz', 'docs/RANKING_BACKGROUND.md', 'results/../code/data/gsm8k_validation.jsonl'):
            with self.assertRaises(ValueError):
                h.path(self.root, name)
        self.put(h.BASE + 'SEARCH_CLOSED.json', {'search_closed': False, 'created_at_utc': '2026-01-01T00:00:00Z'})
        with self.assertRaisesRegex(ValueError, 'Explicit SEARCH_CLOSED'):
            self.select()


if __name__ == '__main__':
    unittest.main(verbosity=2)
