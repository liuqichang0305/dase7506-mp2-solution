"""Synthetic interruption/recovery, nested fit, artifact and isolation checks."""
from copy import deepcopy
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mp2.research_v3 import training as t
from mp2.research_v3.models import build_model, export_state
from mp2.research_v3.preprocess import fit_preprocessor


class TrainingV3Tests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=t.ROOT / 'tmp', prefix='v3-synthetic-')
        self.folder = Path(self.temporary.name)
        rng = np.random.default_rng(59)
        self.x = rng.normal(size=(18, 4, 2064))
        self.groups = np.tile(np.array([0, 0, 1, 2]), (18, 1))
        self.labels = (self.groups == (np.arange(18) % 3)[:, None]).astype(float)
        self.config = {'experiment_id': 'synthetic', 'family': 'mlp', 'width': 8, 'hidden': 4, 'dropout': .25,
                       'objective': 'group', 'reduction': 'full', 'optimizer': 'muon', 'muon_lr': .01,
                       'lr': .001, 'weight_decay': .001, 'ema_decay': .9, 'preprocess': 'robust_stats',
                       'seed': 59, 'batch_size': 4, 'max_epochs': 3, 'patience': 5, 'permute_candidates': True,
                       'decoder': {'mode': 'group_lse', 'alpha': 0., 'tau': 1., 'mask': True}}

    def tearDown(self):
        self.temporary.cleanup()

    def fit(self, name, config=None, interrupt=None, stopping=True):
        return t.fit_neural(self.x, self.labels, self.groups, np.arange(12), np.arange(12, 18) if stopping else None,
            config or self.config, 59, self.folder / name, self.folder / f'{name}.json',
            {'source': 'synthetic-code-hash', 'data': 'synthetic-only', 'fold': 0},
            max_new_epochs=interrupt)

    def test_epoch_resume_is_exact_with_muon_ema_dropout_and_permutation(self):
        with redirect_stdout(io.StringIO()):
            uninterrupted, prep, info = self.fit('full')
            with self.assertRaises(t.TrainingPaused):
                self.fit('resumed', interrupt=1)
            checkpoint = torch.load(self.folder / 'resumed/current.pt', weights_only=True)
            self.assertEqual(checkpoint['epoch'], 1)
            self.assertFalse(checkpoint['complete'])
            resumed, resumed_prep, resumed_info = self.fit('resumed')
        for key in uninterrupted:
            np.testing.assert_array_equal(uninterrupted[key], resumed[key])
        for key in prep:
            np.testing.assert_array_equal(prep[key], resumed_prep[key])
        self.assertEqual(info['best_epoch'], resumed_info['best_epoch'])
        self.assertEqual(info['optimizer_steps'], resumed_info['optimizer_steps'])
        self.assertEqual(info['ema_updates'], resumed_info['ema_updates'])
        self.assertEqual(resumed_info['resumed_from_epoch'], 1)
        first = json.loads((self.folder / 'full.json').read_text())['history']
        second = json.loads((self.folder / 'resumed.json').read_text())['history']
        self.assertEqual([r['inner_stop_loss'] for r in first], [r['inner_stop_loss'] for r in second])
        with self.assertRaises(ValueError), redirect_stdout(io.StringIO()):
            self.fit('resumed', {**self.config, 'lr': .002})

    def test_scaler_uses_fit_partition_and_tabm_raw_export(self):
        self.x[12:] += 30.
        cfg = {**self.config, 'family': 'tabm', 'width': 4, 'depth': 1, 'k': 2, 'optimizer': 'adamw',
               'max_epochs': 2, 'preprocess': 'standard', 'ema_decay': None}
        with redirect_stdout(io.StringIO()):
            state, prep, info = self.fit('tabm', cfg)
        expected = fit_preprocessor(self.x[:12], 'standard')
        np.testing.assert_array_equal(prep['mean'], expected['mean'])
        self.assertFalse(np.allclose(prep['mean'], self.x.mean(axis=(0, 1))))
        raw = t.predict_features(self.x, cfg, state, prep)
        self.assertEqual(raw.shape, (18, 2, 4))
        self.assertEqual(info['evaluation_weights'], 'live')

    def test_global_training_lock_blocks_live_and_preserves_stale(self):
        with patch.object(t, 'ROOT', self.folder):
            with t.training_lock():
                with self.assertRaises(RuntimeError):
                    with t.training_lock():
                        self.fail('second trainer entered')
            lock = self.folder / 'tmp/research_v3/trainer.lock'
            self.assertFalse(lock.exists())
            lock.write_text(json.dumps({'pid': 2147483647, 'create_time': 0., 'command_line': []}))
            with t.training_lock():
                self.assertTrue(lock.is_file())
            self.assertEqual(len(list(lock.parent.glob('trainer.stale-*.json'))), 1)

    def test_closed_and_frozen_stage_guards_and_safe_ids(self):
        out = self.folder / 'results'
        out.mkdir()
        for name in ('.', '..', '../escaped', 'MixedCase', 'contains-dash'):
            with self.assertRaises(ValueError):
                t.normalize_config({'id': name})
        with patch.object(t, 'ROOT', self.folder), patch.object(t, 'OUT', out):
            t.check_stage_permission()
            with self.assertRaises(RuntimeError):
                t.check_stage_permission('synthetic')
            (out / 'SEARCH_CLOSED.json').write_text('{}')
            with self.assertRaises(RuntimeError):
                t.check_stage_permission()
            (out / 'FINALISTS_LOCK.json').write_text(json.dumps({'allowed_final_fit_experiments': ['synthetic']}))
            t.check_stage_permission('synthetic')
            with self.assertRaises(RuntimeError):
                t.check_stage_permission('unregistered')
            freeze = self.folder / 'submission/v3/FREEZE.json'
            freeze.parent.mkdir(parents=True)
            freeze.write_text('{}')
            with self.assertRaises(RuntimeError):
                t.check_stage_permission('synthetic')

    def test_data_and_ranking_audit_hook_blocks_before_open(self):
        code = "import sys; sys.path.insert(0,'code'); from mp2.research_v3.training import install_training_guard,ROOT; install_training_guard(); paths=['code/data/gsm8k_validation.jsonl','code/data/gsm8k_test.jsonl','results/leaderboard-review.json','docs/RANKING_BACKGROUND.md']; blocked=0\nfor path in paths:\n try: open(ROOT/path,'r')\n except PermissionError: blocked+=1\nassert blocked==4; print('blocked',blocked)"
        result = subprocess.run([sys.executable, '-c', code], cwd=t.ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('blocked 4', result.stdout)
        with self.assertRaises(PermissionError):
            t._check_access(t.DATA / 'gsm8k_train.jsonl', 'w')

    def test_partial_bundle_is_preserved_and_complete_bundle_never_overwritten(self):
        artifacts, out = self.folder / 'artifacts', self.folder / 'results'
        target = artifacts / 'synthetic' / 'fold0'
        target.mkdir(parents=True)
        (target / 'weights.npz').write_bytes(b'incomplete synthetic bytes')
        model = build_model(self.config)
        state, prep = export_state(model), fit_preprocessor(self.x[:12])
        with patch.object(t, 'ASSETS', artifacts), patch.object(t, 'OUT', out), redirect_stdout(io.StringIO()):
            t._save_or_verify_bundle(target, self.config, state, prep)
            preserved = list(target.parent.glob('fold0.incomplete-*'))
            self.assertEqual(len(preserved), 1)
            self.assertEqual((preserved[0] / 'weights.npz').read_bytes(), b'incomplete synthetic bytes')
            before = t.digest(target / 'weights.npz')
            t._save_or_verify_bundle(target, self.config, state, prep)
            self.assertEqual(before, t.digest(target / 'weights.npz'))
            changed = deepcopy(state)
            changed[next(iter(changed))] += 1.
            with self.assertRaises(ValueError):
                t._save_or_verify_bundle(target, self.config, changed, prep)

    def test_completed_threefold_reuse_verifies_artifacts_without_retraining(self):
        out, assets, states = self.folder / 'results', self.folder / 'artifacts', self.folder / 'states'
        out.mkdir()
        folds = []
        for number in range(3):
            held = np.arange(number * 6, (number + 1) * 6)
            fitting = np.setdiff1d(np.arange(18), held)
            folds.append({'fold': number, 'holdout': held.tolist(), 'outer_fit': fitting.tolist(),
                          'inner_fit': fitting[:8].tolist(), 'inner_stop': fitting[8:].tolist()})
        t.write_json(out / 'common_folds.json', {'synthetic': True, 'folds': folds})
        provenance = {'source_files': {'synthetic': 'stable'}, 'data_files': {'synthetic_train': 'stable'},
                      'folds_sha256': t.digest(out / 'common_folds.json'), 'git_commit': 'informational'}
        cfg = {**self.config, 'optimizer': 'adamw', 'max_epochs': 1, 'ema_decay': None}
        with patch.object(t, 'OUT', out), patch.object(t, 'ASSETS', assets), patch.object(t, 'STATE_ROOT', states), \
             patch.object(t, 'data_and_folds', return_value=(self.x, self.labels, self.groups, folds)), \
             patch.object(t, 'provenance', return_value=provenance), redirect_stdout(io.StringIO()):
            result = t.run_cv(cfg)
            self.assertEqual(result['status'], 'complete')
            summary_hash = t.digest(out / 'synthetic/summary.json')
            with patch.object(t, 'fit_neural', side_effect=AssertionError('Completed run must not retrain')):
                reused = t.run_cv(cfg)
            self.assertEqual(result, reused)
            self.assertEqual(summary_hash, t.digest(out / 'synthetic/summary.json'))
            score_path = out / 'synthetic/fold0-scores.npy'
            values = np.load(score_path)
            values[0, 0] += .01
            np.save(score_path, values)
            with self.assertRaises(ValueError):
                t.run_cv(cfg)


if __name__ == '__main__':
    unittest.main(verbosity=2)
