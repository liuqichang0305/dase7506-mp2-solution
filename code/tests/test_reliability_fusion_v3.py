"""Synthetic future-fusion reader checks; no official data or real mixture run."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code'))
sys.path.insert(0, str(ROOT / 'code/tools'))
import numpy as np
from mp2.selection_v2 import group_probabilities
import reliability_v3
import reliability_fusion_v3 as reader


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


class FusionReaderTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='mp2-fusion-reader-synthetic-')
        self.root = Path(self.temporary.name)
        self.patches = [patch.object(reader, 'ROOT', self.root), patch.object(reliability_v3, 'ROOT', self.root)]
        for item in self.patches:
            item.start()
        self.results = self.root / 'results/research_v3'
        self.folder = self.results / 'fixed_mixture'
        self.folder.mkdir(parents=True)
        n = 36
        fold = np.arange(n) % 3
        groups = np.tile(np.arange(4), (n, 1))
        labels = np.tile([1., 0., 0., 0.], (n, 1))
        manifest = {'questions': n, 'folds': [{'fold': i, 'holdout': np.flatnonzero(fold == i).tolist()} for i in range(3)]}
        write(self.results / 'common_folds.json', manifest)
        decoder = {'mode': 'group_lse', 'alpha': 0., 'tau': 1., 'mask': True}
        self.keys = ['v3:synthetic_a', 'v3:synthetic_b']
        descriptions, probabilities = [], []
        for index, key in enumerate(self.keys):
            directory = self.results / key.split(':')[1]
            directory.mkdir()
            correct = np.zeros(n, bool)
            if index == 0:
                correct[:18] = True
            else:
                correct[:8] = True
                correct[18:28] = True
            raw = np.zeros((n, 4))
            raw[np.arange(n), np.where(correct, 0, 1)] = 3.
            probability = group_probabilities(raw, groups)
            probabilities.append(probability)
            config = {'experiment_id': key.split(':')[1], 'family': 'mlp', 'seed': 3036900957, 'decoder': decoder}
            registration = {'config': config, 'folds_sha256': sha(self.results / 'common_folds.json'),
                            'data_files': {'code/data/gsm8k_train.jsonl': 'a' * 64}}
            write(directory / 'registration.json', registration)
            reports = []
            for number in range(3):
                rows = fold == number
                filenames = [f'fold{number}-scores.npy', f'fold{number}-raw-scores.npy']
                for name in filenames:
                    np.save(directory / name, raw[rows], allow_pickle=False)
                reports.append({'fold': number, 'status': 'complete', 'output_sha256': {name: sha(directory / name) for name in filenames}})
            np.savez(directory / 'oof.npz', groups=groups, labels=labels, fold=fold, scores=raw, raw_scores=raw)
            write(directory / 'summary.json', {'status': 'complete', 'config': config, 'folds': reports,
                                              'oof_sha256': sha(directory / 'oof.npz')})
            descriptions.append({'method': key, 'input_sha256': {name: sha(directory / name)
                for name in ('registration.json', 'summary.json', 'oof.npz')}})
        prior = {'created_at_utc': '2026-01-01T00:00:00+00:00', 'methods': descriptions,
                 'complementarity': [{'methods': self.keys, 'gate_passed': True}]}
        gate_path = self.results / 'prior/diagnostic.json'
        write(gate_path, prior)
        self.registration = {'status': 'registered', 'purpose': 'single_fixed_equal_probability_oof',
            'methods': self.keys, 'weights': [.5, .5],
            'oof_sha256': {item['method']: item['input_sha256']['oof.npz'] for item in descriptions},
            'gate_evidence': {'path': gate_path.relative_to(self.root).as_posix(), 'sha256': sha(gate_path)},
            'registered_at_utc': '2026-01-01T00:01:00+00:00', 'hypothesis': 'Synthetic loader test only'}
        write(self.folder / 'fusion-registration.json', self.registration)
        probability = np.mean(probabilities, axis=0)
        choices = probability.argmax(1).astype(np.int64)
        self.arrays = {'probabilities': probability, 'choices': choices,
                       'correct': labels[np.arange(n), choices].astype(bool), 'fold': fold,
                       'groups': groups, 'labels': labels}
        np.savez(self.folder / 'fusion-oof.npz', **self.arrays)
        self.diagnostic = {'status': 'complete', 'created_at_utc': '2026-01-01T00:02:00+00:00',
            'methods': descriptions, 'folds_sha256': sha(self.results / 'common_folds.json'),
            'fusion': {'status': 'complete', 'methods': self.keys, 'weights': [.5, .5],
                       'gate': {'gate_passed': True}, 'registration_file': 'fusion-registration.json',
                       'registration_sha256': sha(self.folder / 'fusion-registration.json'),
                       'oof_sha256': sha(self.folder / 'fusion-oof.npz')}}
        self.save_diagnostic()

    def tearDown(self):
        for item in reversed(self.patches):
            item.stop()
        self.temporary.cleanup()

    def save_diagnostic(self):
        write(self.folder / 'diagnostic.json', self.diagnostic)

    def save_arrays_and_update_hash(self):
        np.savez(self.folder / 'fusion-oof.npz', **self.arrays)
        self.diagnostic['fusion']['oof_sha256'] = sha(self.folder / 'fusion-oof.npz')
        self.save_diagnostic()

    def test_valid_actual_chosen_event_schema_and_complete_registration_identity(self):
        value = reader.load_fusion_oof(self.folder)
        self.assertEqual(value['split'], 'train_oof')
        self.assertEqual(value['selector'], {'kind': 'fixed_equal_probability', 'specification': self.registration})
        np.testing.assert_array_equal(value['choices'], self.arrays['choices'])
        np.testing.assert_array_equal(value['correct'], self.arrays['correct'])
        np.testing.assert_array_equal(value['confidence'], self.arrays['probabilities'][np.arange(36), self.arrays['choices']])
        self.assertFalse((self.root / 'code/data').exists())
        self.assertTrue(value['source_sha256'])

    def test_not_run_and_unequal_weights_rejected(self):
        original = deepcopy(self.diagnostic)
        for change in ({'status': 'not_run'}, {'weights': [.8, .2]}):
            self.diagnostic = deepcopy(original)
            self.diagnostic['fusion'].update(change)
            self.save_diagnostic()
            with self.assertRaises(ValueError):
                reader.load_fusion_oof(self.folder)

    def test_changed_member_archive_rejected_before_use(self):
        path = self.results / 'synthetic_a/oof.npz'
        path.write_bytes(path.read_bytes() + b'tamper')
        with self.assertRaisesRegex(ValueError, 'provenance changed'):
            reader.load_fusion_oof(self.folder)

    def test_rehashed_wrong_average_and_wrong_choice_rejected(self):
        self.arrays['probabilities'][0, 0] -= .001
        self.arrays['probabilities'][0, 1] += .001
        self.save_arrays_and_update_hash()
        with self.assertRaisesRegex(ValueError, 'equal group-probability'):
            reader.load_fusion_oof(self.folder)
        self.arrays['probabilities'][0, 0] += .001
        self.arrays['probabilities'][0, 1] -= .001
        self.arrays['choices'][0] = 1
        self.arrays['correct'][0] = False
        self.save_arrays_and_update_hash()
        with self.assertRaisesRegex(ValueError, 'representative rule'):
            reader.load_fusion_oof(self.folder)

    def test_rehashed_group_label_and_fold_mismatch_rejected(self):
        original = {key: value.copy() for key, value in self.arrays.items()}
        for name in ('groups', 'labels', 'fold'):
            self.arrays = {key: value.copy() for key, value in original.items()}
            self.arrays[name].flat[0] = 1 if name != 'labels' else 0
            self.save_arrays_and_update_hash()
            with self.assertRaisesRegex(ValueError, 'do not align'):
                reader.load_fusion_oof(self.folder)


if __name__ == '__main__':
    result = unittest.TextTestRunner(stream=sys.stdout, verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__]))
    raise SystemExit(0 if result.wasSuccessful() else 1)
