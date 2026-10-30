"""Synthetic confidence/event tests; no official data or model training."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import reliability_v3 as r
from mp2.research_v3.selection import collapse_scores, member_probabilities, select_raw


class ReliabilityTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=r.ROOT / 'tmp', prefix='confidence-synthetic-')
        self.folder = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def oof(self, family='tabm'):
        path = self.folder / family
        path.mkdir()
        groups = np.tile([0, 0, 1, 2], (12, 1))
        rng = np.random.default_rng(84)
        raw = rng.normal(size=(12, 2, 4) if family == 'tabm' else (12, 4))
        decoder = {'mode': 'group_lse', 'alpha': .5, 'tau': 1., 'mask': True}
        choices = np.asarray(select_raw(raw, groups, decoder))
        chosen_groups = groups[np.arange(12), choices]
        target = np.where(np.arange(12) % 3 == 0, (chosen_groups + 1) % 3, chosen_groups)
        labels = (groups == target[:, None]).astype(float)
        fold = np.arange(12) % 3
        manifest = {'questions': 12, 'folds': [{'fold': i, 'holdout': np.flatnonzero(fold == i).tolist()} for i in range(3)]}
        fold_path = self.folder / 'common_folds.json'
        if not fold_path.exists():
            r.write_new(fold_path, manifest)
        config = {'experiment_id': family, 'family': family, 'decoder': decoder, 'seed': r.SEED}
        registration = {'config': config, 'folds_sha256': r.digest(fold_path),
                        'data_files': {'code/data/gsm8k_train.jsonl': '1' * 64}}
        r.write_new(path / 'registration.json', registration)
        scores = collapse_scores(raw, groups, decoder)
        np.savez(path / 'oof.npz', scores=scores, raw_scores=raw, groups=groups, labels=labels, fold=fold)
        reports = []
        for i in range(3):
            files = {}
            for suffix, values in [('scores', scores), ('raw-scores', raw)]:
                file = path / f'fold{i}-{suffix}.npy'
                np.save(file, values[fold == i])
                files[file.name] = r.digest(file)
            reports.append({'fold': i, 'status': 'complete', 'output_sha256': files})
        r.write_new(path / 'summary.json', {'status': 'complete', 'config': config, 'folds': reports,
                                          'oof_sha256': r.digest(path / 'oof.npz')})
        return path, raw, groups, labels, choices

    def test_actual_chosen_group_not_maximum_and_all_none(self):
        groups = np.array([[0, 0, 1, -1], [-1, -1, -1, -1]])
        probability = np.array([[.8, .2, 0, 0], [0, 0, 0, 0]])
        np.testing.assert_array_equal(r.chosen_confidence(groups, probability, [2, 0]), [.2, 0])
        with self.assertRaises(ValueError):
            r.chosen_confidence(groups, probability, [4, 0])
        with self.assertRaises(ValueError):
            r.chosen_confidence(groups, probability + .01, [2, 0])

    def test_bins_brier_and_risk_coverage_exact(self):
        result = r.metrics([0, 1, 1, 0], [0, .1, .9, 1])
        self.assertAlmostEqual(result['brier'], .455)
        self.assertEqual([b['n'] for b in result['bins']], [1, 1, 0, 0, 0, 0, 0, 0, 0, 2])
        self.assertIsNone(result['bins'][2]['accuracy'])
        self.assertAlmostEqual(result['ece_10_equal_width'], .45)
        self.assertEqual(result['risk_coverage']['ordered_original_rows'], [3, 2, 1, 0])
        np.testing.assert_allclose(result['risk_coverage']['risk'], [1, .5, 1 / 3, .5])
        tied = r.metrics([1, 0, 1], [.5, .5, .5])
        self.assertEqual(tied['risk_coverage']['ordered_original_rows'], [0, 1, 2])

    def test_mapping_fixed_penalty_finite_endpoints_and_no_choices_changed(self):
        correct = np.array([0, 0, 1, 0, 1, 1, 1, 1])
        confidence = np.array([0, .1, .3, .5, .6, .8, .9, 1.])
        mapping = r.fit_mapping(correct, confidence)
        self.assertEqual(mapping['C'], 1.)
        self.assertEqual(mapping['seed'], r.SEED)
        calibrated = r.apply_mapping(confidence, mapping)
        self.assertTrue(np.isfinite(calibrated).all())
        self.assertTrue(np.all((calibrated > 0) & (calibrated < 1)))
        from sklearn.linear_model import LogisticRegression
        reference = LogisticRegression(C=1., l1_ratio=0., solver='lbfgs', max_iter=1000, tol=1e-10, random_state=r.SEED)
        reference.fit(r._logit(confidence), correct)
        np.testing.assert_allclose(calibrated, reference.predict_proba(r._logit(confidence))[:, 1], atol=1e-14)
        with self.assertRaisesRegex(ValueError, 'both correct and incorrect'):
            r.fit_mapping(np.ones(8), confidence)

    def test_tabm_oof_uses_actual_member_probabilities_and_preserves_input(self):
        path, raw, groups, labels, choices = self.oof()
        before = r.digest(path / 'oof.npz')
        events = r.load_train_oof(path)
        expected = member_probabilities(raw, groups, {'mode': 'group_lse', 'alpha': .5, 'tau': 1., 'mask': True})
        np.testing.assert_array_equal(events['group_probabilities'], expected)
        np.testing.assert_array_equal(events['choices'], choices)
        self.assertFalse(np.allclose(expected, member_probabilities(raw.mean(axis=1), groups,
            {'mode': 'group_lse', 'alpha': .5, 'tau': 1., 'mask': True})))
        output = self.folder / 'fitted'
        report = r.fit(path, output)
        self.assertIn('IN-SAMPLE', report['interpretation'])
        self.assertEqual(report['choices_changed'], 0)
        self.assertEqual(r.digest(path / 'oof.npz'), before)
        with np.load(output / 'confidence-events.npz') as saved:
            np.testing.assert_array_equal(saved['choices'], choices)
        with self.assertRaises(FileExistsError):
            r.fit(path, output)

    def test_oof_tamper_and_incomplete_folds_rejected(self):
        path, *_ = self.oof()
        fold_file = path / 'fold0-raw-scores.npy'
        raw = np.load(fold_file)
        raw[0, 0, 0] += 1
        np.save(fold_file, raw)
        with self.assertRaisesRegex(ValueError, 'immutable held-out'):
            r.load_train_oof(path)
        with self.assertRaisesRegex(ValueError, 'Complete three-fold'):
            r._validate_folds(np.zeros(12, dtype=int), {'questions': 12})

    def test_legacy_v2_without_archive_hash_checks_each_saved_fold(self):
        path, _, _, _, choices = self.oof('mlp')
        with np.load(path / 'oof.npz') as archive:
            data = {key: archive[key].copy() for key in archive.files if key != 'raw_scores'}
        np.savez(path / 'oof.npz', **data)
        summary = json.loads((path / 'summary.json').read_text())
        del summary['oof_sha256']
        for report in summary['folds']:
            report['scores_sha256'] = report.pop('output_sha256')[f"fold{report['fold']}-scores.npy"]
        (path / 'summary.json').write_text(json.dumps(summary))
        events = r.load_train_oof(path)
        np.testing.assert_array_equal(events['choices'], choices)
        self.assertEqual(events['source_sha256']['oof.npz'], r.digest(path / 'oof.npz'))

    def test_fixed_equal_probability_oof_and_selected_confidence_check(self):
        single, raw, groups, labels, choices = self.oof('mlp')
        events = r.load_train_oof(single)
        output = self.folder / 'mixture'
        output.mkdir()
        weights = [.5, .5]
        np.savez(output / 'oof-mixture.npz', groups=groups, labels=labels, fold=events['fold'],
                 probability=events['group_probabilities'], choices=choices, correct=events['correct'],
                 selected_probability=events['confidence'])
        registration = {'split': 'train OOF only', 'specification': {'weights': weights, 'first': 'a', 'second': 'b',
                         'folds_sha256': events['folds_sha256']}}
        summary = {'status': 'complete', 'ensemble_evaluated': True, 'weights': weights,
                   'oof_mixture_sha256': r.digest(output / 'oof-mixture.npz')}
        r.write_new(output / 'registration.json', registration)
        r.write_new(output / 'summary.json', summary)
        result = r.load_train_oof(output)
        np.testing.assert_array_equal(result['confidence'], events['confidence'])
        np.testing.assert_array_equal(result['choices'], choices)
        with np.load(output / 'oof-mixture.npz') as archive:
            data = {key: archive[key].copy() for key in archive.files}
        data['selected_probability'] += .1
        np.savez(output / 'oof-mixture.npz', **data)
        summary['oof_mixture_sha256'] = r.digest(output / 'oof-mixture.npz')
        (output / 'summary.json').write_text(json.dumps(summary))
        with self.assertRaisesRegex(ValueError, 'actual selected group'):
            r.load_train_oof(output)

    def test_external_events_synthetic_roundtrip_and_validation_lock(self):
        path, *_ = self.oof('mlp')
        fitted = self.folder / 'calibration'
        r.fit(path, fitted)
        events = r.load_train_oof(path)
        record = {'schema': r.EVENT_SCHEMA, 'split': 'synthetic', 'selector': events['selector'],
                  'source_sha256': {'synthetic_input': 'a' * 64}, **{key: events[key].tolist() for key in
                  ('groups', 'group_probabilities', 'choices', 'correct')}}
        prediction = self.folder / 'events.json'
        r.write_new(prediction, record)
        report = r.evaluate(prediction, fitted, self.folder / 'evaluation')
        self.assertEqual(report['choices_changed'], 0)
        self.assertEqual(report['split'], 'synthetic')
        record['split'] = 'test'
        prediction.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError, 'test is disabled'):
            r.load_prediction_events(prediction)
        record['split'] = 'validation'
        prediction.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError, 'selection lock'):
            r.load_prediction_events(prediction)
        # This is a synthetic temporary lock, never the real campaign lock.
        lock = self.folder / 'results/research_v3/FINALISTS_LOCK.json'
        lock.parent.mkdir(parents=True)
        r.write_new(lock, {'synthetic': True})
        record['selection_lock'] = {'path': str(lock), 'sha256': r.digest(lock)}
        prediction.write_text(json.dumps(record))
        with patch.object(r, 'ROOT', self.folder):
            loaded = r.load_prediction_events(prediction)
        np.testing.assert_array_equal(loaded['choices'], events['choices'])

    def test_npz_events_hash_choice_alias_and_actual_confidence(self):
        path, *_ = self.oof('mlp')
        events = r.load_train_oof(path)
        archive = self.folder / 'events.npz'
        arrays = {key: events[key] for key in ('groups', 'group_probabilities', 'choices', 'correct', 'confidence')}
        arrays['selected'] = arrays['choices'].copy()
        np.savez(archive, **arrays)
        record = {'schema': r.EVENT_SCHEMA, 'split': 'synthetic', 'selector': events['selector'],
                  'source_sha256': {'synthetic_input': 'b' * 64},
                  'predictions_npz': {'path': archive.name, 'sha256': r.digest(archive)}}
        prediction = self.folder / 'event-manifest.json'
        r.write_new(prediction, record)
        loaded = r.load_prediction_events(prediction)
        np.testing.assert_array_equal(loaded['confidence'], events['confidence'])
        arrays['selected'][0] = (arrays['selected'][0] + 1) % 4
        np.savez(archive, **arrays)
        with self.assertRaisesRegex(ValueError, 'NPZ hash mismatch'):
            r.load_prediction_events(prediction)
        record['predictions_npz']['sha256'] = r.digest(archive)
        prediction.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError, 'Selected alias'):
            r.load_prediction_events(prediction)
        arrays['selected'] = arrays['choices'].copy()
        arrays['confidence'] = arrays['confidence'] + .01
        np.savez(archive, **arrays)
        record['predictions_npz']['sha256'] = r.digest(archive)
        prediction.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError, 'actual chosen group'):
            r.load_prediction_events(prediction)

    def test_real_results_require_unique_selected_model_and_matching_selector(self):
        path, *_ = self.oof('mlp')
        events = r.load_train_oof(path)
        target = self.folder / 'results/research_v3/mlp'
        target.mkdir(parents=True)
        events['oof_directory'] = str(target)
        with patch.object(r, 'ROOT', self.folder):
            with self.assertRaises(FileNotFoundError):
                r.selected_fit_guard(events)
            summary = target.parent / 'validation_summary.json'
            r.write_new(summary, {'schema': 'mp2-validation-confirmation-v3', 'selected': {'name': 'mlp'}})
            selected = self.folder / 'submission/v3/SELECTED_MODEL.json'
            selected.parent.mkdir(parents=True)
            record = {'schema': 'mp2-selected-model-v3', 'name': 'mlp',
                      'validation_summary_sha256': r.digest(summary),
                      'oof_directory': 'results/research_v3/mlp', 'selector': events['selector']}
            r.write_new(selected, record)
            r.selected_fit_guard(events)
            self.assertEqual(events['source_sha256']['selected_model_json'], r.digest(selected))
            record['selector'] = {'kind': 'different_model'}
            selected.write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError, 'unique selected method'):
                r.selected_fit_guard(events)


if __name__ == '__main__':
    unittest.main()
