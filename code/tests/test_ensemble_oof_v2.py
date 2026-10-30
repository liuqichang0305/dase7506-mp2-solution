"""Synthetic ensemble-gate and probability/index tests; no official bank access."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import ensemble_oof_v2 as ensemble


class EnsembleOOFTests(unittest.TestCase):
    def test_gate_threshold_and_failure(self):
        a, b = np.zeros(1000, bool), np.zeros(1000, bool)
        a[:500], b[10:510] = True, True
        gate = ensemble.complementarity_gate(a, b, np.ones(1000, bool))
        self.assertTrue(gate['passed'])
        self.assertEqual(gate['first_only_correct'], 10)
        self.assertEqual(gate['second_only_correct'], 10)
        self.assertEqual(gate['hard_oracle_gain_over_stronger_percentage_points'], 1.)
        self.assertFalse(ensemble.complementarity_gate(a, a, np.ones(1000, bool))['passed'])
        self.assertFalse(ensemble.complementarity_gate(a, b, np.zeros(1000, bool))['passed'])
        a, b = np.r_[a, np.zeros(1000, bool)], np.r_[b, np.zeros(1000, bool)]
        self.assertFalse(ensemble.complementarity_gate(a, b, np.ones(2000, bool))['passed'])

    def test_probability_average_is_not_logit_average(self):
        groups = np.array([[0, 1, 2, -1]])
        first = np.array([[10., 9., 0., 1000.]])
        second = np.array([[0., 9., 10., 1000.]])
        result = ensemble.equal_probability_mixture(first, second, groups, {}, {})
        self.assertEqual(result['choices'], [0])
        # The central group wins raw-logit averaging but loses probability averaging.
        self.assertEqual(int(np.argmax(((first+second)/2)[0, :3])), 1)
        np.testing.assert_allclose(result['probability'], (result['first_probability']+result['second_probability'])/2, atol=0, rtol=0)
        self.assertEqual(result['probability'][0, 3], 0.)
        self.assertAlmostEqual(result['probability'].sum(), 1.)

    def test_original_index_ties_none_and_actual_probability(self):
        groups = np.array([[2, 1, 2, -1], [-1, -1, -1, -1], [0, 1, 0, 1]])
        probability = np.array([[0., .2, .8, 0.], [0., 0., 0., 0.], [.5, .5, 0., 0.]])
        scores = np.array([[1., 100., 3., 1000.], [1., 5., 100., 1000.], [0., 1., 2., 3.]])
        choices, confidence = ensemble.choose_from_group_probabilities(probability, groups, scores)
        self.assertEqual(choices, [2, 0, 2])
        self.assertTrue(all(type(i) is int for i in choices))
        np.testing.assert_array_equal(confidence, [.8, 0., .5])
        with self.assertRaises(ValueError):
            ensemble.choose_from_group_probabilities(np.ones((1, 4)) / 4, np.array([[0, 1, 2, -1]]), scores[:1])

    def test_member_alpha_tau_and_candidate_permutation(self):
        groups = np.array([[0, 0, 1, 2]])
        first = np.array([[1., .9, 1.4, -.7]])
        second = np.array([[.2, .3, 1.2, -.9]])
        result = ensemble.equal_probability_mixture(first, second, groups, {'alpha': 1., 'tau': .7}, {'alpha': .5, 'tau': 1.2})
        p = np.array([2, 0, 3, 1])
        changed = ensemble.equal_probability_mixture(first[:, p], second[:, p], groups[:, p], {'alpha': 1., 'tau': .7}, {'alpha': .5, 'tau': 1.2})
        self.assertEqual([int(p[j]) for j in changed['choices']], result['choices'])
        np.testing.assert_allclose(changed['probability'], result['probability'], atol=1e-15)

    def test_failed_gate_persists_before_any_probability_mixture(self):
        n = 30
        groups = np.tile(np.array([0, 1, 2, 3]), (n, 1))
        labels = np.tile(np.array([1., 0., 0., 0.]), (n, 1))
        fold = np.repeat(np.arange(3), 10)

        def member(name):
            return {'experiment_id': name, 'groups': groups, 'labels': labels, 'fold': fold,
                    'correct': np.ones(n, bool), 'input_sha256': {'oof.npz': name},
                    'registration': {'folds_sha256': 'same-folds', 'data_files': {'gsm8k_train.jsonl': 'train-hash'}},
                    'config': {'decoder': {'mode': 'group_lse', 'alpha': 0., 'tau': 1.}}}

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / 'result'
            with patch.object(ensemble, 'load_experiment', side_effect=lambda _, name: member(name)), patch.object(ensemble, 'equal_probability_mixture') as mixing:
                result = ensemble.evaluate_pair('first', 'second', output, root)
                self.assertFalse(result['ensemble_evaluated'])
                self.assertEqual(result['status'], 'not_run_complementarity_gate_failed')
                mixing.assert_not_called()
                self.assertFalse(json.loads((output / 'complementarity_gate.json').read_text())['passed'])
                self.assertTrue((root / 'ENSEMBLE_PAIR_LOCK.json').is_file())
                with self.assertRaises(FileExistsError):
                    ensemble.evaluate_pair('first', 'second', output, root)
                with self.assertRaises(ValueError):
                    ensemble.evaluate_pair('first', 'third', root / 'different-pair', root)
                self.assertFalse((root / 'different-pair').exists())

    def test_passed_gate_records_actual_synthetic_mixture_and_resources(self):
        n = 60
        groups = np.tile(np.array([0, 1, 2, 3]), (n, 1))
        labels = np.tile(np.array([1., 0., 0., 0.]), (n, 1))
        fold = np.repeat(np.arange(3), 20)

        def member(name):
            correct = np.zeros(n, bool)
            correct[:30] = True
            if name == 'second':
                correct[:] = False
                correct[15:45] = True
            scores = np.full((n, 4), -10.)
            scores[np.arange(n), np.where(correct, 0, 1)] = 2.
            return {'experiment_id': name, 'groups': groups, 'labels': labels, 'fold': fold,
                    'scores': scores, 'choices': np.where(correct, 0, 1), 'correct': correct,
                    'input_sha256': {'oof.npz': name},
                    'registration': {'folds_sha256': 'same-folds', 'data_files': {'gsm8k_train.jsonl': 'train-hash'}},
                    'config': {'decoder': {'mode': 'group_lse', 'alpha': 0., 'tau': 1.}},
                    'summary': {'training_seconds': 12., 'folds': [
                        {'fold': i, 'checkpoint': 'synthetic-only', 'numeric_asset_bytes': 40, 'parameters': 4}
                        for i in range(3)]}}

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / 'result'
            original_mix = ensemble.equal_probability_mixture

            def checked_mixture(*args):
                self.assertTrue(json.loads((output / 'complementarity_gate.json').read_text())['passed'])
                return original_mix(*args)

            with patch.object(ensemble, 'load_experiment', side_effect=lambda _, name: member(name)), \
                    patch.object(ensemble, 'audit', return_value={'uncompressed_bytes': 100}), \
                    patch.object(ensemble, 'equal_probability_mixture', side_effect=checked_mixture) as mixing:
                result = ensemble.evaluate_pair('first', 'second', output, root)
            self.assertTrue(result['ensemble_evaluated'])
            self.assertEqual(result['status'], 'complete')
            self.assertEqual(mixing.call_count, 1)
            self.assertEqual(result['additional_training_seconds'], 0.)
            self.assertEqual(result['existing_training_seconds_reused'], 24.)
            self.assertEqual(result['assets_per_fold'][0]['member_audited_uncompressed_bytes_sum'], 200)
            self.assertEqual(len(result['paired_comparisons']), 2)
            with np.load(output / 'oof-mixture.npz', allow_pickle=False) as stored:
                np.testing.assert_allclose(stored['probability'], (stored['first_probability']+stored['second_probability'])/2)
                self.assertEqual(int(stored['correct'].sum()), result['metrics']['overall']['correct'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
