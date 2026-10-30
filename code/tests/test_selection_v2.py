"""Synthetic behavior and independent finite-difference checks; no bank files."""
import importlib.util
import itertools
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mp2.selection_v2 import (answer_groups, group_logits, group_probabilities,
                              score_loss_gradient, select_candidates,
                              selection_details, torch_loss)


def bank(texts):
    return [{'id': f'synthetic-{i}', 'question': 'synthetic only',
             'candidates': [{'index': j, 'text': text} for j, text in enumerate(row)]}
            for i, row in enumerate(texts)]


class SelectionV2Tests(unittest.TestCase):
    def test_official_parser_equivalence_and_none(self):
        data = bank([['#### 1/2', '#### 0.50', r'\boxed{2/4}', 'no numeric answer'],
                     ['none', 'unparseable', 'no answer', 'also none']])
        groups = answer_groups(data)
        np.testing.assert_array_equal(groups, [[0, 0, 0, -1], [-1, -1, -1, -1]])
        scores = np.array([[1., 2., 3., 100.], [8., 9., 10., 11.]])
        for mode in ('candidate_max', 'group_max', 'group_lse'):
            detail = selection_details(scores, groups, mode)
            self.assertEqual(detail['chosen'], [2, 0])
            self.assertEqual(detail['all_unparseable_rows'], [1])
            np.testing.assert_array_equal(detail['chosen_group_probability'], [1., 0.])
            np.testing.assert_array_equal(detail['group_probability_margin'], [1., 0.])
            self.assertTrue(all(type(j) is int for j in detail['chosen']))
        self.assertEqual(select_candidates(scores, groups, mask=False), [3, 3])
        self.assertTrue(np.isfinite(group_probabilities(scores, groups)).all())

    def test_lse_formula_and_normalization(self):
        groups = np.array([[0, 0, 1, -1]])
        scores = np.array([[1., .9, 1.4, 100.]])
        self.assertEqual(select_candidates(scores, groups, 'candidate_max'), [2])
        self.assertEqual(select_candidates(scores, groups, 'group_lse'), [0])
        self.assertAlmostEqual(group_logits(scores, groups)[0, 0], np.logaddexp(1., .9))
        self.assertEqual(select_candidates(scores, groups, 'group_lse', alpha=1), [2])
        self.assertAlmostEqual(group_logits(scores, groups, alpha=.5)[0, 0], np.logaddexp(1., .9)-.5*np.log(2))
        details = selection_details(scores, groups, 'candidate_max')
        self.assertLess(details['chosen_group_probability'][0], .5)
        self.assertLess(details['group_probability_margin'][0], 0)

    def test_group_max_equivalence_including_ties(self):
        rng = np.random.default_rng(443)
        scores = rng.integers(-2, 3, size=(300, 4)).astype(float)
        groups = rng.integers(-1, 4, size=(300, 4))
        self.assertEqual(select_candidates(scores, groups, 'candidate_max'),
                         select_candidates(scores, groups, 'group_max'))
        # A group whose first member is earlier need not contain the first max.
        self.assertEqual(select_candidates([[0., 2., 2., 1.]], np.array([[0, 1, 0, 1]]), 'group_max'), [1])

    def test_permutation_question_order_ids_and_n(self):
        texts = [['#### 10', '#### 10', '#### 20', 'none'], ['#### 1', '#### 2', '#### 2', '#### 3']]
        data = bank(texts)
        scores = np.array([[1., .9, 1.4, 100.], [.1, .3, 1.3, -.2]])
        for mode in ('candidate_max', 'group_max', 'group_lse'):
            expected = select_candidates(scores, answer_groups(data), mode)
            for permutation in itertools.permutations(range(4)):
                p = np.asarray(permutation)
                changed = bank([[row[j] for j in p] for row in texts])
                actual = select_candidates(scores[:, p], answer_groups(changed), mode)
                self.assertEqual([int(p[j]) for j in actual], expected)
            for row in data:
                row['id'] = 'changed-id-is-ignored'
            self.assertEqual(select_candidates(scores[::-1], answer_groups(data[::-1]), mode), expected[::-1])
            self.assertEqual(select_candidates(scores[:1], answer_groups(data[:1]), mode), expected[:1])
            self.assertEqual(select_candidates(np.empty((0, 4)), answer_groups([]), mode), [])

    def test_one_to_four_positives_and_objective_equivalence(self):
        scores = np.array([[.1, -.2, 1.2, .5]] * 4)
        labels = np.array([[int(j < positives) for j in range(4)] for positives in range(1, 5)])
        groups = np.array([[0 if j < positives else j-positives+1 for j in range(4)] for positives in range(1, 5)])
        group_loss, group_grad = score_loss_gradient(scores, labels, groups, 'group')
        old_loss, old_grad = score_loss_gradient(scores, labels, groups, 'listwise')
        self.assertAlmostEqual(group_loss, old_loss, places=12)
        np.testing.assert_allclose(group_grad, old_grad, atol=1e-15)
        for alpha in (0., .5, 1.):
            loss, gradient = score_loss_gradient(scores[-1:], labels[-1:], groups[-1:], alpha=alpha)
            self.assertAlmostEqual(loss, 0., places=14)
            np.testing.assert_array_equal(gradient, np.zeros((1, 4)))

    def test_numpy_gradient_finite_differences_and_normalization(self):
        scores = np.array([[1.2, -.3, .1, .8], [.4, -.9, .7, -.4], [.1, .2, .3, .4]])
        groups = np.array([[0, 0, 1, -1], [0, 1, 2, 3], [0, 0, -1, 0]])
        labels = np.array([[1., 1., 0., 0.], [0., 1., 0., 0.], [1., 1., 0., 1.]])
        eps = 1e-6
        for objective, alpha, tau, reduction in itertools.product(('group', 'listwise', 'bce'), (0., .5, 1.), (1., .7), ('full', 'hard')):
            loss, gradient = score_loss_gradient(scores, labels, groups, objective, alpha, tau, reduction)
            numerical = np.empty_like(scores)
            for index in np.ndindex(scores.shape):
                plus, minus = scores.copy(), scores.copy()
                plus[index] += eps
                minus[index] -= eps
                numerical[index] = (score_loss_gradient(plus, labels, groups, objective, alpha, tau, reduction)[0]
                                    - score_loss_gradient(minus, labels, groups, objective, alpha, tau, reduction)[0])/(2*eps)
            np.testing.assert_allclose(gradient, numerical, atol=3e-9, rtol=1e-6)
            self.assertTrue(np.isfinite(loss))
        full, grad_full = score_loss_gradient(scores, labels, groups)
        hard, grad_hard = score_loss_gradient(scores, labels, groups, reduction='hard')
        self.assertAlmostEqual(hard, full * 3/2, places=12)
        np.testing.assert_allclose(grad_hard, grad_full * 3/2, atol=1e-15)

    def test_extreme_scores_and_input_guards(self):
        scores = np.array([[10000., -10000., -9999., 9999.]])
        groups = np.array([[0, 1, 1, -1]])
        labels = np.array([[0., 1., 1., 0.]])
        for objective in ('group', 'listwise', 'bce'):
            loss, grad = score_loss_gradient(scores, labels, groups, objective)
            self.assertTrue(np.isfinite(loss))
            self.assertTrue(np.isfinite(grad).all())
        with self.assertRaises(ValueError):
            score_loss_gradient(scores, [[1., 1., 0., 0.]], groups)
        with self.assertRaises(ValueError):
            select_candidates([[0., 1., 2., np.nan]], groups)
        with self.assertRaises(ValueError):
            select_candidates(scores, groups, tau=0)
        with self.assertRaises(ValueError):
            select_candidates(scores, groups, 'group_lse', mask=False)

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'PyTorch optional training dependency not installed')
    def test_torch_loss_and_autograd_match_independent_numpy(self):
        import torch
        torch.set_num_threads(1)
        scores = np.array([[1.2, -.3, .1, .8], [.4, -.9, .7, -.4], [.1, .2, .3, .4], [10000., -10000., -9999., 9999.]])
        groups = np.array([[0, 0, 1, -1], [0, 1, 2, 3], [0, 0, -1, 0], [0, 1, 1, -1]])
        labels = np.array([[1., 1., 0., 0.], [0., 1., 0., 0.], [1., 1., 0., 1.], [0., 1., 1., 0.]])
        for objective, alpha, reduction in itertools.product(('group', 'listwise', 'bce'), (0., .5, 1.), ('full', 'hard')):
            expected_loss, expected_gradient = score_loss_gradient(scores, labels, groups, objective, alpha, .7, reduction)
            tensor = torch.tensor(scores, dtype=torch.float64, requires_grad=True)
            actual_loss = torch_loss(tensor, torch.tensor(labels), torch.tensor(groups), objective, alpha, .7, reduction)
            actual_loss.backward()
            self.assertAlmostEqual(float(actual_loss.detach()), expected_loss, places=9)
            np.testing.assert_allclose(tensor.grad.numpy(), expected_gradient, atol=2e-10, rtol=1e-8)
        # No hard rows: connected zero with finite, exactly zero gradients.
        tensor = torch.zeros((2, 4), dtype=torch.float64, requires_grad=True)
        loss = torch_loss(tensor, torch.ones_like(tensor), torch.zeros((2, 4), dtype=torch.long), reduction='hard')
        loss.backward()
        self.assertEqual(float(loss.detach()), 0.)
        np.testing.assert_array_equal(tensor.grad.numpy(), np.zeros((2, 4)))


if __name__ == '__main__':
    unittest.main(verbosity=2)
