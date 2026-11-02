"""Synthetic objective and probability-collapse tests; no official data IO."""
import os
from pathlib import Path
import sys
import unittest

for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from mp2.research_v3.selection import (COLLAPSED_DECODER, collapse_scores,
                                      member_loss, member_probabilities, select_raw)
from mp2.selection_v2 import group_probabilities, select_candidates, torch_loss

torch.set_num_threads(1)


def pairwise_reference(scores, labels, groups, reduction):
    values, keep = [], []
    for row, label, group in zip(scores, labels, groups):
        positive = np.flatnonzero((label > 0) & (group >= 0))
        negative = np.flatnonzero((label == 0) & (group >= 0))
        pairs = [np.logaddexp(0., row[j] - row[i]) for i in positive for j in negative]
        values.append(float(np.mean(pairs)) if pairs else 0.)
        keep.append(reduction == 'full' or len(set(group[group >= 0])) >= 2)
    values, keep = np.asarray(values), np.asarray(keep, dtype=bool)
    return float(values[keep].mean()) if keep.any() else 0.


class ObjectiveTests(unittest.TestCase):
    def test_each_member_loss_is_averaged_not_mean_logits(self):
        groups = torch.tensor([[0, 1, 2, 3], [0, 0, 1, -1]])
        labels = torch.tensor([[1., 0., 0., 0.], [1., 1., 0., 0.]], dtype=torch.float64)
        raw = torch.tensor([[[8., -2., 0., 1.], [-4., 5., 0., 1.]],
                            [[4., 2., -2., 9.], [-3., -2., 7., 9.]]], dtype=torch.float64, requires_grad=True)
        for objective in ('group', 'listwise', 'bce', 'pairwise'):
            with self.subTest(objective=objective):
                actual = member_loss(raw, labels, groups, objective=objective, reduction='full')
                expected = torch.stack([member_loss(raw[:, j], labels, groups,
                    objective=objective, reduction='full') for j in range(raw.shape[1])]).mean()
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                incorrect = member_loss(raw.mean(1), labels, groups, objective=objective, reduction='full')
                self.assertGreater(abs(float((actual - incorrect).detach())), .01)
                gradient = torch.autograd.grad(actual, raw, retain_graph=True)[0]
                self.assertTrue(torch.isfinite(gradient).all())
                self.assertTrue(all(float(gradient[:, j].abs().sum()) > 0 for j in range(raw.shape[1])))

    def test_pairwise_question_average_and_finite_difference(self):
        scores = np.array([[.3, -.4, 1.2, -.8], [1.3, -2., .6, 8.], [2., 1., 0., -.5]])
        labels = np.array([[1., 0., 0., 0.], [1., 1., 0., 0.], [1., 1., 1., 1.]])
        groups = np.array([[0, 1, 2, 3], [0, 0, 1, -1], [0, 0, 0, 0]])
        for reduction in ('full', 'hard'):
            with self.subTest(reduction=reduction):
                tensor = torch.tensor(scores, dtype=torch.float64, requires_grad=True)
                actual = member_loss(tensor, torch.tensor(labels), torch.tensor(groups),
                                     objective='pairwise', reduction=reduction)
                self.assertAlmostEqual(float(actual.detach()), pairwise_reference(scores, labels, groups, reduction), places=14)
                gradient = torch.autograd.grad(actual, tensor)[0].numpy()
                numeric = np.empty_like(scores)
                for index in np.ndindex(scores.shape):
                    plus, minus = scores.copy(), scores.copy()
                    plus[index] += 1e-6
                    minus[index] -= 1e-6
                    numeric[index] = (pairwise_reference(plus, labels, groups, reduction)
                                      - pairwise_reference(minus, labels, groups, reduction)) / 2e-6
                np.testing.assert_allclose(gradient, numeric, rtol=2e-6, atol=1e-9)
                self.assertEqual(gradient[1, 3], 0.)  # Invalid candidate is excluded.
                np.testing.assert_array_equal(gradient[2], np.zeros(4))

    def test_all_positive_empty_and_malformed_positive(self):
        for reduction in ('full', 'hard'):
            for n in (0, 3):
                scores = torch.randn(n, 4, dtype=torch.float64, requires_grad=True)
                loss = member_loss(scores, torch.ones_like(scores), torch.zeros(n, 4, dtype=torch.long),
                                   objective='pairwise', reduction=reduction)
                self.assertEqual(float(loss.detach()), 0.)
                loss.backward()
                torch.testing.assert_close(scores.grad, torch.zeros_like(scores), rtol=0, atol=0)
        with self.assertRaises(ValueError):
            member_loss(torch.zeros(1, 4), torch.tensor([[1., 0., 0., 0.]]),
                        torch.full((1, 4), -1), objective='pairwise')


class ProbabilityTests(unittest.TestCase):
    def test_collapse_probabilities_and_permutation(self):
        rng = np.random.default_rng(3036901991)
        groups = np.array([[0, 1, 2, 3], [0, 0, 1, -1], [0, 1, 0, 1],
                           [-1, -1, -1, -1], [-1, 0, 0, 1], [0, 0, 0, 0]])
        raw = rng.normal(size=(len(groups), 4, 4))
        permutation = np.array([2, 0, 3, 1])
        for alpha in (0., .5, 1.):
            for tau in (.7, 1., 2.):
                decoder = dict(COLLAPSED_DECODER, alpha=alpha, tau=tau)
                with self.subTest(alpha=alpha, tau=tau):
                    expected = np.mean([group_probabilities(raw[:, j], groups, alpha, tau) for j in range(4)], axis=0)
                    actual = member_probabilities(raw, groups, decoder)
                    np.testing.assert_array_equal(actual, expected)
                    collapsed = collapse_scores(raw, groups, decoder)
                    self.assertTrue(np.isfinite(collapsed).all())
                    np.testing.assert_allclose(group_probabilities(collapsed, groups), expected, rtol=1e-14, atol=1e-15)
                    chosen = select_raw(raw, groups, decoder)
                    self.assertEqual(chosen, select_candidates(collapsed, groups, **COLLAPSED_DECODER))
                    moved = select_raw(raw[..., permutation], groups[:, permutation], decoder)
                    mapped = permutation[moved]
                    for i in range(len(groups)):
                        if (groups[i] >= 0).any():
                            self.assertEqual(int(mapped[i]), chosen[i])
                    self.assertEqual(chosen[3], 0)
                    self.assertTrue(all(type(x) is int for x in chosen))
                    self.assertEqual(select_raw(raw[:0], groups[:0], decoder), [])
                    self.assertEqual(collapse_scores(raw[:0], groups[:0], decoder).shape, (0, 4))

    def test_exact_ties_group_first_position_and_member_zero_representative(self):
        groups = np.array([[0, 0, 1, 1], [0, 1, 0, 1], [0, 0, 0, 0], [-1, -1, -1, -1]])
        raw = np.array([[[0., 2., 0., 2.], [2., 0., 2., 0.]],
                        [[0., 2., 2., 0.], [2., 0., 0., 2.]],
                        [[2., 2., 2., 2.], [9., 8., 7., 6.]],
                        [[3., 2., 1., 0.], [0., 1., 2., 3.]]])
        expected = [1, 2, 0, 0]
        self.assertEqual(select_raw(raw, groups, COLLAPSED_DECODER), expected)
        self.assertEqual(select_candidates(collapse_scores(raw, groups, COLLAPSED_DECODER), groups,
                                          **COLLAPSED_DECODER), expected)
        probability = member_probabilities(raw, groups, COLLAPSED_DECODER)
        np.testing.assert_array_equal(probability[0, :2], [.5, .5])
        np.testing.assert_array_equal(probability[1, :2], [.5, .5])
        np.testing.assert_array_equal(probability[-1], np.zeros(4))


if __name__ == '__main__':
    result = unittest.TextTestRunner(stream=sys.stdout, verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__]))
    raise SystemExit(0 if result.wasSuccessful() else 1)
