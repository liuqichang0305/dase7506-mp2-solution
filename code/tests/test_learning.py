"""Numerical and split-boundary tests; synthetic examples only."""
import os
for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[variable] = '1'
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mp2.training import load_development, loss_gradient


class LearningTests(unittest.TestCase):
    def test_development_rejects_test_before_any_loading(self):
        with self.assertRaisesRegex(ValueError, 'only train or validation'):
            load_development('test')

    def test_listwise_preserves_all_positive_group(self):
        x = np.arange(8, dtype=float).reshape(1, 4, 2)
        loss, grad = loss_gradient(np.array([.2, -.4, .8]), x, np.ones((1, 4)), 'listwise', 0)
        self.assertAlmostEqual(loss, 0, places=12)
        np.testing.assert_allclose(grad, 0, atol=1e-12)

    def test_listwise_gradient_with_multiple_positives(self):
        x = np.arange(16, dtype=float).reshape(2, 4, 2) / 10
        y = np.array([[1, 0, 1, 0], [0, 1, 1, 1]], dtype=float)
        theta = np.array([.3, -.1, .2])
        _, analytic = loss_gradient(theta, x, y, 'listwise', .03)
        numeric = []
        for j in range(len(theta)):
            offset = np.zeros_like(theta)
            offset[j] = 1e-5
            left, _ = loss_gradient(theta - offset, x, y, 'listwise', .03)
            right, _ = loss_gradient(theta + offset, x, y, 'listwise', .03)
            numeric.append((right - left) / 2e-5)
        np.testing.assert_allclose(analytic, numeric, rtol=1e-5, atol=1e-7)

    def test_listwise_extreme_scores_remain_finite(self):
        x = np.array([[[1000.], [-1000.], [-1000.], [-1000.]]])
        y = np.array([[0., 1., 0., 0.]])
        loss, gradient = loss_gradient(np.array([1., 0.]), x, y, 'listwise', 0)
        self.assertTrue(np.isfinite(loss))
        self.assertTrue(np.isfinite(gradient).all())
        self.assertAlmostEqual(loss, 2000., places=8)


if __name__ == '__main__':
    unittest.main()
