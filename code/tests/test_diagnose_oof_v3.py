"""Synthetic checks of descriptive slices and the pre-fusion evidence gate."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import numpy as np
from diagnose_oof_v3 import complementarity, evaluate_registered_fusion, fixed_slices, method_path


def record(name, event):
    return {'key': name, 'correct': np.asarray(event, dtype=bool)}


class DiagnosticTests(unittest.TestCase):
    def test_pair_gate_integer_boundary_and_reciprocal_requirement(self):
        n = 1000
        left, right = np.zeros(n, bool), np.zeros(n, bool)
        left[:500] = True
        right[:490] = True
        right[500:510] = True
        rows = [record('left', left), record('right', right)]
        result = complementarity(rows, np.ones(n, bool))
        self.assertTrue(result['gate_passed'])
        self.assertEqual(result['overall']['member_unique_correct'], [10, 10])
        self.assertEqual(result['hard']['oracle_gain_over_strongest_percentage_points'], 1.)
        right[509] = False
        self.assertFalse(complementarity([rows[0], record('right', right)], np.ones(n, bool))['gate_passed'])
        self.assertFalse(complementarity(rows, np.zeros(n, bool))['gate_passed'])

    def test_three_member_requires_unique_increment_from_each(self):
        events = np.zeros((3, 1000), bool)
        events[:, :700] = True
        for i in range(3):
            events[i, 700 + 10 * i:710 + 10 * i] = True
        result = complementarity([record(str(i), event) for i, event in enumerate(events)], np.ones(1000, bool))
        self.assertTrue(result['gate_passed'])
        self.assertEqual(result['overall']['member_unique_correct'], [10, 10, 10])
        self.assertEqual(result['hard']['oracle_gain_over_strongest_percentage_points'], 2.)
        events[2] = events[0]
        self.assertFalse(complementarity([record(str(i), event) for i, event in enumerate(events)], np.ones(1000, bool))['gate_passed'])

    def test_fixed_slices_partition_all_questions_without_mutation(self):
        bank = [{'question': 'word ' * length, 'candidates': [{'text': 'x ' * (length + i)} for i in range(4)]}
                for length in (0, 39, 40, 79, 80, 119, 120, 200)]
        groups = np.tile([0, 0, 1, -1], (len(bank), 1))
        labels = np.tile([1., 1., 0., 0.], (len(bank), 1))
        slices, hard, definitions = fixed_slices(bank, labels, groups)
        self.assertTrue(hard.all())
        for dimension in set(row[0] for row in slices):
            masks = np.stack([mask for name, _, mask in slices if name == dimension])
            np.testing.assert_array_equal(masks.sum(0), np.ones(len(bank)))
        self.assertEqual(definitions['question_words'], [0, 40, 80, 120, None])
        selected = {bucket: int(mask.sum()) for dimension, bucket, mask in slices if dimension == 'question_words'}
        self.assertEqual(selected, {'[0,40)': 2, '[40,80)': 2, '[80,120)': 2, '[120,inf)': 2})

    def test_unregistered_fusion_and_traversal_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'registration.json'
            path.write_text(json.dumps({'methods': ['x', 'y']}))
            with self.assertRaisesRegex(ValueError, 'separate explicit'):
                evaluate_registered_fusion(path, {}, np.zeros(2, bool), Path(temporary))
        for key in ('v3:../other', 'v2:a/b', 'v3:..', 'test:foo', 'v3:'):
            with self.assertRaises(ValueError):
                method_path(key)


if __name__ == '__main__':
    result = unittest.TextTestRunner(stream=sys.stdout, verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__]))
    raise SystemExit(0 if result.wasSuccessful() else 1)
