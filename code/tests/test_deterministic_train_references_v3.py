"""No course data: deterministic majority cross-check on generated parseable text."""
import itertools
from pathlib import Path
import sys
import unittest
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code/tools'))
from deterministic_train_references_v3 import majority_from_groups, majority_choices, answer_groups


class DeterministicReferenceTests(unittest.TestCase):
    def test_all_625_synthetic_parse_vote_patterns_and_ties(self):
        patterns = list(itertools.product((None, 1, 2, 3, 4), repeat=4))
        bank = [{'candidates': [{'text': 'unparseable' if value is None else f'#### {value}'} for value in row]} for row in patterns]
        np.testing.assert_array_equal(majority_from_groups(answer_groups(bank)), majority_choices(bank))

    def test_none_exclusion_first_occurrence_and_all_none(self):
        groups = np.array([[-1, 0, 1, -1], [0, 1, 1, 0], [-1, -1, -1, -1], [0, 1, 1, 1]])
        np.testing.assert_array_equal(majority_from_groups(groups), [1, 0, 0, 1])

    def test_reject_invalid_shape_and_group_values(self):
        for value in (np.zeros((3, 3), dtype=int), np.zeros((3, 4), dtype=float), np.full((1, 4), -2)):
            with self.assertRaises(ValueError): majority_from_groups(value)


if __name__ == '__main__':
    unittest.main(verbosity=2)
