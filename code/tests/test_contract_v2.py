"""The contract audit must attribute ties to group decisions, not raw maxima."""
import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.check_contract_v2 import admissible_tie_choices


class GroupDecisionAuditTests(unittest.TestCase):
    def test_group_winner_can_differ_from_raw_maximum(self):
        result = admissible_tie_choices(np.array([[1.0, .9, 1.4, -2]]), np.array([[0, 0, 1, -1]]), {'mode': 'group_lse'})
        self.assertEqual(result, [{0}])

    def test_group_and_within_group_ties(self):
        result = admissible_tie_choices(np.zeros((1, 4)), np.array([[0, 0, 1, 1]]), {'mode': 'group_lse'})
        self.assertEqual(result, [{0, 1, 2, 3}])

    def test_invalid_high_score_does_not_create_tie(self):
        result = admissible_tie_choices(np.array([[9., 1., 0., -1.]]), np.array([[-1, 0, 1, 2]]), {'mode': 'candidate_max'})
        self.assertEqual(result, [{1}])

    def test_all_none_has_documented_positional_exception(self):
        result = admissible_tie_choices(np.arange(4)[None], np.full((1, 4), -1), {'mode': 'group_lse'})
        self.assertEqual(result, [{0, 1, 2, 3}])


if __name__ == '__main__':
    unittest.main()
