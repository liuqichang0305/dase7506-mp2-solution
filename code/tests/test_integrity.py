"""Small boundary checks: parser semantics, safe assets, and test freeze gate."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE / 'tools'))
from common import parse_answer, validate_choices
from audit_assets import audit
from evaluate import verify_freeze


class IntegrityTests(unittest.TestCase):
    def test_official_numeric_equivalence(self):
        self.assertEqual(parse_answer('#### 1/2'), parse_answer('#### 0.5'))
        self.assertEqual(parse_answer('#### 1,000'), '1000')
        self.assertIsNone(parse_answer('No numeric final marker'))
        self.assertIsNone(parse_answer('#### 1/0'))

    def test_python_int_contract(self):
        bank = [{'id': 'a'}]
        validate_choices(bank, ['a'], [0])
        for invalid in [True, np.int64(0), -1, 4]:
            with self.assertRaises(ValueError):
                validate_choices(bank, ['a'], [invalid])

    def test_compression_does_not_reduce_counted_assets(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            values = np.zeros(100000, dtype=np.float64)
            np.savez_compressed(root / 'weights.npz', weight=values)
            (root / 'config.json').write_text('{}', encoding='utf-8')
            result = audit(root)
            self.assertGreaterEqual(result['uncompressed_bytes'], values.nbytes + 2)
            row = next(x for x in result['files'] if x['path'] == 'weights.npz')
            self.assertGreater(row['uncompressed_bytes'], row['disk_bytes'])
            self.assertEqual(row['sha256'], hashlib.sha256((root / row['path']).read_bytes()).hexdigest())

    def test_test_gate_rejects_unfrozen_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            freeze = root / 'freeze.json'
            freeze.write_text(json.dumps({'schema': 'dase7506-mp2-freeze-v1', 'frozen': False}), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'frozen=true'):
                verify_freeze(root, freeze, root / 'code/solution.py', root / 'bundle')


if __name__ == '__main__':
    unittest.main()
