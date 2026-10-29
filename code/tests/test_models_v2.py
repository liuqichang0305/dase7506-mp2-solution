"""Synthetic architecture/export/contract tests; no official data is opened."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[variable] = '1'
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from mp2.features import build_features
from mp2.models_v2 import (build_model, export_state, load_bundle, numpy_forward,
                           parameter_count, save_bundle, score_with_bundle)
from mp2.selection_v2 import answer_groups, select_candidates
from solution_v2 import predict, predict_confidence

HAS_TORCH = importlib.util.find_spec('torch') is not None
DECODER = {'mode': 'group_lse', 'alpha': 0.5, 'tau': 1.0, 'mask': True}


def synthetic_bank(n):
    result = []
    for i in range(n):
        texts = ['reasoning #### 2', 'other reasoning #### 2.0', 'third #### 3', 'unparseable']
        if i % 3 == 1:
            texts = ['#### 1', '#### 2', '#### 3', '#### 4']
        if i % 3 == 2:
            texts = ['#### 5'] * 4
        result.append({'id': f'synthetic-{i}', 'question': 'Synthetic input',
                       'candidates': [{'text': text} for text in texts]})
    return result


class NumericBundleTests(unittest.TestCase):
    def test_linear_preserves_float64_and_rejects_invalid_bundle(self):
        rng = np.random.default_rng(1907)
        bank = synthetic_bank(7)
        embeddings = rng.normal(size=(7, 4, 1024))
        features = build_features(bank, embeddings, 'relation')
        mean = rng.normal(size=2064)
        scale = rng.uniform(0.1, 2.0, size=2064)
        state = {'weight': rng.normal(size=2064), 'bias': np.array(0.01234567890123456)}
        with tempfile.TemporaryDirectory() as path:
            save_bundle(path, {'family': 'linear'}, state, mean, scale, DECODER)
            expected = ((features - mean) / scale) @ state['weight'] + state['bias']
            np.testing.assert_array_equal(score_with_bundle(bank, embeddings, load_bundle(path)), expected)
            self.assertEqual(parameter_count({'family': 'linear'}), 2065)
            self.assertEqual(predict([], {'embeddings': np.empty((0, 4, 1024))}, path), [])
            with self.assertRaises(FileExistsError):
                save_bundle(path, {'family': 'linear'}, state, mean, scale, DECODER)
            config_path = Path(path) / 'config.json'
            config = json.loads(config_path.read_text())
            config['schema'] = 'mp2-linear-v1'
            config_path.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, 'schema'):
                load_bundle(path)

    def test_inference_import_has_no_torch_dependency(self):
        code = ('import sys; sys.path.insert(0, ' + repr(str(Path(__file__).resolve().parents[1]))
                + '); import solution_v2; assert "torch" not in sys.modules')
        result = subprocess.run([sys.executable, '-c', code], text=True, capture_output=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)


@unittest.skipUnless(HAS_TORCH, 'Torch training dependency is not installed')
class NeuralArchitectureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(1)
        cls.torch = torch

    def test_numpy_equivalence_parameter_count_and_permutation(self):
        rng = np.random.default_rng(991)
        x = rng.normal(size=(9, 4, 2064)).astype(np.float32)
        groups = answer_groups(synthetic_bank(len(x)))
        permutation = np.array([2, 0, 3, 1])
        configs = [{'family': family} for family in ('mlp', 'resmlp', 'deepsets', 'attention')]
        configs += [{'family': 'deepsets', 'context': False}, {'family': 'resmlp', 'blocks': 2}]
        for cfg in configs:
            with self.subTest(config=cfg):
                self.torch.manual_seed(590)
                model = build_model(cfg).eval()
                with self.torch.no_grad():
                    expected = model(self.torch.from_numpy(x)).numpy()
                state = export_state(model)
                self.assertEqual(parameter_count(cfg), sum(value.numel() for value in model.parameters()))
                actual = numpy_forward(x, cfg, state)
                np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=3e-6)
                for mode in ('candidate_max', 'group_max', 'group_lse'):
                    self.assertEqual(select_candidates(actual, groups, mode=mode),
                                     select_candidates(expected, groups, mode=mode))
                changed = numpy_forward(x[:, permutation], cfg, state)
                np.testing.assert_allclose(changed, actual[:, permutation], rtol=3e-5, atol=3e-6)
                reordered = numpy_forward(x[::-1], cfg, state)
                np.testing.assert_allclose(reordered, actual[::-1], rtol=3e-5, atol=3e-6)
                self.assertEqual(numpy_forward(x[:0], cfg, state).shape, (0, 4))
                for name, value in state.items():
                    if name == 'head.weight':
                        value *= np.float32(1e-5)
                    if name == 'head.bias':
                        value.fill(0)
                model.load_state_dict({name: self.torch.from_numpy(value) for name, value in state.items()})
                with self.torch.no_grad():
                    tiny_expected = model(self.torch.from_numpy(x)).numpy()
                tiny_actual = numpy_forward(x, cfg, state)
                np.testing.assert_allclose(tiny_actual, tiny_expected, rtol=3e-5, atol=3e-11)
                self.assertEqual(tiny_actual.argmax(axis=1).tolist(), tiny_expected.argmax(axis=1).tolist())

    def test_end_to_end_export_ids_batch_permutation_and_alignment(self):
        rng = np.random.default_rng(519)
        bank = synthetic_bank(11)
        embeddings = rng.normal(size=(len(bank), 4, 1024)).astype(np.float32)
        features = build_features(bank, embeddings, 'relation')
        mean = features.mean(axis=(0, 1))
        scale = features.std(axis=(0, 1))
        scale[scale < 1e-8] = 1
        for family in ('mlp', 'resmlp', 'deepsets', 'attention'):
            with self.subTest(family=family), tempfile.TemporaryDirectory() as path:
                self.torch.manual_seed(610)
                cfg = {'family': family}
                model = build_model(cfg).eval()
                save_bundle(path, cfg, export_state(model), mean, scale, DECODER)
                baseline = predict(bank, {'embeddings': embeddings}, path)
                self.assertTrue(all(type(value) is int and 0 <= value <= 3 for value in baseline))
                for n in (0, 1, 3, 7, len(bank)):
                    self.assertEqual(predict(bank[:n], {'embeddings': embeddings[:n]}, path), baseline[:n])
                renamed = copy.deepcopy(bank)
                for i, row in enumerate(renamed):
                    row['id'] = str(983123 - i)
                self.assertEqual(predict(renamed, {'embeddings': embeddings}, path), baseline)
                permutation = np.array([2, 0, 3, 1])
                changed = copy.deepcopy(bank)
                for row in changed:
                    row['candidates'] = [row['candidates'][j] for j in permutation]
                selected = predict(changed, {'embeddings': embeddings[:, permutation]}, path)
                self.assertEqual(permutation[selected].tolist(), baseline)
                self.assertEqual(predict(bank[::-1], {'embeddings': embeddings[::-1]}, path), baseline[::-1])
                probability = predict_confidence(bank, {'embeddings': embeddings}, path)
                self.assertEqual(probability.shape, (len(bank),))
                self.assertTrue(np.isfinite(probability).all())
                self.assertTrue(((probability >= 0) & (probability <= 1)).all())
                with self.assertRaises(ValueError):
                    predict(bank, {'embeddings': embeddings[:-1]}, path)
                bad = embeddings.copy()
                bad[0, 0, 0] = np.nan
                with self.assertRaises(ValueError):
                    predict(bank, {'embeddings': bad}, path)


if __name__ == '__main__':
    unittest.main()
