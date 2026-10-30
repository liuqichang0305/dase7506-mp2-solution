"""Synthetic V3 preprocessing, architecture and export tests. No official data IO."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

for variable in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[variable] = '1'
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from mp2.features import build_features
from mp2.research_v3.models import (build_model, export_state, load_bundle, numpy_forward,
                                    parameter_count, save_bundle, score_with_bundle,
                                    validate_state)
from mp2.research_v3.preprocess import fit_preprocessor, transform_features, validate_preprocessor
from mp2.selection_v2 import answer_groups, group_probabilities, select_candidates

torch.set_num_threads(1)
DECODER = {'mode': 'group_lse', 'alpha': 0., 'tau': 1., 'mask': True}


def bank(n):
    return [{'id': f'synthetic-{i}', 'question': 'Synthetic input', 'candidates': [
        {'text': text} for text in ('reason #### 2', 'other #### 2.0', 'third #### 3', 'unparseable')
    ]} for i in range(n)]


def nonzero_model(config):
    torch.manual_seed(590)
    model = build_model(config).eval()
    if config['family'] == 'realmlp':
        # The specified zero output initialization must not make parity vacuous.
        with torch.no_grad():
            model.head.weight.normal_(0, .4)
            model.head.bias.fill_(.2)
    return model


class PreprocessingTests(unittest.TestCase):
    def test_standard_matches_legacy_and_batch_isolation(self):
        values = np.random.default_rng(73).normal(size=(13, 4, 2064))
        values[..., 0] = 8.
        state = fit_preprocessor(values[:9])
        flat = values[:9].reshape(-1, 2064)
        scale = flat.std(0)
        scale[scale < 1e-8] = 1.
        expected = ((values - flat.mean(0)) / scale).astype(np.float32)
        np.testing.assert_array_equal(transform_features(values, state), expected)
        before = {key: value.copy() if isinstance(value, np.ndarray) else value for key, value in state.items()}
        modified = values.copy()
        modified[9:] += 100000.
        np.testing.assert_array_equal(transform_features(modified[:9], state), expected[:9])
        for key in ('mean', 'scale', 'clip_mask', 'constant_mask'):
            np.testing.assert_array_equal(state[key], before[key])

    def test_robust_formula_constant_and_range_fallback(self):
        values = np.random.default_rng(74).normal(size=(5, 4, 2064))
        values[..., -1] = 7.
        values[..., -2] = 0.
        values[0, 0, -2] = 10.  # IQR=0, half-range fallback=5.
        for mode in ('robust_stats', 'robust_all'):
            with self.subTest(mode=mode):
                state = fit_preprocessor(values, mode)
                self.assertEqual(state['scale'][-2], 5.)
                self.assertTrue(state['constant_mask'][-1])
                check = values.copy()
                check[..., -1] = 9000.  # Author rule maps training-constant dimensions to zero.
                transformed = transform_features(check, state, dtype=np.float64)
                self.assertTrue((transformed[..., -1] == 0).all())
                mask = state['clip_mask']
                z = (check - state['mean']) / state['scale']
                z[..., -1] = 0.
                expected = z.copy()
                expected[..., mask] = z[..., mask] / np.sqrt(1 + (z[..., mask] / 3) ** 2)
                np.testing.assert_allclose(transformed, expected, rtol=1e-14, atol=1e-14)
                self.assertTrue((abs(transformed[..., mask]) <= 3).all())
                np.testing.assert_array_equal(transform_features(check[:2], state),
                                              transform_features(check, state)[:2])
        invalid = fit_preprocessor(values, 'robust_stats')
        invalid['clip_mask'][0] = True
        with self.assertRaises(ValueError):
            validate_preprocessor(invalid)
        with self.assertRaises(ValueError):
            fit_preprocessor(values[:0])


class ArchitectureTests(unittest.TestCase):
    def test_export_parity_permutations_and_feature_subsets(self):
        x = np.random.default_rng(75).normal(size=(9, 4, 2064)).astype(np.float32)
        groups = answer_groups(bank(len(x)))
        permutation = np.array([2, 0, 3, 1])
        configs = [{'family': family, 'input_features': subset}
                   for family in ('tabm', 'realmlp') for subset in ('relation', 'embedding_stats', 'stats')]
        configs += [{'family': 'tabm', 'k': 3, 'width': 17, 'depth': 2, 'start_scaling_init': 'normal'},
                    {'family': 'realmlp', 'activation': 'mish', 'width': 31}]
        for config in configs:
            with self.subTest(config=config):
                model = nonzero_model(config)
                state = export_state(model)
                validate_state(config, state)
                self.assertEqual(parameter_count(config), sum(x.numel() for x in model.parameters()))
                with torch.no_grad():
                    expected = model(torch.from_numpy(x)).numpy()
                actual = numpy_forward(x, config, state)
                np.testing.assert_allclose(actual, expected, rtol=5e-5, atol=2e-6)
                print('parity', json.dumps(config, sort_keys=True), 'max_abs', float(abs(actual - expected).max()))
                np.testing.assert_allclose(numpy_forward(x[:, permutation], config, state),
                                          actual[..., permutation], rtol=5e-5, atol=2e-6)
                np.testing.assert_allclose(numpy_forward(x[::-1], config, state), actual[::-1],
                                          rtol=5e-5, atol=2e-6)
                self.assertEqual(numpy_forward(x[:0], config, state).shape, (0,) + actual.shape[1:])
                if actual.ndim == 3:
                    a = np.mean([group_probabilities(actual[:, j], groups) for j in range(actual.shape[1])], 0)
                    e = np.mean([group_probabilities(expected[:, j], groups) for j in range(expected.shape[1])], 0)
                    np.testing.assert_allclose(a, e, rtol=5e-5, atol=2e-6)
                    self.assertEqual(a.argmax(1).tolist(), e.argmax(1).tolist())
                else:
                    self.assertEqual(select_candidates(actual, groups, **DECODER),
                                     select_candidates(expected, groups, **DECODER))

    def test_tabm_initialization_independent_losses_and_realmlp_input_gradients(self):
        x = torch.from_numpy(np.random.default_rng(76).normal(size=(5, 4, 2064)).astype(np.float32))
        torch.manual_seed(77)
        tabm = build_model({'family': 'tabm', 'dropout': 0})
        self.assertTrue(set(tabm.blocks[0].r.detach().numpy().ravel()) == {-1., 1.})
        self.assertFalse(torch.equal(tabm.blocks[0].r[0], tabm.blocks[0].r[1]))
        self.assertTrue(torch.equal(tabm.blocks[1].r, torch.ones_like(tabm.blocks[1].r)))
        self.assertTrue(torch.equal(tabm.blocks[0].s, torch.ones_like(tabm.blocks[0].s)))
        scores = tabm(x)
        self.assertGreater(float((scores[:, 0] - scores[:, 1]).detach().abs().max()), 1e-5)
        positive = torch.tensor([[1, 1, 0, 0]] * len(x), dtype=torch.bool)
        losses = [(-torch.logsumexp(torch.log_softmax(scores[:, member], -1).masked_fill(~positive, -torch.inf), -1)).mean()
                  for member in range(4)]
        loss = torch.stack(losses).mean()
        loss.backward()
        for member in range(4):
            self.assertGreater(float(tabm.head.weight.grad[member].abs().sum()), 0.)
            self.assertGreater(float(tabm.blocks[0].r.grad[member].abs().sum()), 0.)
        real = build_model({'family': 'realmlp'})
        self.assertTrue(torch.equal(real(x), torch.zeros(5, 4)))
        optimizer = torch.optim.Adam(real.parameters(), lr=.01)
        for _ in range(2):
            optimizer.zero_grad()
            torch.nn.functional.cross_entropy(real(x), torch.zeros(5, dtype=torch.long)).backward()
            optimizer.step()
        self.assertGreater(float(real.input_scale.grad.abs().sum()), 0.)
        self.assertTrue(torch.isfinite(real(x)).all())


class BundleTests(unittest.TestCase):
    def test_roundtrip_alignment_and_no_overwrite(self):
        data = bank(137)  # Includes the second inference chunk.
        embeddings = np.random.default_rng(78).normal(size=(len(data), 4, 1024))
        features = build_features(data, embeddings, 'relation')
        for family in ('tabm', 'realmlp', 'deepsets'):
            with self.subTest(family=family), tempfile.TemporaryDirectory() as folder:
                config = {'family': family}
                model = nonzero_model(config)
                prep = fit_preprocessor(features[:53], 'robust_stats')
                state = export_state(model)
                save_bundle(folder, config, state, prep['mean'], prep['scale'], DECODER, preprocess=prep)
                bundle = load_bundle(folder)
                actual = score_with_bundle(data, embeddings, bundle)
                expected = numpy_forward(transform_features(features, prep), config, state)
                np.testing.assert_allclose(actual, expected, rtol=5e-5, atol=2e-6)
                for n in (0, 1, 3, 137):
                    np.testing.assert_allclose(score_with_bundle(data[:n], embeddings[:n], bundle), actual[:n],
                                              rtol=5e-5, atol=2e-6)
                changed = copy.deepcopy(data)
                for i, row in enumerate(changed):
                    row['id'] = f'arbitrary-{137 - i}'
                np.testing.assert_array_equal(score_with_bundle(changed, embeddings, bundle), actual)
                with self.assertRaises(FileExistsError):
                    save_bundle(folder, config, state, prep['mean'], prep['scale'], DECODER, preprocess=prep)
                with self.assertRaises(ValueError):
                    score_with_bundle(data, embeddings[:-1], bundle)
                invalid = embeddings.copy()
                invalid[0, 0, 0] = np.nan
                with self.assertRaises(ValueError):
                    score_with_bundle(data, invalid, bundle)
                path = Path(folder) / 'config.json'
                saved = json.loads(path.read_text())
                saved['schema'] = 'mp2-research-v2'
                path.write_text(json.dumps(saved))
                with self.assertRaisesRegex(ValueError, 'schema'):
                    load_bundle(folder)

    def test_linear_float64_and_numpy_only_import(self):
        rng = np.random.default_rng(79)
        data = bank(7)
        embeddings = rng.normal(size=(7, 4, 1024))
        features = build_features(data, embeddings, 'relation')
        prep = fit_preprocessor(features)
        state = {'weight': rng.normal(size=2064), 'bias': np.array(.0123456789012345)}
        with tempfile.TemporaryDirectory() as folder:
            save_bundle(folder, {'family': 'linear'}, state, prep['mean'], prep['scale'], DECODER)
            expected = ((features - prep['mean']) / prep['scale']) @ state['weight'] + state['bias']
            np.testing.assert_array_equal(score_with_bundle(data, embeddings, load_bundle(folder)), expected)
        script = ('import sys; sys.path.insert(0, ' + repr(str(Path(__file__).resolve().parents[1]))
                  + '); import mp2.research_v3.models; assert "torch" not in sys.modules')
        result = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main(verbosity=2)
