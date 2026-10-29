"""Synthetic optimizer/EMA partition, update, and exact-resume checks."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mp2.research_v3.optimizers import EMA, build_optimizers


class SharedBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(4, 4))
        self.r = nn.Parameter(torch.ones(2, 4))
        self.s = nn.Parameter(torch.ones(2, 4))
        self.bias = nn.Parameter(torch.zeros(2, 4))


class GroupingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(5, 4), nn.Linear(4, 4)])
        self.blocks = nn.ModuleList([SharedBlock()])
        self.norm = nn.LayerNorm((2, 4))
        self.embedding = nn.Embedding(5, 4)
        self.input_scale = nn.Parameter(torch.ones(5))
        self.head = nn.Linear(4, 2)
        self.frozen = nn.Parameter(torch.zeros(2, 2), requires_grad=False)


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(5, 4), nn.Linear(4, 4)])
        self.head = nn.Linear(4, 1)

    def forward(self, x):
        for layer in self.layers:
            x = torch.tanh(layer(x))
        return self.head(x)


def train_step(model, optimizer, x, y):
    optimizer.zero_grad(set_to_none=True)
    loss = (model(x) - y).square().mean()
    loss.backward()
    optimizer.step()
    return loss.detach()


class OptimizersV3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)

    def test_hidden_matrix_partition_head_norm_and_tabm_adapters(self):
        model = GroupingModel()
        bundle = build_optimizers(model, {'optimizer': 'muon', 'lr': .0003, 'muon_lr': .02, 'weight_decay': .01})
        assignment = {name: group for group in bundle.group_report for name in group['param_names']}
        muon_names = {name for name, group in assignment.items() if group['role'] == 'muon_hidden_matrix'}
        self.assertEqual(muon_names, {'layers.0.weight', 'layers.1.weight', 'blocks.0.weight'})
        self.assertEqual(assignment['head.weight']['role'], 'adamw_auxiliary')
        for name in ('norm.weight', 'norm.bias', 'embedding.weight', 'input_scale', 'head.bias', 'blocks.0.bias'):
            self.assertEqual(assignment[name]['weight_decay'], 0.)
        for name in ('blocks.0.r', 'blocks.0.s'):
            self.assertEqual(assignment[name]['role'], 'adamw_auxiliary')
        self.assertNotIn('frozen', assignment)
        parameters = [p for g in bundle.param_groups for p in g['params']]
        self.assertEqual(len(parameters), len({id(p) for p in parameters}))
        self.assertEqual({id(p) for p in parameters}, {id(p) for p in model.parameters() if p.requires_grad})

    def test_tied_head_weight_is_excluded_from_muon(self):
        model = TinyModel()
        model.head = nn.Linear(4, 4)
        model.head.weight = model.layers[1].weight
        bundle = build_optimizers(model, {'optimizer': 'muon'})
        muon_names = {name for group in bundle.group_report if group['role'] == 'muon_hidden_matrix' for name in group['param_names']}
        self.assertEqual(muon_names, {'layers.0.weight'})

    def test_explicit_realmlp_learning_rate_multipliers(self):
        model = GroupingModel()
        bundle = build_optimizers(model, {'optimizer': 'adamw', 'lr': .01, 'beta2': .95,
                                         'input_scale_lr_multiplier': 6., 'bias_lr_multiplier': .1})
        assignment = {name: group for group in bundle.group_report for name in group['param_names']}
        self.assertAlmostEqual(assignment['input_scale']['lr'], .06)
        self.assertAlmostEqual(assignment['layers.0.bias']['lr'], .001)
        self.assertAlmostEqual(assignment['layers.0.weight']['lr'], .01)
        self.assertEqual(bundle.optimizers[0].defaults['betas'], (.9, .95))

    def test_adamw_matches_direct_torch_with_identical_groups(self):
        torch.manual_seed(7)
        model = TinyModel()
        reference = deepcopy(model)
        bundle = build_optimizers(model, {'optimizer': 'adamw', 'lr': .003, 'weight_decay': .02})
        named = dict(reference.named_parameters())
        reference_groups = [{'params': [named[name] for name in group['param_names']],
                             'lr': group['lr'], 'weight_decay': group['weight_decay']}
                            for group in bundle.group_report]
        direct = torch.optim.AdamW(reference_groups, lr=.003, weight_decay=.02, foreach=False, fused=False)
        x, y = torch.randn(7, 5), torch.randn(7, 1)
        for _ in range(3):
            train_step(model, bundle, x, y)
            train_step(reference, direct, x, y)
        for name, parameter in model.state_dict().items():
            torch.testing.assert_close(parameter, reference.state_dict()[name], atol=0, rtol=0)

    def test_both_optimizers_update_and_exactly_resume(self):
        for optimizer_name in ('adamw', 'muon'):
            torch.manual_seed(11)
            model = TinyModel()
            initial = deepcopy(model.state_dict())
            config = {'optimizer': optimizer_name, 'lr': .003, 'muon_lr': .02, 'weight_decay': .01}
            bundle = build_optimizers(model, config)
            x, y = torch.randn(7, 5), torch.randn(7, 1)
            train_step(model, bundle, x, y)
            train_step(model, bundle, x, y)
            checkpoint = {'model': deepcopy(model.state_dict()), 'optimizer': bundle.state_dict()}
            self.assertFalse(torch.equal(initial['layers.0.weight'], model.layers[0].weight))
            self.assertFalse(torch.equal(initial['head.weight'], model.head.weight))
            train_step(model, bundle, x, y)
            resumed = TinyModel()
            resumed.load_state_dict(checkpoint['model'])
            resumed_bundle = build_optimizers(resumed, config)
            resumed_bundle.load_state_dict(checkpoint['optimizer'])
            train_step(resumed, resumed_bundle, x, y)
            self.assertEqual(resumed_bundle.step_count, bundle.step_count)
            for name, parameter in model.state_dict().items():
                self.assertTrue(torch.isfinite(parameter).all())
                torch.testing.assert_close(parameter, resumed.state_dict()[name], atol=0, rtol=0)

    def test_ema_update_order_buffers_and_same_trajectory(self):
        model = nn.Linear(2, 1)
        model.register_buffer('running', torch.zeros(1))
        model.register_buffer('counter', torch.zeros((), dtype=torch.long))
        ema = EMA(model, decay=.5)
        with torch.no_grad():
            model.weight.fill_(2.)
            model.running.fill_(2.)
            model.counter.fill_(1)
        ema.update(model)
        torch.testing.assert_close(ema.module.weight, torch.full_like(model.weight, 2.))
        with torch.no_grad():
            model.weight.fill_(4.)
            model.running.fill_(6.)
            model.counter.fill_(2)
        ema.update(model)
        torch.testing.assert_close(ema.module.weight, torch.full_like(model.weight, 3.))
        self.assertEqual(float(ema.module.running), 4.)
        self.assertEqual(int(ema.module.counter), 2)
        self.assertEqual(float(model.weight[0, 0].detach()), 4.)
        self.assertTrue(all(not p.requires_grad for p in ema.module.parameters()))
        self.assertFalse(ema.module.training)
        self.assertEqual(set(ema.export_state_dict()), set(model.state_dict()))
        with self.assertRaises(ValueError):
            ema.update(deepcopy(model))

    def test_ema_matches_pytorch_averagedmodel_and_resumes(self):
        from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
        torch.manual_seed(23)
        model = TinyModel()
        ema = EMA(model, decay=.99)
        official = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(.99), use_buffers=True)
        for step in range(4):
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.add_(.01 * (step + 1))
            ema.update(model)
            official.update_parameters(model)
        for name, parameter in ema.module.state_dict().items():
            torch.testing.assert_close(parameter, official.module.state_dict()[name], atol=1e-7, rtol=1e-7)
        saved = ema.state_dict()
        resumed = EMA(model, decay=.99)
        resumed.load_state_dict(saved)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.add_(.5)
        ema.update(model)
        resumed.update(model)
        self.assertEqual(ema.num_updates, resumed.num_updates)
        for name, parameter in ema.module.state_dict().items():
            torch.testing.assert_close(parameter, resumed.module.state_dict()[name], atol=0, rtol=0)

    def test_fail_fast_and_checkpoint_guards(self):
        model = TinyModel()
        bundle = build_optimizers(model, {'optimizer': 'muon'})
        initial = deepcopy(model.state_dict())
        for parameter in model.parameters():
            parameter.grad = torch.ones_like(parameter)
        model.head.bias.grad.fill_(float('nan'))
        with self.assertRaises(FloatingPointError):
            bundle.step()
        for name, parameter in model.state_dict().items():
            torch.testing.assert_close(parameter, initial[name], atol=0, rtol=0)
        with self.assertRaises(ValueError):
            build_optimizers(model, {'optimizer': 'muon', 'muon_lr': 0})
        with self.assertRaises(ValueError):
            build_optimizers(nn.Linear(4, 1), {'optimizer': 'muon'})
        with self.assertRaises(ValueError):
            EMA(nn.BatchNorm1d(4))
        ema = EMA(model)
        with self.assertRaises(ValueError):
            ema.export_state_dict()
        with self.assertRaises(ValueError):
            build_optimizers(model, {'optimizer': 'adamw'}).load_state_dict(bundle.state_dict())


if __name__ == '__main__':
    unittest.main(verbosity=2)
