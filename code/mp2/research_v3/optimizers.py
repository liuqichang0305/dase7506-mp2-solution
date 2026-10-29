"""Training-only AdamW/Muon grouping and same-trajectory EMA for V3.

Muon delegates to the installed PyTorch implementation. No optimizer state or
EMA history belongs in an inference bundle; export only the chosen model state.
See docs/research_v3/OPTIMIZER_SOURCES.md for sources and adaptation boundaries.
"""
from __future__ import annotations

from copy import deepcopy
import math
import weakref

import torch
from torch import nn


_HEAD_COMPONENTS = {'head', 'heads', 'output', 'output_layer', 'classifier', 'logits'}
_EMBEDDING_COMPONENTS = {'embedding', 'embeddings', 'num_embeddings', 'numerical_embeddings'}
_NORM_TYPES = (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
               nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d)
if hasattr(nn, 'RMSNorm'):
    _NORM_TYPES += (nn.RMSNorm,)


def _positive(value, name, allow_zero=False):
    value = float(value)
    if not math.isfinite(value) or value < 0 or (value == 0 and not allow_zero):
        raise ValueError(f'{name} must be finite and {"nonnegative" if allow_zero else "positive"}')
    return value


def _configuration(config):
    name = str(config.get('optimizer', 'adamw')).lower()
    if name not in ('adamw', 'muon'):
        raise ValueError(f'Unsupported optimizer: {name}')
    betas = tuple(float(x) for x in config.get('betas', (config.get('beta1', .9), config.get('beta2', .999))))
    if len(betas) != 2 or any(not math.isfinite(x) or not 0 <= x < 1 for x in betas):
        raise ValueError('AdamW betas must be two finite values in [0,1)')
    result = {'optimizer': name, 'lr': _positive(config.get('lr', 3e-4), 'lr'),
              'weight_decay': _positive(config.get('weight_decay', .001), 'weight_decay', True),
              'betas': betas, 'eps': _positive(config.get('eps', 1e-8), 'eps'),
              'input_scale_lr_multiplier': _positive(config.get('input_scale_lr_multiplier', 1.), 'input_scale_lr_multiplier'),
              'bias_lr_multiplier': _positive(config.get('bias_lr_multiplier', 1.), 'bias_lr_multiplier')}
    if name == 'muon':
        momentum = float(config.get('muon_momentum', config.get('momentum', .95)))
        if not math.isfinite(momentum) or not 0 <= momentum < 1:
            raise ValueError('Muon momentum must be finite and in [0,1)')
        steps = config.get('muon_ns_steps', config.get('ns_steps', 5))
        if type(steps) is not int or steps < 1:
            raise ValueError('Muon Newton-Schulz steps must be a positive integer')
        adjust = config.get('muon_adjust_lr_fn', 'original')
        if adjust not in ('original', 'match_rms_adamw', 'spectral_unclamped'):
            raise ValueError('Unsupported Muon rectangular-matrix scaling')
        result.update({'muon_lr': _positive(config.get('muon_lr', .02), 'muon_lr'),
                       'muon_weight_decay': _positive(config.get('muon_weight_decay', result['weight_decay']), 'muon_weight_decay', True),
                       'muon_momentum': momentum, 'muon_nesterov': bool(config.get('muon_nesterov', True)),
                       'muon_ns_steps': steps, 'muon_eps': _positive(config.get('muon_eps', 1e-7), 'muon_eps'),
                       'muon_adjust_lr_fn': adjust})
    return result


def _parameter_partition(model, configuration):
    modules = dict(model.named_modules(remove_duplicate=False))
    aliases = {}
    for name, parameter in model.named_parameters(remove_duplicate=False):
        aliases.setdefault(id(parameter), []).append(name)
    groups = {}
    seen = set()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        parameter_aliases = aliases[id(parameter)]
        norm_or_embedding = False
        is_bias = False
        is_input_scale = False
        hidden_weight = parameter.ndim == 2 and min(parameter.shape) > 1
        for alias in parameter_aliases:
            path, _, leaf = alias.rpartition('.')
            owner = modules[path]
            components = set(path.lower().split('.'))
            is_norm = isinstance(owner, _NORM_TYPES)
            is_embedding = isinstance(owner, (nn.Embedding, nn.EmbeddingBag)) or bool(components & _EMBEDDING_COMPONENTS)
            norm_or_embedding |= is_norm or is_embedding
            is_bias |= leaf == 'bias'
            is_input_scale |= leaf == 'input_scale'
            # Custom TabM shared weights also have leaf `weight`; member r/s,
            # bias and three-dimensional per-member heads never enter Muon.
            hidden_weight &= leaf == 'weight' and not (components & _HEAD_COMPONENTS) and not is_norm and not is_embedding
        use_muon = configuration['optimizer'] == 'muon' and hidden_weight
        if use_muon:
            role = 'muon_hidden_matrix'
            lr, decay = configuration['muon_lr'], configuration['muon_weight_decay']
        else:
            role = 'adamw_auxiliary' if configuration['optimizer'] == 'muon' else 'adamw'
            lr = configuration['lr']
            if is_bias:
                lr *= configuration['bias_lr_multiplier']
            if is_input_scale:
                lr *= configuration['input_scale_lr_multiplier']
            decay = 0. if is_bias or norm_or_embedding or parameter.ndim < 2 else configuration['weight_decay']
        key = (role, lr, decay)
        group = groups.setdefault(key, {'role': role, 'lr': lr, 'weight_decay': decay, 'params': [], 'param_names': [], 'shapes': []})
        if id(parameter) in seen:
            raise ValueError('A trainable parameter was assigned more than once')
        seen.add(id(parameter))
        group['params'].append(parameter)
        group['param_names'].append(name)
        group['shapes'].append(list(parameter.shape))
    if not seen:
        raise ValueError('Model has no trainable parameters')
    expected = {id(p) for p in model.parameters() if p.requires_grad}
    if seen != expected:
        raise ValueError('Optimizer parameter partition is incomplete')
    if configuration['optimizer'] == 'muon' and not any(g['role'] == 'muon_hidden_matrix' for g in groups.values()):
        raise ValueError('Muon requested but no eligible hidden two-dimensional weight matrix exists')
    return list(groups.values())


class OptimizerBundle:
    """A deterministic wrapper around disjoint PyTorch optimizer instances.

    It offers the training loop's zero_grad/step/state_dict API. This is not a
    torch.optim.Optimizer subclass; use its actual optimizers if a scheduler
    requiring that class is explicitly introduced. No scheduler is implicit.
    """
    def __init__(self, optimizers, optimizer_names, configuration, group_report):
        self.optimizers = tuple(optimizers)
        self.optimizer_names = tuple(optimizer_names)
        self.configuration = deepcopy(configuration)
        self.group_report = deepcopy(group_report)
        self.step_count = 0

    @property
    def param_groups(self):
        return [group for optimizer in self.optimizers for group in optimizer.param_groups]

    def zero_grad(self, set_to_none=True):
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=set_to_none)

    def step(self, closure=None):
        if closure is not None:
            raise ValueError('OptimizerBundle requires explicit forward/backward; closures are unsupported')
        # Validate before either optimizer mutates weights, so a bad gradient
        # cannot silently produce a half-updated Muon/AdamW pair.
        for group in self.param_groups:
            for parameter in group['params']:
                if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                    raise FloatingPointError('Nonfinite gradient before optimizer step')
        for optimizer in self.optimizers:
            optimizer.step()
        self.step_count += 1

    def state_dict(self):
        return {'schema': 'mp2-v3-optimizer-state-v1', 'configuration': deepcopy(self.configuration),
                'optimizer_names': list(self.optimizer_names), 'group_report': deepcopy(self.group_report),
                'step_count': self.step_count,
                'optimizer_states': [deepcopy(optimizer.state_dict()) for optimizer in self.optimizers]}

    def load_state_dict(self, state):
        if state.get('schema') != 'mp2-v3-optimizer-state-v1':
            raise ValueError('Unsupported optimizer checkpoint schema')
        if state['configuration'] != self.configuration or state['optimizer_names'] != list(self.optimizer_names) or state['group_report'] != self.group_report:
            raise ValueError('Optimizer checkpoint configuration or named parameter partition differs')
        if len(state['optimizer_states']) != len(self.optimizers) or type(state['step_count']) is not int or state['step_count'] < 0:
            raise ValueError('Malformed optimizer checkpoint')
        for optimizer, saved in zip(self.optimizers, state['optimizer_states']):
            optimizer.load_state_dict(deepcopy(saved))
        self.step_count = state['step_count']


def build_optimizers(model, config):
    """Build AdamW, or native Muon for hidden matrices plus auxiliary AdamW.

    All parameters have exactly one optimizer. Head, bias, normalization,
    feature-embedding and non-matrix parameters are never Muon parameters.
    Config explicitly separates `muon_lr` from auxiliary AdamW `lr`.
    """
    configuration = _configuration(config)
    partition = _parameter_partition(model, configuration)
    reports = [{key: deepcopy(value) for key, value in group.items() if key != 'params'} |
               {'parameter_count': sum(p.numel() for p in group['params'])} for group in partition]
    optimizers, names = [], []
    muon_groups = [{k: v for k, v in group.items() if k not in ('role', 'shapes')} for group in partition if group['role'] == 'muon_hidden_matrix']
    adam_groups = [{k: v for k, v in group.items() if k not in ('role', 'shapes')} for group in partition if group['role'] != 'muon_hidden_matrix']
    if muon_groups:
        if not hasattr(torch.optim, 'Muon'):
            raise RuntimeError('This experiment requires a PyTorch version exposing torch.optim.Muon')
        optimizers.append(torch.optim.Muon(muon_groups, lr=configuration['muon_lr'],
            weight_decay=configuration['muon_weight_decay'], momentum=configuration['muon_momentum'],
            nesterov=configuration['muon_nesterov'], ns_steps=configuration['muon_ns_steps'],
            eps=configuration['muon_eps'], adjust_lr_fn=configuration['muon_adjust_lr_fn']))
        names.append('torch.optim.Muon')
    if adam_groups:
        optimizers.append(torch.optim.AdamW(adam_groups, lr=configuration['lr'], weight_decay=configuration['weight_decay'],
                                            betas=configuration['betas'], eps=configuration['eps'], foreach=False, fused=False))
        names.append('torch.optim.AdamW')
    return OptimizerBundle(optimizers, names, configuration, reports)


class EMA:
    """A detached average of one live training model, updated after each step.

    The first update copies the first post-step state. Subsequent floating
    parameters/buffers use decay*average+(1-decay)*live; integer buffers copy
    the live value. BatchNorm is rejected because these models do not implement
    the additional fitting-partition-only BN-statistic recalibration it needs.
    """
    def __init__(self, model, decay=.99):
        self.decay = float(decay)
        if not math.isfinite(self.decay) or not 0 <= self.decay < 1:
            raise ValueError('EMA decay must be finite and in [0,1)')
        if any(isinstance(module, nn.modules.batchnorm._BatchNorm) for module in model.modules()):
            raise ValueError('EMA BatchNorm requires explicit train-only statistic recalibration; unsupported in this recipe')
        self._source = weakref.ref(model)
        self.module = deepcopy(model).eval()
        self.module.requires_grad_(False)
        self.num_updates = 0

    def _check_source(self, model):
        if self._source() is not model:
            raise ValueError('EMA can only update from its own live training trajectory')
        average, live = self.module.state_dict(), model.state_dict()
        if set(average) != set(live) or any(average[key].shape != live[key].shape or average[key].dtype != live[key].dtype for key in average):
            raise ValueError('EMA model structure or dtype changed')
        if any(value.is_floating_point() and not torch.isfinite(value).all() for value in live.values()):
            raise FloatingPointError('EMA source contains nonfinite state')
        return average, live

    @torch.no_grad()
    def update(self, model):
        average, live = self._check_source(model)
        for name, target in average.items():
            source = live[name].detach().to(device=target.device)
            if self.num_updates == 0 or not target.is_floating_point():
                target.copy_(source)
            else:
                target.mul_(self.decay).add_(source, alpha=1. - self.decay)
        self.num_updates += 1
        self.module.eval()

    def state_dict(self):
        return {'schema': 'mp2-v3-ema-state-v1', 'decay': self.decay, 'num_updates': self.num_updates,
                'model_state': {name: value.detach().clone() for name, value in self.module.state_dict().items()}}

    def load_state_dict(self, state):
        if state.get('schema') != 'mp2-v3-ema-state-v1' or state['decay'] != self.decay:
            raise ValueError('EMA schema or decay differs from the registered trajectory')
        if type(state['num_updates']) is not int or state['num_updates'] < 0:
            raise ValueError('Invalid EMA update count')
        current, saved = self.module.state_dict(), state['model_state']
        if set(current) != set(saved):
            raise ValueError('EMA checkpoint model keys differ')
        for name in current:
            if current[name].shape != saved[name].shape or current[name].dtype != saved[name].dtype:
                raise ValueError(f'EMA checkpoint tensor differs: {name}')
            if saved[name].is_floating_point() and not torch.isfinite(saved[name]).all():
                raise ValueError(f'EMA checkpoint tensor is nonfinite: {name}')
        self.module.load_state_dict(saved, strict=True)
        self.module.eval()
        self.num_updates = state['num_updates']

    def export_state_dict(self):
        """Only averaged model tensors, without optimizer/history metadata."""
        if self.num_updates < 1:
            raise ValueError('EMA has not received any post-step model update')
        return {name: value.detach().cpu().clone() for name, value in self.module.state_dict().items()}
