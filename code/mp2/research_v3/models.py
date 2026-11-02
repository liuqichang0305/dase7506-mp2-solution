"""Original compact TabM/RealMLP adaptations and versioned NumPy-only export.

The v2 architectures are reused without changing their implementations. New
modules follow cited mathematical structures, not author-package training APIs.
No model constructor or inference routine reads task data or reference files.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .. import models_v2 as legacy
from .preprocess import (DIM, preprocess_from_bundle, transform_features,
                         validate_preprocessor)

SCHEMA = 'mp2-research-v3'
RESERVED = frozenset(('mean', 'scale', 'prep_clip_mask', 'prep_constant_mask'))
LEGACY_FAMILIES = frozenset(('linear', 'mlp', 'resmlp', 'deepsets', 'attention'))


def model_config(config):
    family = config.get('family', 'mlp')
    if family in LEGACY_FAMILIES:
        return legacy.model_config(config)
    if family not in ('tabm', 'realmlp'):
        raise ValueError(f'Unknown v3 architecture: {family}')
    cfg = {'family': family, 'width': int(config.get('width', 128)),
           'depth': int(config.get('depth', 3)), 'k': int(config.get('k', 4)),
           'dropout': float(config.get('dropout', 0.1 if family == 'tabm' else 0.0)),
           'input_features': config.get('input_features', 'relation'),
           'start_scaling_init': config.get('start_scaling_init', 'random-signs'),
           'activation': config.get('activation', 'selu' if family == 'realmlp' else 'relu')}
    if min(cfg['width'], cfg['depth'], cfg['k']) < 1 or not 0 <= cfg['dropout'] < 1:
        raise ValueError('Invalid architecture dimensions or dropout')
    if cfg['input_features'] not in ('relation', 'embedding_stats', 'stats'):
        raise ValueError('Unknown input feature subset')
    if cfg['start_scaling_init'] not in ('random-signs', 'normal'):
        raise ValueError('Unknown TabM first-scaling initialization')
    if cfg['activation'] not in ('relu', 'selu', 'mish'):
        raise ValueError('Unknown activation')
    return cfg


def feature_indices(config):
    subset = config.get('input_features', 'relation')
    if subset == 'relation':
        return np.arange(DIM)
    if subset == 'embedding_stats':
        return np.r_[np.arange(1024), np.arange(DIM - 16, DIM)]
    if subset == 'stats':
        return np.arange(DIM - 16, DIM)
    raise ValueError('Unknown input feature subset')


def build_model(config):
    cfg = model_config(config)
    if cfg['family'] in LEGACY_FAMILIES:
        return legacy.build_model(cfg)
    import torch
    from torch import nn
    import torch.nn.functional as F

    indices = feature_indices(cfg).tolist()
    d_input, width, depth, members = len(indices), cfg['width'], cfg['depth'], cfg['k']

    def activation(x):
        if cfg['activation'] == 'relu':
            return torch.relu(x)
        if cfg['activation'] == 'selu':
            return F.selu(x)
        return x * torch.tanh(F.softplus(x))

    class MemberLinear(nn.Module):
        def __init__(self, d_in, d_out, first=False):
            super().__init__()
            self.weight = nn.Parameter(torch.empty(d_out, d_in))
            self.r = nn.Parameter(torch.ones(members, d_in))
            self.s = nn.Parameter(torch.ones(members, d_out))
            self.bias = nn.Parameter(torch.empty(members, d_out))
            bound = d_in ** -0.5
            nn.init.uniform_(self.weight, -bound, bound)
            with torch.no_grad():
                self.bias.copy_(torch.empty(d_out).uniform_(-bound, bound))
                if first:
                    if cfg['start_scaling_init'] == 'random-signs':
                        self.r.copy_(torch.where(torch.rand_like(self.r) < 0.5, -1.0, 1.0))
                    else:
                        self.r.normal_()

        def forward(self, x):
            return ((x * self.r) @ self.weight.T) * self.s + self.bias

    class MemberHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.empty(members, 1, width))
            self.bias = nn.Parameter(torch.empty(members, 1))
            bound = width ** -0.5
            nn.init.uniform_(self.weight, -bound, bound)
            nn.init.uniform_(self.bias, -bound, bound)

        def forward(self, x):
            # Inputs are [N,candidate,member,width]; heads remain independent.
            return (x * self.weight[:, 0, :]).sum(dim=-1) + self.bias[:, 0]

    class TangentLinear(nn.Module):
        def __init__(self, d_in, d_out, zero=False):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(d_out, d_in))
            self.bias = nn.Parameter(torch.randn(d_out))
            self.multiplier = d_in ** -0.5
            if zero:
                nn.init.zeros_(self.weight)
                nn.init.zeros_(self.bias)

        def forward(self, x):
            return (x @ self.weight.T) * self.multiplier + self.bias

    class Verifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.architecture = cfg
            self.drop = nn.Dropout(cfg['dropout'])
            if cfg['family'] == 'tabm':
                self.blocks = nn.ModuleList(MemberLinear(d_input if i == 0 else width, width, first=i == 0)
                                            for i in range(depth))
                self.head = MemberHead()
            else:
                self.input_scale = nn.Parameter(torch.ones(d_input))
                self.layers = nn.ModuleList(TangentLinear(d_input if i == 0 else width, width)
                                            for i in range(depth))
                self.head = TangentLinear(width, 1, zero=True)

        def forward(self, x):
            if x.ndim != 3 or tuple(x.shape[1:]) != (4, DIM):
                raise ValueError('Expected preprocessed [N,4,2064] features')
            x = x[..., indices]
            if cfg['family'] == 'tabm':
                # Member-specific R is applied before the first feature mixing.
                h = x.unsqueeze(2).expand(-1, -1, members, -1)
                for block in self.blocks:
                    h = self.drop(activation(block(h)))
                return self.head(h).transpose(1, 2)  # [N,K,4], never mean logits.
            h = x * self.input_scale
            for layer in self.layers:
                h = self.drop(activation(layer(h)))
            return self.head(h).squeeze(-1)

    return Verifier()


def export_state(model):
    return legacy.export_state(model)


def expected_state_shapes(config):
    cfg = model_config(config)
    if cfg['family'] in LEGACY_FAMILIES:
        return legacy.expected_state_shapes(cfg)
    d_input, width, depth, members = len(feature_indices(cfg)), cfg['width'], cfg['depth'], cfg['k']
    shapes = {}
    if cfg['family'] == 'tabm':
        for i in range(depth):
            d_in = d_input if i == 0 else width
            for suffix, shape in [('weight', (width, d_in)), ('r', (members, d_in)),
                                  ('s', (members, width)), ('bias', (members, width))]:
                shapes[f'blocks.{i}.{suffix}'] = shape
        shapes['head.weight'] = (members, 1, width)
        shapes['head.bias'] = (members, 1)
    else:
        shapes['input_scale'] = (d_input,)
        for i in range(depth):
            shapes[f'layers.{i}.weight'] = (width, d_input if i == 0 else width)
            shapes[f'layers.{i}.bias'] = (width,)
        shapes['head.weight'] = (1, width)
        shapes['head.bias'] = (1,)
    return shapes


def parameter_count(config):
    return int(sum(np.prod(shape, dtype=np.int64) for shape in expected_state_shapes(config).values()))


def validate_state(config, state):
    shapes = expected_state_shapes(config)
    if set(shapes) != set(state):
        raise ValueError('V3 state keys do not match architecture')
    for name, shape in shapes.items():
        array = np.asarray(state[name])
        if array.shape != shape or array.dtype.kind != 'f' or not np.isfinite(array).all():
            raise ValueError(f'Invalid v3 parameter: {name}')


def numpy_forward(x, config, state):
    cfg = model_config(config)
    if cfg['family'] in LEGACY_FAMILIES:
        return legacy.numpy_forward(x, cfg, state)
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 3 or x.shape[1:] != (4, DIM) or not np.isfinite(x).all():
        raise ValueError('Expected finite preprocessed [N,4,2064] features')

    def activation(z):
        if cfg['activation'] == 'relu':
            return np.maximum(z, np.float32(0))
        if cfg['activation'] == 'selu':
            # The min prevents overflow in the unused positive exp branch.
            negative = np.float32(1.6732632423543772) * np.expm1(np.minimum(z, 0))
            return np.float32(1.0507009873554805) * np.where(z > 0, z, negative)
        return z * np.tanh(np.logaddexp(np.float32(0), z))

    h = x[..., feature_indices(cfg)]
    if cfg['family'] == 'tabm':
        h = np.broadcast_to(h[:, :, None, :], (len(x), 4, cfg['k'], h.shape[-1]))
        for i in range(cfg['depth']):
            prefix = f'blocks.{i}'
            h = activation(((h * state[prefix + '.r']) @ state[prefix + '.weight'].T)
                           * state[prefix + '.s'] + state[prefix + '.bias'])
        result = ((h * state['head.weight'][:, 0, :]).sum(axis=-1) + state['head.bias'][:, 0]).transpose(0, 2, 1)
    else:
        h = h * state['input_scale']
        for i in range(cfg['depth']):
            prefix = f'layers.{i}'
            h = activation((h @ state[prefix + '.weight'].T) * np.float32(h.shape[-1] ** -0.5)
                           + state[prefix + '.bias'])
        result = ((h @ state['head.weight'].T) * np.float32(h.shape[-1] ** -0.5) + state['head.bias']).squeeze(-1)
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite v3 scores')
    return result


def save_bundle(path, config, state, mean, scale, decoder, preprocess=None):
    folder = Path(path)
    if (folder / 'config.json').exists() or (folder / 'weights.npz').exists():
        raise FileExistsError('Refusing to overwrite an existing v3 bundle')
    cfg = model_config(config)
    validate_state(cfg, state)
    if preprocess is None:
        preprocess = {'mode': 'standard', 'mean': np.asarray(mean, dtype=np.float64),
                      'scale': np.asarray(scale, dtype=np.float64),
                      'clip_mask': np.zeros(DIM, bool), 'constant_mask': np.zeros(DIM, bool)}
    validate_preprocessor(preprocess)
    if not np.array_equal(mean, preprocess['mean']) or not np.array_equal(scale, preprocess['scale']):
        raise ValueError('Bundle scaler differs from supplied preprocessing state')
    record = {'schema': SCHEMA, 'features': 'relation', 'architecture': cfg, 'decoder': dict(decoder),
              'preprocess': {'mode': preprocess['mode'], 'smooth_clip_bound': 3.0},
              'output': 'member_candidate_scores' if cfg['family'] == 'tabm' else 'candidate_scores'}
    folder.mkdir(parents=True, exist_ok=True)
    temporary = folder / 'weights.pending.npz'
    np.savez(temporary, mean=preprocess['mean'], scale=preprocess['scale'],
             prep_clip_mask=np.asarray(preprocess['clip_mask'], dtype=np.uint8),
             prep_constant_mask=np.asarray(preprocess['constant_mask'], dtype=np.uint8), **state)
    temporary.replace(folder / 'weights.npz')
    temporary_config = folder / 'config.pending.json'
    temporary_config.write_text(json.dumps(record, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    temporary_config.replace(folder / 'config.json')
    return record


def load_bundle(checkpoint):
    if checkpoint is None:
        raise ValueError('V3 inference requires an explicit checkpoint')
    folder = Path(checkpoint)
    config = json.loads((folder / 'config.json').read_text(encoding='utf-8'))
    if config.get('schema') != SCHEMA or config.get('features') != 'relation':
        raise ValueError('Unsupported v3 schema/features; v1/v2 bundles require their original interfaces')
    if config.get('preprocess', {}).get('smooth_clip_bound') != 3.0:
        raise ValueError('Unsupported smooth-clipping bound')
    with np.load(folder / 'weights.npz', allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    if not RESERVED <= set(arrays):
        raise ValueError('Missing v3 preprocessing arrays')
    # Validate masks before converting them to bool (otherwise arbitrary integers
    # could silently become valid masks).
    preprocess = {'mode': config['preprocess']['mode'], 'mean': arrays['mean'], 'scale': arrays['scale'],
                  'clip_mask': arrays['prep_clip_mask'], 'constant_mask': arrays['prep_constant_mask']}
    validate_preprocessor(preprocess)
    validate_state(config['architecture'], {name: value for name, value in arrays.items() if name not in RESERVED})
    return config, arrays


def score_with_bundle(bank, embeddings, bundle):
    from ..features import build_features
    config, arrays = bundle
    raw = build_features(bank, embeddings, 'relation')
    cfg = config['architecture']
    dtype = np.float64 if cfg['family'] == 'linear' else np.float32
    shape = (len(bank), cfg['k'], 4) if cfg['family'] == 'tabm' else (len(bank), 4)
    output = np.empty(shape, dtype=dtype)
    preprocess = preprocess_from_bundle(config, arrays)
    state = {name: value for name, value in arrays.items() if name not in RESERVED}
    for start in range(0, len(bank), 128):
        stop = min(start + 128, len(bank))
        x = transform_features(raw[start:stop], preprocess, dtype=dtype)
        output[start:stop] = numpy_forward(x, cfg, state)
    return output
