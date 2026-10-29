"""Small randomly initialized verifiers and a NumPy-only inference equivalent.

Architectures are original task-specific implementations inspired by MLP/ResNet,
Deep Sets and Set Transformer. No source or pretrained weights are imported from
those projects. Torch is imported only by ``build_model`` for training.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

SCHEMA = 'mp2-research-v2'
INPUT_DIM = 2064


def model_config(config):
    """Resolve the small architecture vocabulary, independent of training knobs."""
    family = config.get('family', 'mlp')
    if family not in ('linear', 'mlp', 'resmlp', 'deepsets', 'attention'):
        raise ValueError(f'Unknown v2 architecture: {family}')
    result = {
        'family': family,
        'width': int(config.get('width', 128)),
        'hidden': int(config.get('hidden', 64)),
        'dropout': float(config.get('dropout', 0.1)),
        'blocks': int(config.get('blocks', 1)),
        'expansion': int(config.get('expansion', 2)),
        'heads': int(config.get('heads', 4)),
        'context': bool(config.get('context', True)),
        'layer_norm_eps': 1e-5,
    }
    if any(result[key] < 1 for key in ('width', 'hidden', 'blocks', 'expansion', 'heads')):
        raise ValueError('Architecture dimensions must be positive')
    if not 0 <= result['dropout'] < 1:
        raise ValueError('Dropout must be in [0,1)')
    if family == 'attention' and result['width'] % result['heads']:
        raise ValueError('Attention width must be divisible by heads')
    return result


def build_model(config):
    """Build a torch module taking standardized [batch,4,2064] features."""
    import torch
    from torch import nn

    cfg = model_config(config)
    family, width = cfg['family'], cfg['width']

    class ResidualBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = nn.LayerNorm(width, eps=cfg['layer_norm_eps'])
            self.fc1 = nn.Linear(width, width * cfg['expansion'])
            self.fc2 = nn.Linear(width * cfg['expansion'], width)
            self.drop = nn.Dropout(cfg['dropout'])

        def forward(self, x):
            return x + self.fc2(self.drop(torch.relu(self.fc1(self.norm(x)))))

    class Verifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.architecture = cfg
            self.drop = nn.Dropout(cfg['dropout'])
            if family == 'linear':
                self.weight = nn.Parameter(torch.zeros(INPUT_DIM))
                self.bias = nn.Parameter(torch.zeros(()))
            elif family == 'mlp':
                self.fc1 = nn.Linear(INPUT_DIM, width)
                self.fc2 = nn.Linear(width, cfg['hidden'])
                self.head = nn.Linear(cfg['hidden'], 1)
            elif family == 'resmlp':
                self.input = nn.Linear(INPUT_DIM, width)
                self.blocks = nn.ModuleList(ResidualBlock() for _ in range(cfg['blocks']))
                self.head = nn.Linear(width, 1)
            elif family == 'deepsets':
                self.phi1 = nn.Linear(1024, width)
                self.phi2 = nn.Linear(width, width)
                context_dim = 5 * width if cfg['context'] else width
                self.rho = nn.Linear(context_dim + 16, width)
                self.head = nn.Linear(width, 1)
            else:
                # Exactly four candidate tokens; no candidate position encoding.
                self.input = nn.Linear(1024, width)
                self.norm1 = nn.LayerNorm(width, eps=cfg['layer_norm_eps'])
                self.qkv = nn.Linear(width, 3 * width)
                self.out = nn.Linear(width, width)
                self.norm2 = nn.LayerNorm(width, eps=cfg['layer_norm_eps'])
                self.ff1 = nn.Linear(width, 2 * width)
                self.ff2 = nn.Linear(2 * width, width)
                self.rho = nn.Linear(width + 16, cfg['hidden'])
                self.head = nn.Linear(cfg['hidden'], 1)

        def forward(self, x):
            if x.ndim != 3 or tuple(x.shape[1:]) != (4, INPUT_DIM):
                raise ValueError('Expected standardized features [N,4,2064]')
            if family == 'linear':
                return x @ self.weight + self.bias
            if family == 'mlp':
                h = self.drop(torch.relu(self.fc1(x)))
                h = self.drop(torch.relu(self.fc2(h)))
            elif family == 'resmlp':
                h = self.input(x)
                for block in self.blocks:
                    h = block(h)
            elif family == 'deepsets':
                h = torch.relu(self.phi2(self.drop(torch.relu(self.phi1(x[..., :1024])))))
                if cfg['context']:
                    mean = h.mean(dim=1, keepdim=True).expand_as(h)
                    maximum = h.max(dim=1, keepdim=True).values.expand_as(h)
                    h = torch.cat((h, mean, maximum, torch.abs(h - mean), h * mean), dim=-1)
                h = self.drop(torch.relu(self.rho(torch.cat((h, x[..., -16:]), dim=-1))))
            else:
                h = self.input(x[..., :1024])
                q, k, v = self.qkv(self.norm1(h)).chunk(3, dim=-1)
                heads, depth = cfg['heads'], width // cfg['heads']
                q = q.reshape(-1, 4, heads, depth).transpose(1, 2)
                k = k.reshape(-1, 4, heads, depth).transpose(1, 2)
                v = v.reshape(-1, 4, heads, depth).transpose(1, 2)
                attention = torch.softmax((q @ k.transpose(-1, -2)) / (depth ** 0.5), dim=-1)
                context = (self.drop(attention) @ v).transpose(1, 2).reshape(-1, 4, width)
                h = h + self.drop(self.out(context))
                h = h + self.ff2(self.drop(torch.relu(self.ff1(self.norm2(h)))))
                h = self.drop(torch.relu(self.rho(torch.cat((h, x[..., -16:]), dim=-1))))
            return self.head(h).squeeze(-1)

    return Verifier()


def export_state(model):
    """Copy only numeric tensors; no pickle or executable checkpoint payload."""
    return {name: tensor.detach().cpu().numpy().astype(np.float32, copy=True)
            for name, tensor in model.state_dict().items()}


def expected_state_shapes(config):
    cfg = model_config(config)
    family, width, hidden = cfg['family'], cfg['width'], cfg['hidden']
    result = {}

    def linear(name, din, dout):
        result[name + '.weight'] = (dout, din)
        result[name + '.bias'] = (dout,)

    def norm(name):
        result[name + '.weight'] = (width,)
        result[name + '.bias'] = (width,)

    if family == 'linear':
        return {'weight': (INPUT_DIM,), 'bias': ()}
    if family == 'mlp':
        linear('fc1', INPUT_DIM, width)
        linear('fc2', width, hidden)
        linear('head', hidden, 1)
    elif family == 'resmlp':
        linear('input', INPUT_DIM, width)
        for i in range(cfg['blocks']):
            norm(f'blocks.{i}.norm')
            linear(f'blocks.{i}.fc1', width, width * cfg['expansion'])
            linear(f'blocks.{i}.fc2', width * cfg['expansion'], width)
        linear('head', width, 1)
    elif family == 'deepsets':
        linear('phi1', 1024, width)
        linear('phi2', width, width)
        linear('rho', (5 * width if cfg['context'] else width) + 16, width)
        linear('head', width, 1)
    else:
        linear('input', 1024, width)
        norm('norm1')
        linear('qkv', width, 3 * width)
        linear('out', width, width)
        norm('norm2')
        linear('ff1', width, 2 * width)
        linear('ff2', 2 * width, width)
        linear('rho', width + 16, hidden)
        linear('head', hidden, 1)
    return result


def validate_state(config, state):
    expected = expected_state_shapes(config)
    if set(state) != set(expected):
        raise ValueError('Checkpoint parameter keys do not match the architecture')
    for name, shape in expected.items():
        value = np.asarray(state[name])
        if value.shape != shape or value.dtype.kind != 'f' or not np.isfinite(value).all():
            raise ValueError(f'Invalid numeric parameter: {name}')


def parameter_count(config):
    return int(sum(np.prod(shape, dtype=np.int64) for shape in expected_state_shapes(config).values()))


def numpy_forward(x, config, state):
    """Evaluation-mode equivalent of the torch module (dropout disabled)."""
    cfg = model_config(config)
    # Keep the legacy SciPy linear path in float64 for exact reproduction.
    x = np.asarray(x, dtype=np.float64 if cfg['family'] == 'linear' else np.float32)
    if x.ndim != 3 or x.shape[1:] != (4, INPUT_DIM) or not np.isfinite(x).all():
        raise ValueError('Expected finite standardized features [N,4,2064]')
    family, width = cfg['family'], cfg['width']

    def linear(h, name):
        return h @ state[name + '.weight'].T + state[name + '.bias']

    def relu(h):
        return np.maximum(h, np.float32(0))

    def norm(h, name):
        centered = h - h.mean(axis=-1, keepdims=True)
        normalized = centered / np.sqrt(np.mean(centered * centered, axis=-1, keepdims=True) + np.float32(cfg['layer_norm_eps']))
        return normalized * state[name + '.weight'] + state[name + '.bias']

    if family == 'linear':
        result = x @ state['weight'] + state['bias']
        if not np.isfinite(result).all():
            raise ValueError('Nonfinite v2 linear scores')
        return result
    if family == 'mlp':
        h = relu(linear(relu(linear(x, 'fc1')), 'fc2'))
    elif family == 'resmlp':
        h = linear(x, 'input')
        for i in range(cfg['blocks']):
            prefix = f'blocks.{i}'
            h = h + linear(relu(linear(norm(h, prefix + '.norm'), prefix + '.fc1')), prefix + '.fc2')
    elif family == 'deepsets':
        h = relu(linear(relu(linear(x[..., :1024], 'phi1')), 'phi2'))
        if cfg['context']:
            mean = np.broadcast_to(h.mean(axis=1, keepdims=True), h.shape)
            maximum = np.broadcast_to(h.max(axis=1, keepdims=True), h.shape)
            h = np.concatenate((h, mean, maximum, np.abs(h - mean), h * mean), axis=-1)
        h = relu(linear(np.concatenate((h, x[..., -16:]), axis=-1), 'rho'))
    else:
        h = linear(x[..., :1024], 'input')
        q, k, v = np.split(linear(norm(h, 'norm1'), 'qkv'), 3, axis=-1)
        heads, depth = cfg['heads'], width // cfg['heads']
        q = q.reshape(len(x), 4, heads, depth).transpose(0, 2, 1, 3)
        k = k.reshape(len(x), 4, heads, depth).transpose(0, 2, 1, 3)
        v = v.reshape(len(x), 4, heads, depth).transpose(0, 2, 1, 3)
        logits = (q @ k.swapaxes(-1, -2)) / np.float32(depth ** 0.5)
        attention = np.exp(logits - logits.max(axis=-1, keepdims=True))
        attention /= attention.sum(axis=-1, keepdims=True)
        context = (attention @ v).transpose(0, 2, 1, 3).reshape(len(x), 4, width)
        h = h + linear(context, 'out')
        h = h + linear(relu(linear(norm(h, 'norm2'), 'ff1')), 'ff2')
        h = relu(linear(np.concatenate((h, x[..., -16:]), axis=-1), 'rho'))
    result = linear(h, 'head').squeeze(-1)
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite v2 model scores')
    return result


def _validate_scaler(mean, scale):
    for name, value in (('mean', mean), ('scale', scale)):
        if value.shape != (INPUT_DIM,) or value.dtype.kind != 'f' or not np.isfinite(value).all():
            raise ValueError(f'Invalid v2 scaler {name}')
    if np.any(scale <= 0):
        raise ValueError('Scaler must be positive')


def save_bundle(path, config, state, mean, scale, decoder):
    """Write a new numeric inference bundle, refusing to overwrite artifacts."""
    folder = Path(path)
    if (folder / 'config.json').exists() or (folder / 'weights.npz').exists():
        raise FileExistsError(f'Refusing to overwrite v2 bundle: {folder}')
    cfg = model_config(config)
    validate_state(cfg, state)
    mean, scale = np.asarray(mean, dtype=np.float64), np.asarray(scale, dtype=np.float64)
    _validate_scaler(mean, scale)
    bundle_config = {'schema': SCHEMA, 'features': 'relation', 'architecture': cfg, 'decoder': dict(decoder)}
    folder.mkdir(parents=True, exist_ok=True)
    np.savez(folder / 'weights.npz', mean=mean, scale=scale, **state)
    (folder / 'config.json').write_text(json.dumps(bundle_config, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    return bundle_config


def load_bundle(checkpoint):
    if checkpoint is None:
        raise ValueError('A v2 checkpoint directory is required')
    folder = Path(checkpoint)
    config = json.loads((folder / 'config.json').read_text(encoding='utf-8'))
    if config.get('schema') != SCHEMA or config.get('features') != 'relation':
        raise ValueError('Unsupported v2 checkpoint schema or features')
    with np.load(folder / 'weights.npz', allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    if 'mean' not in arrays or 'scale' not in arrays:
        raise ValueError('V2 checkpoint is missing its fitting-only scaler')
    _validate_scaler(arrays['mean'], arrays['scale'])
    validate_state(config['architecture'], {name: value for name, value in arrays.items() if name not in ('mean', 'scale')})
    return config, arrays


def score_with_bundle(bank, embeddings, bundle):
    from .features import build_features
    config, arrays = bundle
    features = build_features(bank, embeddings, 'relation')
    state = {name: value for name, value in arrays.items() if name not in ('mean', 'scale')}
    dtype = np.float64 if config['architecture']['family'] == 'linear' else np.float32
    result = np.empty((len(bank), 4), dtype=dtype)
    # This bounds intermediate hidden states independently of the input bank size.
    for start in range(0, len(bank), 256):
        stop = min(start + 256, len(bank))
        x = ((features[start:stop] - arrays['mean']) / arrays['scale']).astype(dtype)
        result[start:stop] = numpy_forward(x, config['architecture'], state)
    return result
