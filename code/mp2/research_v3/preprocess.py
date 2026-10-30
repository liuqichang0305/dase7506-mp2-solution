"""Fold-fitted numeric preprocessing; no data IO and no implicit batch refitting.

Robust modes follow the median/IQR, range fallback and smooth-clipping equations
in RealMLP. Only the caller-supplied fitting partition is used by fit_preprocessor.
"""
from __future__ import annotations

import numpy as np

DIM = 2064
MODES = ('standard', 'robust_stats', 'robust_all')


def _features(features):
    values = np.asarray(features, dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] != (4, DIM) or not np.isfinite(values).all():
        raise ValueError('Preprocessing expects finite [N,4,2064] features')
    return values


def fit_preprocessor(features, mode='standard'):
    if mode not in MODES:
        raise ValueError(f'Unsupported preprocessing: {mode}')
    values = _features(features)
    if not len(values):
        raise ValueError('Cannot fit preprocessing on an empty partition')
    flat = values.reshape(-1, DIM)
    mean, scale = flat.mean(axis=0), flat.std(axis=0)
    scale[scale < 1e-8] = 1.0
    clip_mask = np.zeros(DIM, dtype=bool)
    constant_mask = np.zeros(DIM, dtype=bool)
    if mode != 'standard':
        clip_mask[-16:] = True
        if mode == 'robust_all':
            clip_mask[:] = True
        chosen = flat[:, clip_mask]
        median = np.median(chosen, axis=0)
        quartiles = np.quantile(chosen, [0.25, 0.75], axis=0)
        denominator = quartiles[1] - quartiles[0]
        fallback = 0.5 * (chosen.max(axis=0) - chosen.min(axis=0))
        denominator = np.where(denominator == 0, fallback, denominator)
        constant_mask[clip_mask] = denominator == 0
        mean[clip_mask] = median
        scale[clip_mask] = np.where(denominator == 0, 1.0, denominator)
    return {'mode': mode, 'mean': mean, 'scale': scale,
            'clip_mask': clip_mask, 'constant_mask': constant_mask}


def validate_preprocessor(preprocess):
    if preprocess.get('mode') not in MODES:
        raise ValueError('Invalid preprocessing mode')
    for key in ('mean', 'scale'):
        value = np.asarray(preprocess[key])
        if value.shape != (DIM,) or value.dtype.kind != 'f' or not np.isfinite(value).all():
            raise ValueError(f'Invalid preprocessing array: {key}')
    if np.any(preprocess['scale'] <= 0):
        raise ValueError('Preprocessing scales must be positive')
    for key in ('clip_mask', 'constant_mask'):
        value = np.asarray(preprocess[key])
        if value.shape != (DIM,) or not np.isin(value, (0, 1)).all():
            raise ValueError(f'Invalid preprocessing mask: {key}')
    expected = np.zeros(DIM, bool)
    if preprocess['mode'] == 'robust_stats':
        expected[-16:] = True
    elif preprocess['mode'] == 'robust_all':
        expected[:] = True
    if not np.array_equal(preprocess['clip_mask'], expected):
        raise ValueError('Smooth-clipping mask does not match preprocessing mode')
    if np.any(np.asarray(preprocess['constant_mask'], bool) & ~expected):
        raise ValueError('Only robust-transformed constant dimensions may be zeroed')


def transform_features(features, preprocess, dtype=np.float32):
    validate_preprocessor(preprocess)
    values = _features(features)
    result = (values - preprocess['mean']) / preprocess['scale']
    result[..., np.asarray(preprocess['constant_mask'], bool)] = 0.0
    mask = np.asarray(preprocess['clip_mask'], bool)
    # hypot implements z/sqrt(1+(z/3)^2) without squaring large finite z.
    selected = result[..., mask]
    result[..., mask] = selected / np.hypot(1.0, selected / 3.0)
    result = result.astype(dtype)
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite transformed features')
    return result


def preprocess_from_bundle(config, arrays):
    return {'mode': config['preprocess']['mode'], 'mean': arrays['mean'], 'scale': arrays['scale'],
            'clip_mask': arrays['prep_clip_mask'].astype(bool),
            'constant_mask': arrays['prep_constant_mask'].astype(bool)}
