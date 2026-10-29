"""Safe numeric checkpoint loading and NumPy-only inference."""
import json
from pathlib import Path

import numpy as np
from .features import build_features, parsed_answers


def load_bundle(checkpoint):
    if checkpoint is None:
        raise ValueError('A checkpoint directory is required for this learned method')
    folder = Path(checkpoint)
    config = json.loads((folder / 'config.json').read_text(encoding='utf-8'))
    if config.get('schema') != 'mp2-linear-v1':
        raise ValueError('Unsupported checkpoint schema')
    with np.load(folder / 'weights.npz', allow_pickle=False) as data:
        arrays = {k: data[k].copy() for k in data.files}
    d = 1024 if config['features'] == 'embedding' else 2064
    for key in ('mean', 'scale', 'weight'):
        if arrays[key].shape != (d,) or not np.isfinite(arrays[key]).all():
            raise ValueError(f'Invalid checkpoint array: {key}')
    if arrays['bias'].shape != () or not np.isfinite(arrays['bias']).all() or np.any(arrays['scale'] <= 0):
        raise ValueError('Invalid checkpoint scale or bias')
    return config, arrays


def score_with_bundle(bank, embeddings, bundle):
    config, arrays = bundle
    features = build_features(bank, embeddings, config['features'])
    scores = ((features - arrays['mean']) / arrays['scale']) @ arrays['weight'] + arrays['bias']
    if not np.isfinite(scores).all():
        raise ValueError('Nonfinite model scores')
    return scores


def confidence_features(bank, scores):
    if scores.shape != (len(bank), 4):
        raise ValueError('Confidence requires four scores per question')
    ordered = np.sort(scores, axis=1)
    chosen = np.argmax(scores, axis=1)
    agreement = np.array([sum(a is not None and a == row[j] for a in row) / 4
                          for row, j in zip(parsed_answers(bank), chosen)])
    return np.column_stack([ordered[:, -1], ordered[:, -1] - ordered[:, -2], agreement])


def calibrated_confidence(bank, scores, bundle):
    _, arrays = bundle
    if 'calibration_weight' not in arrays:
        raise ValueError('This checkpoint has no trained calibrator')
    x = confidence_features(bank, scores)
    logits = ((x - arrays['calibration_mean']) / arrays['calibration_scale']) @ arrays['calibration_weight'] + arrays['calibration_bias']
    return 1 / (1 + np.exp(-np.clip(logits, -700, 700)))
