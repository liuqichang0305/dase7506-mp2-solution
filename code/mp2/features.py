"""Deterministic features from passed inputs only; never reads labels or IDs."""
from collections import Counter
import re

import numpy as np
from common import parse_answer


def validate_inputs(bank, embeddings):
    e = np.asarray(embeddings)
    if e.shape != (len(bank), 4, 1024):
        raise ValueError('Expected embeddings of shape [N, 4, 1024]')
    if not np.issubdtype(e.dtype, np.number) or not np.isfinite(e).all():
        raise ValueError('Embeddings must be finite numeric values')
    for row in bank:
        if len(row['candidates']) != 4 or any(not isinstance(c['text'], str) for c in row['candidates']):
            raise ValueError('Each question requires four candidate texts')
    return e.astype(np.float64, copy=False)


def parsed_answers(bank):
    return [[parse_answer(c['text']) for c in r['candidates']] for r in bank]


def hard_mask(bank):
    return np.array([len({a for a in row if a is not None}) >= 2 for row in parsed_answers(bank)], dtype=bool)


def majority_choices(bank):
    result = []
    for answers in parsed_answers(bank):
        counts = Counter(a for a in answers if a is not None)
        votes = [counts[a] if a is not None else 0 for a in answers]
        result.append(int(np.argmax(votes)))
    return result


def build_features(bank, embeddings, mode):
    e = validate_inputs(bank, embeddings)
    if mode == 'embedding':
        return e
    if mode != 'relation':
        raise ValueError(f'Unknown feature mode: {mode}')
    n = len(bank)
    if n == 0:
        return np.empty((0, 4, 2064), dtype=np.float64)
    mean = e.mean(axis=1, keepdims=True)
    normed = e / np.maximum(np.linalg.norm(e, axis=-1, keepdims=True), 1e-12)
    cosine = np.einsum('nid,njd->nij', normed, normed)
    peer = cosine.copy()
    for j in range(4):
        peer[:, j, j] = np.nan
    stats = np.zeros((n, 4, 16), dtype=np.float64)
    stats[:, :, 0] = np.nanmean(peer, axis=2)
    stats[:, :, 1] = np.nanmin(peer, axis=2)
    stats[:, :, 2] = np.nanmax(peer, axis=2)
    for i, (row, answers) in enumerate(zip(bank, parsed_answers(bank))):
        counts = Counter(a for a in answers if a is not None)
        maximum = max(counts.values(), default=0)
        lengths = np.array([len(c['text']) for c in row['candidates']], dtype=float)
        for j, candidate in enumerate(row['candidates']):
            text = candidate['text']
            a = answers[j]
            count = counts[a] if a is not None else 0
            stats[i, j, 3:] = [
                float(a is not None), count / 4, float(count == maximum and count > 0),
                len(counts) / 4, sum(v == maximum for v in counts.values()) / 4,
                np.log1p(len(text)), np.log1p(len(text.split())),
                np.log1p(len(re.findall(r'\d+(?:\.\d+)?', text))),
                np.log1p(text.count('\n')), float('####' in text), float('\\boxed{' in text),
                np.log1p(text.count('=')), np.log1p(len(text)) - np.log1p(lengths).mean(),
            ]
    return np.concatenate([e, np.abs(e - mean), stats], axis=-1)
