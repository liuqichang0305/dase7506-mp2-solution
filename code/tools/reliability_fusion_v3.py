"""Strict reader for a previously registered/completed V3 OOF fusion.

This module does not evaluate new mixtures, fit calibration, select methods or
read official banks. It verifies the saved fixed mixture against its existing
member OOF arrays before returning report-only chosen-answer confidence events.
"""
from __future__ import annotations

from datetime import datetime
import hashlib
import json
from pathlib import Path
import re

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _json(path):
    path = Path(path).resolve()
    if path.is_relative_to((ROOT / 'code/data').resolve()) or any(
            token in path.name.lower() for token in ('leaderboard', 'ranking_background')):
        raise ValueError('Fusion confidence cannot open official banks/ranking files')
    return json.loads(path.read_text(encoding='utf-8'))


def _method_folder(key):
    if not isinstance(key, str) or re.fullmatch(r'v[23]:[a-z0-9_]+', key) is None:
        raise ValueError('Invalid fixed fusion member identity')
    version, name = key.split(':')
    return ROOT / 'results' / ('research_' + version) / name


def _instant(value):
    result = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if result.tzinfo is None:
        raise ValueError('Fusion provenance timestamps must be timezone-aware')
    return result


def load_fusion_oof(folder):
    """Return the same event schema as reliability_v3.load_train_oof."""
    # Lazy import avoids a dispatch/import cycle. Members must be singles.
    from reliability_v3 import chosen_confidence, load_train_oof
    folder = Path(folder).resolve()
    if not folder.is_relative_to((ROOT / 'results/research_v3').resolve()):
        raise ValueError('Fixed V3 fusion evidence must be within its research results directory')
    diagnostic_path = folder / 'diagnostic.json'
    diagnostic = _json(diagnostic_path)
    fusion = diagnostic.get('fusion', {})
    if diagnostic.get('status') != 'complete' or fusion.get('status') != 'complete':
        raise ValueError('No completed fixed fusion exists; not_run is not a confidence source')
    if fusion.get('registration_file') != 'fusion-registration.json':
        raise ValueError('Expected the immutable copied fusion registration')
    registration_path, archive_path = folder / 'fusion-registration.json', folder / 'fusion-oof.npz'
    if _sha(registration_path) != fusion.get('registration_sha256') or _sha(archive_path) != fusion.get('oof_sha256'):
        raise ValueError('Fusion registration/archive hash mismatch')
    registration = _json(registration_path)
    keys = registration.get('methods')
    if not isinstance(keys, list) or len(keys) not in (2, 3) or len(set(keys)) != len(keys):
        raise ValueError('Fusion requires two or three distinct fixed members')
    if (registration.get('status') != 'registered' or registration.get('purpose') != 'single_fixed_equal_probability_oof'
            or registration.get('weights') != [1. / len(keys)] * len(keys)
            or fusion.get('methods') != keys or fusion.get('weights') != registration['weights']):
        raise ValueError('Only the exact preregistered equal-probability fusion is supported')
    descriptions = diagnostic.get('methods', [])
    by_key = {item['method']: item for item in descriptions}
    if len(by_key) != len(descriptions) or any(key not in by_key for key in keys):
        raise ValueError('Missing/duplicated fusion member provenance')
    with np.load(archive_path, allow_pickle=False) as archive:
        if set(archive.files) != {'probabilities', 'choices', 'correct', 'fold', 'groups', 'labels'}:
            raise ValueError('Incomplete or unknown fusion OOF arrays')
        saved = {name: archive[name].copy() for name in archive.files}
    events, raw_scores = [], []
    source_hashes = {name: _sha(folder / name) for name in ('diagnostic.json', 'fusion-registration.json', 'fusion-oof.npz')}
    expected_oof = {}
    for key in keys:
        member_folder = _method_folder(key)
        registration_member = _json(member_folder / 'registration.json')
        summary = _json(member_folder / 'summary.json')
        config = registration_member['config']
        if (summary.get('status') != 'complete' or summary.get('ensemble_evaluated')
                or (member_folder / 'diagnostic.json').exists() or summary['config'] != config
                or config['experiment_id'] != key.split(':')[1] or config['seed'] != 3036900957):
            raise ValueError('A fusion member must be a complete fixed-seed single-model OOF')
        if config['decoder'].get('mode') != 'group_lse' or not config['decoder'].get('mask', True):
            raise ValueError('Fusion member decoder must match masked group-probability deployment')
        for name in ('registration.json', 'summary.json', 'oof.npz'):
            actual = _sha(member_folder / name)
            if by_key[key]['input_sha256'].get(name) != actual:
                raise ValueError('Member OOF provenance changed after fusion')
            source_hashes[key + '/' + name] = actual
        expected_oof[key] = source_hashes[key + '/oof.npz']
        child = load_train_oof(member_folder)
        with np.load(member_folder / 'oof.npz', allow_pickle=False) as archive:
            labels = archive['labels'].copy()
            raw = archive['raw_scores'].copy() if 'raw_scores' in archive.files else archive['scores'].copy()
        if (not np.array_equal(saved['labels'], labels) or not np.array_equal(saved['groups'], child['groups'])
                or not np.array_equal(saved['fold'], child['fold']) or child['folds_sha256'] != diagnostic['folds_sha256']):
            raise ValueError('Fusion/member labels, groups or common folds do not align')
        events.append(child)
        raw_scores.append(raw)
    if registration.get('oof_sha256') != expected_oof:
        raise ValueError('Registration refers to different member OOF predictions')
    gate_reference = registration['gate_evidence']
    relative_gate = Path(gate_reference['path'])
    gate_path = (ROOT / relative_gate).resolve()
    if relative_gate.is_absolute() or not gate_path.is_relative_to((ROOT / 'results/research_v3').resolve()) or _sha(gate_path) != gate_reference['sha256']:
        raise ValueError('Prior gate evidence is outside the allowed scope or changed')
    prior = _json(gate_path)
    prior_hashes = {item['method']: item['input_sha256']['oof.npz'] for item in prior['methods']}
    if (any(prior_hashes.get(key) != value for key, value in expected_oof.items())
            or not any(row['methods'] == keys and row['gate_passed'] for row in prior['complementarity'])
            or not _instant(prior['created_at_utc']) <= _instant(registration['registered_at_utc']) <= _instant(diagnostic['created_at_utc'])):
        raise ValueError('The fixed mixture lacks a prior aligned registered complementarity gate')
    source_hashes['prior_gate_evidence'] = _sha(gate_path)
    member_correct = np.stack([event['correct'] for event in events])
    hard = np.array([len(set(row[row >= 0])) >= 2 for row in saved['groups']])
    unique = [int((member_correct[i] & ~np.delete(member_correct, i, axis=0).any(0)).sum()) for i in range(len(keys))]
    gain = int(member_correct[:, hard].any(0).sum()) - int(member_correct[:, hard].sum(1).max())
    if min(unique) < 10 or not hard.any() or 100 * gain < int(hard.sum()) or not fusion.get('gate', {}).get('gate_passed'):
        raise ValueError('Saved mixture does not satisfy the actual reciprocal/oracle gate')
    probability = np.mean([event['group_probabilities'] for event in events], axis=0)
    if not np.allclose(saved['probabilities'], probability, rtol=1e-12, atol=1e-12):
        raise ValueError('Saved fusion is not the registered equal group-probability average')
    groups = saved['groups']
    valid = groups >= 0
    first = np.where(valid, np.take_along_axis(probability, np.maximum(groups, 0), axis=1), -np.inf).argmax(1)
    winner = groups[np.arange(len(groups)), first]
    anchor = raw_scores[0][:, 0] if raw_scores[0].ndim == 3 else raw_scores[0]
    choices = np.where(valid & (groups == winner[:, None]), anchor, -np.inf).argmax(1)
    choices = np.where(valid.any(1), choices, 0)
    if not np.array_equal(saved['choices'], choices):
        raise ValueError('Saved fusion choices differ from the fixed first-member representative rule')
    correct = saved['labels'][np.arange(len(labels)), choices].astype(bool)
    if not np.array_equal(saved['correct'], correct):
        raise ValueError('Saved fusion correctness differs from its actual selected answer')
    confidence = chosen_confidence(groups, saved['probabilities'], choices)
    return {'groups': groups, 'group_probabilities': saved['probabilities'], 'choices': choices,
            'correct': correct, 'confidence': confidence, 'fold': saved['fold'],
            'selector': {'kind': 'fixed_equal_probability', 'specification': registration},
            'split': 'train_oof', 'source_sha256': source_hashes,
            'folds_sha256': diagnostic['folds_sha256'], 'oof_directory': str(folder)}
