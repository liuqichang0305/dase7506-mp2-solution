"""Offline V3 inference: NumPy only, explicit bundles, no reference-data access.

TabM averages answer-group probabilities across its members. An optional frozen
manifest supports two or three equally weighted model probabilities. The first
model's first member supplies the representative index inside the winning group.
"""
import os

# Set before NumPy import in a fresh grading process. A caller that has already
# imported numerical libraries remains responsible for its process thread cap.
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
              'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS',
              'OMP_THREAD_LIMIT'):
    os.environ[_name] = '1'

import json
from pathlib import Path
import numpy as np

from mp2 import models_v2
from mp2.research_v3 import models
from mp2.research_v3.selection import (COLLAPSED_DECODER, collapse_scores,
                                      member_probabilities, select_raw)
from mp2.selection_v2 import answer_groups

ENSEMBLE_SCHEMA = 'mp2-research-v3-ensemble'


def load_inference_checkpoint(checkpoint):
    """Load one numeric bundle or a self-contained, fixed equal-probability set.

    Ensemble config: schema, aggregation='equal_group_probability',
    representative='first_model_first_member', members=['member0','member1'].
    Members must be distinct directories below the manifest directory. Nested
    ensembles, arbitrary weights, absolute paths and external assets are refused.
    """
    if checkpoint is None:
        raise ValueError('An explicit frozen checkpoint is required')
    folder = Path(checkpoint).resolve()
    config = json.loads((folder / 'config.json').read_text(encoding='utf-8'))
    if config.get('schema') == models.SCHEMA:
        return {'kind': 'single', 'path': folder, 'members': [('v3', models.load_bundle(folder))]}
    if config.get('schema') == models_v2.SCHEMA:
        return {'kind': 'single', 'path': folder, 'members': [('v2', models_v2.load_bundle(folder))]}
    if config.get('schema') != ENSEMBLE_SCHEMA:
        raise ValueError('Unsupported inference checkpoint schema')
    if set(config) != {'schema', 'aggregation', 'representative', 'members'}:
        raise ValueError('Unsupported manifest fields; only fixed equal-probability members are allowed')
    if config.get('aggregation') != 'equal_group_probability' or config.get('representative') != 'first_model_first_member':
        raise ValueError('Unsupported ensemble decision rule')
    members = config.get('members')
    if not isinstance(members, list) or len(members) not in (2, 3):
        raise ValueError('An equal-probability manifest requires two or three members')
    loaded, seen = [], set()
    for member in members:
        if not isinstance(member, str) or Path(member).is_absolute():
            raise ValueError('Manifest members must be relative directory paths')
        target = (folder / member).resolve()
        if target == folder or not target.is_relative_to(folder) or target in seen:
            raise ValueError('Manifest members must be distinct contained directories')
        seen.add(target)
        child = load_inference_checkpoint(target)
        if child['kind'] != 'single':
            raise ValueError('Nested ensembles are not supported')
        item = child['members'][0]
        decoder = item[1][0]['decoder']
        if decoder.get('mode') != 'group_lse' or not decoder.get('mask', True):
            raise ValueError('Ensemble members require masked group-probability decoding')
        loaded.append(item)
    return {'kind': 'ensemble', 'path': folder, 'manifest': config, 'members': loaded}


def checkpoint_asset_paths(checkpoint):
    """All actual numeric/config files required by the frozen inference entry."""
    loaded = load_inference_checkpoint(checkpoint)
    folder = loaded['path']
    if loaded['kind'] == 'single':
        return [folder / 'config.json', folder / 'weights.npz']
    files = [folder / 'config.json']
    for member in loaded['manifest']['members']:
        files.extend([folder / member / 'config.json', folder / member / 'weights.npz'])
    return files


def _score(bank, arrays, item):
    version, bundle = item
    scorer = models.score_with_bundle if version == 'v3' else models_v2.score_with_bundle
    return scorer(bank, arrays['embeddings'], bundle)


def _probability_choice(probability, anchor, groups):
    """Exact deterministic group-probability tie, then first-model score tie."""
    valid = groups >= 0
    expanded = np.take_along_axis(probability, np.maximum(groups, 0), axis=1)
    first = np.where(valid, expanded, -np.inf).argmax(axis=1)
    winner = groups[np.arange(len(groups)), first]
    in_group = valid & (groups == winner[:, None])
    chosen = np.where(in_group, anchor, -np.inf).argmax(axis=1)
    return [int(x) for x in np.where(valid.any(axis=1), chosen, 0)]


def _probability_scores(probability, anchor, groups):
    """Finite 2D diagnostics whose group-LSE recovers the fused probabilities."""
    result = np.full(groups.shape, -1e30, dtype=np.float64)
    for group in range(4):
        member = groups == group
        active = member.any(axis=1)
        representative = np.where(member, anchor, -np.inf).argmax(axis=1)
        rows = np.flatnonzero(active)
        result[rows, representative[rows]] = np.log(np.maximum(probability[rows, group], np.finfo(np.float64).tiny))
    return result


def prediction_details(bank, arrays, checkpoint):
    """Analysis information derived exclusively from supplied candidate inputs."""
    loaded = load_inference_checkpoint(checkpoint)
    groups = answer_groups(bank)
    probabilities, raw_scores = [], []
    for item in loaded['members']:
        raw = _score(bank, arrays, item)
        raw_scores.append(raw)
        probabilities.append(member_probabilities(raw, groups, item[1][0]['decoder']))
    probability = np.mean(probabilities, axis=0)
    first = raw_scores[0]
    anchor = np.asarray(first[:, 0] if first.ndim == 3 else first, dtype=np.float64)
    if loaded['kind'] == 'single':
        decoder = loaded['members'][0][1][0]['decoder']
        chosen = select_raw(first, groups, decoder)
        scores = collapse_scores(first, groups, decoder)
        effective_decoder = dict(COLLAPSED_DECODER if first.ndim == 3 else decoder)
    else:
        chosen = _probability_choice(probability, anchor, groups)
        scores = _probability_scores(probability, anchor, groups)
        effective_decoder = dict(COLLAPSED_DECODER)
    chosen_array = np.asarray(chosen, dtype=int)
    selected_group = groups[np.arange(len(groups)), chosen_array]
    confidence = np.zeros(len(groups), dtype=np.float64)
    valid = selected_group >= 0
    confidence[valid] = probability[np.flatnonzero(valid), selected_group[valid]]
    other = probability.copy()
    other[np.flatnonzero(valid), selected_group[valid]] = 0.
    return {'chosen': chosen, 'chosen_group_probability': confidence,
            'group_probability_margin': confidence - other.max(axis=1),
            'group_probabilities': probability, 'candidate_scores': scores,
            'representative_scores': anchor, 'effective_decoder': effective_decoder,
            'all_unparseable_rows': np.flatnonzero((groups < 0).all(axis=1)).tolist(),
            'raw_member_scores': raw_scores, 'kind': loaded['kind']}


def score_members(bank, arrays, checkpoint):
    """Raw [N,4] or [N,K,4] arrays, one per manifest model (one for singles)."""
    loaded = load_inference_checkpoint(checkpoint)
    return [_score(bank, arrays, member) for member in loaded['members']]


def score_candidates(bank, arrays, checkpoint):
    """2D diagnostic logits; decisions use the exact probability-average rule."""
    return prediction_details(bank, arrays, checkpoint)['candidate_scores']


def predict(bank, arrays, checkpoint):
    return prediction_details(bank, arrays, checkpoint)['chosen']


def predict_confidence(bank, arrays, checkpoint):
    """Uncalibrated probability of the actually selected numeric answer group."""
    return prediction_details(bank, arrays, checkpoint)['chosen_group_probability']
