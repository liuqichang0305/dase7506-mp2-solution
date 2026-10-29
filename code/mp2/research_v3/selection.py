"""Candidate/member score decoding without labels or data access."""
import numpy as np

from ..selection_v2 import group_probabilities, select_candidates

COLLAPSED_DECODER = {'mode': 'group_lse', 'alpha': 0., 'tau': 1., 'mask': True}


def oof_decoder(config):
    return dict(COLLAPSED_DECODER if config['family'] == 'tabm' else config['decoder'])


def member_probabilities(raw_scores, groups, decoder):
    raw_scores = np.asarray(raw_scores)
    if raw_scores.ndim == 2:
        return group_probabilities(raw_scores, groups, alpha=decoder.get('alpha', 0.), tau=decoder.get('tau', 1.))
    if raw_scores.ndim != 3 or raw_scores.shape[0] != len(groups) or raw_scores.shape[2] != 4 or raw_scores.shape[1] < 1:
        raise ValueError('Member scores require [N,K,4]')
    if decoder.get('mode', 'group_lse') != 'group_lse' or not decoder.get('mask', True):
        raise ValueError('TabM probability averaging requires masked group-LSE member decoding')
    probabilities = [group_probabilities(raw_scores[:, i], groups, alpha=decoder.get('alpha', 0.), tau=decoder.get('tau', 1.))
                     for i in range(raw_scores.shape[1])]
    return np.mean(probabilities, axis=0)


def collapse_scores(raw_scores, groups, decoder):
    """Represent a member probability average as equivalent 2D candidate logits.

    Place each group's log probability on member zero's best candidate; other
    candidates get a finite negligible score. Group-LSE(alpha=0,tau=1) then
    recovers the group probability without a second within-group LSE rounding.
    Probabilities below float64 tiny are floored only to keep saved scores finite.
    """
    raw_scores = np.asarray(raw_scores)
    if raw_scores.ndim == 2:
        if raw_scores.shape != np.asarray(groups).shape or not np.isfinite(raw_scores).all():
            raise ValueError('Invalid candidate scores')
        return raw_scores.copy()
    probability = member_probabilities(raw_scores, groups, decoder)
    first = np.asarray(raw_scores[:, 0], dtype=np.float64)
    groups = np.asarray(groups)
    result = np.full(groups.shape, -1e30, dtype=np.float64)
    for group in range(4):
        member = groups == group
        active = member.any(axis=1)
        representative = np.where(member, first, -np.inf).argmax(axis=1)
        rows = np.flatnonzero(active)
        result[rows, representative[rows]] = np.log(np.maximum(probability[rows, group], np.finfo(np.float64).tiny))
    if not np.isfinite(result).all():
        raise ValueError('Member probability collapse produced nonfinite scores')
    return result


def select_raw(raw_scores, groups, decoder):
    raw_scores = np.asarray(raw_scores)
    if raw_scores.ndim == 2:
        return select_candidates(raw_scores, groups, **decoder)
    probability = member_probabilities(raw_scores, groups, decoder)
    groups = np.asarray(groups)
    valid = groups >= 0
    expanded = np.take_along_axis(probability, np.maximum(groups, 0), axis=1)
    first = np.where(valid, expanded, -np.inf).argmax(axis=1)
    winner = groups[np.arange(len(groups)), first]
    chosen = np.where(valid & (groups == winner[:, None]), raw_scores[:, 0], -np.inf).argmax(axis=1)
    return [int(j) for j in np.where(valid.any(axis=1), chosen, 0)]


def member_loss(raw_scores, labels, groups, **kwargs):
    """Mean of per-member objectives, never objective of averaged scores."""
    from ..selection_v2 import torch_loss
    def single(scores):
        if kwargs.get('objective') != 'pairwise':
            return torch_loss(scores, labels, groups, **kwargs)
        import torch
        import torch.nn.functional as F
        if scores.shape != labels.shape or groups.shape != labels.shape or scores.ndim != 2 or scores.shape[1] != 4:
            raise ValueError('Pairwise loss requires aligned [N,4] tensors')
        if bool(torch.any((labels > 0) & (groups < 0))):
            raise ValueError('Positive pairwise candidates must have valid parsed answers')
        positive = (labels > 0) & (groups >= 0)
        negative = (labels == 0) & (groups >= 0)
        pairs = positive[:, :, None] & negative[:, None, :]
        count = pairs.sum(dim=(1, 2))
        losses = (F.softplus(scores[:, None, :] - scores[:, :, None]) * pairs).sum(dim=(1, 2)) / count.clamp_min(1)
        reduction = kwargs.get('reduction', 'full')
        if reduction == 'full':
            return losses.mean() if len(losses) else scores.sum() * 0
        if reduction != 'hard':
            raise ValueError('Unknown pairwise reduction')
        membership = groups[:, :, None] == torch.arange(4, device=groups.device)[None, None, :]
        hard = membership.any(dim=1).sum(dim=1) >= 2
        return (losses * hard).sum() / hard.sum().clamp_min(1)
    if raw_scores.ndim == 2:
        return single(raw_scores)
    if raw_scores.ndim != 3 or raw_scores.shape[0] != labels.shape[0] or raw_scores.shape[-1] != 4:
        raise ValueError('TabM loss requires [N,K,4] scores')
    return sum(single(raw_scores[:, i]) for i in range(raw_scores.shape[1])) / raw_scores.shape[1]
