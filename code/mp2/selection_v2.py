"""Answer-group selection and training objectives, using the official parser.

Group IDs are local to each question, ordered by first occurrence. Invalid
parses have ID -1, never a shared answer group. Inference uses no references.
The all-unparseable fallback is original index 0, reported by diagnostics.
"""
import numpy as np

from common import parse_answer


def answer_groups(bank):
    """Return int64 [N,4] group IDs, or -1 for an unparseable candidate."""
    groups = np.full((len(bank), 4), -1, dtype=np.int64)
    for i, row in enumerate(bank):
        candidates = row['candidates']
        if len(candidates) != 4 or any(not isinstance(c['text'], str) for c in candidates):
            raise ValueError('Each question requires four candidate texts')
        seen = {}
        for j, candidate in enumerate(candidates):
            answer = parse_answer(candidate['text'])
            if answer is not None:
                if answer not in seen:
                    seen[answer] = len(seen)
                groups[i, j] = seen[answer]
    return groups


def _checked(scores, groups, alpha=0.0, tau=1.0):
    scores = np.asarray(scores, dtype=np.float64)
    groups = np.asarray(groups)
    if scores.ndim != 2 or scores.shape[1:] != (4,) or groups.shape != scores.shape:
        raise ValueError('Scores and groups must both have shape [N,4]')
    if not np.isfinite(scores).all():
        raise ValueError('Scores must be finite')
    if not np.issubdtype(groups.dtype, np.integer) or np.any((groups < -1) | (groups > 3)):
        raise ValueError('Group IDs must be integers in {-1,0,1,2,3}')
    if not np.isfinite(alpha) or not np.isfinite(tau) or tau <= 0:
        raise ValueError('alpha must be finite and tau must be finite and positive')
    return scores, groups


def _group_terms(scores, groups, alpha, tau):
    # Last two dimensions are candidate, group. Empty groups use safe finite
    # intermediates; only the final group logits contain -inf.
    member = groups[:, :, None] == np.arange(4)[None, None, :]
    count = member.sum(axis=1)
    active = count > 0
    scaled = scores / tau
    maxima = np.where(member, scaled[:, :, None], -np.inf).max(axis=1)
    maxima = np.where(active, maxima, 0.0)
    exponent = np.exp(np.where(member, scaled[:, :, None] - maxima[:, None, :], -np.inf))
    sums = exponent.sum(axis=1)
    safe_sums = np.where(active, sums, 1.0)
    logits = np.where(active, maxima + np.log(safe_sums) - alpha * np.log(np.maximum(count, 1)), -np.inf)
    within = exponent / safe_sums[:, None, :]
    return logits, within, active


def group_logits(scores, groups, alpha=0.0, tau=1.0):
    """Padded [N,4] group LSE minus alpha*log(size); missing groups are -inf."""
    scores, groups = _checked(scores, groups, alpha, tau)
    return _group_terms(scores, groups, alpha, tau)[0]


def _group_softmax(logits, active):
    maximum = np.max(logits, axis=1, keepdims=True)
    maximum = np.where(active.any(axis=1, keepdims=True), maximum, 0.0)
    weights = np.exp(logits - maximum)
    return weights / np.maximum(weights.sum(axis=1, keepdims=True), 1.0)


def group_probabilities(scores, groups, alpha=0.0, tau=1.0):
    """Normalized group probabilities; all-unparseable rows return all zeros.

    These are model probabilities, not calibrated correctness probabilities.
    Correlated trajectories are not assumed to supply independent evidence.
    """
    scores, groups = _checked(scores, groups, alpha, tau)
    logits, _, active = _group_terms(scores, groups, alpha, tau)
    return _group_softmax(logits, active)


def select_candidates(scores, groups, mode='candidate_max', alpha=0.0, tau=1.0, mask=True):
    """Return original candidate indices as Python ints.

    Max ties use the smallest original candidate index. Group-LSE ties use the
    group whose smallest original candidate index is lowest, then its highest
    scoring candidate (smallest index on a within-group tie). This tie policy
    intentionally cannot be permutation equivariant on indistinguishable ties.
    """
    scores, groups = _checked(scores, groups, alpha, tau)
    if mode not in ('candidate_max', 'group_max', 'group_lse'):
        raise ValueError(f'Unsupported selector: {mode}')
    if mode != 'candidate_max' and not mask:
        raise ValueError('Answer-group selection requires the parseability mask')
    if mode == 'candidate_max' and not mask:
        return [int(j) for j in scores.argmax(axis=1)]
    valid = groups >= 0
    if mode in ('candidate_max', 'group_max'):
        # Algebraically equivalent, including the cross-group tie convention.
        chosen = np.where(valid, scores, -np.inf).argmax(axis=1)
    else:
        logits, _, _ = _group_terms(scores, groups, alpha, tau)
        # Repeat group scores on members so a group-level tie is resolved by
        # its first original candidate, independent of arbitrary group IDs.
        expanded = np.take_along_axis(logits, np.maximum(groups, 0), axis=1)
        expanded = np.where(valid, expanded, -np.inf)
        first = expanded.argmax(axis=1)
        winner = groups[np.arange(len(groups)), first]
        in_winner = valid & (groups == winner[:, None])
        chosen = np.where(in_winner, scores, -np.inf).argmax(axis=1)
    chosen = np.where(valid.any(axis=1), chosen, 0)
    return [int(j) for j in chosen]


def selection_details(scores, groups, mode='candidate_max', alpha=0.0, tau=1.0, mask=True):
    """Diagnostics aligned with the actual chosen answer, not max softmax.

    Margin is chosen group probability minus the largest other group
    probability. For a unique group it is 1, for no group 0; it may be negative
    if candidate-max disagrees with group-LSE. An unparseable raw selection has
    probability 0 and margin minus the largest valid group probability.
    """
    scores, groups = _checked(scores, groups, alpha, tau)
    choices = select_candidates(scores, groups, mode, alpha, tau, mask)
    probability = group_probabilities(scores, groups, alpha, tau)
    selected_group = groups[np.arange(len(groups)), np.asarray(choices, dtype=int)]
    chosen_prob = np.zeros(len(groups), dtype=np.float64)
    good = selected_group >= 0
    chosen_prob[good] = probability[np.flatnonzero(good), selected_group[good]]
    others = probability.copy()
    others[np.flatnonzero(good), selected_group[good]] = 0.0
    return {'chosen': choices, 'chosen_group_probability': chosen_prob,
            'group_probability_margin': chosen_prob - others.max(axis=1),
            'all_unparseable_rows': np.flatnonzero((groups < 0).all(axis=1)).tolist()}


def _checked_labels(labels, scores, groups, objective):
    labels = np.asarray(labels, dtype=np.float64)
    if labels.shape != scores.shape or not np.isfinite(labels).all() or not np.isin(labels, (0, 1)).all():
        raise ValueError('Labels must be binary [N,4]')
    if np.any(labels.sum(axis=1) < 1):
        raise ValueError('Every training question requires a positive candidate')
    if objective == 'group':
        if np.any((labels > 0) & (groups < 0)):
            raise ValueError('A positive candidate must have a valid parsed answer')
        member = groups[:, :, None] == np.arange(4)[None, None, :]
        count = member.sum(axis=1)
        positive = (member * labels[:, :, None]).sum(axis=1)
        if np.any((positive > 0) & (positive != count)) or np.any((positive > 0).sum(axis=1) != 1):
            raise ValueError('Exactly one answer group must be wholly correct')
    return labels


def score_loss_gradient(scores, labels, groups, objective='group', alpha=0.0, tau=1.0, reduction='full'):
    """Mean unregularized question loss and derivative with respect to scores.

    Group CE always excludes invalid parses. Legacy listwise and BCE keep all
    four candidates. 'hard' averages only questions with >=2 valid groups;
    zero hard questions return zero data loss/gradient. L2/AdamW belongs to the
    caller and must be reconsidered when changing the loss denominator.
    """
    scores, groups = _checked(scores, groups, alpha, tau)
    labels = _checked_labels(labels, scores, groups, objective)
    if objective not in ('group', 'listwise', 'bce') or reduction not in ('full', 'hard'):
        raise ValueError('Unsupported objective or reduction')
    n = len(scores)
    if n == 0:
        return 0.0, np.zeros_like(scores)
    member = groups[:, :, None] == np.arange(4)[None, None, :]
    active = member.any(axis=1)
    selected = np.ones(n, dtype=bool) if reduction == 'full' else active.sum(axis=1) >= 2
    normalizer = max(int(selected.sum()), 1)
    if objective == 'group':
        logits, within, active = _group_terms(scores, groups, alpha, tau)
        probability = _group_softmax(logits, active)
        gold = (member * labels[:, :, None]).sum(axis=1) > 0
        maximum = logits.max(axis=1, keepdims=True)
        shifted = logits - maximum
        losses = np.log(np.exp(shifted).sum(axis=1)) - np.where(gold, shifted, 0.0).sum(axis=1)
        derivative = ((probability - gold)[:, None, :] * within).sum(axis=2) / tau
    elif objective == 'listwise':
        maximum = scores.max(axis=1, keepdims=True)
        weights = np.exp(scores - maximum)
        positive_scores = np.where(labels > 0, scores, -np.inf)
        positive_maximum = positive_scores.max(axis=1, keepdims=True)
        positive_weights = np.exp(positive_scores - positive_maximum)
        sums = weights.sum(axis=1, keepdims=True)
        positive_sums = positive_weights.sum(axis=1, keepdims=True)
        losses = (maximum + np.log(sums) - positive_maximum - np.log(positive_sums))[:, 0]
        derivative = weights / sums - positive_weights / positive_sums
    else:
        losses = (np.logaddexp(0.0, scores) - labels * scores).mean(axis=1)
        exponent = np.exp(-np.abs(scores))
        sigmoid = np.where(scores >= 0, 1.0 / (1.0 + exponent), exponent / (1.0 + exponent))
        derivative = (sigmoid - labels) / 4
    return float(losses[selected].sum() / normalizer), derivative * (selected / normalizer)[:, None]


def torch_loss(scores, labels, groups, objective='group', alpha=0.0, tau=1.0, reduction='full'):
    """Differentiable equivalent of score_loss_gradient; PyTorch is training-only.

    Callers must supply aligned binary labels and group IDs from answer_groups.
    Empty group LSE is evaluated on safe zero inputs before masking its final
    logit, avoiding undefined gradients from logsumexp([-inf,...,-inf]).
    """
    import torch
    import torch.nn.functional as F

    if scores.ndim != 2 or scores.shape[1] != 4 or labels.shape != scores.shape or groups.shape != scores.shape:
        raise ValueError('Scores, labels and groups require [N,4]')
    if objective not in ('group', 'listwise', 'bce') or reduction not in ('full', 'hard'):
        raise ValueError('Unsupported objective or reduction')
    if not np.isfinite(alpha) or not np.isfinite(tau) or tau <= 0:
        raise ValueError('Invalid alpha or tau')
    if scores.shape[0] == 0:
        return scores.sum() * 0
    member = groups[:, :, None] == torch.arange(4, device=groups.device)[None, None, :]
    count = member.sum(dim=1)
    active = count > 0
    if objective == 'group':
        scaled = scores[:, :, None] / tau
        masked = scaled.masked_fill(~member, -torch.inf)
        safe = torch.where(active[:, None, :], masked, torch.zeros_like(masked))
        logits = torch.logsumexp(safe, dim=1) - alpha * count.clamp_min(1).to(scores.dtype).log()
        logits = logits.masked_fill(~active, -torch.inf)
        gold = (member * labels[:, :, None]).sum(dim=1) > 0
        # Official training has one positive group. Raise clearly for malformed
        # callers instead of returning an all-None NaN objective.
        if not bool(torch.all(gold.sum(dim=1) == 1)) or bool(torch.any((labels > 0) & (groups < 0))):
            raise ValueError('Exactly one valid gold answer group is required')
        positive_count = (member * labels[:, :, None]).sum(dim=1)
        if bool(torch.any((positive_count > 0) & (positive_count != count))):
            raise ValueError('Answer-group labels must be consistent')
        selected_logits = torch.where(gold, logits, torch.zeros_like(logits)).sum(dim=1)
        losses = torch.logsumexp(logits, dim=1) - selected_logits
    elif objective == 'listwise':
        if not bool(torch.all((labels > 0).any(dim=1))):
            raise ValueError('Every training question requires a positive candidate')
        losses = torch.logsumexp(scores, dim=1) - torch.logsumexp(scores.masked_fill(labels <= 0, -torch.inf), dim=1)
    else:
        losses = F.binary_cross_entropy_with_logits(scores, labels.to(scores.dtype), reduction='none').mean(dim=1)
    if reduction == 'full':
        return losses.mean()
    hard = active.sum(dim=1) >= 2
    return (losses * hard.to(scores.dtype)).sum() / hard.sum().clamp_min(1)
