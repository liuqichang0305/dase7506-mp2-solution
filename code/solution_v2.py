"""Research-v2 interface: offline, NumPy-only, with numeric answer-group decoding."""
from mp2.models_v2 import load_bundle, score_with_bundle
from mp2.selection_v2 import answer_groups, select_candidates, selection_details


def score_candidates(bank, arrays, checkpoint):
    return score_with_bundle(bank, arrays['embeddings'], load_bundle(checkpoint))


def predict(bank, arrays, checkpoint):
    bundle = load_bundle(checkpoint)
    scores = score_with_bundle(bank, arrays['embeddings'], bundle)
    return select_candidates(scores, answer_groups(bank), **bundle[0]['decoder'])


def predict_confidence(bank, arrays, checkpoint):
    """Uncalibrated probability of the actually selected answer group.

    This is a model-derived analysis signal, not a guaranteed correctness
    probability. It deliberately does not reuse the old candidate-max calibrator.
    """
    bundle = load_bundle(checkpoint)
    scores = score_with_bundle(bank, arrays['embeddings'], bundle)
    details = selection_details(scores, answer_groups(bank), **bundle[0]['decoder'])
    return details['chosen_group_probability']
