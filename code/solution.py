"""Frozen candidate selection. Inference reads only passed inputs and checkpoint."""
from mp2.model import load_bundle, score_with_bundle, calibrated_confidence


def score_candidates(bank, arrays, checkpoint):
    return score_with_bundle(bank, arrays['embeddings'], load_bundle(checkpoint))


def predict(bank, arrays, checkpoint):
    scores = score_candidates(bank, arrays, checkpoint)
    return [int(j) for j in scores.argmax(axis=1)]


def predict_confidence(bank, arrays, checkpoint):
    bundle = load_bundle(checkpoint)
    scores = score_with_bundle(bank, arrays['embeddings'], bundle)
    return calibrated_confidence(bank, scores, bundle)
