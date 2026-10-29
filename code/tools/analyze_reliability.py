"""Validation-only reliability diagnostics; never reads or selects on test."""
from __future__ import annotations

import os
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_key] = "1"

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

CODE = Path(__file__).resolve().parents[1]
ROOT = CODE.parent
sys.path.insert(0, str(CODE))
from common import DATA, load_bank, load_inputs, parse_answer, read_jsonl
from mp2.features import hard_mask, majority_choices, parsed_answers
from solution import predict_confidence, score_candidates


def labels_for_development(bank, split):
    if split not in ("train", "validation"):
        raise ValueError("This analysis supports only train/validation")
    refs = read_jsonl(DATA / f"gsm8k_{split}.jsonl")
    if len(refs) != len(bank):
        raise ValueError("Reference length mismatch")
    for ref, row in zip(refs, bank):
        if ref["id"] != row["id"] or ref["question"] != row["question"]:
            raise ValueError("Reference alignment mismatch")
    answers = parsed_answers(bank)
    gold = [parse_answer(r["answer"]) for r in refs]
    if any(a is None for a in gold):
        raise ValueError("Reference did not parse")
    y = np.array([[a is not None and a == g for a in row] for row, g in zip(answers, gold)], dtype=bool)
    if not y.any(axis=1).all():
        raise ValueError("A development question has no positive candidate")
    return y, gold, answers


def subset(name, mask, correct, majority):
    n = int(mask.sum())
    return {"name": name, "n": n, "correct": int(correct[mask].sum()),
            "accuracy": float(100 * correct[mask].mean()) if n else None,
            "majority_accuracy": float(100 * majority[mask].mean()) if n else None}


def bootstrap(a, b, seed=3036900957, replicates=5000):
    """Paired question-level resampling, with all four candidates kept together."""
    rng = np.random.default_rng(seed)
    delta = (a.astype(float) - b.astype(float)) * 100
    estimates = np.array([delta[rng.integers(0, len(delta), len(delta))].mean() for _ in range(replicates)])
    return {"difference_percentage_points": float(delta.mean()), "percentile_95_interval": np.quantile(estimates, [.025, .975]).tolist(), "replicates": replicates, "seed": seed, "unit": "question", "interpretation": "Descriptive paired interval; repeated model selection can make validation estimates optimistic."}


def distribution(bank, y, answers):
    lengths = np.array([len(c["text"]) for row in bank for c in row["candidates"]])
    return {"n": len(bank), "correct_candidate_counts": {str(k): int((y.sum(axis=1) == k).sum()) for k in range(1, 5)},
            "random_expected_accuracy": float(100 * y.mean()),
            "parseable_candidate_fraction": sum(a is not None for row in answers for a in row) / (4 * len(bank)),
            "hard_n": int(hard_mask(bank).sum()),
            "candidate_char_length": {"min": int(lengths.min()), "median": float(np.median(lengths)), "mean": float(lengths.mean()), "max": int(lengths.max())},
            "valid_distinct_answer_counts": {str(k): sum(len({a for a in row if a is not None}) == k for row in answers) for k in range(5)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--development", type=Path, default=ROOT / "results/development")
    parser.add_argument("--output", type=Path, default=ROOT / "results/reliability")
    parser.add_argument("--baseline-checkpoint", type=Path)
    args = parser.parse_args()
    checkpoint = args.checkpoint.resolve(strict=True)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    bank, arrays = load_inputs("validation")
    y, gold, answers = labels_for_development(bank, "validation")
    scores = score_candidates(bank, arrays, checkpoint)
    choices = scores.argmax(axis=1)
    confidence = predict_confidence(bank, arrays, checkpoint)
    if confidence.shape != (len(bank),) or not np.isfinite(confidence).all() or np.any((confidence < 0) | (confidence > 1)):
        raise ValueError("Calibrated confidence must be finite probabilities per question")
    correct = y[np.arange(len(bank)), choices]
    vote = np.array(majority_choices(bank))
    vote_correct = y[np.arange(len(bank)), vote]
    hard = hard_mask(bank)
    bins = []
    for i in range(10):
        low, high = i / 10, (i + 1) / 10
        mask = (confidence >= low) & ((confidence < high) if i < 9 else (confidence <= high))
        bins.append({"low": low, "high": high, "n": int(mask.sum()),
                     "mean_confidence": float(confidence[mask].mean()) if mask.any() else None,
                     "accuracy_fraction": float(correct[mask].mean()) if mask.any() else None})
    ece = sum(b["n"] / len(bank) * abs(b["mean_confidence"] - b["accuracy_fraction"]) for b in bins if b["n"])
    order = np.argsort(-confidence, kind="stable")
    coverage = np.arange(1, len(bank) + 1) / len(bank)
    risk = 1 - np.cumsum(correct[order]) / np.arange(1, len(bank) + 1)
    length = np.array([np.mean([len(c["text"]) for c in row["candidates"]]) for row in bank])
    parse_failure = np.array([any(a is None for a in row) for row in answers])
    same_answer = np.array([len({a for a in row if a is not None}) == 1 for row in answers])
    subsets = [subset("all validation", np.ones(len(bank), bool), correct, vote_correct),
               subset("validation-hard", hard, correct, vote_correct),
               subset("not hard", ~hard, correct, vote_correct),
               subset("selected index agrees with majority", choices == vote, correct, vote_correct),
               subset("selected index differs from majority", choices != vote, correct, vote_correct),
               subset("one distinct valid answer", same_answer, correct, vote_correct),
               subset("at least two distinct valid answers", hard, correct, vote_correct),
               subset("any parse failure", parse_failure, correct, vote_correct),
               subset("all candidates parse", ~parse_failure, correct, vote_correct)]
    for k in range(1, 5):
        subsets.append(subset(f"{k} correct candidates", y.sum(axis=1) == k, correct, vote_correct))
    for title, mask in (("mean candidate length < 500 characters", length < 500), ("mean candidate length 500-999 characters", (length >= 500) & (length < 1000)), ("mean candidate length >= 1000 characters", length >= 1000)):
        subsets.append(subset(title, mask, correct, vote_correct))
    case_specs = [("both model and majority wrong", ~correct & ~vote_correct),
                  ("model wrong, majority correct", ~correct & vote_correct),
                  ("model correct, majority wrong", correct & ~vote_correct),
                  ("additional high-confidence error", ~correct)]
    cases, used = [], set()
    for title, mask in case_specs:
        indices = np.flatnonzero(mask)
        if title.startswith("additional high-confidence"):
            indices = indices[np.argsort(-confidence[indices])]
        chosen_i = next((int(i) for i in indices if int(i) not in used), None)
        if chosen_i is None:
            continue
        used.add(chosen_i)
        row = bank[chosen_i]
        cases.append({"category": title, "validation_index": chosen_i, "id": row["id"], "question": row["question"],
                      "selected_index": int(choices[chosen_i]), "majority_index": int(vote[chosen_i]), "selected_final_answer": answers[chosen_i][choices[chosen_i]],
                      "reference_final_answer": gold[chosen_i], "all_candidate_final_answers": answers[chosen_i],
                      "correct_candidate_indices": np.flatnonzero(y[chosen_i]).tolist(), "confidence": float(confidence[chosen_i]),
                      "selected_text": row["candidates"][choices[chosen_i]]["text"],
                      "attribution_status": "Observed answer/selection mismatch only; intermediate reasoning has not been manually verified. No process-supervision label was created."})
    train_bank = load_bank("train")
    train_y, _, train_answers = labels_for_development(train_bank, "train")
    summary = {"split": "validation", "checkpoint": str(checkpoint.relative_to(ROOT)) if checkpoint.is_relative_to(ROOT) else str(checkpoint),
               "checkpoint_hashes": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(checkpoint.iterdir()) if p.is_file()},
               "n": len(bank), "correct": int(correct.sum()), "accuracy": float(100 * correct.mean()),
               "hard_n": int(hard.sum()), "hard_accuracy": float(100 * correct[hard].mean()),
               "majority_accuracy": float(100 * vote_correct.mean()), "brier": float(np.mean((confidence - correct) ** 2)), "ece": float(ece),
               "bins_rule": "10 equal-width bins; [lower,upper), final upper inclusive; empty bins omitted from ECE", "bins": bins,
               "coverage": coverage.tolist(), "risk": risk.tolist(), "aurc": float(risk.mean()),
               "paired_bootstrap_vs_majority": bootstrap(correct, vote_correct), "subsets": subsets,
               "data_diagnostics": {"train": distribution(train_bank, train_y, train_answers), "validation": distribution(bank, y, answers)},
               "case_selection": "First bank-order eligible cases per declared category; highest-confidence error chosen by confidence; one case per question.",
               "cases": cases, "calibration_note": "Loaded from the trained checkpoint. Fitting provenance must be verified against search_summary.json and oof_folds.json; this script performs no fitting.",
               "limitation": "Validation was used for model selection; calibration/effect estimates are descriptive and not an independent final-test claim."}
    if args.baseline_checkpoint:
        base = score_candidates(bank, arrays, args.baseline_checkpoint).argmax(axis=1)
        base_correct = y[np.arange(len(bank)), base]
        summary["paired_bootstrap_vs_learning_baseline"] = bootstrap(correct, base_correct)
        summary["learning_baseline_accuracy"] = float(100 * base_correct.mean())
    np.savez(out / "validation_diagnostics.npz", choices=choices, confidence=confidence, selected_correct=correct, majority_choices=vote)
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    os.environ["MPLCONFIGDIR"] = str(out / ".matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.5), layout="constrained")
    active = [b for b in bins if b["n"]]
    axes[0].plot([0, 1], [0, 1], "--", color="#a3adb9", linewidth=1)
    axes[0].plot([b["mean_confidence"] for b in active], [b["accuracy_fraction"] for b in active], "o-", color="#087f8c")
    for b in active:
        axes[0].annotate(str(b["n"]), (b["mean_confidence"], b["accuracy_fraction"]), xytext=(2, 6), textcoords="offset points", fontsize=8)
    axes[0].set(xlim=(0, 1), ylim=(0, 1.04), xlabel="Mean predicted probability", ylabel="Observed answer accuracy", title="Reliability (annotations: bin n)")
    axes[1].plot(coverage, risk, color="#087f8c", linewidth=1.6)
    axes[1].axhline(1 - correct.mean(), color="#a3adb9", linestyle="--", label="Full-coverage risk")
    axes[1].set(xlim=(0, 1), ylim=(0, max(.5, float(risk.max()) + .03)), xlabel="Coverage (fraction retained)", ylabel="Error rate among retained questions", title="Risk versus coverage")
    axes[1].legend(fontsize=8)
    fig.savefig(out / "reliability.png", dpi=180)
    plt.close(fig)
    (out / "failure_cases.md").write_text("# Validation diagnostic cases\n\nNo intermediate reasoning was manually certified. These are answer-level observations, not additional training labels.\n\n" + "\n\n".join(f"## {c['category']} (validation row {c['validation_index']})\n\nQuestion: {c['question']}\n\nSelected candidate: {c['selected_index']}; parsed answer: {c['selected_final_answer']}; reference: {c['reference_final_answer']}; confidence: {c['confidence']:.4f}.\n\n{c['attribution_status']}\n\n```text\n{c['selected_text']}\n```" for c in cases), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("n", "accuracy", "hard_n", "hard_accuracy", "brier", "ece")}))


if __name__ == "__main__":
    main()
