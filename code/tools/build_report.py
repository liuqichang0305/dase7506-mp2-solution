"""Build a <=10-page English report from recorded development evidence.

Never loads benchmark data or test predictions. A final aggregate official score
is read only when explicitly passed with --official-score. Missing evidence is
labelled pending, never replaced by invented results. Run from the project root.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone, timedelta
from html import escape
import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image, KeepTogether
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[2]
NAVY = colors.HexColor("#17304a")
TEAL = colors.HexColor("#087f8c")
GRAY = colors.HexColor("#536171")
LIGHT = colors.HexColor("#edf4f6")


def read(path, required=False):
    path = Path(path)
    if not path.exists():
        if required:
            raise FileNotFoundError(path)
        return None
    return json.loads(path.read_text(encoding="utf-8-sig"))


def fmt(value, digits=2):
    return "pending" if value is None else f"{value:.{digits}f}"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=ROOT / "reports/report.pdf")
    p.add_argument("--development", type=Path, default=ROOT / "results/development/search_summary.json")
    p.add_argument("--reliability", type=Path, default=ROOT / "results/reliability/summary.json")
    p.add_argument("--official-score", type=Path)
    p.add_argument("--timing", type=Path)
    p.add_argument("--assets", type=Path, default=ROOT / "results/asset-audit.json")
    p.add_argument("--freeze", type=Path, default=ROOT / "submission/PRE_TEST_FREEZE.json")
    args = p.parse_args()
    dev = read(args.development, True)
    rel = read(args.reliability, True)
    final = read(args.official_score, True) if args.official_score else None
    timing = read(args.timing, True) if args.timing else None
    assets, freeze = read(args.assets), read(args.freeze)
    if final and (final.get("split") != "test" or not 0 <= final.get("accuracy", -1) <= 100):
        raise ValueError("Explicit final score must be the official test aggregate JSON")
    selected = next(r for r in dev["validation"] if r["experiment_id"] == dev["recommended_learned"])
    config = dev["recommended_config"]
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="BodyMP2", fontName="Helvetica", fontSize=10, leading=14.4, textColor=NAVY, spaceAfter=8))
    styles.add(ParagraphStyle(name="SmallMP2", parent=styles["BodyMP2"], fontSize=8.4, leading=11, textColor=GRAY, spaceAfter=5))
    styles.add(ParagraphStyle(name="TitleMP2", fontName="Helvetica-Bold", fontSize=27, leading=31, textColor=NAVY, spaceAfter=15))
    styles.add(ParagraphStyle(name="H1MP2", fontName="Helvetica-Bold", fontSize=18, leading=22, textColor=NAVY, spaceAfter=14))
    styles.add(ParagraphStyle(name="H2MP2", fontName="Helvetica-Bold", fontSize=11.5, leading=15, textColor=TEAL, spaceBefore=8, spaceAfter=6))
    styles.add(ParagraphStyle(name="CellMP2", parent=styles["BodyMP2"], fontSize=8.6, leading=11, spaceAfter=0))
    story, editable = [], []
    width = A4[0] - 36 * mm

    def text(value, style="BodyMP2"):
        editable.append(value)
        story.append(Paragraph(escape(str(value)).replace("\n", "<br/>"), styles[style]))

    def heading(title):
        text(title, "H1MP2")

    def sub(title):
        text(title, "H2MP2")

    def table(rows, col_widths=None):
        editable.extend(" | ".join(str(c) for c in r) for r in rows)
        cells = [[Paragraph(escape(str(c)), styles["CellMP2"]) for c in r] for r in rows]
        t = Table(cells, colWidths=col_widths, repeatRows=1, hAlign="LEFT")
        t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), LIGHT), ("TEXTCOLOR", (0, 0), (-1, 0), NAVY),
                              ("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 7),
                              ("RIGHTPADDING", (0, 0), (-1, -1), 7), ("TOPPADDING", (0, 0), (-1, -1), 6),
                              ("BOTTOMPADDING", (0, 0), (-1, -1), 6), ("LINEBELOW", (0, 0), (-1, 0), .7, TEAL),
                              ("LINEBELOW", (0, 1), (-1, -1), .3, colors.HexColor("#d7e0e6"))]))
        story.extend([t, Spacer(1, 7)])

    def new():
        story.append(PageBreak())

    text("DASE7506 / MINI-PROJECT 2", "SmallMP2")
    text("Selecting Reliable\nLLM Reasoning", "TitleMP2")
    text("A small linear verifier with candidate comparison and a multi-positive objective", "H2MP2")
    text("Student ID: 3036900957  |  GitHub: liuqichang0305", "SmallMP2")
    text("Evidence status: " + ("frozen official test aggregate included" if final else "development report; final test aggregate not yet supplied") + ". Generated " + datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M UTC+8"), "SmallMP2")
    sub("Abstract")
    text(f"We select one of four supplied mathematical solutions using a shared regularized linear scorer. A six-configuration training-only screen precedes three learned validation comparisons. The selected relation/listwise model scores {selected['correct']}/{selected['n']} ({selected['accuracy']:.4f}%) on validation and {selected['hard_correct']}/{selected['hard_n']} ({selected['hard_accuracy']:.4f}%) on its predefined disagreement subset. The embedding-only BCE baseline scores 89.5990%; the 1.1278 percentage-point gain has a paired bootstrap interval crossing zero. Adding relation features with BCE is a negative result. A separate training-OOF calibrator estimates selected-answer correctness. The method uses no online generation, extra pretrained model or answer lookup.")
    if final:
        text(f"The frozen method obtains {final['correct']}/{final['n']} = {final['accuracy']:.8f}% in the official full-test scorer. This is a final-answer selection score, not certification of all reasoning steps.")
    sub("Task and protocol")
    text("The official bank supplies 4,755 training, 798 validation and 467 test questions, each with four candidate texts and fixed 1,024-dimensional whole-trace representations. Every question has at least one correct candidate and may have several. predict(bank, arrays, checkpoint) returns Python integers 0..3 in input order. The unchanged official numeric parser defines correctness, including equivalent fractions and decimals.")
    text("Only train is used to fit weights, standardization and calibration. Validation selects among prespecified candidates and supports diagnostic analysis. Test is reserved for the frozen method. The predefined validation-hard subset contains at least two distinct non-null parsed answers, matching a stated test construction condition without using test content. Its distribution need not match test.")
    text("Official release: bb801f4f2e7629d270df765ab12c333b2dc616bb\nProtocol: dase7506-mp2-open-hard-verifier-v5", "SmallMP2")

    new(); heading("1. Method and learning objective")
    sub("From frozen representations to a shared score")
    text("The embedding baseline uses x_j = e_j in R^1024. The relation model concatenates e_j, the elementwise absolute difference |e_j - mean(e)|, and 16 input-derived statistics, producing 2,064 features per candidate. All candidates share one linear map. Train-only per-feature means and standard deviations standardize each vector; near-constant scales are set to one.")
    table([["Feature block", "Inputs and interpretation"], ["1,024 embedding coordinates", "Official whole-trace representation; contains reasoning information."], ["1,024 absolute deviations", "Candidate distance from its question's group mean, coordinate by coordinate."], ["3 cosine statistics", "Mean, minimum and maximum similarity with the other three candidates."], ["5 answer statistics", "Parseability, answer frequency, majority membership, distinct-answer count and number of tied largest groups."], ["8 text statistics", "Log character/word/number/newline/equality counts, two final-answer markers, and relative log character length."]], [width*.32, width*.68])
    sub("Multi-positive supervision")
    text("A candidate is positive exactly when its non-null official parsed answer equals the aligned training reference. BCE retains all positives. The selected alternative minimizes the average group loss below plus (lambda/2)||w||^2, with lambda = 0.1:")
    text("L_i = logsumexp(s_i0, s_i1, s_i2, s_i3) - logsumexp({s_ij : y_ij = 1})", "SmallMP2")
    text("A numerically stable implementation subtracts maxima separately for both sums. All-positive questions contribute zero listwise data loss and gradient; this is an objective property. The optimization uses L-BFGS-B from zero initialization, at most 200 iterations. No neural network or exported pretrained weight is used. Inference applies NumPy operations and takes the first maximum on ties.")
    sub("Confidence is a separate prediction")
    text("Three grouped training folds generate out-of-fold scores with preprocessing fitted inside each fold. A small L2-regularized logistic calibrator (lambda = 0.01) uses the maximum score, top-two margin and selected-answer agreement fraction to predict whether the selected answer is correct. It fits 4,755 OOF events. It never fits validation and does not change candidate choices. A four-way maximum softmax is not treated as a correctness probability.")

    new(); heading("2. Experiments and selection")
    text("A fixed grouped 80/20 training split (3,804/951 questions; seed 3036900957) compares three families at L2 strengths {0.01, 0.1}. The best inner-holdout setting per family is fitted on all train, then scored on validation with the official evaluator. Full validation accuracy is the selection criterion; hard accuracy and resources are diagnostics. Two deterministic baselines use candidate 0 and majority parsed answer; invalid parses do not vote and ties use the first index.")
    rows = [["Method", "Correct / n", "Val. %", "Hard %", "Train s"]]
    for r in dev["validation"]:
        label = {"fixed0": "Candidate 0", "majority": "Answer majority", "embedding-bce-l2-0.01": "Embedding / BCE (.01)", "relation-bce-l2-0.1": "Relation / BCE (.1)", "relation-listwise-l2-0.1": "Relation / listwise (.1)"}.get(r["experiment_id"], r["experiment_id"])
        rows.append([label, f"{r['correct']} / {r['n']}", fmt(r["accuracy"]), fmt(r["hard_accuracy"]), fmt(r.get("training_seconds"), 3) if "training_seconds" in r else "none"])
    table(rows, [width*.37, width*.19, width*.14, width*.14, width*.16])
    text("Hard-subset denominator: 355 questions for every row. All rows use identical development inputs and parser. Training seconds cover the model fit as instrumented; complete search wall time, including screening, final fits, validation processes and OOF calibration, is reported below.", "SmallMP2")
    sub("Ablation and negative results")
    text("Adding relation/text features under BCE reduces validation accuracy from 89.60% to 88.22%. Switching the relation model to the multi-positive listwise objective reaches 90.73%. This supports examining the objective jointly with the features; it does not establish a causal contribution for each of the 16 statistics. The selected model is kept without an ensemble. Whole-trace embeddings remain in all learned variants, so none is a question-and-answer-only ablation.")
    text("The relation/listwise L2=.01 inner-screen run reaches the 200-iteration cap without convergence and is not selected. Eleven of twelve selector fits converge; the chosen full fit and its three OOF fits converge. The failed optimizer status is retained in the experiment log.")
    sub("Uncertainty and stability")
    for key, label in [("majority", "answer majority"), ("embedding-bce-l2-0.01", "embedding BCE")]:
        b = dev["paired_bootstrap"][key]; lo, hi = b["paired_bootstrap_95_interval"]
        text(f"Versus {label}: {b['difference_percentage_points']:+.4f} percentage points; paired 95% percentile interval [{lo:.4f}, {hi:.4f}] ({dev['bootstrap_repetitions']} question-level resamples; seed {dev['bootstrap_seed']}).")
    text("The interval against the learned baseline includes zero. Intervals are descriptive after model selection, not corrected for selection bias. Chosen-model OOF accuracies are " + ", ".join(f"{r['accuracy']:.4f}%" for r in dev["selected_train_only_oof"]) + ". These folds establish limited stability, not independent repetitions of the whole search.")

    new(); heading("3. Reliability of the selected answer")
    text(f"On 798 validation questions, the train-OOF calibrator has Brier score {rel['brier']:.6f} and ECE {rel['ece']:.6f}. ECE uses 10 equal-width bins [lower, upper), with 1 included in the final bin; empty bins are omitted. These measure final-answer correctness of the selected candidate. They do not measure correctness of intermediate steps.")
    chart = args.reliability.parent / "reliability.png"
    if chart.exists():
        story.append(Image(str(chart), width=width, height=width * 3.5 / 9.2))
        editable.append("Figure: results/reliability/reliability.png")
    text("Left: validation accuracy against predicted probability; annotations give bin sample counts. Right: errors among the highest-confidence fraction. The official task still requires a prediction for every question; the curve is diagnostic, not an abstention-based submitted score.", "SmallMP2")
    table([["Confidence bin", "n", "Mean p", "Answer accuracy %"]] + [[f"[{b['low']:.1f}, {b['high']:.1f}" + ("]" if b["high"] == 1 else ")"), b["n"], fmt(b["mean_confidence"], 3), fmt(100*b["accuracy_fraction"]) if b["n"] else "empty"] for b in rel["bins"] if b["n"]], [width*.3, width*.15, width*.25, width*.3])
    text("Calibration did not fit this validation set, but selection of the underlying model used validation. The OOF models also fit fewer questions than the final model, so their confidence distribution can shift. Small-bin estimates are noisy. High estimated confidence does not guarantee correctness; failure examples below retain this limitation.")

    new(); heading("4. Data composition and failures")
    tr, va = rel["data_diagnostics"]["train"], rel["data_diagnostics"]["validation"]
    table([["Diagnostic", "Train", "Validation"], ["Questions", tr["n"], va["n"]], ["Correct candidates: 1 / 2 / 3 / 4", " / ".join(str(tr["correct_candidate_counts"][str(k)]) for k in range(1,5)), " / ".join(str(va["correct_candidate_counts"][str(k)]) for k in range(1,5))], ["Expected random selection %", fmt(tr["random_expected_accuracy"]), fmt(va["random_expected_accuracy"])], ["Parseable candidate %", fmt(100*tr["parseable_candidate_fraction"]), fmt(100*va["parseable_candidate_fraction"])], ["Hard questions", tr["hard_n"], va["hard_n"]]], [width*.48, width*.26, width*.26])
    text("Random selection is about 75.7% on validation because many groups contain several positives; 25% would be an incorrect baseline. Train/validation diagnostics are not fitting statistics for a new model.")
    keep = {"validation-hard", "not hard", "any parse failure", "all candidates parse", "1 correct candidates", "2 correct candidates", "3 correct candidates", "4 correct candidates"}
    table([["Validation subgroup", "n", "Selected %", "Majority %"]] + [[r["name"], r["n"], fmt(r["accuracy"]), fmt(r["majority_accuracy"])] for r in rel["subsets"] if r["name"] in keep], [width*.48, width*.12, width*.2, width*.2])
    sub("Interpretation")
    text("Most residual failures occur in disagreement groups. With exactly one correct candidate, selection accuracy is 50.43%, versus 23.08% for majority, leaving substantial room for error. On groups with any parse failure, the selected model is slightly worse than majority. The complete subset table, including length groups and method agreement, is preserved in reliability/summary.json. These overlapping subsets must not be added together.")
    text("Agreement by candidate index differs from agreement by final answer: several indices may produce the same answer. The saved diagnostic table explicitly labels index agreement to avoid confusing the two events.", "SmallMP2")

    new(); heading("5. Examples and resource cost")
    sub("Validation examples: explanatory AI inspection")
    text("These observations use validation records already scored by the official parser. They are hypotheses about failure mechanisms, not new supervision, a correction of the benchmark, or a human verification of reasoning. No method was changed in response to these report examples.")
    cases = {c["validation_index"]: c for c in rel["cases"]}
    descriptions = {
        29: "The candidate lists 6 cracked eggs, then groups 3 broken eggs with them as 9 cracked. This suggests category confusion despite locally valid arithmetic. The selected parsed answer is 6; the official reference is 9.",
        3: "The selected answer matches the official reference (13), but the text invents 9:00-9:40 and calls it a 1-hour-20-minute interval. A successful final answer therefore does not certify every statement.",
        586: "The candidate finds 70 boys present, then subtracts 70 from the total 250 while omitting girls from the present count. Its parsed answer is 180; the official reference is 40, despite estimated confidence around 0.95."
    }
    for index, explanation in descriptions.items():
        if index in cases:
            c = cases[index]
            text(f"Validation row {index} | {c['category']} | confidence {c['confidence']:.3f}", "H2MP2")
            text(explanation)
    if 32 in cases:
        text("A further paperclip example (row 32) exposes a possible wording/reference interpretation tension. We retain the official reference and do not adjudicate or relabel it. Full questions and selected traces are saved in failure_cases.md.", "SmallMP2")
    sub("Measured local costs")
    table([["Quantity", "Recorded value"], ["Hardware / OS", "Intel Core i9-12900H; Windows 11; 20 logical CPUs. CPU training and inference; GPU unused."], ["Complete development search", f"{dev['wall_seconds']:.3f} s wall time"], ["Selected model full training fit", f"{selected['training_seconds']:.3f} s"], ["Selected model validation prediction + scoring", f"{selected['full_local_wall_seconds']:.3f} s, fresh sequential processes"], ["Additional inference assets", f"{assets['uncompressed_bytes']:,} bytes ({assets['uncompressed_bytes']/2**20:.6f} MiB), audited" if assets else f"Final uncompressed audit pending; recommended bundle physical files: {dev['recommended_asset_file_bytes']:,} bytes"], ["Final test prediction + scoring", f"{timing['total_wall_seconds']:.3f} s, local measurement" if timing else "Not supplied to this report build"]], [width*.45, width*.55])
    text("The official limits are 16 MiB uncompressed and 30 s on a four-thread Xeon Platinum 8457C. All scientific pools are configured to one thread and processes run sequentially; threadpool evidence is retained separately. Local timing is not reference-machine certification. No inference network request, generation, pretrained-weight download, or reference-file read is needed.", "SmallMP2")

    new(); heading("6. Frozen result and reproduction")
    if final:
        table([["Official frozen test metric", "Value"], ["Selection accuracy (%)", repr(final["accuracy"])], ["Correct / complete bank", f"{final['correct']} / {final['n']}"], ["Score evidence", str(args.official_score).replace(str(ROOT), ".")], ["Full code freeze commit", freeze.get("source_commit", "not supplied") if freeze else "not supplied"]], [width*.45, width*.55])
        text("Only this frozen method is reported on test. Test performance was not used to choose features, objective, regularization, calibration, seed, or checkpoint. The official aggregate score was supplied to this report after freezing; this report generator never opens test inputs, references or predictions.")
        text("Test deliberately requires answer disagreement, whereas full validation includes easy agreement groups. Validation-hard (79.4366%) is a more relevant diagnostic comparison than full validation (90.7268%), but the same construction condition does not guarantee identical distributions or equal accuracy. The test result was accepted without further tuning.")
    else:
        text("Final testing is pending in this report build. No test score, final submission, or successful reproduction is claimed. The report can be regenerated from the same frozen evidence with an explicit --official-score argument after the authorized scorer completes.")
    if freeze:
        text("Freeze evidence: " + str(args.freeze).replace(str(ROOT), ".") + ". It records exact source and checkpoint hashes, the official data version, seed, selection reason and dependencies. Later documentation/result commits may differ from the source freeze; the inference files and assets must match its hashes.", "SmallMP2")
    sub("How to reproduce")
    text("Use Python 3.12 and a fresh virtual environment; install the pinned project requirements. Fetch the official release and run its integrity checker. Use code/solution.py with artifacts/final_bundle in the unchanged official predict.py, followed by score_predictions.py. Exact offline commands and hash checks are in README.md. Retraining is optional for reproducing the score; code/train.py --config code/configs/final.json --output <new-directory> --calibrate reproduces the training procedure using train only.")
    sub("Limits and disclosure")
    text("The scorer is linear over fixed whole-trace features and can learn surface heuristics. Comparison/text features are tested as a block; no arithmetic checker or process verifier is implemented. A small, selected validation set limits precision. No audit-bank result or instructor verification is claimed. User understanding and manual review remain for the student; they have not been assumed complete.")
    text("Codex supplied substantive implementation, experiment orchestration, literature reading, diagnostics and report writing. Internal review by other AI agents is not independent human approval or teacher certification. Official benchmark utilities, data and parser are reused unchanged with their retained license and attribution. See docs/AI_DISCLOSURE.md and docs/UNDERSTAND_ZH.md.")
    sub("References")
    text("[1] K. Cobbe et al. Training Verifiers to Solve Math Word Problems. arXiv:2110.14168, 2021. https://arxiv.org/abs/2110.14168\n[2] X. Wang et al. Self-Consistency Improves Chain of Thought Reasoning in Language Models. ICLR, 2023. https://arxiv.org/abs/2203.11171\n[3] C. Guo, G. Pleiss, Y. Sun, K. Q. Weinberger. On Calibration of Modern Neural Networks. ICML, PMLR 70:1321-1330, 2017. https://proceedings.mlr.press/v70/guo17a.html", "SmallMP2")
    text("Literature reading scopes, adaptations and limitations are recorded in docs/literature_review.md. These sources motivated the design; their published gains are not transferred to this task.", "SmallMP2")

    def frame(canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(TEAL); canvas.setLineWidth(.7)
        canvas.line(18*mm, A4[1]-15*mm, A4[0]-18*mm, A4[1]-15*mm)
        canvas.setFont("Helvetica", 8); canvas.setFillColor(GRAY)
        canvas.drawString(18*mm, 11*mm, "DASE7506 MP2  |  3036900957  |  Evidence-based course report")
        canvas.drawRightString(A4[0]-18*mm, 11*mm, str(doc.page))
        canvas.restoreState()

    document = SimpleDocTemplate(str(output), pagesize=A4, leftMargin=18*mm, rightMargin=18*mm, topMargin=22*mm, bottomMargin=20*mm, title="Selecting Reliable LLM Reasoning", author="3036900957; substantive Codex assistance disclosed")
    document.build(story, onFirstPage=frame, onLaterPages=frame)
    reader = PdfReader(output)
    if len(reader.pages) > 10:
        raise ValueError(f"Report exceeds ten pages: {len(reader.pages)}")
    (output.parent / "report_content.md").write_text("\n\n".join(editable) + "\n", encoding="utf-8")
    evidence = {"pages": len(reader.pages), "development": str(args.development), "reliability": str(args.reliability), "official_score": str(args.official_score) if args.official_score else None, "timing": str(args.timing) if args.timing else None, "assets": str(args.assets) if assets else None, "freeze": str(args.freeze) if freeze else None, "visual_qa": "pending render and visual inspection", "output": str(output)}
    (output.parent / "build_record.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    print(json.dumps(evidence))


if __name__ == "__main__":
    main()
