DASE7506 / MINI-PROJECT 2

Selecting Reliable
LLM Reasoning

A small linear verifier with candidate comparison and a multi-positive objective

Student ID: 3036900957  |  GitHub: liuqichang0305

Evidence status: frozen official test aggregate included. Generated 2026-10-29 14:44 UTC+8

Abstract

We select one of four supplied mathematical solutions using a shared regularized linear scorer. A six-configuration training-only screen precedes three learned validation comparisons. The selected relation/listwise model scores 724/798 (90.7268%) on validation and 282/355 (79.4366%) on its predefined disagreement subset. The embedding-only BCE baseline scores 89.5990%; the 1.1278 percentage-point gain has a paired bootstrap interval crossing zero. Adding relation features with BCE is a negative result. A separate training-OOF calibrator estimates selected-answer correctness. The method uses no online generation, extra pretrained model or answer lookup.

The frozen method obtains 348/467 = 74.51820128% in the official full-test scorer. This is a final-answer selection score, not certification of all reasoning steps.

Task and protocol

The official bank supplies 4,755 training, 798 validation and 467 test questions, each with four candidate texts and fixed 1,024-dimensional whole-trace representations. Every question has at least one correct candidate and may have several. predict(bank, arrays, checkpoint) returns Python integers 0..3 in input order. The unchanged official numeric parser defines correctness, including equivalent fractions and decimals.

Only train is used to fit weights, standardization and calibration. Validation selects among prespecified candidates and supports diagnostic analysis. Test is reserved for the frozen method. The predefined validation-hard subset contains at least two distinct non-null parsed answers, matching a stated test construction condition without using test content. Its distribution need not match test.

Official release: bb801f4f2e7629d270df765ab12c333b2dc616bb
Protocol: dase7506-mp2-open-hard-verifier-v5

1. Method and learning objective

From frozen representations to a shared score

The embedding baseline uses x_j = e_j in R^1024. The relation model concatenates e_j, the elementwise absolute difference |e_j - mean(e)|, and 16 input-derived statistics, producing 2,064 features per candidate. All candidates share one linear map. Train-only per-feature means and standard deviations standardize each vector; near-constant scales are set to one.

Feature block | Inputs and interpretation

1,024 embedding coordinates | Official whole-trace representation; contains reasoning information.

1,024 absolute deviations | Candidate distance from its question's group mean, coordinate by coordinate.

3 cosine statistics | Mean, minimum and maximum similarity with the other three candidates.

5 answer statistics | Parseability, answer frequency, majority membership, distinct-answer count and number of tied largest groups.

8 text statistics | Log character/word/number/newline/equality counts, two final-answer markers, and relative log character length.

Multi-positive supervision

A candidate is positive exactly when its non-null official parsed answer equals the aligned training reference. BCE retains all positives. The selected alternative minimizes the average group loss below plus (lambda/2)||w||^2, with lambda = 0.1:

L_i = logsumexp(s_i0, s_i1, s_i2, s_i3) - logsumexp({s_ij : y_ij = 1})

A numerically stable implementation subtracts maxima separately for both sums. All-positive questions contribute zero listwise data loss and gradient; this is an objective property. The optimization uses L-BFGS-B from zero initialization, at most 200 iterations. No neural network or exported pretrained weight is used. Inference applies NumPy operations and takes the first maximum on ties.

Confidence is a separate prediction

Three grouped training folds generate out-of-fold scores with preprocessing fitted inside each fold. A small L2-regularized logistic calibrator (lambda = 0.01) uses the maximum score, top-two margin and selected-answer agreement fraction to predict whether the selected answer is correct. It fits 4,755 OOF events. It never fits validation and does not change candidate choices. A four-way maximum softmax is not treated as a correctness probability.

2. Experiments and selection

A fixed grouped 80/20 training split (3,804/951 questions; seed 3036900957) compares three families at L2 strengths {0.01, 0.1}. The best inner-holdout setting per family is fitted on all train, then scored on validation with the official evaluator. Full validation accuracy is the selection criterion; hard accuracy and resources are diagnostics. Two deterministic baselines use candidate 0 and majority parsed answer; invalid parses do not vote and ties use the first index.

Method | Correct / n | Val. % | Hard % | Train s

Candidate 0 | 624 / 798 | 78.20 | 54.93 | none

Answer majority | 688 / 798 | 86.22 | 69.01 | none

Embedding / BCE (.01) | 715 / 798 | 89.60 | 77.18 | 3.335

Relation / BCE (.1) | 704 / 798 | 88.22 | 73.52 | 3.425

Relation / listwise (.1) | 724 / 798 | 90.73 | 79.44 | 2.971

Hard-subset denominator: 355 questions for every row. All rows use identical development inputs and parser. Training seconds cover the model fit as instrumented; complete search wall time, including screening, final fits, validation processes and OOF calibration, is reported below.

Ablation and negative results

Adding relation/text features under BCE reduces validation accuracy from 89.60% to 88.22%. Switching the relation model to the multi-positive listwise objective reaches 90.73%. This supports examining the objective jointly with the features; it does not establish a causal contribution for each of the 16 statistics. The selected model is kept without an ensemble. Whole-trace embeddings remain in all learned variants, so none is a question-and-answer-only ablation.

The relation/listwise L2=.01 inner-screen run reaches the 200-iteration cap without convergence and is not selected. Eleven of twelve selector fits converge; the chosen full fit and its three OOF fits converge. The failed optimizer status is retained in the experiment log.

Uncertainty and stability

Versus answer majority: +4.5113 percentage points; paired 95% percentile interval [2.3810, 6.6416] (2000 question-level resamples; seed 3036900959).

Versus embedding BCE: +1.1278 percentage points; paired 95% percentile interval [-0.7519, 3.1328] (2000 question-level resamples; seed 3036900959).

The interval against the learned baseline includes zero. Intervals are descriptive after model selection, not corrected for selection bias. Chosen-model OOF accuracies are 89.4637%, 88.4543%, 88.6435%. These folds establish limited stability, not independent repetitions of the whole search.

3. Reliability of the selected answer

On 798 validation questions, the train-OOF calibrator has Brier score 0.067713 and ECE 0.021821. ECE uses 10 equal-width bins [lower, upper), with 1 included in the final bin; empty bins are omitted. These measure final-answer correctness of the selected candidate. They do not measure correctness of intermediate steps.

Figure: results/reliability/reliability.png

Left: validation accuracy against predicted probability; annotations give bin sample counts. Right: errors among the highest-confidence fraction. The official task still requires a prediction for every question; the curve is diagnostic, not an abstention-based submitted score.

Confidence bin | n | Mean p | Answer accuracy %

[0.1, 0.2) | 2 | 0.188 | 0.00

[0.3, 0.4) | 1 | 0.391 | 100.00

[0.4, 0.5) | 36 | 0.464 | 47.22

[0.5, 0.6) | 28 | 0.548 | 75.00

[0.6, 0.7) | 15 | 0.649 | 73.33

[0.7, 0.8) | 73 | 0.769 | 73.97

[0.8, 0.9) | 69 | 0.835 | 85.51

[0.9, 1.0] | 574 | 0.968 | 97.74

Calibration did not fit this validation set, but selection of the underlying model used validation. The OOF models also fit fewer questions than the final model, so their confidence distribution can shift. Small-bin estimates are noisy. High estimated confidence does not guarantee correctness; failure examples below retain this limitation.

4. Data composition and failures

Diagnostic | Train | Validation

Questions | 4755 | 798

Correct candidates: 1 / 2 / 3 / 4 | 711 / 732 / 1070 / 2242 | 117 / 123 / 179 / 379

Expected random selection % | 75.46 | 75.69

Parseable candidate % | 95.42 | 95.08

Hard questions | 2114 | 355

Random selection is about 75.7% on validation because many groups contain several positives; 25% would be an incorrect baseline. Train/validation diagnostics are not fitting statistics for a new model.

Validation subgroup | n | Selected % | Majority %

validation-hard | 355 | 79.44 | 69.01

not hard | 443 | 99.77 | 100.00

any parse failure | 126 | 83.33 | 84.92

all candidates parse | 672 | 92.11 | 86.46

1 correct candidates | 117 | 50.43 | 23.08

2 correct candidates | 123 | 87.80 | 83.74

3 correct candidates | 179 | 99.44 | 100.00

4 correct candidates | 379 | 100.00 | 100.00

Interpretation

Most residual failures occur in disagreement groups. With exactly one correct candidate, selection accuracy is 50.43%, versus 23.08% for majority, leaving substantial room for error. On groups with any parse failure, the selected model is slightly worse than majority. The complete subset table, including length groups and method agreement, is preserved in reliability/summary.json. These overlapping subsets must not be added together.

Agreement by candidate index differs from agreement by final answer: several indices may produce the same answer. The saved diagnostic table explicitly labels index agreement to avoid confusing the two events.

5. Examples and resource cost

Validation examples: explanatory AI inspection

These observations use validation records already scored by the official parser. They are hypotheses about failure mechanisms, not new supervision, a correction of the benchmark, or a human verification of reasoning. No method was changed in response to these report examples.

Validation row 29 | model wrong, majority correct | confidence 0.447

The candidate lists 6 cracked eggs, then groups 3 broken eggs with them as 9 cracked. This suggests category confusion despite locally valid arithmetic. The selected parsed answer is 6; the official reference is 9.

Validation row 3 | model correct, majority wrong | confidence 0.562

The selected answer matches the official reference (13), but the text invents 9:00-9:40 and calls it a 1-hour-20-minute interval. A successful final answer therefore does not certify every statement.

Validation row 586 | additional high-confidence error | confidence 0.954

The candidate finds 70 boys present, then subtracts 70 from the total 250 while omitting girls from the present count. Its parsed answer is 180; the official reference is 40, despite estimated confidence around 0.95.

A further paperclip example (row 32) exposes a possible wording/reference interpretation tension. We retain the official reference and do not adjudicate or relabel it. Full questions and selected traces are saved in failure_cases.md.

Measured local costs

Quantity | Recorded value

Hardware / OS | Intel Core i9-12900H; Windows 11; 20 logical CPUs. CPU training and inference; GPU unused.

Complete development search | 56.069 s wall time

Selected model full training fit | 2.971 s

Selected model validation prediction + scoring | 1.497 s, fresh sequential processes

Additional inference assets | 51,923 bytes (0.049518 MiB), audited

Final test prediction + scoring | 0.706 s, local measurement

The official limits are 16 MiB uncompressed and 30 s on a four-thread Xeon Platinum 8457C. All scientific pools are configured to one thread and processes run sequentially; threadpool evidence is retained separately. Local timing is not reference-machine certification. No inference network request, generation, pretrained-weight download, or reference-file read is needed.

6. Frozen result and reproduction

Official frozen test metric | Value

Selection accuracy (%) | 74.51820128479658

Correct / complete bank | 348 / 467

Score evidence | results\test-score.json

Full code freeze commit | 8602992842d444524ab4148991da2f1dc88f316f

Only this frozen method is reported on test. Test performance was not used to choose features, objective, regularization, calibration, seed, or checkpoint. The official aggregate score was supplied to this report after freezing; this report generator never opens test inputs, references or predictions.

Test deliberately requires answer disagreement, whereas full validation includes easy agreement groups. Validation-hard (79.4366%) is a more relevant diagnostic comparison than full validation (90.7268%), but the same construction condition does not guarantee identical distributions or equal accuracy. The test result was accepted without further tuning.

Freeze evidence: .\submission\PRE_TEST_FREEZE.json. It records exact source and checkpoint hashes, the official data version, seed, selection reason and dependencies. Later documentation/result commits may differ from the source freeze; the inference files and assets must match its hashes.

How to reproduce

Use Python 3.12 and a fresh virtual environment; install the pinned project requirements. Fetch the official release and run its integrity checker. Use code/solution.py with artifacts/final_bundle in the unchanged official predict.py, followed by score_predictions.py. Exact offline commands and hash checks are in README.md. Retraining is optional for reproducing the score; code/train.py --config code/configs/final.json --output <new-directory> --calibrate reproduces the training procedure using train only.

Limits and disclosure

The scorer is linear over fixed whole-trace features and can learn surface heuristics. Comparison/text features are tested as a block; no arithmetic checker or process verifier is implemented. A small, selected validation set limits precision. No audit-bank result or instructor verification is claimed. User understanding and manual review remain for the student; they have not been assumed complete.

Codex supplied substantive implementation, experiment orchestration, literature reading, diagnostics and report writing. Internal review by other AI agents is not independent human approval or teacher certification. Official benchmark utilities, data and parser are reused unchanged with their retained license and attribution. See docs/AI_DISCLOSURE.md and docs/UNDERSTAND_ZH.md.

References

[1] K. Cobbe et al. Training Verifiers to Solve Math Word Problems. arXiv:2110.14168, 2021. https://arxiv.org/abs/2110.14168
[2] X. Wang et al. Self-Consistency Improves Chain of Thought Reasoning in Language Models. ICLR, 2023. https://arxiv.org/abs/2203.11171
[3] C. Guo, G. Pleiss, Y. Sun, K. Q. Weinberger. On Calibration of Modern Neural Networks. ICML, PMLR 70:1321-1330, 2017. https://proceedings.mlr.press/v70/guo17a.html

Literature reading scopes, adaptations and limitations are recorded in docs/literature_review.md. These sources motivated the design; their published gains are not transferred to this task.
