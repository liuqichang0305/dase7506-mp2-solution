# Independent V2 review

Reviewer: separate AI audit agent, not an instructor or human peer reviewer. Initial review on 2026-10-29 UTC+8; final development-package review completed on 2026-10-30 UTC+8. **Development, freeze, clean reproduction and eight-page report pass the checks described below.** New formal test evaluation and course score updates remain pending an explicit applicable course process. The dated sections retain the review history and its resolved findings.

## Scope and evidence already checked

- Latest official rules and the precise unresolved formal-test gate: `rules_and_sources.md`, retrieved evidence `results/research_v2/rules_evidence.json`. New formal test scoring remains disabled. General submission updates do not themselves establish a post-evaluation redevelopment procedure.
- Preservation: 92 source/asset/log/report checks against the old freeze or `d9585ce4ea52877a448ff146fac5d942ac71b7a1` all matched at review time. See `rules_preservation_check.json`. No old log, release asset, parser, scorer or search guard was changed.
- Shared folds: each of 4,755 train questions appears in exactly one outer holdout; each inner fit/stopping pair is disjoint and partitions its outer training set; no outer holdout intersects either. Verified from actual persisted memberships, not only code inspection: `rules_fold_review.json`.
- `training_v2.py` loads train only. The early-stop scaler is fit on inner-fit questions; outer holdout labels do not choose epochs. A fresh scaler/model is fit on outer training questions for the inner-selected epoch count. Final epochs derive from the three inner choices. No validation/test fit or early-stop path was found in this source review.
- Architecture inspection: shared MLP and residual candidate heads; Deep Sets pools only within the four-candidate question; optional attention has four tokens and no positional embeddings. These implement the intended permutation behavior. Training-only torch imports are lazy. Numeric NPZ loading disables pickle, validates parameter keys/shapes/finiteness and requires the new schema.
- Selector inspection: invalid parses have group ID -1, not a shared answer value; valid groups use the unchanged official parser. Candidate/group-max are equivalent. Group-LSE has explicit alpha/tau, deterministic original-index ties, valid-mask enforcement and index-0 all-None fallback. Group CE distinguishes masked/normalized aggregation from original listwise; alpha=0, tau=1 and all-valid is the equivalence condition. Multi-positive candidates in the correct answer group are not made negative examples.
- New `code/tools/check_contract_v2.py` independently copies v2 inference, parser, package and assets into a directory without official data/references. It guards external reads, reference reads, writes, sockets and subprocesses. It checks arbitrary N/IDs, question/candidate ordering, actual group-decision ties, None behavior, Python-int indices, actual chosen-group confidence, no torch import and native thread-pool count. Four audit-logic tests passed (`rules_contract_unit_tests_clean.log`). The zero-weight synthetic fixture passed the full harness (`rules_contract_synthetic_fixture.json`); this is infrastructure verification, not final-model accuracy evidence.

## Findings and resolution requirements

1. **Old audit harness could audit the wrong method.** The original tool hardcodes `solution.py` and uses raw-candidate ties. Resolved by adding the separate v2 harness above; old tool preserved unchanged. V2 tie attribution uses winning-group logits and within-group maxima, with a documented positional all-None exception.
2. **Preregistered all-None wording differed from code.** The initial plan said first score maximum while selector used original index 0. Reported to the implementation lead before neural experiments; resolved in the plan as fixed original index 0, matching implementation and audit.
3. **Hard-loss regularization is not universally equivalent.** Linear full-to-hard normalization requires the fraction in the actual fitting subset. The implementation computes that fraction. The plan now explicitly distinguishes AdamW decay scaling as an experimental control from exact linear objective rescaling. The report must retain this distinction. Minibatch hard-loss training logs weighted by total batch size are batch-weighted optimization summaries, not an exact full-dataset hard mean.
4. **Export evidence must include trained finalists.** Synthetic and fit-row checks are useful but do not alone prove that all final validation choices match torch. Require full validation score tolerance and choice consistency before claiming the export preserves the trained selector, especially near ties.

## Final sign-off still required

Check actual completed experiment registrations, shared-fold summaries for all main families, fixed-seed stability, decoder/loss/architecture ablations and any ensemble complementarity gate. Check that the finalist lock precedes validation access, no search restarts after confirmation, all table numbers originate from logs, and missing experiments are marked unrun. Check final assets including configurations against the 16 MiB limit, fresh-process prediction-plus-scoring time, isolated final-model inference, immutable V2 freeze hashes, clean-environment validation reproduction, <=10-page V2 report and PR. No new test result, score update or peer-review completion is approved by this initial review.

## Follow-up audit at 2026-10-29 09:23 UTC

`results/research_v2/rules_oof_review_1.json` records an independent reconstruction of 16 completed experiments at this time. Each has all three common folds, the same fold-manifest SHA, consistent registered configuration, byte-consistent saved fold and concatenated OOF scores, and recomputed full/hard correctness counts matching its summary. All recorded training-source hashes match the current core files. The audit read train OOF arrays only. Stability and ablations were still in progress, so this is not a final count of experiments.

The completed families include MLP, ResMLP, Deep Sets and four-candidate attention, plus old linear, decoder/objective controls and stronger regularization. The primary analysis has 26 rows but 13 fitted configurations: decoder variants reuse models and are correctly labelled as additional decoding analyses, not independent training runs. The linear full/hard normalization control has identical selected correctness events, which is consistent with the scaled-objective identity and is not presented as a gain.

`check_export_v2.py` was reviewed: it permits only validation candidate inputs and the manifest under the official data directory, rejects gold/other-split reads, reconstructs the trained torch model from numeric assets, compares every candidate score and every chosen index, and records group margins. It reports no accuracy. Its synthetic smoke test and model tests are separate from the future trained-finalist full-validation check.

The one-pair ensemble analysis correctly requires common aligned train OOF predictions, fixed deployment seed, bidirectional error complementarity and hard oracle gain before computing an equal group-probability mixture. The oracle is labelled diagnostic; the pair/weights lock prevents changing to another pair under a new destination. A selected deployment ensemble must still have its entire wrapper, asset collection and inference interface audited, not merely the members separately.

Budget deviation is material and retained: `attention_regularized_s1` returned successfully after 1124.3837074 seconds, exceeding the initial batch's 900-second cap; its fitting logs sum to 1098.3893508 seconds. The parent stopped before another experiment and resumed only the three still-pending fixed-seed configurations. `BUDGET_DEVIATION.md` documents the cap breach rather than claiming strict enforcement. The original log and completed predictions remain intact. The broader 60-minute additional budget was not yet exhausted at this review.

Two confirmation-tool improvements were requested before any finalist validation: verify an existing `VALIDATION_OPENED.json` refers to the same lock hash on resume; and filter/stop for failed asset/time checks rather than interpreting a subprocess `status=passed` as resource eligibility. `evaluate.py` distinguishes process success from `local_under_30_seconds`, which must be checked explicitly. No finalist lock, new-model validation-open marker or new validation summary existed at the audit timestamp.

## Development sign-off at 2026-10-29 09:46 UTC

**Development evidence passes; immutable source freeze and clean public-clone reproduction remain pending.** Full machine-readable verification: `results/research_v2/rules_development_final_review.json`; fixed-seed and ensemble checks: `rules_stability_ensemble_review.json`.

All **19 fitted configurations completed the same three outer folds**. Registered core source hashes still match, and their saved per-fold scores reconstruct the OOF arrays and all reported full/hard correct counts. Attention and Deep Sets each retain all three fixed seeds 3036900957/58/59. Final deployment uses the predeclared seed 3036900957, not the best seed. The two final diagnostic ablations were completed before the search closed and were explicitly ineligible as new finalists.

The one equal-probability ensemble passed the preregistered complementarity gate: the members uniquely corrected 71 and 38 of each other's questions; the hard oracle gain was 1.7975 percentage points. Its actual hard result was 1674/2114, versus 1687/2114 for fixed-seed Attention, so it was rejected. Its stored probabilities exactly match the half-and-half aligned group-probability rule. No extra pair, weight search or deployment wrapper is needed.

The sequence is verified from saved records and hashes:

| Event | UTC |
|---|---|
| Last CV configuration complete | 09:31:07.797814 |
| Two finalist configurations selected; search closed | 09:32:41.286922 |
| Final Attention training complete | 09:33:47.942515 |
| Final Deep Sets training complete | 09:34:30.421836 |
| Final inference assets and source bytes sealed | 09:34:31.826350 |
| New-model validation access opened | 09:35:57.841761 |
| Validation confirmation complete | 09:36:06.208803 |

The selection-file hash in the final lock, the lock hash in the validation-open marker, all four finalist asset hashes and all eight inference-source hashes match actual files. Exactly two new models were evaluated on validation. Both identified confirmation-tool findings were fixed before this access: resumed confirmation checks the original lock hash; resource eligibility requires both asset and local runtime checks.

The audit independently checked the existing selected-index predictions against validation references after the locked confirmation, without running new model predictions or changing a choice rule:

| Finalist | Full validation | Hard | Single positive | Fresh prediction + scorer | All inference assets |
|---|---:|---:|---:|---:|---:|
| Attention | 738/798 | 295/355 | 70/117 | 1.7623151 s | 1,130,539 bytes |
| Deep Sets | 740/798 | 297/355 | 68/117 | 1.7397279 s | 963,670 bytes |

The selected Deep Sets model follows the recorded hard-count/full-count/resource rule. Against frozen v1 it gains 16 full-validation questions and 15 hard questions. These are development results; comparing two finalists and earlier OOF selection still creates selection optimism. The two-question validation difference between the new finalists is not proof of an underlying universal superiority.

For **both trained finalists**, the complete 798-question NumPy/Torch export audit has 798 identical choices, no scores outside tolerance and maximum absolute score error 1.9073486328125e-6. The complete validation isolation audits pass with one native thread, no forbidden reads/network/processes/writes, no torch import, and the actual chosen-group confidence event. Recorded research training finished before both resource measurements began. This establishes no overlap with recorded project training; it is not an exhaustive OS activity trace or reference-Xeon certification.

The 92 old release/source/log/report preservation checks pass again. No new test was run or read, no course Issue was updated, and no claim about a new leaderboard rank is justified. The only remaining sign-off items are the committed V2 freeze, clean no-retraining reproduction and final delivery artifacts/PR.

## Final development-package sign-off, 2026-10-30 UTC+8

The immutable source freeze passes independent verification. All 18 source files and both selected assets match both the working bytes and the Git objects at `36f9617175321a2861e78bb2f4ffcde643c4fc03`. The selected checkpoint is `artifacts/research_v2/final_deepsets`; the freeze explicitly sets `formal_test_allowed=false`. See `results/research_v2/rules_freeze_review.json`.

The separate public clone and newly created environment were inspected directly, in addition to reading the reproduction summary. The clone is at the exact frozen commit and all 20 frozen source/asset hashes match. Its actual installed package inventory is only `numpy==2.5.3` (apart from environment tooling), and its official data directory contains validation inputs/references, manifest and attribution files, with no train or test files. The preserved checkpoint, without retraining, reproduces **740/798**, and all **798 selected indices** exactly equal the original confirmation. Fresh prediction plus independent official scoring takes **0.7323425 seconds** in that run; the earlier **1.7397279 seconds** remains preserved. The isolated inference contract passes in the NumPy-only environment. Thread environment settings are all one; optional native-pool introspection is not installed there, so that check is not falsely claimed for the new environment. The original inference environment independently measured one native pool thread. Evidence: `REPRODUCTION_RESULT.json`, `clean-reproduction/`, `fresh-reproduction-contract.json`, and the independent `rules_reproduction_review.json`.

The final `reports/research_v2/report_v2.pdf` contains **eight pages**, within the course's ten-page maximum. All eight rendered pages were visually inspected with no clipping, overlap, broken tables or missing glyphs observed. Report tables and text were checked against saved experiment, validation, bootstrap, asset, timing and reproduction records; the source-artifact hashes in its build record match. It retains the failed objective/context/ensemble hypotheses, the batch-budget deviation, the stronger Attention OOF evidence versus the small validation advantage selecting Deep Sets, the prior v1 test exposure, and the limits of local resource checks. `rules_report_review.json` records the exact reviewed PDF hash. The initial report remains unchanged.

**No unresolved implementation or evidence issue blocks a reviewable development PR.** This conclusion covers the frozen development package and reproducibility; it does not authorize new test evaluation, promise a leaderboard rank, certify the reference Xeon machine, update the course grade, or mark instructor-opened peer review complete. Publishing the reviewable PR is the delivery step; the old formal Issue #294 and its matching immutable release remain intact.

Final online rule check completed at 2026-10-30 00:08:19 UTC+8: the official repository commit is unchanged. Website HEAD advanced to `63ee1cb984b9d3229196a14fc0d2312ba7704dd6` for a leaderboard snapshot, while all five checked submission/workflow/publication files retain identical blob hashes. MP2 peer review is still unreleased. This metadata check did not reopen research or use new leaderboard results; the pending formal-test gate remains unchanged.
