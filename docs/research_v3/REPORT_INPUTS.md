# Evidence-only final report inputs

`reports/research_v3/prepare_report_inputs.py` consumes saved aggregates using only the standard library. It imports no training/inference module and does not render PDF. Actual invocation waits for the unique final frozen test. Synthetic tests use temporary metadata and opaque prediction/array bytes, never official data.

Output matches the root-owned builder: schema `mp2-v3-final-report-inputs-v1`, `title`, `pages:[{blocks:[{kind,text,...}]}]`, and `evidence_sha256:{repository_relative_path:sha256}`. Additional `derived_aggregates` and `final_facts` make descriptive arithmetic inspectable. Nine content pages cover saved counts, all registered rounds, mechanism controls, seed stability, stored bootstrap, fixed fusion, confidence, resources, reproduction and disclosure. A tenth page embeds the two saved-evidence PNGs. Actual PDF pagination and every rendered page still require QA; any overflow must be revised before publication.

| Optional flag | Default repository-relative aggregate |
|---|---|
| `--analysis` | `results/research_v3/analysis/final_search/analysis.json` |
| `--v2-analysis` | `results/research_v2/analysis-final/analysis.json` |
| `--fusion-analysis` | `results/research_v3/fixed_fusion_analysis.json` |
| `--confidence-fit` | `results/research_v3/confidence_calibration/report.json` |
| `--confidence-validation` | `results/research_v3/confidence_validation/report.json` |
| `--figures` | `reports/research_v3/figures/figure_manifest.json` |

The corresponding complete Markdown experiment table, common folds, terminal queue, every round decision, fixed-fusion registry/diagnostics, selected-confidence preparation, source notes and fixed final records are required. The generator checks the entire ten-file test output inventory, selected/validation/freeze/lock/review/reproduction bindings, source/assets and final-fit records. Binary OOF/events/predictions are hash-bound only, never decoded.

Run `prepare_figures.py` on the saved final analysis/confidence records first. Its manifest schema is `mp2-v3-report-figures-v1`, with `scope=saved_final_evidence_only`, `evidence_sha256` covering the exact same four aggregate inputs and its generator source, and two `figures` entries containing path/hash, height 235 points, caption and typed generation-record reference. The report input generator binds both PNGs and receipts and refuses synthetic-preview scope or figures made from a different analysis or confidence report.

Both per-figure receipts must also use the production scope and `mp2-v3-figure-generation-v1` schema. Their fixed PNG paths/order, byte hashes, captions, height, PNG format, no-raw-read flag, current generator hash and exact two-input source subsets must agree. A valid top-level manifest cannot promote a synthetic or differently sourced generation receipt.

Reproduction retains typed `score_result`, `timing_result` and `environment_result` references. Confidence requires all ten bins and saved risk-coverage arrays, unchanged choices, exact selector, fixed C/seed/counts and explicit train-fit `IN-SAMPLE` interpretation. Historical V2 fallback is supported.

`results/research_v3/selected_validation_comparison.json` is also mandatory. It supplies saved question-paired overall/hard comparisons against frozen V1 and the historical V2 selection. The generator checks selected-model identity/counts, the original manifest hash, 5,000 repetitions, seeds 3036901059/1060, source/evidence hashes and the explicit post-selection validation caveat. It displays the saved intervals without loading predictions or recomputing bootstrap. This report-only analysis is not an additional model selection or official scoring run.

Only descriptive arithmetic is performed: percentages, sums of existing fit/attempt costs, and seed mean/sample SD for identical configurations. Grouping removes only `seed`, `id`, `experiment_id`; other configuration differences stay separate. Names never supply hyperparameters. The highest hard-mean primary-seed representative within each registered round is explicitly selection-optimistic; the complete table preserves all methods. Stored bootstrap intervals are reused without reading labels.

Every final-analysis row is reconciled with its hash-bound original summary: identity/config/seed, pooled counts, individual fold counts, fold mean/SD, recorded fit cost, parameter count and numeric bytes. All preserved V2 analysis input hashes are checked. The original R01 ledger uses legacy `id/full_correct/full_n` keys and omitted `analysis_sha256`; it remains unchanged. The generator binds the current saved analysis, cross-checks its values against the final summaries, and explicitly records/prints that a historical hash receipt was missing. It does not pretend to reconstruct that missing past receipt. Round representatives come from the canonical table and queue, not missing legacy fields.

Defaults outputs are `reports/research_v3/report_inputs.json` and `reports/research_v3/understand_final_zh.md`; both must be absent. `--output` and `--chinese-output` allow a new review version without overwriting evidence. The explanation draft `docs/research_v3/UNDERSTAND_ZH.md` is preserved.

Missing final evidence, partial test, hash changes, nonterminal jobs, missing rounds, inconsistent identity and official-bank/background access all fail explicitly. The tool cannot authorize new training, testing, selection, networking or peer review. Initial synthetic verification: eight tests passed in 1.055 seconds, `results/research_v3/report-inputs-synthetic-tests.log`; subsequent integration receipts are separate.

After figure, paired-validation and independent-audit corrections, 13/13 synthetic tests passed in 2.370 seconds (`report-inputs-audit-repaired-tests.log`). These include the untouched legacy-ledger case, secondary-cost inconsistency, canonical path traversal rejection, calibration identity, stale figures, complete-output binding and refusal to overwrite. No production report inputs, holdout evaluation or PDF were generated.

The final typed-figure integration passed 14/14 synthetic tests in 3.207 seconds (`report-inputs-final-typed-figures-tests.log`), adding independently hashed receipt scope/source/path rejection. Earlier receipts remain preserved.
