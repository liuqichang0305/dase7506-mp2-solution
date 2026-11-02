# Post-deadline research archive report

This separate workflow records the missed formal deadline and the actual completed research. It does not create a test result, new score submission or automatic artifact replacement. The original `prepare_report_inputs.py` / `build_report.py` remain TEST_COMPLETE-gated. The V1 Issue294 and its historical348/467 are expressly distinguished from the archived V3 model.

New files:

- `reports/research_v3/prepare_closeout_inputs.py`: standard-library aggregate reader; no models, raw banks, array decoding, fitting, scoring, networking or PDF rendering.
- `reports/research_v3/build_closeout.py`: separate evidence validator and PDF builder using the preserved V2 typography. It recomputes report text from saved aggregate records and rejects edited pages or a formal-report schema.
- `code/tests/test_closeout_report_v3.py`: temporary synthetic metadata, opaque array/prediction bytes, no actual evaluation or PDF.

The input producer waits for actual `SEARCH_CLOSED`, `FINALISTS_LOCK`, `validation_summary`, `SELECTED_MODEL`, `FREEZE`, `REVIEW_RESULT` and `REPRODUCTION_RESULT`, plus `FORMAL_SUBMISSION_BLOCKED`, `RESEARCH_CLOSEOUT` and the preserved `submission/SUBMISSION_VERIFICATION.json`. Missing records fail before any final aggregate is parsed. Presence of V3 `TEST_OPENED`, `TEST_COMPLETE`, `test-score` or `SUBMISSION_SUCCESS` contradicts the no-test/no-update scope and fails. No fake formal completion is accepted or written.

The archive uses the genuine strict freeze schema with these additional requirements:

```json
{
  "frozen": true,
  "test_evaluated_before_freeze": false,
  "research_only": true,
  "formal_test_allowed": false,
  "formal_submission_blocked_sha256": "SHA256 of actual block record"
}
```

All existing selection, source, asset, training-recipe, review and clean-reproduction references remain required. The report hashes source/assets and original aggregate files without importing method code. Actual validation population is798; the fresh-environment official validation count must match the selected saved count. The timing shown is explicitly validation reproduction, never V3 test latency.

Root creates `submission/v3/RESEARCH_CLOSEOUT.json` only after the actual research archive freeze:

```json
{
  "schema": "mp2-v3-post-deadline-closeout-v1",
  "created_at_utc": "actual aware ISO timestamp after freeze",
  "freeze_sha256": "actual FREEZE.json hash",
  "selected_model_sha256": "actual SELECTED_MODEL.json hash",
  "formal_submission_blocked_sha256": "actual FORMAL_SUBMISSION_BLOCKED.json hash",
  "rules_audit_path": "repository-relative saved current official-rule audit",
  "rules_audit_sha256": "actual audit hash",
  "research_only": true,
  "missed_deadline": true,
  "no_new_test": true,
  "no_new_score_update": true,
  "prior_issue_number": 294,
  "prior_issue_url": "https://github.com/xudongwu-0/xudongwu-0.github.io/issues/294",
  "incomplete_stability_ids": ["all actual incomplete nondeployment-seed job IDs"],
  "monitor_gap_disclosure": "actual monitoring gaps, late closure and elapsed-time interpretation"
}
```

`FORMAL_SUBMISSION_BLOCKED.json` must contain actual post-deadline `created_at_utc`, `missed_deadline:true`, a nonempty `reason`, and either both `no_new_test:true` / `no_new_score_update:true`, or `no_new_test_or_score_update:true`. The input tool checks that its timestamp precedes the closeout record. Public-rule evidence is hash-bound; the report does not invent a teacher exception or ask for another approval gate.

Default evidence is `analysis/final_search/analysis.json`, `research_v2/analysis-final/analysis.json` and `fixed_fusion_analysis_complete.json`; CLI `--analysis`, `--v2-analysis` and `--fusion-analysis` allow explicit alternative saved aggregate paths. Complete current train rows are reconciled with original per-experiment summaries, terminal queue and historical round analyses. Every actual completed row is printed (layout bounded to36 V3 and20 historical V2 rows), four fixed fusions must have full saved rejected diagnostics, and the incomplete seed IDs must equal the terminal queue. Saved attempt logs are bound; recorded elapsed times and missing observations are disclosed separately from optimization fit time. No uninterrupted24-hour CPU/monitoring claim is made.

The ten intended pages cover deadline status, representation and deterministic references, all V3 experiments over two pages, all historical V2 experiments, decoder/objective/seed controls, rounds and interruptions, four rejected fusions with stored paired uncertainty, actual offline validation and clean reproduction, and reproducibility/disclosures. R05 matched comparison reuses the source-reviewed historical-snapshot/completion/registration checks; no bootstrap is rerun. The capacity/RNG confound and search-selection optimism stay explicit. No CV or validation statistic is called a leaderboard rank.

If the saved rule retains the historical fallback, the report says so explicitly; when every locked candidate's overall/hard counts are no higher, it states that no locked V3 candidate improved these validation counts. This is derived from actual aggregates, not a promise or rewritten research outcome. If optional confidence files are present, all four final records (OOF report, calibration, validation report, selected-event preparation) must be complete and consistently bound. Brier/ECE/AURC are printed only from those stored metrics; fitted-OOF results are labeled IN-SAMPLE and choices must remain unchanged. With no optional confidence records, the report states only that the diagnostic is not included; partial records fail rather than supplying placeholder numbers.

After actual final evidence is complete, root runs:

```powershell
.venv/Scripts/python.exe reports/research_v3/prepare_closeout_inputs.py
.venv/Scripts/python.exe reports/research_v3/build_closeout.py
```

Defaults are `reports/research_v3/closeout_inputs.json`, `reports/research_v3/closeout_zh.md` and `reports/research_v3/research_closeout/research_closeout_v3.pdf`. JSON/Chinese output paths can be changed with `--output` / `--chinese-output`; use `--inputs` / `--output-directory` for a separate reviewed render attempt. Existing outputs are never overwritten. Build metadata records source/input/output hashes, actual page count and pending visual QA. A rendered count over10 fails and keeps the attempt for correction.

The PDF skill was read for this artifact workflow; root already successfully marked the single create operation, so no second marker is run. No actual PDF has been generated by this implementation task. Root must render every actual page to PNG, inspect it individually, fix overflow/legibility in a new preserved attempt, and record visual QA before publishing. Peer review remains a separate actual workflow; the report does not assert its completion or continued closure.
