# Final saved-evidence figures

Implementation status: synthetic-tested only. Actual final evidence has not been
plotted. No raw candidate bank, reference, NPZ event archive, test score or
leaderboard is read by this tool. It never trains, scores, calibrates, selects
a model, or creates a PDF.

After the four completed saved reports exist, run:

```powershell
.venv/Scripts/python.exe reports/research_v3/prepare_figures.py
```

The fixed inputs are:

- `results/research_v3/analysis/final_search/analysis.json`
- `results/research_v3/campaign_queue.json`
- `results/research_v3/confidence_calibration/report.json`
- `results/research_v3/confidence_validation/report.json`

Missing evidence, a nonterminal campaign, omitted completed V3 methods, selector
mismatch, changed chosen outcomes, incomplete confidence curves, inconsistent
counts or missing fit costs cause rejection before the final output directory is
created. No path inside the input JSON is followed. All four JSONs and the
generator source receive SHA-256 bindings, checked again after rendering.

`reports/research_v3/figures/cv_fit_cost_vs_hard_accuracy.png` shows every complete
common-three-fold method/seed row. The x-axis is recorded `training_seconds`, a
fitting wall-clock cost on a log scale. It is not inference latency or a new
hardware-controlled speed comparison; historical V2 costs are read from their
saved evidence. The y-axis is pooled hard-subset accuracy, verified against
correct/count. Colors distinguish V2/V3 and circle/triangle markers distinguish
the fixed deployment seed from the registered two extra seeds. Partial rows are
excluded. Failed/pruned jobs, recorded nonzero exits and interrupted attempts
are separately disclosed; their missing costs are not imputed or silently added
to successful-method fit times. Every plotted method ID/seed/value is retained
in its generation record for lookup against the full experiment table.

`chosen_event_confidence.png` has two rows: train OOF, where the calibration map
was fitted and its calibration diagnostics are optimistic, and validation, an
already-used holdout. Columns show raw/calibrated ten-bin reliability and full
saved risk-coverage curves. Empty bins are omitted rather than assigned invented
zero accuracy; marker area encodes within-set bin frequency. Exact bin sample
counts remain in the JSON generation records. Full curve lengths, cumulative
binary error counts, endpoints, bin totals, ECE and AURC are checked against the
saved summaries. Curves retain the saved stable tie ordering. These plots do
not choose an abstention threshold or claim untouched-test generalization.

Both PNGs are 7.1 by 3.26 inches at 300 dpi (2130 by 978 pixels). Each has a
`*.generation.json` containing PNG SHA-256, source hashes, chart semantics,
relevant counts, renderer version and a pending final-input visual-QA marker.
The `figure_manifest.json` schema is `mp2-v3-report-figures-v1`, with
`evidence_sha256` and two `figures` entries containing `path`, `sha256`,
`height: 235`, `caption`, and `generation_record: {path, sha256}`. The report
input generator consumes this manifest and appends one tenth page containing
the two figures; it still enforces the final ten-page limit. Actual report
rendering and every-page visual QA remain required.

Existing figure outputs are never overwritten. Failed rendering leaves a
`FIGURES_INCOMPLETE.json` record and cannot produce a usable final manifest.
Preserve an earlier directory before an explicitly reviewed layout rebuild.

Synthetic checks: `code/tests/test_prepare_figures_v3.py`, 9/9 passed in 3.518
seconds with exit zero. They cover complete PNG/manifest output, terminal/common
CV consistency, no invented costs/empty-bin points, full curve/count integrity,
selector/choice isolation, and missing-final-evidence rejection. Temporary
unit-test trees are deleted. Two clearly labeled QA preview trees are retained
under `tmp/research_v3/synthetic_figures_preview_01` and `_02`; every preview is
watermarked `SYNTHETIC FIXTURE` and its manifest says `synthetic_fixture_only`.
Preview 02 matches the final generator source and both PNGs were visually
inspected. No synthetic PNG or manifest exists in the final figure directory.
