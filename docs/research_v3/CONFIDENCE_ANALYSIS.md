# Optional confidence analysis

`code/tools/reliability_v3.py` is a report-only tool. Training, method selection and
`solution_v3.predict` do not import or load its fitted map. It never changes a
candidate index. Its files are outside the inference bundle; any future use by a
prediction path would require counting all loaded assets in that bundle audit.

Run `fit` only once the final selected method is unique. It reads complete train
OOF evidence, not official data files. No real OOF calibration has been run as part
of implementing this tool. All initial tests use generated arrays.

```powershell
.venv\Scripts\python.exe code/tools/reliability_v3.py fit --oof-dir results/research_v3/SELECTED_ID --output results/research_v3/SELECTED_ID_confidence_fit
.venv\Scripts\python.exe code/tools/reliability_v3.py evaluate --predictions PATH_TO_VALIDATION_EVENTS_MANIFEST.json --calibration results/research_v3/SELECTED_ID_confidence_fit --output results/research_v3/SELECTED_ID_confidence_validation
```

Every output directory must be new. Existing evidence is never overwritten.
The second command consumes a separately produced, selection-locked validation
event manifest. It does not read official validation inputs/references itself.
Test events are rejected.

## Confidence and fixed fitting rule

Raw confidence is `group_probabilities[row, groups[row, choices[row]]]`. An
unparseable chosen candidate has confidence zero. This is the probability of the
actual returned answer group, which may differ from the maximum group probability
for a candidate-max selector. TabM probabilities are averaged across members
after each member's group decoding; logits are never averaged for this purpose.

The single scalar feature is `logit(clip(confidence, 1e-6, 1-1e-6))`. The map is
scikit-learn logistic regression with fixed `C=1`, L2-regularized slope,
unpenalized intercept, LBFGS, tolerance `1e-10`, maximum 1000 iterations and seed
`3036900957`. There is no hyperparameter search. Both correct and incorrect
examples are required, and failure to converge raises an error. Only the fitted
coefficient/intercept and JSON metadata are saved; no estimator pickle is loaded.

Calibration metrics on the OOF outcomes used to fit this map are **in-sample**.
They are optimistic even when the underlying model predictions are OOF; prior
OOF model selection can add further optimism. Only the later locked validation
assessment can inform generalization. It must not be used to refit the mapping,
modify choices, or select a new model.

## Saved diagnostics

`calibration.json` records coefficient, intercept, fixed settings, versions, seed,
selector, source hashes and the in-sample caveat. `report.json` contains raw and
calibrated binary Brier scores, ECE, all ten equal-width bin counts/means/accuracy,
and the full risk-coverage trace. Bins are `[0,.1), ..., [.9,1]`; empty bins have
zero count and null means. ECE is the count-weighted absolute calibration gap.

The risk-coverage trace retains confidence-ranked examples from one to N, with
empirical error risk at each coverage. Equal confidences use stable original row
order; partial coverage within a tied group is arbitrary. The reported AURC is
the arithmetic mean of these N risks. This is a descriptive ranking diagnostic,
not a deployed abstention rule. No threshold is tuned. `confidence-events.npz`
preserves groups, probabilities, choices, outcomes and both confidences.

## Accepted OOF evidence

A single-model folder needs `summary.json`, `registration.json`, `oof.npz`, all
three saved fold-score files, and its parent's `common_folds.json`. Configuration,
train-only provenance, common fold hash, held-out membership and fold score hashes
are checked. V3 also requires the summary's archive hash and raw-score hashes.
Legacy V2 has no summary archive hash; its saved fold scores are checked and the
current archive hash is bound into the calibration evidence. Labels within a
parsed answer group must agree. No official labels are loaded separately.

A fixed equal-probability mixture uses `oof-mixture.npz`, the completed ensemble
summary and train-OOF registration, and the common fold manifest. Its immutable
archive hash, equal two- or three-member weights, registered split and held-out
memberships are checked. Saved correctness and selected probability must match
the actual recorded candidate. This tool does not search pairs or weights.

## Validation event schema

The validation confirmation tool provides a JSON object with:

```json
{
  "schema": "mp2-v3-confidence-events-v1",
  "split": "validation",
  "selector": {"kind": "single_model", "experiment_id": "SELECTED_ID", "family": "tabm", "decoder": {"mode": "group_lse", "alpha": 0.0, "tau": 1.0, "mask": true}, "member_probability_average": true},
  "source_sha256": {"official_predictions": "64 lowercase hexadecimal characters"},
  "selection_lock": {"path": "absolute path to an existing V3 selection lock", "sha256": "64 lowercase hexadecimal characters"},
  "predictions_npz": {"path": "validation-events.npz", "sha256": "64 lowercase hexadecimal characters"}
}
```

NPZ arrays are `groups[N,4]` (integers -1..3), `group_probabilities[N,4]`,
`choices[N]` (original integer indices), and `correct[N]` (binary). Optional
`selected` must exactly equal `choices`; optional `confidence` must match the
computed chosen-group probability. Extra IDs/subset flags are permitted. Instead
of NPZ, the same four required arrays may be embedded directly in JSON. NPZ paths
are relative to the manifest, unless absolute. Their SHA256 is verified before
loading without pickle. Source hashes supplied by the confirmation tool are
recorded as declared provenance; this analysis does not reopen those sources.

`selector` must exactly equal the fitted `calibration.json` selector. Single-model
metadata follows the example. Mixture metadata is
`{"kind":"fixed_equal_probability","specification":<OOF registration.specification>}`.
The selection lock must be one of `results/research_v3/FINALISTS_LOCK.json`,
`results/research_v3/FINALISTS_SELECTION.json`, or `submission/v3/FREEZE.json`, and
its actual hash must match. The upstream confirmation tool is responsible for
binding the selected artifact and official scoring results to that lock.

Synthetic manifests use `split: "synthetic"` and need no selection lock. Test and
other split names are rejected. The implementation tests use a temporary fake
lock, never modifying the real campaign locks.
