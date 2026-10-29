# V3 one-way validation confirmation

Status: implemented and synthetic-tested only. No actual V3 validation/test has been opened by this work.

Run from the repository root, after train-only search is closed and final bundles are fixed:

```powershell
.venv/Scripts/python.exe code/tools/confirm_validation_v3.py
```

There is no split, output, candidate, seed, or checkpoint override. All paths below are repository-relative unless explicitly described otherwise. UTF-8 JSON files are immutable once consumed.

## Required pre-validation records

`results/research_v3/SEARCH_CLOSED.json` must contain `search_closed: true` and an aware UTC `created_at_utc`.

`results/research_v3/FINALISTS_SELECTION.json`:

```json
{
  "schema": "mp2-finalists-selection-v3",
  "created_at_utc": "<after search closure, before final fit>",
  "deployment_seed": 3036900957,
  "allowed_final_fit_experiments": ["chosen_experiment_s0"],
  "finalists": [{
    "name": "final_candidate_a",
    "component_ids": ["chosen_experiment_s0"],
    "selector": {
      "kind": "single_model",
      "experiment_id": "chosen_experiment_s0",
      "family": "<family>",
      "decoder": {"mode": "group_lse", "alpha": 0.0, "tau": 1.0, "mask": true},
      "member_probability_average": false
    }
  }]
}
```

Exactly one or two distinct new candidates are allowed. A fixed equal-probability candidate may have two or three components; `component_ids` order is the inference manifest order. `allowed_final_fit_experiments` is the exact unique union of **new** component IDs, excluding a copied immutable historical V2 component. The selector object must exactly match the OOF reliability identity: a single model uses the object above (TabM sets member probability average true); a fusion uses `{"kind":"fixed_equal_probability","specification":<registered specification>}`. It is not the candidate display name.

`results/research_v3/FINALISTS_LOCK.json`:

```json
{
  "schema": "mp2-finalists-lock-v3",
  "search_closed": true,
  "created_at_utc": "<after final training>",
  "selection_sha256": "<SHA256 of FINALISTS_SELECTION.json>",
  "search_closed_sha256": "<SHA256 of SEARCH_CLOSED.json>",
  "data_manifest_sha256": "<SHA256 of code/data/manifest.json>",
  "baseline_freeze_sha256": "<SHA256 of results/research_v2/V2_FREEZE.json>",
  "inference_source_sha256": {"code/solution_v3.py": "<hash>", "<all dependencies>": "<hash>"},
  "asset_sha256": {"artifacts/research_v3/final_candidate_a/config.json": "<hash>", "<every candidate bundle file>": "<hash>"},
  "finalists": [{
    "name": "final_candidate_a",
    "method": "code/solution_v3.py",
    "checkpoint": "artifacts/research_v3/final_candidate_a",
    "selector": "<same structured object as selection>",
    "components": [{
      "kind": "new",
      "experiment_id": "chosen_experiment_s0",
      "seed": 3036900957,
      "bundle_subdir": ".",
      "final_fit_path": "results/research_v3/chosen_experiment_s0/final_fit.json",
      "final_fit_sha256": "<hash>"
    }]
  }]
}
```

For a fusion component, `bundle_subdir` is its actual contained member directory. A historical component uses `kind: "historical_v2"`, its real experiment ID, `freeze_path` and `freeze_sha256`, rather than a new final-fit report. It retains its actual historical seed; the fixed V3 deployment seed applies to newly trained components.

Use the `INFERENCE_SOURCE` constant in the tool as the minimum source hash inventory. It includes the full reviewed inference dependency closure, the asset auditor, this confirmation tool, and exact unchanged official common/predict/scoring sources. Additional source hashes may be included. The candidate asset mapping must exactly equal the union of all files in the locked candidate directories.

Final-fit report validation binds config/seed, official train-only data hashes, training sources, common folds, exported assets, and (for neural models) the full-train epoch state path/hash and phase. The V2 fallback source and asset freeze is verified without reevaluating its holdout.

## Execution and restart

The tool takes the trainer's cooperative resource lock and refuses any active project scheduler/training process. It verifies all locks before creating `VALIDATION_OPENED.json`. Official prediction and separate official scoring run in fresh external processes, sharing a 30-second deadline. Windows launcher and actual child identities are captured by PID plus creation time; timeout cleanup includes the captured tree. CPU, RSS and OS thread counts are sampled every 10 ms and disclosed as samples.

Each candidate saves original prediction/scoring JSON and stdout/stderr, timing/asset records, and a hash-bound `evaluation_complete.json`. A completed candidate resumes without rerunning. Partial evidence is preserved and refused automatically; it requires explicit internal review of the same frozen candidate. A changed lock or a third candidate is rejected. Summary reuse verifies all recorded evidence hashes.

After both official runs finish, the tool creates report-only selected-answer confidence events and hard/single-positive counts; these cannot change preprocessing, weights, fusion, or choices. The NPZ reference in `validation-events.json` is relative to that JSON's directory. The selection lock path is repository-relative, so run the reliability evaluator from repository root.

The summary applies the preregistered rule: a new model must be resource eligible, hard-correct >=297 and full-correct >=740, with at least one strict improvement; compare hard/full/bytes/latency. Otherwise keep historical V2 Deep Sets. It does not reopen search or enable test. Independent selected-model export/contract checks, final freeze, and clean reproduction remain separate.

## Measured Windows resource limitation

The synthetic tests revealed that even an empty Python predictor can have more than four sampled OS threads across the Windows venv launcher, actual child and monitoring coordinator despite every BLAS/OMP environment limit being one. The current tool records the measured counts and conservatively marks such local eligibility false. It does not equate configured numerical-library threads with observed total OS threads or silently certify reference Xeon hardware. A compliant execution arrangement or explicitly documented rule interpretation is needed before claiming the thread condition is met.

Tests: `code/tests/test_confirm_validation_v3.py` uses temporary synthetic trees only, including real harmless sleep children, process-tree timeout cleanup, actual independent synthetic predictor/scorer processes, data-read guards in a separate process, source/asset/lock tampering, immutable resumption and selection boundaries. No official validation/test files were loaded.
