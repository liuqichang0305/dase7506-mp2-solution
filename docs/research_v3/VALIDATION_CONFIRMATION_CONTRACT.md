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
      "final_fit_sha256": "<hash>",
      "cv_summary_sha256": "<hash of the same experiment's summary.json>"
    }]
  }]
}
```

For a fusion component, `bundle_subdir` is its actual contained member directory. A historical component uses `kind: "historical_v2"`, its real experiment ID, `freeze_path` and `freeze_sha256`, rather than a new final-fit report. It retains its actual historical seed; the fixed V3 deployment seed applies to newly trained components.

Use the `INFERENCE_SOURCE` constant in the tool as the minimum source hash inventory. It includes the full reviewed inference dependency closure, the asset auditor, this confirmation tool, and exact unchanged official common/predict/scoring sources. Additional source hashes may be included. The candidate asset mapping must exactly equal the union of all files in the locked candidate directories.

Before any holdout access, the actual inference loader checks every model/scaler/numeric state and manifest. Single-model selector identity is reconstructed from the real bundle architecture/decoder and component experiment ID. Final-fit report validation binds config/seed, official train-only data hashes, the exact eleven reviewed training sources, common folds, the hash of the selected complete three-fold summary (all folds complete with 1,585 held-out questions, matching config/provenance), the exact complete exported component file inventory, and (for neural models) the full-train epoch state path/hash and phase. The V2 fallback source and asset freeze is verified without reevaluating its holdout.

An ensemble finalist additionally supplies `fusion_registration_path`, `fusion_registration_sha256`, `fusion_result_path`, and `fusion_result_sha256`, all in `results/research_v3`. The canonical selector specification is the entire independent registration JSON: status registered, purpose single_fixed_equal_probability_oof, ordered methods such as v3:experiment and v2:historical_experiment, exact equal weights, component OOF hashes, gate-evidence path/hash, and registration UTC. It must match actual component order; prior passing complementarity must precede registration, and the bound diagnostic result must report that exact fusion complete. A diagnostic containing fusion status not_run is rejected. Fusion preparation support does not imply any real mixture has been evaluated or chosen.

## Execution and restart

The tool takes the trainer's cooperative resource lock and refuses any active project scheduler/training process. It verifies all locks before creating `VALIDATION_OPENED.json`. Official prediction and separate official scoring run in fresh external processes, sharing a 30-second deadline. It launches the underlying Python directly with `-S` and the active venv's explicit site-packages path, avoiding the extra Windows venv launcher. A measured wrapper executes the unchanged official script via `runpy` and records actual `threadpoolctl` library pools. Startup, imports, wrappers, complete prediction, independent scoring and library inspection all count in the combined elapsed time. PID/creation-time identities and any children are monitored; timeout cleanup includes the captured tree. CPU, RSS and OS thread counts are sampled every 10 ms and disclosed as samples.

Each candidate saves original prediction/scoring JSON and stdout/stderr, timing/asset records, and a hash-bound `evaluation_complete.json`. A completed candidate resumes without rerunning. Partial evidence is preserved and refused automatically; it requires explicit internal review of the same frozen candidate. A changed lock or a third candidate is rejected. Summary reuse verifies all recorded evidence hashes.

After both official runs finish, the tool creates report-only selected-answer confidence events and hard/single-positive counts; these cannot change preprocessing, weights, fusion, or choices. The NPZ reference in `validation-events.json` is relative to that JSON's directory. The selection lock path is repository-relative, so run the reliability evaluator from repository root.

The summary applies the preregistered rule: a new model must be resource eligible, hard-correct >=297 and full-correct >=740, with at least one strict improvement; compare hard/full/bytes/latency. Otherwise keep historical V2 Deep Sets. It does not reopen search or enable test. Independent selected-model export/contract checks, final freeze, and clean reproduction remain separate.

## Measured Windows resource arrangement

The initial synthetic test revealed seven sampled OS threads across the Windows venv launcher and actual Python, despite every BLAS/OMP limit being one. The external monitor has another four OS threads and is recorded separately; it is not the submitted prediction/scoring program. A direct-base-Python `-S` control using explicit venv packages reduced the actual task tree to one process and four sampled OS threads, with OpenBLAS reporting one thread. Raw outputs, process identities and commands are preserved in `results/research_v3/audit_validation_process_probe/`.

The implemented eligibility gate still requires at most four sampled task-process OS threads, at most one thread in every actually loaded scientific library pool, <=16 MiB complete assets, and <=30 seconds complete prediction plus separate scoring. Prediction and scoring never overlap. Raw task and monitor-inclusive counts are both disclosed; a configuration limit is never substituted for a measured count. No local measurement certifies the reference Xeon hardware. After independent-review repairs, the suite passed 13/13 in 5.435 seconds (`audit_validation_complete_synthetic_tests.log`); earlier unsuccessful fixture/overstrict-monitor assertions and the independent counterexamples are preserved with their true return codes.

Tests: `code/tests/test_confirm_validation_v3.py` uses temporary synthetic trees only, including real harmless sleep children, process-tree timeout cleanup, actual independent synthetic predictor/scorer processes, data-read guards in a separate process, source/asset/lock tampering, immutable resumption and selection boundaries. No official validation/test files were loaded.

Final independent software review: `results/research_v3/independent-confirmation-review-final/REVIEW.md`, source SHA256 `5eac728edc8ecab5a07b20246c5637a1992ffb359084246a2515e150246de319`, 13/13 synthetic checks passed in 5.422 seconds. This verifies the gate implementation; actual final checkpoints, locked validation and reference-hardware eligibility remain separate evidence.

The later timing-only revision adds `started_at_utc` immediately before subprocess creation and `completed_at_utc` after its lifetime is complete. Its final source SHA256 is `a3928fdd9b58bfd5d3036f51510a866516d5aab8a8289892212769aecf558aff`; 13/13 synthetic tests passed in 5.650 seconds (`audit_validation_formal_scope_tests.*`). Monotonic timing still controls the shared deadline. This revision was independently read by the freeze/reproduction integrator, and the separate direct-base NumPy worker integration also passed (`independent-final-worker-smoke/`).

The formal worker arrangement is direct base Python with child creation denied. PID sampling can retain and clean up observed descendants, but cannot reconstruct detached ancestry if all intermediate launchers disappear before the first sample. Earlier four-level launcher failure logs are preserved as `audit_validation_phase_timestamps_tests.*` and `audit_validation_orphan_recovery_tests.*`; this limitation is disclosed rather than claimed solved by the supported direct-worker arrangement. Sampling uses a nominal 10 ms delay plus process-query overhead; CPU/RSS/thread figures are observations rather than exact integrals or guarantees about every instantaneous peak. The external coordinator's own threads remain separate from task eligibility.
