# Clean selected-model reproduction

Implementation status: code and synthetic verification only. No actual public
clone, package installation, validation run, or test run has been performed by
this implementation task.

After `SEARCH_CLOSED.json` and the unique `SELECTED_MODEL.json` exist, commit and
push all selected source/assets and this reproducer to the public repository.
Before `FREEZE.json` or any new test opening, stop the project trainer/scheduler
and run exactly:

```powershell
.venv/Scripts/python.exe scripts/reproduce_selected_v3.py --attempt final-selected-01
```

The script has no method, model, seed, split, package, commit, or destination
override. It uses the current exact HEAD, the fixed public repository URL,
`code/solution_v3.py`, `artifacts/research_v3/final_selected`, and validation only.
It holds the trainer's existing exclusive resource lock throughout setup and
execution. The root coordinator first verifies the closed search, immutable
selection, unchanged official scripts, full source/asset inventory, committed
bytes, and original locked validation-output hashes.

A new checkout and venv are allocated beneath `ROOT/tmp` with unique names.
Clone runs over the fixed public HTTPS URL with interactive credentials and
inherited user Git configuration disabled. Checkout must equal the source HEAD
exactly. The complete source and asset inventories must match both their Git
blobs and the copied files. The reproducer itself must also be committed.
Fresh venv installation is pinned to `numpy==2.5.3`, `threadpoolctl==3.7.0`, and
`psutil==7.2.2`; pip/setuptools may remain as installation tooling. The actual
environment probe records prefix, executable, direct base executable, package
versions, site-package paths and NumPy version. Any Torch/TensorFlow availability,
extra scientific dependency, wrong pin, or old-environment path is rejected.

The checkout may contain only manifest, the three official validation files,
and optional data `LICENSE.txt`/`NOTICE.txt`. Train/test or any unapproved data
file causes rejection before copying. The four required files are copied from
the original official data directory, verified against manifest hashes, never
overwritten, and recorded with absolute paths, byte counts and hashes. Optional
committed license/notice files also receive hashes.

Prediction and independent official scoring run sequentially in fresh direct
base-Python `-S` processes with only the new venv's explicit `PYTHONPATH`.
The shared `confirm_validation_v3.run_process` monitors the task process tree;
the external coordinator is separately disclosed. The same 30-second deadline
includes both process startups, imports, wrappers, official input/model loading,
prediction output, independent scoring and pool inspection. The task must have
at most four sampled OS threads, each loaded scientific library one thread, and
at most four total configured scientific threads. Setup clone/install/probe
time is separately logged and is not inference time. Local timing is not
reference-Xeon certification.

Phase-specific audit hooks deny all network/socket and subprocess operations,
prior research/submission results, unapproved data or external file reads, and
input/asset writes. Prediction can read candidate inputs and manifest but cannot
open validation gold. Only the separate scoring worker can read gold and saved
predictions. A caught forbidden-access error still makes the phase fail. The
root compares all 798 original ordered IDs and selected candidate indices and
official correct/count/accuracy fields, without choosing another model or tuning.

Every attempt saves raw setup/phase logs and immutable evidence under
`results/research_v3/clean_reproduction/<attempt>`. A passed attempt writes
`reproduction-result.json`, with the exact typed fields expected by
`freeze_v3._gates`: `predictions_result`, `score_result`, `timing_result`, and
`environment_result`, each a `{path, sha256}` reference also in `evidence_files`.
The original canonical locked validation prediction/score files are included
in that evidence map. Root reviews this record before placing its final wrapper
at `submission/v3/REPRODUCTION_RESULT.json`; this script never writes that path.

Failures retain `STARTED.json`, raw logs, partial outputs, and `FAILED.json`.
No directory is deleted or overwritten. A later attempt requires a new name
and a specific engineering explanation, for example:

```powershell
.venv/Scripts/python.exe scripts/reproduce_selected_v3.py --attempt final-selected-02 --engineering-reason "Fixed the identified environment setup failure; model and assets are unchanged."
```

This is a recovery instruction, not permission to repeat for a better runtime
or change selected assets. Any existing completed attempt prevents another run;
its hash-bound evidence is retained for final freeze review.

Synthetic checks are in `code/tests/test_reproduce_selected_v3.py`. They cover
the full coordinator with fake clone/install and fake official outputs, real
isolated guard processes, 798-choice equality, package/provenance validation,
train/test exclusion, no-overwrite behavior, resource-command structure, search
closure, and failure preservation. All inputs live in temporary synthetic
directories; real course data and production results are not loaded.

Actual synthetic result: 13/13 tests passed in 1.494 seconds, with command exit
code zero. The log is
`results/research_v3/clean-reproduction-synthetic-tests-final.log`. Earlier logs
are retained, including an initial PowerShell stderr-pipeline failure despite
the Python test runner reporting success; the final command routes the runner
to stdout explicitly. These checks establish implementation behavior only and
do not constitute the final real-data reproduction gate.
