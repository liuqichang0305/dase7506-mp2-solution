# Final freeze and one-shot official test

These tools implement the already-authorized final evaluation. No real test is
run during their development. Tests use temporary generated files and simulated
official-process responses; their numbers are not course results.

```powershell
.venv\Scripts\python.exe code/tools/freeze_v3.py
.venv\Scripts\python.exe submission/v3/evaluate_final.py
```

There are no method, checkpoint, seed, split or output-directory override flags.
The method is `code/solution_v3.py`, the deployment bundle is
`artifacts/research_v3/final_selected`, and all final-test outputs go to
`submission/v3`. The second command is for the root agent only, after final
freeze and the course-permitted evaluation process. No separate teacher-email
approval gate is introduced.

## Selection and byte-identical deployment copy

First complete `results/research_v3/validation_summary.json` under the closed
search and immutable finalist lock. Copy its uniquely selected bundle, including
every manifest/member file, to `artifacts/research_v3/final_selected` without
changing any bytes. Historical V2 fallback is supported by the V3 NumPy wrapper.
Run final contract/export audit and fresh-environment validation reproduction
against that exact copied bundle.

The root agent writes `submission/v3/SELECTED_MODEL.json`:

```json
{
  "schema": "mp2-selected-model-v3",
  "created_at_utc": "timezone-aware time after validation completion",
  "name": "exact validation_summary.selected.name",
  "historical_fallback": false,
  "method": "code/solution_v3.py",
  "checkpoint": "artifacts/research_v3/final_selected",
  "origin_checkpoint": "exact validation_summary.selected.checkpoint",
  "deployment_seed": 3036900957,
  "selector": {"kind": "single_model", "experiment_id": "EXPERIMENT_ID", "family": "tabm", "decoder": {"mode": "group_lse", "alpha": 0.0, "tau": 1.0, "mask": true}, "member_probability_average": true},
  "oof_directory": "results/research_v3/EXPERIMENT_ID",
  "validation_summary_sha256": "SHA256 of completed validation summary",
  "components": []
}
```

For a new finalist, `components` and `selector` are copied **exactly** from its
`FINALISTS_LOCK.json` row, including order and provenance fields. A fusion uses
the full registered fixed-probability specification as its selector; see the
confidence-analysis document. For historical fallback, components is exactly:

```json
[{"kind":"historical_v2","experiment_id":"V2_SELECTED_ID","bundle_subdir":".","freeze_path":"results/research_v2/V2_FREEZE.json","freeze_sha256":"actual V2 freeze SHA256"}]
```

Its selector identifies the original single V2 model with
`member_probability_average:false`; its OOF directory is the corresponding V2
experiment. The freezer independently reapplies the fixed validation selection
rule, verifies the existing finalist/baseline provenance, and checks that the
entire deployment-copy inventory equals the chosen original inventory.

## Independent review and fresh reproduction records

The root agent records completed work, rather than requesting an additional
approval. Both records live in `submission/v3` and require:

- `passed: true`, `method: "code/solution_v3.py"`, and the fixed `checkpoint`.
- `created_at_utc`, after the unique selected record and before the freeze.
- `selected_model_sha256`: the exact selected-record SHA256.
- `source_files`: the full map returned by `freeze_v3.source_hashes(ROOT)`.
- `assets`: the full map returned by `freeze_v3.asset_hashes(ROOT)[0]`.
- `evidence_files`: a nonempty map of repository-relative audit/reproduction
  evidence paths to SHA256. These files must exist and match.

`REVIEW_RESULT.json` additionally requires:

```json
{
  "schema": "mp2-v3-review-result-v1",
  "review_kind": "independent_ai_internal",
  "contract_result": {"path": "results/research_v3/final-selected-contract.json", "sha256": "actual SHA256"},
  "export_result": {"path": "results/research_v3/final-selected-export.json", "sha256": "actual SHA256"},
  "dependency_audit": {
    "passed": true,
    "no_hidden_parameters": true,
    "external_runtime_reads": "bundle_and_inputs_only",
    "inference_dependencies": ["numpy==ACTUAL_VERSION"]
  }
}
```

`REPRODUCTION_RESULT.json` additionally requires:

```json
{
  "schema": "mp2-v3-reproduction-result-v1",
  "split": "validation",
  "fresh_environment": true,
  "choices_match": true,
  "official_score_match": true,
  "source_commit": "full reproduced source commit",
  "predictions_result": {"path": "results/research_v3/clean-reproduction/validation-predictions.json", "sha256": "actual SHA256"},
  "score_result": {"path": "results/research_v3/clean-reproduction/validation-score.json", "sha256": "actual SHA256"},
  "timing_result": {"path": "results/research_v3/clean-reproduction/validation-timing.json", "sha256": "actual SHA256"},
  "environment_result": {"path": "results/research_v3/clean-reproduction/environment.json", "sha256": "actual SHA256"}
}
```

Every typed reference has exactly `path` and `sha256`, and must also appear with
that hash in the record's `evidence_files`. A `passed` declaration and an arbitrary
text file are insufficient. The freezer parses the real tool reports:

- Contract: `check_contract_v3.py` passed on all 798 validation inputs, fixed
  `final_selected` checkpoint, exact asset hashes and method hash, required
  isolation/choice/permutation checks all passed, references unread, at most four
  numerical-pool threads.
- Export: `check_export_v3.py` passed on all 798 validation inputs, same checkpoint
  and assets, every member's scores within tolerance, group probabilities within
  tolerance, and 798/798 identical choices with no differing rows.
- Reproduction: all 798 IDs and original indices exactly equal the original
  locked validation prediction file. Protocol/split and the independent scorer's
  `correct`, `n` and `accuracy` must match the original score. The canonical
  original files are under `results/research_v3/validation/SELECTED_NAME/`, or
  `research_v2` for fallback; both must also be in `evidence_files`.
- Resource report: the `confirm_validation_v3` timing schema, passed and locally
  eligible; prediction plus scoring ≤30 seconds; task OS threads ≤4; exactly two
  successful, non-timeout direct-base `-S` official validation commands. Their
  `started_at_utc`/`completed_at_utc` prove prediction ended before scoring began.
  Report `completed_at_utc` precedes the reproduction record. Both numerical-pool
  reports show no Torch, one thread per pool, at most four total, and `pythonpath`
  entries entirely inside the fresh environment.

The environment result is produced from the actual new environment/checkout:

```json
{
  "schema": "mp2-v3-clean-environment-v1",
  "captured_at_utc": "after selection, before reproduction record",
  "source_commit": "same reproduced commit",
  "python_prefix": "absolute ROOT/tmp/NEW_ENV",
  "python_executable": "absolute executable inside NEW_ENV",
  "checkout": "absolute ROOT/tmp/NEW_PUBLIC_CHECKOUT",
  "packages": ["numpy==ACTUAL_VERSION", "threadpoolctl==ACTUAL_VERSION", "psutil==ACTUAL_VERSION"],
  "source_files": {"repository-relative source": "actual complete source map"},
  "assets": {"repository-relative asset": "actual complete asset map"},
  "data_manifest_sha256": "original official manifest SHA256",
  "official_data_files": ["candidates/gsm8k_validation.jsonl", "candidates/gsm8k_validation.npz", "gsm8k_validation.jsonl", "manifest.json"]
}
```

The freezer verifies that the environment executable exists, its prefix differs
from the project `.venv`, and the checkout differs from the working tree. Both
must be in the project `tmp` directory. It rehashes every actual checkout source
and asset and the original manifest. The sorted official-data inventory must
contain exactly the four validation/manifest files, optionally `LICENSE.txt` and
`NOTICE.txt`, and no train/test files. Package metadata must contain NumPy and no
Torch/TensorFlow. Measurement helpers are allowed in the environment; deployed
inference remains NumPy-only. This is structured evidence of the completed run,
not a claim that a boolean alone proves a fresh run occurred.

The source closure includes every custom `code/mp2/**/*.py` file, the original
official parser/predictor/scorer, V3 training/inference/preprocessing code, the
resource/contract/export tools and this freeze/evaluation pair. Root requirement
files and `code/requirements*.txt` are included when present, along with the
fusion-confidence readers and fresh-reproduction script used by these checks.
All source and asset bytes must equal their
Git blobs at the freeze commit. Official parser/predictor/scorer hashes must
remain unchanged. The bundle audit counts every file and uncompressed container
contents and must be at most 16 MiB.

The default freeze commit is current `HEAD` and must equal the reproduced commit.
A straightforward order is: commit final method/assets, reproduce that commit,
write review/reproduction evidence, then freeze without first making a separate
evidence-only commit. Later publication commits may add reports/results while
preserving frozen source/assets; the form adapter checks ancestry and bytes.

## Freeze artifact

`submission/v3/FREEZE.json` has schema `dase7506-mp2-final-freeze-v3` and contains
the official adapter's required `frozen:true`,
`test_evaluated_before_freeze:false`, `source_commit`, `source_files`, `assets`
and `created_at_utc`. It also binds selection, validation summary, search closure,
finalist lock, review, reproduction, and original data-manifest hashes. Actual
component final-fit reports supply full configurations, seed, epoch counts,
preprocessing, decoder and EMA/live evaluation-weight provenance. All actual
bundle configuration JSON files are retained in the record.

The freezer reads manifest metadata and completed development evidence, never
raw test input/reference files. Expected test hashes are copied from the manifest.
The known historical V1 aggregate (348/467) is disclosed as already known before
V3; the false pre-freeze test flag refers specifically to the **new V3 test**.
The original submission is preserved. An existing freeze is verified and reused,
never overwritten. A prior `TEST_OPENED.json` prevents creating a new freeze.

## Single official run and measured resource contract

The runner holds the same exclusive resource lock as the trainer/validation
coordinator and refuses live project trainers or schedulers. It validates the
complete freeze and dependency gate before opening any test phase. A durable
`TEST_OPENED.json` is written and flushed first. Prediction then runs in a fresh
direct base-Python `-S` process with explicit active-environment package paths;
independent official scoring starts only after prediction fully exits. This
reuses `confirm_validation_v3.official_command` and `run_process`.

Both official phases share one 30-second wall-clock deadline including startup,
imports, official input/asset loading, prediction, output, scoring and process
wrappers. The task process tree must have at most four sampled OS threads; every
observed numerical pool is configured to one thread, with at most four total
compute threads. PyTorch must not be imported by inference. The measurement
coordinator's additional threads are separately disclosed. Local timing is not
certification on the reference Xeon machine.

Phase-specific runtime guards prohibit network access, official-data writes,
frozen-asset writes, and reading historical research/submission results. The
prediction worker can read only the official test candidate bank/embeddings and
manifest. It cannot open test gold. Only the separate scoring worker may read
test gold and the saved prediction file. Unchanged official code verifies all
input/reference hashes against the frozen manifest.

The runner checks protocol, complete 467-example coverage, unique IDs, original
indices 0–3, and aggregate count/percentage consistency. It retains untouched
official `test-predictions.json` and `test-score.json`, phase stdout/stderr logs,
numerical-pool reports, and `test-timing.json`. It does not print IDs, choices,
correctness arrays or the score into a research-agent conversation. The aggregate
JSON is consumed later for final reporting and the authorized form submission.

Success writes `TEST_COMPLETE.json`, binding every output and the opening marker
to the freeze. Its schema is `mp2-v3-test-complete-v1`, with `status:passed`,
`local_resource_eligible:true`, fixed method/checkpoint/split/n, and
`output_sha256`. A later invocation only verifies those existing hashes and
returns the record; it never calls prediction/scoring again, even after the
deadline. `submit_final.py` requires this passing completion and matching score.

On failure, resource violation, timeout or interruption, the opening marker and
all partial outputs remain, with `TEST_INCOMPLETE.json` when possible. Subsequent
invocations refuse automatic re-evaluation. No candidate, seed, timing attempt
or test-based tuning is offered as a recovery action. Preserve the evidence and
review the precise failure under the course's permitted procedure.
