# DASE7506 MP2: Selecting Reliable LLM Reasoning

Student 3036900957. Individual coursework with disclosed AI assistance. A small CPU verifier selects one of the four supplied candidate solutions. It does not generate answers or call an external model.

## Autonomous research v3 - active train-only study

The current `research/autonomous-v3` branch preserves every historical frozen
method and adds a bounded, resumable campaign. [Draft PR #3](https://github.com/liuqichang0305/dase7506-mp2-solution/pull/3)
contains the implementation and actual evidence. As of 30 October 2026 02:40 UTC,
MLP optimizer controls, small TabM and RealMLP-inspired preprocessing controls
have completed ten shared three-fold configurations. New Attention controls are
running. The [verified R03 table](results/research_v3/analysis/r03_complete/EXPERIMENT_TABLE.md)
includes all completed V2/V3 comparisons, with counts, seed/fold variation,
training costs, asset sizes and question-paired intervals.

The research clock ends at **2026-10-30 16:22:42 UTC** and is not reset on resume.
See the [execution specification](docs/research_v3/AUTONOMOUS_RESEARCH_SPEC.txt),
[search and selection protocol](docs/research_v3/SEARCH_PLAN_V3.md),
[model sources](docs/research_v3/MODELS_SOURCES.md),
[optimizer sources](docs/research_v3/OPTIMIZER_SOURCES.md), and
[current project state](PROJECT_STATE.md).
No new V3 validation or test score has been obtained at this checkpoint.
At most two locked finalists will receive validation confirmation; only the
uniquely selected, reviewed and freshly reproduced bundle can reach final test.
Historical validation use and the known V1 test total are explicitly disclosed.

Resume from the project root with the existing research environment:

```powershell
.venv\Scripts\python.exe scripts/run_campaign_v3.py status
.venv\Scripts\python.exe scripts/run_campaign_v3.py run
```

The scheduler verifies hashes, refuses concurrent trainers, and resumes atomic
epoch checkpoints or reuses completed folds. Read
[the recovery instructions](docs/research_v3/RESUME.txt) before registering
another round. Research commands do not automatically invoke final testing.
Final inference is NumPy-only; pinned development and inference dependencies are
in `code/requirements-training-v3.txt` and `code/requirements-inference-v3.txt`.

## Performance research v2 — development verified, formal re-evaluation pending

The independent `research/performance-v2` branch implements valid-answer masking,
answer-group objectives, MLP/ResMLP, Deep Sets and four-token attention. Nineteen
configuration/seed runs share the same nested train-only three folds. Two
prelocked finalists were confirmed on validation; the selected Deep Sets
checkpoint scores **740/798**, hard **297/355**, compared with frozen v1's
724/798 and 282/355. Its assets total 963,670 bytes and the original local complete
prediction-plus-scoring measurement is 1.7397 seconds. These are development and
local resource results; **no new test score or ranking was obtained**.

See the [complete experiment table](results/research_v2/EXPERIMENT_TABLE.md),
[v2 report](reports/research_v2/report_v2.pdf),
[reproduction commands](docs/research_v2/REPRODUCTION.md),
[Chinese explanation](docs/research_v2/UNDERSTAND_ZH.md),
[freeze](results/research_v2/V2_FREEZE.json), and
[independent implementation review](docs/research_v2/INDEPENDENT_REVIEW.md).
Frozen v2 code/assets: `36f9617175321a2861e78bb2f4ffcde643c4fc03`.
The single equal-weight mixture was evaluated and rejected; unsupported pooling
and loss claims are reported as negative results. V2 did not alter the old course
Issue, frozen assets, original logs or old search guard. The sections below
document the **original submitted v1** and its historical formal test result.

Formal submission: [course Issue #294](https://github.com/xudongwu-0/xudongwu-0.github.io/issues/294), created 29 October 2026 at 15:04:45 UTC+8. The course snapshot recognizes the score as self-reported and artifact links as received; this is not instructor verification. Submitted immutable code/checkpoint commit: `419e1158ba2f83b8c08796429e635f1bf8ab2977`. V1 follow-up commits appended evidence only. See `submission/SUBMISSION_RECORD.md`.

Official benchmark: https://github.com/xudongwu-0/DASE7506-MP2/tree/bb801f4f2e7629d270df765ab12c333b2dc616bb

The official data, numeric parser, prediction runner and scorer are preserved byte for byte. Development fits only training data; validation selects the method. The complete test was scored only after the pre-test freeze.

| Method | Validation correct / 798 | Accuracy |
|---|---:|---:|
| Fixed candidate 0 | 624 | 78.1955% |
| Majority parsed answer | 688 | 86.2155% |
| Embedding-only BCE | 715 | 89.5990% |
| Relation features + BCE | 704 | 88.2206% |
| Relation features + multi-positive listwise (selected) | 724 | 90.7268% |

**Frozen official test: 348/467 = 74.51820128479658%.** This measures final numeric answer correctness, not the validity of every reasoning step. No alternative method was tested on the test bank. Test contains answer-disagreement groups; the prespecified validation-hard subset was 282/355 = 79.4366%. These subsets are not guaranteed to have identical distributions.

The selected model uses 2,064 features: the provided 1,024-dimensional embedding, its absolute deviation from the question's candidate mean, and 16 similarity/agreement/format features. Train-fitted standardization and a shared linear scorer use multi-positive listwise loss, L2=0.1. All positive candidates are retained. A train-only three-fold OOF logistic calibrator estimates correctness of the chosen final answer; it does not change choices.

The six-configuration inner-training screen, three full fits, and three OOF fits took 56.07 seconds locally. Eleven selector fits converged; the unselected listwise L2=.01 screen hit the 200-iteration limit. Relation+BCE was a negative result. The selected method's +1.1278 percentage-point validation advantage over embedding BCE has a paired bootstrap 95% interval [-0.7519, 3.1328]; it is not established as a stable improvement. Repeated selection also makes validation estimates optimistic.

## Install and obtain official data

Use Python **3.12** and Git. The following commands run from the repository root. Select the immutable code commit in your submitted code URL; the inference/training source freeze is `8602992842d444524ab4148991da2f1dc88f316f`. Later commits add reports and records without changing frozen source or weights.

```text
git clone https://github.com/liuqichang0305/dase7506-mp2-solution.git
cd dase7506-mp2-solution
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements-inference.txt
.venv\Scripts\python.exe code\tools\fetch_official.py
.venv\Scripts\python.exe code\tools\verify_official.py --output results\reproduced-official-hashes.json
```

Linux/macOS:

```sh
.venv/bin/python -m pip install -r requirements-inference.txt
.venv/bin/python code/tools/fetch_official.py
.venv/bin/python code/tools/verify_official.py --output results/reproduced-official-hashes.json
```

Only `numpy==2.5.3` is required for inference and the official tools. The download step obtains only the official coursework data at the fixed upstream commit and checks SHA-256. It needs network access; evaluation needs none. Models and configuration are already in `artifacts/final_bundle/`; do not retrain for score reproduction. The fetcher refuses to overwrite a mismatched existing file. On Windows, Git's newline conversion can corrupt official JSONL hashes; this fetcher writes the original bytes and `.gitattributes` keeps supplied source at LF.

## Reproduce without training

Run from the root, selecting the appropriate virtual-environment Python path above:

```powershell
.venv\Scripts\python.exe code\tools\evaluate.py --checkpoint artifacts\final_bundle --split test --output-dir results\reproduced --freeze submission\PRE_TEST_FREEZE.json
```

On Linux/macOS use `.venv/bin/python` and forward slashes. The wrapper verifies frozen source/asset hashes, then starts these two official commands sequentially from `code/`:

```text
python predict.py --method solution.py --checkpoint ../artifacts/final_bundle --split test --output ../results/reproduced/test-predictions.json
python score_predictions.py --split test --predictions ../results/reproduced/test-predictions.json --output ../results/reproduced/test-score.json
```

Use the wrapper for thread limits and complete external timing. It sets OMP/OpenBLAS/MKL/NumExpr/BLIS and other thread limits to 1 before imports, and includes process startup, imports, data, assets, preprocessing, inference and independent scoring. It refuses to overwrite existing results; choose a new output directory for an explicitly intended reproduction. Original official output is `results/test-score.json`. Do not use new test results to change the method.

The initial full local test evaluation was **0.705959 seconds** on Windows 11, Intel Core i9-12900H, Python 3.12.0, 16.82 GB RAM. The isolated inference audit found one OpenBLAS pool with one thread. Local timings do not certify the course's Xeon Platinum 8457C reference machine. All observed timings, including slower development runs, are retained.

## Assets and frozen version

Total additional inference assets, conservatively including NPZ headers/container and configuration: **51,923 bytes = 0.0495186 MiB**, below 16 MiB.

| File | Bytes | SHA-256 |
|---|---:|---|
| `artifacts/final_bundle/weights.npz` | 51,674 | `a785f4a48e2d5708aec9fec29c776e9ce1b7fea1832d064e95b25c926626d462` |
| `artifacts/final_bundle/config.json` | 249 | `24473c3b95b9472b5d844b795edf86c7902d21fe7351604cc680d93d7ead0703` |

See `submission/PRE_TEST_FREEZE.json` for every source hash, configuration, seed, validation selection rationale and timestamp. `results/asset-audit.json` counts uncompressed archive members and discloses array payload bytes. The official runner imports `solution.py`; the unchanged upstream `submission.py` remains an intentionally empty interface and is not used.

## Reproduce training and development

Use a separate fresh checkout for training so that archived experiment records remain intact. Install `requirements-training.txt` (NumPy 2.5.3 and SciPy 1.18.1). On Windows:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements-training.txt
.venv\Scripts\python.exe code\train.py --config code\configs\recommended.json --output artifacts\retrained_bundle --calibrate
```

This fits only official train, with seed 3036900957; the OOF fold seed is 3036900958. It never loads test. No GPU, external data or pretrained weights are required. `--search` implements the prespecified six-configuration screen but refuses to overwrite a completed `results/development/search_summary.json`. The archived original search is complete; it need not be repeated. Training is deterministic in the recorded environment, but numerical convergence can vary across library/hardware versions; the supplied frozen checkpoint is authoritative for score reproduction.

## Audit, report and evidence

`reports/report.pdf` is the English report (7 pages, including references); its editable generator is `code/tools/build_report.py`. `docs/UNDERSTAND_ZH.md` explains the project in Chinese with self-check questions. Validation confidence diagnostics have Brier score 0.067713 and 10-bin ECE 0.021821; these are diagnostics, not guarantees for unseen data.

```powershell
.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.venv\Scripts\python.exe -m unittest discover -s code\tests
.venv\Scripts\python.exe code\tools\audit_assets.py artifacts\final_bundle --output results\reproduced-assets.json
.venv\Scripts\python.exe code\tools\check_contract.py --checkpoint artifacts\final_bundle --output results\reproduced-contract.json
```

Additional documentation and evidence:

- `docs/CODE_REVIEW.md`: independent AI internal review, not a human or instructor approval.
- `docs/REPRODUCTION.md` and `results/reproduction/`: clean-environment reproduction evidence when completed.
- `results/development/`: actual configurations, train folds, outcomes, official validation scores and failures.
- `results/reliability/`: reliability/risk curves, denominators and explanatory validation failure cases.
- `docs/literature_review.md` and `docs/references.bib`: original sources and actual reading scope.
- `submission/SUBMISSION_RECORD.md`: actual submission status and exact public links when issued.
- `submission/PEER_REVIEW_RESUME.txt`: resume only after the instructor actually publishes MP2 links. No future automatic review is promised.

## Attribution and AI disclosure

See `docs/AI_DISCLOSURE.md`. Codex and AI subagents assisted with code, experiment execution, interpretation, reports and internal audits; this does not establish that the student has personally checked or mastered the work. Human understanding checks remain for the student.

Official benchmark tools are reused unchanged. Questions/references derive from GSM8K under the retained `code/data/LICENSE.txt`; attribution is in `code/data/NOTICE.txt`. Provided candidate solutions/embeddings were prepared using Qwen3-0.6B (revision in the official manifest). No Qwen or other pretrained weights were downloaded. The original-paper ideas are adapted to the four supplied candidates and resource constraints; no claimed paper benchmark result is presented as a result of this project.
