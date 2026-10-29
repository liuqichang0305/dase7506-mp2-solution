# Development results and provenance

The prespecified search completed once with exit code 0, using only official train and validation. Total measured search wall time was 56.0689 seconds on the local Windows CPU. All recorded source hashes match the exact source snapshot `d5f149e06970d472a962fc31cbdab8adf3353f3a4400c2088d0f28f92023564d`. `provenance_audit.json` lists every source file hash, training-data hash, environment, actual command and parent-experiment relationship. `experiments.jsonl` retains original records, and `search_summary.json` consolidates them.

| Validation method | Correct / N | Accuracy (%) | Hard correct / N | Hard accuracy (%) |
|---|---:|---:|---:|---:|
| Fixed candidate 0 | 624 / 798 | 78.1955 | 195 / 355 | 54.9296 |
| Majority parsed answer | 688 / 798 | 86.2155 | 245 / 355 | 69.0141 |
| Embedding BCE, L2 0.01 | 715 / 798 | 89.5990 | 274 / 355 | 77.1831 |
| Enriched BCE, L2 0.1 | 704 / 798 | 88.2206 | 261 / 355 | 73.5211 |
| Enriched listwise, L2 0.1 | 724 / 798 | 90.7268 | 282 / 355 | 79.4366 |

The enriched listwise method is the recommended candidate. The enriched BCE method is a negative result relative to embedding BCE. The paired question-bootstrap difference between enriched listwise and embedding BCE is +1.1278 percentage points, with a 95% percentile interval of [-0.7519, 3.1328]; this does not establish a clear improvement. Against majority voting, the difference is +4.5113 points, interval [2.3810, 6.6416]. These intervals do not remove validation model-selection optimism.

Of 12 selector fits, 11 met the optimizer's convergence criterion. The inner-screen enriched listwise model with L2 0.01 reached the prescribed 200-iteration cap (holdout accuracy 86.0147%) and was not selected. Its `optimizer_success: false` and exact message were always retained in the raw log. An AI progress message incorrectly said all fits converged; that claim is explicitly corrected here and in the provenance audit. No optimization was rerun to hide this failure.

Selected-method three-fold train-only OOF accuracies were 89.4637%, 88.4543%, and 88.6435%. Calibration fits the selected-answer correctness event from all 4,755 OOF predictions; it does not change choices. Confidence features are selected maximum score, runner-up margin, and selected parsed-answer agreement fraction. The downstream calibration is logistic BCE with L2 0.01. All selector and calibration normalizers fit only the relevant train partitions.

The recommended checkpoint is `artifacts/experiments/recommended_bundle` (relative to project root). Refit without selecting or evaluating any new method:

```powershell
.venv\Scripts\python.exe code\train.py --config code\configs\recommended.json --output artifacts\retrained_bundle --calibrate
```

Do not rerun `--search` after test access. The search entry point refuses to repeat a completed search. No test input, reference, or score was accessed by this development work.
