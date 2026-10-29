# Internal AI code review

Reviewer: Codex `audit_tools` sub-agent, independent of the model-implementation sub-agent. Date: 2026-10-29 (UTC+8). This is an internal AI review, not a human peer review, GitHub approval, or instructor certification.

## Scope and result

Reviewed `code/solution.py`, `code/mp2/{features,model,training}.py`, `code/train.py`, the prespecified search plan, development results, the recommended checkpoint, and the unchanged official loading/parser/scoring interfaces. No test inputs or test references were parsed, and no test predictions/scoring were run by this reviewer. Official-file verification only hashes bytes for integrity and does not expose labels or examples.

The selected candidate implementation has no remaining blocking issue in the reviewed scope. A complete clean-environment reproduction, final freeze, actual full-test evaluation, public packaging, report, and submission evidence remain separate release gates; this review alone does not establish that they are complete.

## Findings and resolution

1. **Resolved before the first training search: listwise positive underflow.** Initial `loss_gradient` used one maximum over all candidates; a very low-scored positive could underflow to zero, making the positive logsumexp invalid. The implementation agent changed this to independent all-candidate and positive-candidate maximum shifts before the first search was launched. Synthetic extreme-score, finite-difference gradient, and all-positive-group tests now pass. The extreme example `[1000, -1000, -1000, -1000]` with only the second candidate positive returns finite loss 2000 and a finite gradient.
2. **Preserved negative result: one inner-screen optimizer hit its cap.** `relation-listwise-l2-0.01` reached 200 iterations (`optimizer_success=false`, message `STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT`), with inner-holdout accuracy 86.01472134595163%. It was not selected. The selected `relation-listwise-l2-0.1` inner run converged in 38 iterations; its full training run and selected-model OOF runs are separately logged. Do not describe all exploratory fits as converged.
3. **Resolved audit-harness false positive.** The first isolated contract run blocked host/null-device access from NumPy's lazily imported testing helpers, not inference code. The harness now imports its own assertion support before installing the inference restrictions. The failed record is retained at `results/contract-validation-harness-first-attempt.json`. No model or checkpoint was changed for this fix. Both subsequent isolated runs passed.

## Data, fitting, and inference review

- The development loader rejects every split except train and validation before accessing any files. Labels use the unchanged official `parse_answer`, preserve every positive candidate, and require matching row IDs/questions for alignment.
- Every candidate of a question stays in the same inner/OOF fold. Feature standardization fits only the current training rows. Full model parameters fit all and only official train. Validation is used only for the prespecified three-family comparison.
- Candidate features use passed embeddings, candidate texts, within-question absolute deviations, similarity, answer agreement and format/length statistics. No ID, answer lookup, external model, data download, or fixed bank-size branch appears in inference.
- The calibrator fits selected-answer correctness on three train-only OOF folds. It does not alter candidate selection. Raw selector scores are not claimed to be probabilities.
- `solution.py` imports only NumPy-based feature/model code and the official parser. It loads exactly `config.json` and `weights.npz` from the supplied checkpoint using `allow_pickle=False`. No hidden model arrays, external paths, cached predictions, reference loading, or fitting operation were found in the inference source.
- Input validation rejects malformed feature shape and nonfinite embeddings. Missing checkpoint paths fail clearly. Ties use the first maximum; this is not universal candidate-order invariance.

## Executed checks and evidence

Commands below were run from the project root with Python 3.12.0.

```powershell
.venv\Scripts\python.exe -m unittest discover -s code\tests -v
.venv\Scripts\python.exe code\tools\check_contract.py --checkpoint artifacts\experiments\recommended_bundle --input-source validation --output results\contract-validation.json
.venv\Scripts\python.exe code\tools\check_contract.py --checkpoint artifacts\experiments\recommended_bundle --input-source synthetic --output results\contract-synthetic.json
.venv\Scripts\python.exe code\tools\audit_assets.py artifacts\experiments\recommended_bundle --output results\asset-audit-recommended.json
.venv\Scripts\python.exe code\tools\evaluate.py --checkpoint artifacts\experiments\recommended_bundle --split validation --output-dir results\audit-evaluation
python code\tools\verify_official.py --output results\official-hashes.json
```

- **8 unit tests passed**: official numeric equivalence, real Python int output contract, uncompressed asset accounting, an unfrozen-test rejection, development-test rejection, multi-positive gradient, all-positive zero loss/gradient, and extreme-score stability. These are synthetic or interface checks, not additional benchmark searches.
- **Isolated contract checks passed** on 37 validation input rows and 23 synthetic rows, including N=0,1,3,7 and full batch; changed IDs; reordered questions; deterministic repeat; simultaneous candidate text/embedding permutation; score equivariance; malformed/nonfinite inputs; missing checkpoint. Both runs had zero candidate-permutation mismatches and no tested tied maxima. Inference ran in a temporary copied directory without data or references; the Python audit hook prohibited external-file reads, writes, sockets and subprocesses.
- **Threadpool check passed**: one OpenBLAS pool, `num_threads=1`. Evaluation processes run sequentially. All scientific thread environment variables are fixed at 1 before child-process imports.
- **Official file hashes passed**, including the pinned package-manifest hash. No official source/parser/scorer/data bytes were changed by this work.
- **Asset count: 51,923 bytes**, well below 16,777,216. All bundle files and every NPZ member/array are inventoried. Config SHA-256: `24473c3b95b9472b5d844b795edf86c7902d21fe7351604cc680d93d7ead0703`; weights SHA-256: `a785f4a48e2d5708aec9fec29c776e9ce1b7fea1832d064e95b25c926626d462`.
- **Independent official validation:** 724/798, accuracy `90.72681704260651`. Fresh prediction plus independent scoring measured **0.5143444000000272 seconds** locally. Both return codes were 0. Full commands, stdout/stderr, component times, OS/Python and thread settings are in `results/audit-evaluation/validation-timing.json` and adjacent logs. The earlier development run measured 1.4970785000000433 seconds; both runs are retained, not cherry-picked. These local Windows/i9-12900H measurements do not certify the instructor's Xeon reference-machine eligibility.

## Limitations

The file-access hook is an engineering check for this reviewed Python method, not a security sandbox for arbitrary hostile native code. The inference source has also been inspected. Synthetic and 37-row interface checks do not prove correctness for every future input, and selection accuracy does not verify every reasoning step. Validation reuse for selecting three families can produce optimistic estimates; small improvements require the reported paired-bootstrap uncertainty. Clean-directory packaging/reproduction and public-download checks must be recorded independently before release.
