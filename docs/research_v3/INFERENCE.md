# V3 inference and export audits

`code/solution_v3.py` exposes the official `predict(bank, arrays, checkpoint)` interface and returns original candidate positions as Python integers. It reads only the supplied candidate inputs and an explicit frozen numeric checkpoint. It also exposes `predict_confidence` (uncalibrated probability of the actually selected numeric answer group), `score_members` (one raw array per model, including `[N,K,4]` for TabM), `score_candidates` (finite two-dimensional diagnostic logits), and `prediction_details`.

Single V3 bundles support all V3 architectures and the imported legacy architectures. The original V2 bundle schema is also accepted by dispatching to its unchanged loader/scorer; no asset conversion is necessary. A schema mismatch is rejected. Ordinary two-dimensional models retain their registered decoder. TabM averages each member's answer-group probabilities, chooses the highest-probability group, and uses the first member's highest-scoring candidate within that group. Equal group probabilities use the group with the smallest original candidate index; within-group ties use the smallest index. A masked all-unparseable question returns index 0 and confidence 0. IDs do not affect features or selection.

`score_candidates` is an analysis representation, not averaged TabM logits. For TabM and a manifest it places each group's log probability on the designated representative, with finite negligible scores elsewhere. Raw score export and permutation checks use `score_members`; exact decisions use the probability decoder. This avoids making diagnostic log-space roundoff authoritative at exact ties.

## Optional fixed manifest

No fusion is selected by implementing this format. Any deployment choice still requires the preregistered train-OOF complementarity and selection process. The only supported manifest is an equally weighted average of two or three model group probabilities. The directory must contain this exact `config.json` shape plus self-contained member directories:

```json
{
  "schema": "mp2-research-v3-ensemble",
  "aggregation": "equal_group_probability",
  "representative": "first_model_first_member",
  "members": ["member0", "member1"]
}
```

Each member is a single V2/V3 bundle using masked group-LSE decoding. It first averages its own TabM members, if any, so a four-member TabM does not receive four times the ensemble weight. The first listed model and its first member determine only the representative candidate within the winning numeric group. Absolute/external paths, duplicate member paths, nested ensembles, arbitrary weights and unknown manifest fields are rejected. All child numeric/config assets count toward the total resource limit.

## Reproduction commands

Set `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `NUMEXPR_NUM_THREADS`, `BLIS_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS`, and `OMP_THREAD_LIMIT` to `1` before starting Python. The inference module also sets them before its own NumPy import. An embedding host that already imported NumPy must set its process limits before that import. The supplied audit commands set their thread environment before importing numeric libraries.

Run from the repository root, replacing the checkpoint and unused evidence paths:

```text
.venv/Scripts/python.exe code/tools/check_export_v3.py --checkpoint <frozen_bundle> --input-source train --output <new_export_evidence.json>
.venv/Scripts/python.exe code/tools/check_contract_v3.py --checkpoint <frozen_bundle> --input-source train --output <new_contract_evidence.json>
.venv/Scripts/python.exe code/tests/test_inference_v3.py --output <new_synthetic_evidence_directory>
.venv/Scripts/python.exe code/tests/test_selection_v3.py
```

The audit tools default to synthetic inputs. `train` reads candidate inputs only, with a guard forbidding references and other splits. After the final architecture/decoder/assets are locked, the root workflow can replace `train` with `validation` and append `--allow-frozen-validation`. Neither tool accepts test inputs or computes accuracy. Existing evidence and checkpoint assets are never overwritten.

The contract audit copies only code, enumerated frozen numeric assets and candidate inputs into a temporary sandbox, then launches a fresh process. A Python audit hook rejects external reads, reference filenames, writes, networking and subprocess creation during inference. It checks arbitrary N including zero and a chunk boundary, batch independence, ID independence, question/candidate ordering, exact deterministic repeats, chosen-group confidence, malformed input rejection, NumPy-only imports and actual numerical thread pools. Candidate permutation ties are checked against actual group probabilities and representative scores rather than raw candidate maxima.

The export audit reconstructs Torch models from the exact frozen arrays and compares all input rows against NumPy: every raw member score, averaged group probability and returned candidate index. It preserves float64 for linear models, records per-member score errors and asset hashes, and reports diagnostic timings without calling them official inference-plus-scoring results.

Actual development evidence: `results/research_v3/inference-synthetic-audit/summary.json` records 12 passing isolated checks across three new neural families, a V2 linear bundle and synthetic two-/three-model manifests. All 137 choices matched per export case, maximum observed raw score difference `4.917383193969727e-7`, and worker BLAS thread count was one. Temporary synthetic models were removed; no deployment ensemble was created. The real pilot MLP checkpoint also passed full train-input-only export (4,755/4,755 choices equal, maximum score difference `1.430511474609375e-6`) and isolated contract checks in `pilot-export-train.json` and `pilot-contract-train.json`. These checks measure software correctness, not predictive performance.

At this documentation checkpoint, these tasks have opened no validation or test inputs/references. Actual trained TabM/RealMLP full-input export and final frozen validation audits remain pending; passing synthetic models does not mark those later checks complete.
