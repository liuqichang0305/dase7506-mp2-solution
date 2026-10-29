# V3 training, resume and OOF contract

The registered entry is `python code/train_v3.py --config <registered.json>`. It performs common three-fold train CV; `--fold 0|1|2` is an explicit partial run and does not fabricate a full-CV summary. Completed evidence is retained under `results/research_v3/<experiment_id>/`, inference bundles under `artifacts/research_v3/<experiment_id>/fold0..2`, and trusted training states under ignored `tmp/research_v3/training_states/<experiment_id>/`.

`experiment_id` and the alias `id` accept lowercase letters, digits and underscores. Conflicting aliases are rejected. The trainer records normalized and original-input config hashes; content hashes, not a moving Git commit label, control resumption. `training.SOURCE_FILES` is the exact dependency list shared with the scheduler. Source files, four official train/manifest files, and the common-fold manifest are checked. Git commit and dependency versions remain provenance metadata.

## Data and model selection

Only `load_development('train')` is called. The fixed fold manifest is checked against train ID order, disjoint inner/outer membership and exact whitespace-normalized duplicate questions. Input-derived relation features are computed per question. Standard or robust preprocessing fits the exact same inner-fit/outer-fit partition as its model. Each neural outer fold first selects an epoch on its inner stopping set, then reinitializes and trains on the entire outer-fitting set for that fixed number of epochs. Final all-train fitting uses the ceiling of the median inner-selected epoch, with a new all-train preprocessor. Outer holdout labels never choose epochs.

`ema_decay` enables EMA evaluation. Its state is updated after both optimizer components step; the inner stopping objective evaluates EMA, and outer/final refits use the same EMA recipe. No averaging across independently initialized models or differently fitted preprocessing coordinates occurs. The registered learning-rate schedule is currently constant; unsupported schedules are rejected instead of silently ignored. Gradient clipping and any candidate permutation augmentation are explicit config fields. Permutations move features, labels and group IDs together.

TabM returns `[N,K,4]`. Training averages the K separate multi-positive/group/BCE/pairwise losses. Inference averages answer-group probabilities, not raw logits. Raw member scores are preserved. The OOF archive also stores two-dimensional scores representing each mean group probability on the first member's highest-scoring original candidate; other candidates receive negligible finite scores. Consumers must use `summary['oof_decoder']` (canonical masked LSE, alpha=0, tau=1 for TabM). The original member decoder remains in the registered config. Direct group-probability decoding is authoritative; every completed fold checks that the collapsed representation returns the same indices. Exact ties choose the group appearing earliest, then the first member's highest-score candidate, with smallest original index for a within-group tie.

Pairwise loss averages `softplus(negative_score-positive_score)` over valid positive–negative pairs within a question, then over full questions or the input-defined hard subset. Correct candidates do not become each other's negatives. No-pair questions contribute zero, and changing the outer averaging denominator is not described as an architecture improvement.

## Recovery and preservation

Each completed epoch atomically replaces `current.pt` after saving live weights, optimizer states, EMA weights/counter, epoch, best stopping epoch/loss/state, preprocessing tensors, local NumPy generator state, Torch RNG state, constant-schedule state, history, elapsed time, and source/config/data/fold identity. Recovery uses `torch.load(..., map_location='cpu', weights_only=True)` only on the project's own state path. It continues at the next epoch. A partially computed epoch is deliberately replayed; the scheduler's wall-clock/attempt logs account for interrupted work that was not in the last durable epoch record.

The completed-fold records hash raw/collapsed scores and every inference asset. Resume verifies them and reuses them without fitting. A completed summary is not rewritten. Existing complete bundles must match model architecture, preprocessing mode, decoder and all arrays. A weights-only partial V3 bundle is renamed to a preserved `.incomplete-<UTC>` sibling before re-export; old generations are not deleted or overwritten.

`tmp/research_v3/trainer.lock` is created exclusively and records PID, process creation time, command line and a nonce. An active matching process blocks a second trainer. Stale records are renamed and retained. Only the owning process removes its lock. A fresh incomplete lock is treated conservatively as being initialized.

`SEARCH_CLOSED.json` and finalist selection locks prohibit further CV. Final fitting requires an explicitly allowed finalist component. `submission/v3/FREEZE.json` prohibits all subsequent fitting/search; reproduction after freezing uses the prediction entry instead.

The Python audit hook permits only the official train files and public manifest within the official data directory, rejects writes there, rejects leaderboard/ranking-background reads, and prohibits writes to preserved V1/V2 experiment/artifact directories. This is an application-level guard, not OS-level isolation of every agent. The trainer does not import test labels, run validation, invoke the scorer or read ranking feedback.

## Actual critical checks

- Optimizers: 8/8 synthetic tests passed; clean rerun **1.116 seconds**, `optimizer-synthetic-tests-clean.log`.
- Training/recovery: 7/7 synthetic tests passed, **8.569 seconds**, `training-synthetic-tests-final.log`. Includes exact interrupted-vs-continuous Muon+EMA+dropout+permutation weights, fit-only preprocessing, TabM export, real read-hook rejection, live/stale locking, partial bundle preservation and completed three-fold reuse with zero retraining.
- Selection/member losses: 5/5 synthetic tests passed, **1.256 seconds**, `selection-synthetic-tests.log`. Includes independent pairwise finite differences, all-positive/no-pair cases, member-loss averaging, group probability/index equivalence and exact ties.

All logs are under `results/research_v3`. These are implementation checks on synthetic data, not MP2 performance measurements. Real CV results and total campaign wall cost are produced by the separately controlled scheduler.
