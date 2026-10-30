# Performance v2: preregistered development protocol

Registered 2026-10-29 before v2 fitting or new finalist validation access.
Base: `d9585ce4ea52877a448ff146fac5d942ac71b7a1`; submitted release
`419e1158ba2f83b8c08796429e635f1bf8ab2977`. Research branch:
`research/performance-v2`. The old 348/467 aggregate test result and leaderboard
were already known. This is subsequent development, not an untouched experiment.
No test inputs, references, predictions or sample errors will be used in research.
New official test scoring and score updates are disabled pending an explicit
applicable course process. Old search guards, logs, assets and submission stay intact.

## Budget and stopping

Use local CPU, one training process with one compute thread, no purchased compute.
Measure one MLP outer-fold pilot (at most 60 epochs, patience 8), retaining its
results. Use that measurement to set a practical wall budget; absolute research
training cap 6 hours, stop earlier after two hypothesis rounds without stable gain.
Reserve at least 6 hours before the October 31 23:59 UTC+8 deadline. Neural
architecture width is 128; no broad Cartesian search. Per family at most four
structural/hyperparameter configurations. GPU is present (3070 Ti laptop, 8 GB)
but CPU is sufficient unless the measured pilot proves otherwise.

## Data and common folds

Only official train (4,755 questions) is loaded by `training_v2.py`.
Three common question-level outer folds use seed 3036900957, stratified by
positive-candidate count and distinct valid-answer count, rare strata merged.
Each outer training set has a fixed 20% inner stopping subset (seed 3036901057+fold).
Neural scalers are fitted on inner fit rows to choose an epoch by inner objective
loss, then a fresh model/scaler is fitted on the complete outer training set for
that fixed epoch count. Outer labels never choose epochs. Linear models fit the
complete outer training set with fold-local standardization. Inner split and
outer memberships are persisted and checked before reuse. Final all-train epoch
count is the median of the three inner-selected epochs, rounded up.
No scaler, calibrator or feature selection is fit on validation.

## Registered experiments

1. Reproduce old relation/listwise L2=0.1 baseline, old train-inner split and
   frozen validation predictions in a new location. Refit this baseline on the
   common three folds. Decode identical OOF scores with original argmax, valid
   candidate max, group max (equivalence check), and group LSE at tau=1,
   alpha in {0, 0.5, 1}. All-None fallback is fixed original index 0.
2. Linear group CE with alpha {0, 0.5, 1}, full-question normalization, L2=0.1.
   Hard-only group CE alpha=0 uses L2=0.1 / train-hard-fraction, to preserve the
   equivalent objective scale. Report this scaling identity, not as a new gain.
3. MLP 2064-128-64-1, ResMLP width128 one residual block, and Deep Sets
   1024-128-128 phi / candidate mean-max interactions / 128-1 rho.
   Start dropout0.1, AdamW lr0.001, weight_decay0.001, batch128,
   max60 epochs, patience8, clip gradient norm5. Use masked group CE alpha=0,
   full-question normalization and group-LSE decoding. MLP original listwise
   is a loss control. The same score arrays support the registered decoder
   controls; do not claim decoder variants are independently fitted models.
4. If pilot budget permits, one attention model width128, heads4, FFN256,
   no position encoding. TabM is not required and is marked unrun if omitted.
5. Take the best two nonlinear structures by mean outer hard accuracy and add
   fixed seeds 3036900958 and 3036900959; use the average across all three seeds
   for stability comparisons, never choose the lucky seed. Deployment single
   model seed stays 3036900957. On the best family add hard normalization with
   proportionally scaled weight decay and a context ablation if Deep Sets wins;
   at most two additional hypothesis configurations across this round. For
   linear L2, scaling by the fitting subset's hard fraction is an exact
   objective rescaling; AdamW decay scaling is only a controlled comparison,
   not an equivalent optimization trajectory.
6. Examine paired OOF errors of the two strongest families (including baseline).
   Only if each corrects at least 10 errors of the other and their oracle gains
   at least 1 percentage point on hard questions, test equal group-probability
   mixture of two seed3036900957 models. No weight grid/stacker. All mixture
   assets count toward 16 MiB. The mixture is a candidate, not presumed better.

## Selection and one-way validation gate

Primary ranking: mean common-fold hard accuracy; then overall accuracy; then
seed stability and smaller resource cost. Hard = at least two distinct valid
parsed answers, fixed from inputs. Report pooled correct/n, fold mean and sample
SD, plus single-positive diagnostic. Group probabilities treat correlated
trajectories as an aggregation rule, not independent evidence.
Winner and optional second finalist are locked to disk BEFORE loading validation.
At most TWO new finalist checkpoints receive official full validation scoring.
The already-known frozen baseline is also replayed as A0. There is no reopening
of search after validation. If neither finalist beats baseline convincingly,
deliver honest negative/uncertain evidence and preserve v1 as the official release.
Final validation winner uses full/hard counts jointly, resource compliance and
paired uncertainty, not single-positive subgroup selection. Only these finalists
get validation columns; all other table cells say not evaluated by protocol.
Question-paired bootstrap (5,000 draws, seed3036901059) compares principal OOF and
validation differences. Hyperparameter selection makes OOF optimistic; confidence
intervals do not remove adaptive-selection bias. Seeds reuse questions, so they
are not additional independent data.

## Outputs and verification

Append-only experiment events record IDs, config hashes, code/data hashes, folds,
seeds, status, seconds and failures. Completed results are reused only after
hash/provenance checks. Per-fold and final NPZ assets, raw OOF scores, epoch logs,
resource measurements and exact commands are retained in independent v2 paths.
Critical synthetic tests cover group losses/gradients, parser semantics,
all-None fallback, all-positive labels, permutation and NumPy/Torch parity.
Finalist validation uses fresh official prediction and separate scoring
processes, one compute thread, with total elapsed time. Inference is NumPy-only.
Audit all uncompressed assets, no references/network during inference, arbitrary
N/IDs/order, and fresh-environment validation reproduction without retraining.
Record group-probability confidence Brier/ECE and actual chosen-group correctness;
do not reuse v1's calibration. No calibrated-probability claim without evidence.
Freeze v2 source commit and hashes, retain old PDF, add a <=10-page v2 PDF,
disclose AI and architecture references, and open a reviewable PR after fixes.
