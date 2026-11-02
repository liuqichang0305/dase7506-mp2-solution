# Autonomous v3: registered bounded train-only campaign

Started 2026-10-29T16:22:42Z (30 October00:22:42 UTC+8). Preserve v1/v2
methods, freezes, submissions, searches and all existing uncommitted records.
The user explicitly authorizes this subsequent research and one final frozen
test/submission. The old348/467 test aggregate and historical v2 validation
results are known; this is not an untouched holdout experiment.

## Budget, data and continuation

The research clock is anchored to startup and cannot reset on resume:24hours,
ending2026-10-30T16:22:42Z, earlier if the deadline minus8hours is reached.
Deadline2026-10-31T15:59Z. At most96 distinct configurations, at most2 justified
repair retries per failed experiment. One CPU training process, one compute
thread initially. No paid compute. Official inference timing runs without any
training process. Scheduler persists PID/create time, timestamps, status,
commands, stdout/stderr, configuration/data/code hashes and checkpoint location.
Interrupted epochs resume from an atomic trusted local training state; successful
folds/results are verified and reused. Completed experiments are never retrained
merely to select a better outcome.

Pilot `pilot_mlp_adamw_v3` actually completed fold0 in26.866seconds including
loading; inner stopping took7.567seconds and selected epoch6, outer refit
3.699seconds, hard530/705. It is a timing/screening observation, not full CV.
It used preserved v2 primitives; the v3 main table identifies its own implementation.

Only official train supplies labels during research. Exact question strings
after whitespace normalization have no duplicate groups (4,755 unique), so reuse
the previously persisted question-level3fold protocol with fixed inner stopping
sets. V2 OOF may be used as a historical comparator only after source/data/fold
hash verification. New features/scalers/robust clipping/PCA fit inner training
rows, then a fresh complete outer-train fit uses the internally chosen epoch.
No outer labels choose epochs. Final epochs are the median inner choices, rounded
up; preprocessing and weights always come from the same fitting stage.

Research entry points reject validation/test labels and leaderboard/background
files. The one startup rank target is stored separately, never loaded into the
queue or model selector. No leaderboard refresh during development.

## Registered search rounds

Each round adds2--4 hypothesis-specific configurations; no full Cartesian grid.
All main-table configurations complete the shared three outer folds. Initial
seed3036900957 is fixed; stability seeds3036900958/959 are retained, not cherry-picked.

1. Optimizer control: same2064-128-64-1 MLP, dropout0.3, groupCE, standard scaling,
   AdamW lr0.0003 wd0.001; compare EMA0.99, and Muon hidden-matrix lr0.01/auxiliary
   AdamW lr0.0003. The fourth optional control changes only Muon lr to0.02.
2. TabM: k4 width128 depth3 from random initialization; member-specific first
   input scaling before any mixing, member losses averaged independently. Start
   lr0.001 dropout0.1 wd0.001. Compare dropout0.3 and k8 as separate changes.
3. RealMLP-inspired: width128 depth3, NTP/SELU/learnable input scale, AdamW
   beta2=0.95 lr0.01, bias multiplier0.1/input-scale6; compare standard,
   robust-statistics-only and robust-all preprocessing. Median/IQR fallback and
   smooth clipping are fit/applied within each fitting partition. This is a
   task adaptation, not a full reproduction of the author's tuned estimator.
4. Transfer optimizer evidence to the strongest existing interaction family:
   fixed Attention width128 groupCE lr0.0003 dropout0.3 wd0.01; compare EMA0.99,
   Muon with an independently registered matrix LR, and smaller width64.
5. Controlled input/complexity hypotheses: compare embedding-plus-statistics
   versus relation features for the best feed-forward recipe; one smaller width
   and one deeper residual/context variant. Register exact2--4 configs only after
   the preceding OOF evidence, before their fitting.
6. Objective controls on the strongest eligible family: original multi-positive
   listwise, BCE and question-averaged positive/negative pair loss where supported.
   No correct candidate is relabeled negative. Decoder controls reuse scores and
   distinguish refitting from decoding; alpha0/0.5/1,tau1 are a bounded grid.

Next rounds must state an independent hypothesis and exact configs before
execution. Existing v2 mechanism/MLP/interaction evidence is reused, not rerun
mechanically. At least the optimizer and TabM/RealMLP routes must complete before
saturation can be invoked. Engineering continuation threshold:0.5percentage
points in common-fold hard mean. Close candidates get fixed-seed confirmation.
Stop when budget/cap/buffer is reached or, after required coverage,3consecutive
distinct hypothesis rounds fail this increment and no feasible evidence-backed
new hypothesis remains. A high score alone does not stop the campaign.

## Evidence, fusion and selection

Rank eligible methods by resource compliance, hard mean, no unexplained full
accuracy regression, seed stability, then complexity. Report pooled correct/n,
fold mean/sampleSD, single-positive diagnostic, parameters, uncompressed bytes,
fit wall cost, source and dependencies. Question-paired bootstrap5000draws,
seed3036901059; selected OOF remains optimistic. Initial candidate epochs max60,
patience10; fixed changes to epoch cap need internal-curve evidence and registration.

Promising families receive the two fixed extra seeds. Deployment always uses
seed3036900957; the other seeds assess family stability and cannot be selected
for their better score. Incomplete/failed/pruned configurations remain visible
and are not ranked as complete three-fold CV results. At most2--3small members
with reciprocal error corrections and hard oracle potential >=1point may enter
fixed equal-probability fusion. Do not average logits or independent-model
weights. No learned stacker/temperature affects fusion unless its evaluation is
strictly nested inside every outer training set. If that cost is unjustified,
omit learned fusion. Post-selection confidence calibration may fit train-OOF
chosen-answer correctness, but is not used to change selections; its fitted-OOF
calibration metrics must be labeled in-sample, with validation confirmation.

Before new validation, seal at most2new finalists and close search. Historical
v2 Deep Sets740/798,hard297/355 and Attention738/798 are already-known development
comparators, not new evaluations. To replace the historical v2 Deep Sets, a new
finalist must satisfy all resource/contract checks, use the fixed deployment
seed, and have hard_correct >=297/355 AND full_correct >=740/798, with at least
one strict improvement. Among eligible new finalists choose greater hard_correct,
then greater full_correct, then smaller uncompressed assets, then lower measured
fresh-process latency. An exact tie with the historical model retains it. If
neither new finalist qualifies, retain that historical checkpoint and do not
reopen search. No renaming campaigns to reset this confirmation limit.

Complete export equality, input contract, offline/no-reference inference,16MiB
asset audit, full validation fresh-process timing, independent AI review and
new-environment frozen-checkpoint reproduction. Then freeze the unique selected
method and irreversibly close research before official test. New user's final
test authorization applies absent a concrete contrary course restriction. One
official test result, whatever its score/rank, leads only to reporting and the
authorized immutable code/asset submission; it never returns to training.
The report remains <=10pages, with negative results, provenance, AI/source reuse
and prior test exposure disclosed. Peer review waits for actual instructor release.
