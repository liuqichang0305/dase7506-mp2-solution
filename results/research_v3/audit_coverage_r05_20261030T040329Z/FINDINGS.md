# Current train-only coverage review

Snapshot: 2026-10-30 04:03:29 UTC. The queue was unchanged during the audit.
This follows, and does not replace, earlier coverage and Muon-resume audits.

## Verified current evidence

R01–R06 are now complete. R07 has one completed recipe, one running and two
pending; R09 and R10 each have three pending. R11 has exactly two prepared
reduced-feature TabM transfers, not yet in the captured queue. There are 22
completed configurations in the snapshot. All 22 have identical labels/groups/
fold order, the same eleven training-source hashes and official train hashes,
three complete 1,585-question folds and unchanged per-fold output/asset hashes.
The audit independently reconstructed choices from saved raw scores, checked
TabM probability collapse against saved scores, and reproduced every saved
full/hard/single-positive count. It performed no model forward pass or fitting.
All new validation/test opening markers were absent.

The latest R05 diagnostic's 15 sources, 30 inputs and nine outputs match their
completion hashes. Its paired counts and the one reciprocal-error gate agree
with the actual saved OOF choices. Reduced TabM gains 62 questions over its
matched full-feature k4 control, with fold gains +45/+11/+6 and single-positive
gains +41/+3/-2. Its 1,668 hard successes remain below Attention EMA's 1,685 and
the preserved historical Attention's 1,687. A positive conditional paired
interval is not evidence of stable gains over random seeds or of global family
dominance. No bootstrap or diagnostic was rerun by this audit.

No invalidating code defect was found in the examined feature, preprocessing,
loss or saved-score paths. Reduced TabM uses exactly embedding coordinates
0–1023 plus the 16 existing statistics; per-coordinate scaling is fit only on
the appropriate fitting partition, before deterministic coordinate selection.
Member objectives are averaged independently and inference averages aligned
answer-group probabilities. The raw and collapsed OOF choices agree exactly.

## Comparison limits

- The R05 config changes only `input_features`, but 2,064 to 1,040 input
  coordinates also changes first-layer parameter count, initialization variance
  and the RNG sequence consumed by initialization. This is a valid whole-recipe
  ablation, not a causal isolation of information removal or proof that the
  absolute-deviation feature block is harmful.
- R06 listwise/BCE/pairwise differ only in the declared objective from the V3
  Attention AdamW control. Listwise includes all candidates in its denominator;
  BCE includes invalid candidates as negative targets and averages four losses;
  pairwise uses valid positive/negative pairs and averages within each question,
  with zero contribution when no pair exists. Group CE excludes invalid groups.
  These are intentionally different objective/gradient populations and scales.
  Each also uses its own inner stopping objective. Identical LR/decay therefore
  compares fixed recipes, not independently optimized best-case loss families.
- R06 alpha 0.5 changes both loss and decoding, as registered. Its result is a
  joint aligned-policy change, not a decoder-only experiment. All positives stay
  positive. Pairwise gives 1,684 hard but 356 single-positive successes; group
  AdamW gives 1,682/382, and EMA gives 1,685/388. Alpha 0.5 gives 1,663/394. These
  subset tradeoffs are real and must remain visible rather than selecting a
  favorable subset after seeing results.
- Cross-family MLP, Deep Sets and Attention comparisons differ in parameter
  count and effective feature processing. R07's within-family EMA contrasts are
  matched; its cross-family maxima do not isolate the value of interaction.
  Reuse preserved no-context/capacity evidence with its actual limitations.
- The reduced-TabM/Attention-EMA gate (43/60 unique corrections, oracle hard
  +2.0341 pp) justifies the one explicitly prepared fixed equal-probability
  mixture. It does not establish attainable improvement or justify a weight
  grid, replacing members after seeing mixture results, or true-label routing.

## Remaining finite work and stopping

Finish already registered R07/R09/R10 and prepared R11 while the original
budget permits. R11 independently transfers k8 or EMA onto the reduced TabM
control; it is supported by observed family evidence and is not a Cartesian
grid. The one gated mixture, once distinctly registered, must retain its exact
members/weights and count as an additional evaluated configuration. Preserve
both earlier failed mixtures.

There is no current saturation justification: feasible preregistered hypotheses
remain. After this finite coverage, use three chronologically consecutive
completed distinct-hypothesis rounds and their pre-round global incumbent to
test the registered >0.5 pp hard-fold-mean threshold, together with absence of
any remaining feasible evidence-backed hypothesis. A family-local gain below
the global incumbent is not a global threshold crossing. Retry, deterministic
reference computation and stability rounds do not increase this count. The
absolute end remains 2026-10-30 16:22:42 UTC, with delivery buffers unchanged;
96 is a cap rather than a target. Do not invent further grids to fill it.

## Exact minimum stability protocol

1. Wait for the existing base-seed family follow-ups to finish, or document the
   actual budget-based truncation. Select one or two promising **actual completed
   family recipes** using only shared-CV evidence and the registered ordering:
   resource plausibility, hard mean, no unexplained full regression, then
   complexity. Record the base IDs, full configs, hashes and reason before any
   extra-seed result. Current provisional recipes are Attention EMA and reduced
   TabM, but unfinished family rounds can change this before the stability lock.
2. Deep-copy each chosen recipe, changing only experiment ID and base seed to
   3036900958 and 3036900959. Keep all three common outer folds, the existing
   inner partitions and deterministic fold seed offsets. This is exactly two
   additional configurations for one recipe, or four for two recipes.
3. Keep all seed results, including failures, and report paired counts plus
   three-seed mean/sample SD separately from within-run fold SD. Nine correlated
   seed/fold values are not nine independent datasets. Do not change hyperparameters
   after seed results or deploy the luckiest seed. Deployment stays 3036900957.
4. Identical historical V2 recipes already have their fixed repeats and need
   no mechanical rerun; a changed V3 recipe cannot borrow those scores as its
   own stability evidence. If a fixed mixture is retained, do not independently
   cherry-pick member seeds. No validation or leaderboard input enters this step.

The new `scripts/prepare_stability_round_v3.py` was reviewed at SHA256
`f9937daeed428d9709163eb7a9a2e00b594e6dc1e352c726e580658cd56215ef`.
It requires one/two distinct complete base-seed IDs, validates full folds and
summary/config/OOF hashes, emits only the two fixed extra seeds per recipe,
rejects already registered seeds and differing prior specs, and explicitly
marks the round as stability. This is a preparer, not a substitute for the
existing scheduler's provenance, budget and source guards. It was not invoked.

## Subsequently completed deterministic references

Source review found the historical fixed0/majority implementation had been
executed on old validation, but no common-train-fold reference rows existed.
At the parent's explicit instruction, the missing references were computed once
at 04:09:26 UTC with the existing no-holdout guard and verified train OOF labels.
There was no fitting, raw gold/embedding read, new model configuration or
official scoring. The original majority implementation and independent group
counts agree on all 4,755 choices; None has zero votes, tied winning groups use
the earliest original index, and all-None returns 0 (zero such train questions).

| Reference | Full / 4,755 | Hard / 2,114 | Single-positive / 711 |
|---|---:|---:|---:|
| Unconditional original candidate 0 | 3,584 | 1,078 | 177 |
| Valid parsed-answer majority | 4,054 | 1,413 | 121 |

Full subset/fold counts and means/sample SD, selections NPZ, source snapshots,
source/input/fold hashes and immutable completion are in
`results/research_v3/deterministic_train_references/`. Internal computation took
0.8714086 s; the external receipt records 1.2250338 s for the whole process.
These are diagnostic computation costs, not formal inference latency or fit
cost. Three new synthetic tests passed in 0.203 s, including all 625 generated
parse/vote patterns. Add the two rows to the report's train reference table,
clearly label their zero fitted parameters, and count them separately from
learned three-fold training configurations. Final report integration and the
model agent's independent source review remain separate handoffs.
