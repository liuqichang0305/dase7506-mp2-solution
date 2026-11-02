# R15: one improved reduced-TabM EMA / Attention EMA mixture

This is exactly one separately registered follow-up to the saved train-OOF gate
`diagnostic-r11-reduced-ema-vs-attention`. The gate recorded 51 improved-TabM-only
and 49 Attention-only correct questions; hard oracle 1736/2114 versus the stronger
member's 1687/2114, a diagnostic ceiling of +2.3179 percentage points. This does
not establish actual mixture performance. All prior mixture results, including
R12, remain unchanged.

The permanent ordered members are `v3:v3_tabm_embedding_stats_ema_s0` and
`v3:v3_attention_ema_s0`, with model-level weights `[0.5, 0.5]`. TabM first averages
the four internal answer-group probabilities. The winning group's representative
uses the first TabM internal member's highest raw score. Exact ties use earliest
original candidate order and all-None returns index 0. No weighting grid, other
pairs, member substitutions, routing, calibration or holdout access is allowed.

The stronger matched member is now improved TabM EMA: 1687 hard, 4328 overall,
376 single-positive correct. Reject the mixture unless hard strictly increases
and overall does not decrease. Disclose the single-positive change and the
separate comparison with Attention EMA (1685 hard, 4326 overall, 388 single).
For report compatibility, `hard_delta_questions_vs_attention_control` retains
that Attention baseline; the additional `*_vs_matched_control` fields use the
stronger improved TabM. The campaign continuation criterion remains a hard
three-fold mean increase strictly above 0.5 percentage points against the current
best completed BASE957 incumbent, frozen at registration. That inventory includes
all completed V3 recipes, preserved V2 recipes and the three previous actual
mixtures, including R12. Passing a local condition is not automatic selection.

The reduced representation still has 1024 embedding coordinates plus 16
statistics and less first-layer capacity than the 2064-input version. Adding EMA
to the reduced live recipe gained 19 hard questions (+4/+11/+4 across folds) and
12 single-positive questions, at one fixed seed. Neither this nor the oracle gate
establishes seed stability or an architecture-only causal advantage.

The new dedicated helper copies the independently reviewed R12 execution
semantics; R12 source and preregistration are never edited. It pins R12 source
SHA `ddf9af9bc8d4d1483ede52cb3e92f6b71e7bac73831d993686c509bb68bef504`.
Even `prepare` refuses until the real R12 final ledger, execution receipt,
three-result aggregate and their full hash mappings verify successfully.

```powershell
# Only after R12 register/evaluate/summarize actually completed:
.venv/Scripts/python.exe scripts/register_reduced_tabm_ema_attention_fusion_v3.py prepare
# Root operator only, with terminal queue and no scheduler/trainer lock:
.venv/Scripts/python.exe scripts/register_reduced_tabm_ema_attention_fusion_v3.py register
.venv/Scripts/python.exe scripts/register_reduced_tabm_ema_attention_fusion_v3.py evaluate
.venv/Scripts/python.exe scripts/register_reduced_tabm_ema_attention_fusion_v3.py summarize
```

The last three commands use the shared scheduler lock and reject active training.
Registration and evaluation obey the original absolute campaign clock and the
same 96-configuration cap. This mix adds one configuration, zero optimization
runs and zero newly fitted model weights. The direct base Python `-S` worker
uses only the fixed environment's site-packages, one compute thread and the
existing guarded diagnostic CLI. A timeout kills the actual worker; the
diagnostic forbids child processes. No failed or partial evaluation is silently
replayed. Every stage preserves its completed records and failures.

Future actual output locations:

- `docs/research_v3/rounds/r15_reduced_tabm_ema_attention_equal_fusion_preregistration.json`
- `docs/research_v3/rounds/v3_fusion_reduced_tabm_ema_attention_ema_s0.json`
- `results/research_v3/v3_fusion_reduced_tabm_ema_attention_ema_s0/`
- `results/research_v3/fusion_r15_execution/execution.json`
- `results/research_v3/fixed_fusion_analysis_complete.json`
- `results/research_v3/fusion_r15_final_ledger.json`

The new aggregate preserves all three previous result objects and measured
mixing-wall values, then appends the actual fourth result and full four-entry
registry hash. The new round ledger records one configuration only. Immutable
R15 ledger/record snapshots allow later stability-round appends. Recorded mixing
wall time includes startup, checks and output; it is neither inference latency
nor CPU time. The existing report CLI can use
`--fusion-analysis results/research_v3/fixed_fusion_analysis_complete.json`.

`code/tests/test_reduced_tabm_ema_fusion_registration_v3.py` uses only temporary
synthetic fixtures and mocked processes. It checks the R12 prerequisite,
three-old-plus-fourth inventory, immutable identities/hashes, current-incumbent
selection including R12, local rejection control, no overwrite/replay, actual
timing receipt construction and future ledger append compatibility. Synthetic
test success is not evidence that this mixture has actually run.
