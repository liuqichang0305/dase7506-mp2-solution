DASE7506 MP2
Post-deadline research closeout

Student 3036900957 | Research archive only | Not a new course submission

POST-DEADLINE RESEARCH ARCHIVE: V3 FORMAL TEST AND SCORE UPDATE NOT PERFORMED.

The formal deadline was missed. No V3 official test was run, no new score Issue was posted, and no V3 score update is claimed. The earlier V1 Issue #294 remains the submitted result: 348/467 (74.5182%). This historical test score does not evaluate the V3 archive model.

No locked V3 candidate improved either overall or hard validation counts over the preserved V2 fallback. The historical V2 checkpoint is retained byte-for-byte; no accepted replacement is claimed.

The retained research model is deepsets_regularized_s0. Its saved offline validation outcome is 740/798, hard 297/355. Independent review and a fresh-environment validation reproduction passed before archive freeze. These checks establish reproducibility; they neither reopen submission nor establish a test score or rank.

| Evidence | Actual scope |

| --- | --- |

| Complete V3 runs | 35 common-three-fold configurations |

| Preserved V2 runs | 19 historical configurations |

| Fixed fusions | 4 rejected comparisons |

| Incomplete campaign jobs | 4 excluded from complete-CV tables |

| V3 formal test / update | NOT PERFORMED |

Four added-seed jobs were pruned without complete CV. Preserved attempt records include an elapsed wall duration of 186401.74966692924 seconds across 2026-10-30 to 2026-11-01, with long gaps in monitoring; this is not evidence of continuous CPU work or uninterrupted supervision. The underlying cause of the gaps is unestablished. Original nominal attempt and absolute campaign deadlines were not reset; resumption stopped overdue work. The formal closure occurred on 2026-11-02 after both the research budget and course score deadline. The reserved delivery buffer was missed, and no timely V3 submission is claimed.

Deadline: 2026-10-31 16:00 UTC (2026-11-01 00:00 UTC+8). Research closure after expiry is documented, not a claim that every planned experiment finished within the original budget.


---


1. Representation and controlled comparisons

Protocol: 4755 training questions, 798 validation questions and 467 official test questions (V3 test unopened). The common 3 outer folds each hold out 1585; the remaining 3170 split into 2536 inner-fit and 634 stopping questions. Fresh outer refits use all 3170 with inner-selected epochs. All four candidates receive correctness labels; alternate correct answers stay positive. Hard questions have at least two distinct parseable answer groups. OOF-based architecture selection makes reported CV and paired intervals selection-optimistic.

Four candidate answers share an official answer parser and one question-level decision. Relation inputs contain 1,024 embedding coordinates, their 1,024 absolute deviations from the candidate mean, and 16 parsing/vote/text statistics. Reduced inputs omit the deviation block. Features exclude IDs and references. Scalers fit only the fitting partition.

The selector masks unparseable answers, groups equal parsed values and uses group log-sum-exp minus alpha times log group size, at tau=1. Deterministic ties return an original candidate index; all-unparseable rows return index0 with confidence zero. TabM averages member answer-group probabilities, not logits.

| Architecture / control | Implementation tested |

| --- | --- |

| MLP / ResMLP | Shared candidate network / residual transformations |

| Deep Sets / Attention [3,4] | Permutation-compatible candidate context, without positional IDs |

| TabM [1] | Shared matrices with member adapters; average member losses |

| RealMLP-inspired [2] | Input scale, NTP, SELU, zero head and robust preprocessing controls |

| Muon / EMA [5] | Hidden-matrix Muon plus auxiliary AdamW; same-trajectory post-step EMA |

R05 matched TabM recipe: embedding+statistics (1040 inputs) versus full relation (2064) scores 1668/2114 versus 1606/2114 hard questions: +62 (+2.93 pp; saved paired 95% interval [1.51, 4.40] pp), with fold 0/1/2 gains +45/+11/+6; all-question gain +62. This is not a pure feature-information effect: input width changes first-layer capacity and random initialization even at the same fixed seed. The interval is conditional on selected OOF predictions and retains search optimism.

| No-fit train reference | All correct | Hard correct | Hard mean +/- SD |

| --- | --- | --- | --- |

| fixed0 | 3584/4755 | 1078/2114 | 50.992 +/- 2.165 |

| valid_answer_majority | 4054/4755 | 1413/2114 | 66.840 +/- 0.642 |

Fixed0 always returns original index0. Majority ignores None, groups equal official parsed values, breaks ties by earliest original index and returns0 if all are unparseable. Their fold statistics are descriptive partitions, not three fits.


---


2. Complete V3 training experiments (1/2)

All values below are training OOF results on common question folds. Hard fold mean and sample SD are percentages; fit seconds sum the recorded inner/outer optimization cost. Full configuration IDs are immutable identifiers, not encodings of learning rates.

| Experiment ID | All correct | Hard correct | Hard mean +/- SD | Fit s |

| --- | --- | --- | --- | --- |

| v3_tabm_embedding_stats_ema_s0 | 4328/4755 | 1687/2114 | 79.80 +/- 1.94 | 231.3 |

| v3_attention_ema_s0 | 4326/4755 | 1685/2114 | 79.71 +/- 1.27 | 266.3 |

| v3_attention_pairwise_s0 | 4325/4755 | 1684/2114 | 79.66 +/- 0.65 | 210.5 |

| v3_attention_wd01_s0 | 4324/4755 | 1683/2114 | 79.61 +/- 0.80 | 80.4 |

| v3_attention_adamw_s0 | 4323/4755 | 1682/2114 | 79.56 +/- 0.75 | 181.7 |

| v3_realmlp_embedding_stats_drop01_s0 | 4323/4755 | 1682/2114 | 79.56 +/- 0.79 | 143.2 |

| v3_attention_width256_s0 | 4318/4755 | 1677/2114 | 79.33 +/- 1.08 | 186.7 |

| v3_attention_drop05_s0 | 4314/4755 | 1673/2114 | 79.14 +/- 0.57 | 116.6 |

| v3_attention_listwise_s0 | 4313/4755 | 1672/2114 | 79.09 +/- 1.33 | 170.8 |

| v3_attention_small_s0 | 4311/4755 | 1670/2114 | 79.00 +/- 0.65 | 111.1 |

| v3_attention_bce_s0 | 4309/4755 | 1668/2114 | 78.90 +/- 0.85 | 238.0 |

| v3_tabm_k4_embedding_stats_s0 | 4309/4755 | 1668/2114 | 78.90 +/- 2.08 | 279.2 |

| v3_attention_muon_s0 | 4307/4755 | 1666/2114 | 78.81 +/- 0.54 | 2288.8 |

| v3_realmlp_robust_embedding_stats_s0 | 4306/4755 | 1665/2114 | 78.76 +/- 1.14 | 71.6 |

| v3_deepsets_ema_s0 | 4306/4755 | 1665/2114 | 78.76 +/- 1.44 | 123.7 |

| v3_attention_group_alpha05_s0 | 4304/4755 | 1663/2114 | 78.67 +/- 0.48 | 180.5 |

| v3_tabm_embedding_stats_k8_s0 | 4304/4755 | 1663/2114 | 78.67 +/- 1.56 | 265.7 |

| v3_realmlp_embedding_stats_ema_s0 | 4299/4755 | 1658/2114 | 78.43 +/- 1.02 | 167.9 |

All rows, including losses, rates, preprocessing, decoder, source hashes and numeric asset bytes, remain in the bound final_search/EXPERIMENT_TABLE.md and original per-experiment summaries.


---


3. Complete V3 training experiments (2/2)

All values below are training OOF results on common question folds. Hard fold mean and sample SD are percentages; fit seconds sum the recorded inner/outer optimization cost. Full configuration IDs are immutable identifiers, not encodings of learning rates.

| Experiment ID | All correct | Hard correct | Hard mean +/- SD | Fit s |

| --- | --- | --- | --- | --- |

| v3_deepsets_adamw_s0 | 4298/4755 | 1657/2114 | 78.38 +/- 1.32 | 144.3 |

| v3_resmlp_ema_s0 | 4266/4755 | 1625/2114 | 76.87 +/- 1.00 | 67.6 |

| v3_resmlp_adamw_s0 | 4262/4755 | 1621/2114 | 76.68 +/- 0.72 | 47.4 |

| v3_tabm_k8_s0 | 4260/4755 | 1619/2114 | 76.58 +/- 0.73 | 321.9 |

| v3_tabm_k4_ema_s0 | 4260/4755 | 1619/2114 | 76.58 +/- 0.76 | 579.3 |

| v3_realmlp_robust_all_s0 | 4252/4755 | 1611/2114 | 76.21 +/- 0.71 | 73.9 |

| v3_realmlp_robust_lr0003_s0 | 4251/4755 | 1610/2114 | 76.16 +/- 0.73 | 95.4 |

| v3_tabm_k4_lr0003_s0 | 4251/4755 | 1610/2114 | 76.16 +/- 0.57 | 471.8 |

| v3_realmlp_robust_width256_s0 | 4248/4755 | 1607/2114 | 76.02 +/- 1.03 | 111.4 |

| v3_tabm_k4_s0 | 4247/4755 | 1606/2114 | 75.97 +/- 0.93 | 157.0 |

| v3_tabm_k4_drop03_s0 | 4246/4755 | 1605/2114 | 75.92 +/- 0.73 | 185.0 |

| v3_mlp_ema_s0 | 4242/4755 | 1601/2114 | 75.73 +/- 1.15 | 73.3 |

| v3_realmlp_robust_stats_s0 | 4240/4755 | 1599/2114 | 75.64 +/- 1.14 | 50.1 |

| v3_mlp_muon001_s0 | 4239/4755 | 1598/2114 | 75.59 +/- 0.48 | 1079.0 |

| v3_realmlp_standard_s0 | 4238/4755 | 1597/2114 | 75.54 +/- 1.35 | 50.2 |

| v3_mlp_adamw_s0 | 4233/4755 | 1592/2114 | 75.31 +/- 1.04 | 49.5 |

| v3_mlp_muon002_s0 | 4230/4755 | 1589/2114 | 75.17 +/- 0.14 | 790.0 |

Recorded completed-V3 fit time totals 9660.9 s. Recorded process-attempt wall time totals 208968.9 s; 0 attempts lack that field. These clocks have different scopes and are not added. Wall time includes elapsed gaps and startup; it does not establish continuous CPU execution or continuous monitoring.


---


4. Preserved historical V2 experiments

All values below are training OOF results on common question folds. Hard fold mean and sample SD are percentages; fit seconds sum the recorded inner/outer optimization cost. Full configuration IDs are immutable identifiers, not encodings of learning rates.

| Experiment ID | All correct | Hard correct | Hard mean +/- SD | Fit s |

| --- | --- | --- | --- | --- |

| attention_regularized_s0 | 4328/4755 | 1687/2114 | 79.80 +/- 0.72 | 133.3 |

| attention_hard_ablation | 4321/4755 | 1680/2114 | 79.47 +/- 0.89 | 133.3 |

| attention_regularized_s1 | 4319/4755 | 1678/2114 | 79.38 +/- 0.72 | 1098.4 |

| deepsets_regularized_s1 | 4313/4755 | 1672/2114 | 79.09 +/- 1.03 | 59.9 |

| attention_regularized_s2 | 4312/4755 | 1671/2114 | 79.04 +/- 1.39 | 60.2 |

| attention_group_s0 | 4309/4755 | 1668/2114 | 78.90 +/- 1.85 | 41.7 |

| deepsets_regularized_s2 | 4308/4755 | 1667/2114 | 78.85 +/- 1.09 | 59.7 |

| deepsets_no_context | 4296/4755 | 1655/2114 | 78.29 +/- 0.87 | 86.9 |

| deepsets_regularized_s0 | 4295/4755 | 1654/2114 | 78.24 +/- 0.96 | 117.8 |

| deepsets_group_s0 | 4289/4755 | 1648/2114 | 77.96 +/- 1.14 | 34.3 |

| mlp_listwise_s0 | 4253/4755 | 1612/2114 | 76.25 +/- 1.25 | 25.9 |

| mlp_capacity_s0 | 4244/4755 | 1603/2114 | 75.83 +/- 0.84 | 42.8 |

| linear_group_a0 | 4242/4755 | 1601/2114 | 75.73 +/- 1.21 | 14.3 |

| linear_group_hard | 4242/4755 | 1601/2114 | 75.73 +/- 1.21 | 13.0 |

| resmlp_group_s0 | 4241/4755 | 1600/2114 | 75.69 +/- 0.69 | 27.4 |

| mlp_group_s0 | 4240/4755 | 1599/2114 | 75.64 +/- 0.52 | 32.1 |

| linear_group_a0p5 | 4236/4755 | 1595/2114 | 75.45 +/- 0.61 | 14.5 |

| linear_group_a1 | 4229/4755 | 1588/2114 | 75.12 +/- 0.48 | 13.0 |

| linear_original | 4213/4755 | 1576/2114 | 74.55 +/- 0.65 | 11.7 |

Historical V2 recorded fit time: 2020.1 s. Those earlier runs and assets are retained unchanged. They are not additional V3 configurations or newly obtained test results.


---


5. Decoder, objective and stability evidence

| Saved V2 control | Hard correct | Hard mean +/- SD |

| --- | --- | --- |

| linear_original | 1576/2114 | 74.551 +/- 0.651 |

| linear_original::mask_max | 1578/2114 | 74.646 +/- 0.811 |

| linear_original::group_max | 1578/2114 | 74.646 +/- 0.811 |

| linear_original::group_lse_a0 | 1604/2114 | 75.875 +/- 0.369 |

| linear_original::group_lse_a0.5 | 1595/2114 | 75.450 +/- 1.107 |

| linear_original::group_lse_a1 | 1552/2114 | 73.416 +/- 0.622 |

| linear_group_a0 | 1601/2114 | 75.733 +/- 1.210 |

| linear_group_a0p5 | 1595/2114 | 75.450 +/- 0.607 |

| linear_group_a1 | 1588/2114 | 75.118 +/- 0.482 |

| linear_group_hard | 1601/2114 | 75.733 +/- 1.210 |

| mlp_group_s0 | 1599/2114 | 75.638 +/- 0.515 |

| resmlp_group_s0 | 1600/2114 | 75.686 +/- 0.692 |

| deepsets_group_s0 | 1648/2114 | 77.956 +/- 1.139 |

| attention_group_s0 | 1668/2114 | 78.902 +/- 1.853 |

The six linear_original rows reuse identical scores to isolate masking/group decoding. Original listwise training includes all candidates in its denominator and all positives in its numerator; it does not relabel alternate correct answers negative. Group CE, listwise, BCE and valid positive-negative softplus pairs are distinct registered objectives. Hard/full reduction requires corresponding regularization accounting.

| Completed identical recipe | Seed suffixes | Mean hard % | Across-seed SD |

| --- | --- | --- | --- |

| research_v2/attention_regularized_s0 | 957,958,959 | 79.41 | 0.38 |

| research_v2/deepsets_regularized_s0 | 957,958,959 | 78.73 | 0.44 |

Only genuinely completed identical recipes enter this descriptive seed table. Across-seed sample SD is not a confidence interval or fold SD. The deployment seed remains3036900957. Incomplete V3 repeat seeds cannot establish V3 seed stability.


---


6. Research rounds, interruptions and limits

| Registered round | Jobs | Complete | Best base-seed hard mean % |

| --- | --- | --- | --- |

| r01_mlp_optimizer | 4 | 4 | 75.73 |

| r02_tabm | 3 | 3 | 76.58 |

| r03_realmlp_preprocess | 3 | 3 | 76.21 |

| r08_fixed_heterogeneous_fusion | 0 | 0 | No complete primary run |

| r04_attention_optimizer_capacity | 4 | 4 | 79.71 |

| r05_tabm_optimization_features | 3 | 3 | 78.90 |

| r06_attention_objectives | 4 | 4 | 79.66 |

| r07_regularized_interactions | 4 | 4 | 78.76 |

| r09_attention_regularization_capacity | 3 | 3 | 79.61 |

| r10_realmlp_resolution | 3 | 3 | 78.76 |

| r11_tabm_reduced_features_variance | 2 | 2 | 79.80 |

| r12_reduced_tabm_attention_equal_fusion | 0 | 0 | No complete primary run |

| r14_realmlp_reduced_variance | 2 | 2 | 79.56 |

| r15_reduced_tabm_ema_attention_equal_fusion | 0 | 0 | No complete primary run |

| r13_fixed_seed_stability | 4 | 0 | No complete primary run |

| Incomplete job | Terminal status | Last recorded reason |

| --- | --- | --- |

| v3_tabm_embedding_stats_ema_s1 | pruned | attempt_wall_budget |

| v3_tabm_embedding_stats_ema_s2 | pruned | attempt_wall_budget |

| v3_attention_ema_s1 | pruned | absolute_campaign_budget |

| v3_attention_ema_s2 | pruned | Not started: absolute_campaign_budget |

Incomplete jobs and partial folds remain preserved and are excluded from whole-CV rankings. A logged fold seed is the fixed base seed plus the fold offset, not an independently selected repeat. Epoch checkpoints preserve model/optimizer/EMA/RNG state; a resumed attempt is not counted as another completed configuration.

Original planning used a 24-hour absolute research clock and per-attempt budgets. Actual monitoring gaps, late closure and deadline failure are disclosed on page1; no claim of uninterrupted 24-hour CPU work or successful budget enforcement is made. Missing timing is not invented. Historical ledger rows without an analysis hash: r01_mlp_optimizer. Their currently verified summaries are bound without rewriting prior records.


---


7. Negative fixed fusions and paired uncertainty

| Fixed probability mixture | Hard correct | Delta vs named matched member | Decision |

| --- | --- | --- | --- |

| v3_fusion_attention_tabm_s0 | 1667/2114 | -15 vs v3_attention_adamw_s0 | Rejected |

| v3_fusion_attention_realmlp_s0 | 1670/2114 | -12 vs v3_attention_adamw_s0 | Rejected |

| v3_fusion_reduced_tabm_attention_ema_s0 | 1680/2114 | -5 vs v3_attention_ema_s0 | Rejected |

| v3_fusion_reduced_tabm_ema_attention_ema_s0 | 1687/2114 | 0 vs v3_tabm_embedding_stats_ema_s0 | Rejected |

v3_fusion_reduced_tabm_ema_attention_ema_s0: +2 hard questions versus v3_attention_ema_s0 (1685 correct), but +0 versus the matched v3_tabm_embedding_stats_ema_s0 (1687 correct); rejected.

Member training costs are counted once in the campaign. OOF mixing is a deterministic diagnostic, with no new optimization. Formal fresh-process latency was not measured for rejected mixtures.

Each fusion followed saved reciprocal-error/oracle evidence and one registered equal-probability mixture. All four were rejected. No result is rescued by a post-hoc weight, temperature, member, seed or routing sweep. Passing a complementarity gate alone did not guarantee an improvement.

| Base-seed round representative | Hard delta pp | Saved paired 95% interval |

| --- | --- | --- |

| r01_mlp_optimizer | -4.07 | [-5.58, -2.55] |

| r02_tabm | -3.22 | [-4.59, -1.80] |

| r03_realmlp_preprocess | -3.60 | [-5.01, -2.13] |

| r04_attention_optimizer_capacity | -0.09 | [-0.76, 0.57] |

| r05_tabm_optimization_features | -0.90 | [-1.80, 0.00] |

| r06_attention_objectives | -0.14 | [-1.23, 0.99] |

| r07_regularized_interactions | -1.04 | [-2.03, 0.00] |

| r09_attention_regularization_capacity | -0.19 | [-0.57, 0.19] |

| r10_realmlp_resolution | -1.04 | [-2.18, 0.09] |

| r11_tabm_reduced_features_variance | 0.00 | [-0.99, 0.99] |

| r14_realmlp_reduced_variance | -0.24 | [-1.18, 0.71] |

Round representatives are selected from completed train OOF results and compared with preserved V2 Attention predictions. Saved paired intervals are conditional on those predictions and retain search-selection optimism. They are not test intervals, proof of a causal architecture effect, or a ranking claim.


---


8. Offline confirmation and clean reproduction

After search closure, at most two fixed-seed finalists were locked for validation. The predeclared confirmation/fallback rule selected one archive checkpoint. Validation had already informed earlier work; this is not an untouched-holdout estimate. No subsequent test was opened.

| Locked candidate / fallback | All validation | Hard validation |

| --- | --- | --- |

| Historical V2 fallback | 740/798 | 297/355 |

| v3_tabm_embedding_stats_ema_s0 | 735/798 | 292/355 |

| v3_attention_ema_s0 | 736/798 | 293/355 |

No locked V3 candidate improved either overall or hard validation counts over the preserved V2 fallback. The historical V2 checkpoint is retained byte-for-byte; no accepted replacement is claimed.

| Component | Family | Seed | Epochs | Preprocess | Weights |

| --- | --- | --- | --- | --- | --- |

| deepsets_regularized_s0 | deepsets | 3036900957 | 9 | standard | live |

Fresh-environment reproduction retained identical saved choices and official validation score: 740/798. Prediction plus scoring took 2.280391 s; frozen inference assets total 963670 uncompressed bytes. These are validation reproduction measurements, not V3 test latency or acceptance by the course.

Exact configuration, preprocessing, decoder, seed/epoch aggregation and EMA/live weight choice are bound in FREEZE.training_recipes and the original final_fit records. Independent contract/export checks and clean-environment receipts must pass before this report can be generated.

| Chosen-event diagnostic | Brier | ECE10 | AURC |

| --- | --- | --- | --- |

| Train OOF raw | 0.0627 | 0.0191 | 0.0131 |

| Train OOF mapped (in-sample) | 0.0618 | 0.0053 | 0.0131 |

| Used validation raw | 0.0541 | 0.0161 | 0.0101 |

| Used validation mapped | 0.0548 | 0.0201 | 0.0101 |

Confidence is the probability of the actually chosen answer group. The fixed C=1 logistic map fits train OOF only; its fitted-OOF metrics are IN-SAMPLE, and validation was already used for confirmation. Ten equal-width ECE bins retain counts; AURC summarizes saved risk-coverage. The diagnostic changes no choices and is not loaded by formal prediction.


---


9. Reproduction, provenance and honest limits

Archive source commit: eee478b4959e555b6ec8c902b6db8b69f31df508. Model: code/solution_v3.py; assets: artifacts/research_v3/final_selected. For fresh reproduction, use a separate clone checked out at this exact source commit, before the later archive FREEZE metadata, and prepare the fixed official data with code/tools/fetch_official.py; the isolated evaluation checkout receives validation files only. Verify retained source/asset hashes and package versions; then run python scripts/reproduce_selected_v3.py --attempt independent_validation. Its preflight intentionally refuses the current frozen archive directory. Preserve that directory and its guards; do not delete FREEZE or reopen research, test or score updates.

Training regeneration uses the committed configurations and common_folds.json through code/train_v3.py, including fold-local scaling, nested stopping and fresh refits. Existing closed-search guards remain active. Reproduction of archived predictions is separate from reopening model research.

AI assistance was substantive: source inspection, implementation, experiment orchestration, diagnostics, report preparation and internal AI review. Actual logs and hashes support numerical claims; AI review is not instructor approval or a claim of independent human understanding. Author commits, licenses and adaptation details are retained in MODELS_SOURCES.md, OPTIMIZER_SOURCES.md and model-source-evidence.json.

Primary method references

[1] Gorishniy et al. TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling. ICLR 2025. https://arxiv.org/abs/2410.24210

[2] Holzmuller et al. Better by Default: Strong Pre-Tuned MLPs and Boosted Trees on Tabular Data. NeurIPS 2024. https://arxiv.org/abs/2407.04491

[3] Zaheer et al. Deep Sets. NeurIPS 2017. https://arxiv.org/abs/1703.06114

[4] Vaswani et al. Attention Is All You Need. NeurIPS 2017. https://arxiv.org/abs/1706.03762

[5] Gorishniy et al. Benchmarking Optimizers for MLPs in Tabular Deep Learning. 2026 preprint, inspected v1 (Muon/EMA motivation). https://arxiv.org/abs/2604.15297v1

Limitations: missed formal deadline; monitoring gaps; four incomplete V3 seed repeats; selection-optimistic CV/bootstrap; validation reuse; known historical test aggregate; compact task adaptations rather than full source-paper benchmarks; negative fusions; and no new V3 test result. No training or validation statistic establishes a leaderboard position.

Peer review is handled through the instructor's actual released workflow as a separate task. This report neither marks it complete nor postpones it by assumption. Consult its independent current evidence.

Evidence index: final_search/EXPERIMENT_TABLE.md; round_decisions.jsonl; campaign_queue.json; fixed_fusion_analysis_complete.json; deterministic_train_references; R05 paired-and-slices; SELECTED_MODEL/FREEZE/REVIEW_RESULT/REPRODUCTION_RESULT; FORMAL_SUBMISSION_BLOCKED/RESEARCH_CLOSEOUT. All are hash-bound in closeout_inputs.json.

POST-DEADLINE RESEARCH ARCHIVE: V3 FORMAL TEST AND SCORE UPDATE NOT PERFORMED.
