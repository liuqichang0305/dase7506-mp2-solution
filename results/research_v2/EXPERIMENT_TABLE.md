# Actual performance-v2 experiment table

Every fitted row uses the same three outer question folds; ± is sample SD across folds. All seeds and negative results are retained. Decoder-only rows reuse fits and cost. Hyperparameter-selected OOF is not an unbiased final generalization estimate.

| Actual experiment / decoder | Full CV %; correct/n | Hard CV %; correct/n | Structural parameter count | Measured weight parameters | Audited MiB | Fit wall seconds | Validation full; hard; single-positive | Fresh prediction + scoring s |
|---|---|---|---:|---:|---:|---:|---|---:|
| attention_group_s0 | 90.620 ± 0.823; 4309/4755 | 78.902 ± 1.853; 1668/2114 | 273,025 | 273,025 | 1.0782 | 41.689 | 按协议未评估 | — |
| attention_group_s0::candidate_max | 90.242 ± 0.514; 4291/4755 | 78.051 ± 1.149; 1650/2114 | 273,025 | 273,025 | 1.0782 | 41.689 | 按协议未评估 | — |
| attention_hard_ablation | 90.873 ± 0.401; 4321/4755 | 79.470 ± 0.893; 1680/2114 | 273,025 | 273,025 | 1.0782 | 133.294 | 按协议未评估 | — |
| attention_hard_ablation::candidate_max | 90.389 ± 0.364; 4298/4755 | 78.382 ± 0.810; 1657/2114 | 273,025 | 273,025 | 1.0782 | 133.294 | 按协议未评估 | — |
| attention_regularized_s0 | 91.020 ± 0.324; 4328/4755 | 79.801 ± 0.724; 1687/2114 | 273,025 | 273,025 | 1.0782 | 133.297 | 738/798; 295/355; 70/117 | 1.7623 |
| attention_regularized_s0::candidate_max | 90.410 ± 0.437; 4299/4755 | 78.430 ± 0.974; 1658/2114 | 273,025 | 273,025 | 1.0782 | 133.297 | 按协议未评估 | — |
| attention_regularized_s1 | 90.831 ± 0.324; 4319/4755 | 79.376 ± 0.716; 1678/2114 | 273,025 | 273,025 | 1.0782 | 1098.389 | 按协议未评估 | — |
| attention_regularized_s1::candidate_max | 90.263 ± 0.159; 4292/4755 | 78.098 ± 0.355; 1651/2114 | 273,025 | 273,025 | 1.0782 | 1098.389 | 按协议未评估 | — |
| attention_regularized_s2 | 90.683 ± 0.622; 4312/4755 | 79.045 ± 1.390; 1671/2114 | 273,025 | 273,025 | 1.0782 | 60.207 | 按协议未评估 | — |
| attention_regularized_s2::candidate_max | 90.305 ± 0.203; 4294/4755 | 78.193 ± 0.439; 1653/2114 | 273,025 | 273,025 | 1.0782 | 60.207 | 按协议未评估 | — |
| deepsets_group_s0 | 90.200 ± 0.506; 4289/4755 | 77.956 ± 1.139; 1648/2114 | 231,937 | 231,937 | 0.9190 | 34.265 | 按协议未评估 | — |
| deepsets_group_s0::candidate_max | 89.821 ± 0.311; 4271/4755 | 77.105 ± 0.689; 1630/2114 | 231,937 | 231,937 | 0.9190 | 34.265 | 按协议未评估 | — |
| deepsets_no_context | 90.347 ± 0.384; 4296/4755 | 78.287 ± 0.873; 1655/2114 | 166,401 | 166,401 | 0.6690 | 86.942 | 按协议未评估 | — |
| deepsets_no_context::candidate_max | 90.137 ± 0.324; 4286/4755 | 77.814 ± 0.746; 1645/2114 | 166,401 | 166,401 | 0.6690 | 86.942 | 按协议未评估 | — |
| deepsets_regularized_s0 | 90.326 ± 0.420; 4295/4755 | 78.240 ± 0.958; 1654/2114 | 231,937 | 231,937 | 0.9190 | 117.764 | 740/798; 297/355; 68/117 | 1.7397 |
| deepsets_regularized_s0::candidate_max | 90.116 ± 0.364; 4285/4755 | 77.767 ± 0.828; 1644/2114 | 231,937 | 231,937 | 0.9190 | 117.764 | 按协议未评估 | — |
| deepsets_regularized_s1 | 90.705 ± 0.456; 4313/4755 | 79.092 ± 1.034; 1672/2114 | 231,937 | 231,937 | 0.9190 | 59.901 | 按协议未评估 | — |
| deepsets_regularized_s1::candidate_max | 90.326 ± 0.410; 4295/4755 | 78.240 ± 0.939; 1654/2114 | 231,937 | 231,937 | 0.9190 | 59.901 | 按协议未评估 | — |
| deepsets_regularized_s2 | 90.599 ± 0.476; 4308/4755 | 78.855 ± 1.086; 1667/2114 | 231,937 | 231,937 | 0.9190 | 59.654 | 按协议未评估 | — |
| deepsets_regularized_s2::candidate_max | 90.137 ± 0.284; 4286/4755 | 77.814 ± 0.656; 1645/2114 | 231,937 | 231,937 | 0.9190 | 59.654 | 按协议未评估 | — |
| linear_group_a0 | 89.211 ± 0.539; 4242/4755 | 75.733 ± 1.210; 1601/2114 | 2,065 | 2,065 | 0.0486 | 14.332 | 按协议未评估 | — |
| linear_group_a0p5 | 89.085 ± 0.275; 4236/4755 | 75.450 ± 0.607; 1595/2114 | 2,065 | 2,065 | 0.0486 | 14.540 | 按协议未评估 | — |
| linear_group_a1 | 88.938 ± 0.222; 4229/4755 | 75.118 ± 0.482; 1588/2114 | 2,065 | 2,065 | 0.0486 | 13.010 | 按协议未评估 | — |
| linear_group_hard | 89.211 ± 0.539; 4242/4755 | 75.733 ± 1.210; 1601/2114 | 2,065 | 2,065 | 0.0486 | 12.990 | 按协议未评估 | — |
| linear_original | 88.601 ± 0.324; 4213/4755 | 74.551 ± 0.651; 1576/2114 | 2,065 | 2,065 | 0.0486 | 11.718 | 724/798; 282/355; 59/117 (v1 A0 replay) | 1.4555 |
| linear_original::mask_max | 88.728 ± 0.370; 4219/4755 | 74.646 ± 0.811; 1578/2114 | 2,065 | 2,065 | 0.0486 | 11.718 | 按协议未评估 | — |
| linear_original::group_max | 88.728 ± 0.370; 4219/4755 | 74.646 ± 0.811; 1578/2114 | 2,065 | 2,065 | 0.0486 | 11.718 | 按协议未评估 | — |
| linear_original::group_lse_a0 | 89.274 ± 0.167; 4245/4755 | 75.875 ± 0.369; 1604/2114 | 2,065 | 2,065 | 0.0486 | 11.718 | 按协议未评估 | — |
| linear_original::group_lse_a0.5 | 89.085 ± 0.501; 4236/4755 | 75.450 ± 1.107; 1595/2114 | 2,065 | 2,065 | 0.0486 | 11.718 | 按协议未评估 | — |
| linear_original::group_lse_a1 | 88.181 ± 0.284; 4193/4755 | 73.416 ± 0.622; 1552/2114 | 2,065 | 2,065 | 0.0486 | 11.718 | 按协议未评估 | — |
| mlp_capacity_s0 | 89.253 ± 0.380; 4244/4755 | 75.828 ± 0.838; 1603/2114 | 232,190 | 232,190 | 0.9195 | 42.775 | 按协议未评估 | — |
| mlp_capacity_s0::candidate_max | 88.580 ± 0.189; 4212/4755 | 74.314 ± 0.408; 1571/2114 | 232,190 | 232,190 | 0.9195 | 42.775 | 按协议未评估 | — |
| mlp_group_s0 | 89.169 ± 0.222; 4240/4755 | 75.638 ± 0.515; 1599/2114 | 272,641 | 272,641 | 1.0738 | 32.114 | 按协议未评估 | — |
| mlp_group_s0::candidate_max | 88.665 ± 0.222; 4216/4755 | 74.503 ± 0.516; 1575/2114 | 272,641 | 272,641 | 1.0738 | 32.114 | 按协议未评估 | — |
| mlp_listwise_s0 | 89.443 ± 0.565; 4253/4755 | 76.254 ± 1.253; 1612/2114 | 272,641 | 272,641 | 1.0738 | 25.850 | 按协议未评估 | — |
| mlp_listwise_s0::candidate_max | 88.875 ± 0.544; 4226/4755 | 74.977 ± 1.205; 1585/2114 | 272,641 | 272,641 | 1.0738 | 25.850 | 按协议未评估 | — |
| resmlp_group_s0 | 89.190 ± 0.311; 4241/4755 | 75.686 ± 0.692; 1600/2114 | 330,625 | 330,625 | 1.2961 | 27.359 | 按协议未评估 | — |
| resmlp_group_s0::candidate_max | 88.749 ± 0.410; 4220/4755 | 74.693 ± 0.906; 1579/2114 | 330,625 | 330,625 | 1.2961 | 27.359 | 按协议未评估 | — |
| Equal group-probability mixture (rejected) | 90.747 ± 0.324; 4315/4755 | 79.186 ± 0.741; 1674/2114 | 504,962 | 504,962 | 1.9972 members only | 251.061 reused | 按协议未评估 | — |
| TabM | 未运行 | 未运行 | — | — | — | — | 按协议未评估 | — |

Structural counts are architecture arithmetic, not estimated accuracy; measured parameters come from saved numeric state arrays. MiB includes configuration and uncompressed NPZ accounting for one fold bundle; the mixture is the actual sum of its two members and was never packaged/deployed. Final selected bundle is independently audited at963,670bytes. All per-fold raw bytes/costs are in analysis-final/analysis.json.

The common-fold fit total is 2020.093 wall seconds, including the documented long interval in attention_regularized_s1. It is not CPU-only time; see docs/research_v2/BUDGET_DEVIATION.md.

The selected Deep Sets validation gain over frozen v1 is16questions (+2.005percentage points), question-paired bootstrap95% interval[0.376,3.512]points. Hard gain is15questions (+4.225points), interval[0.845,7.887]. These intervals do not remove model-selection bias. Attention has the higher seed-averaged train OOF result; Deep Sets wins this confirmation by only2questions. Neither result establishes a new test score or rank.

Negative controls: group-training did not beat merely decoding old linear scores by group; MLP group loss did not beat original listwise; hard normalization did not improve Attention; removing Deep Sets learned pooling changed hard correctness by only+1; the single equal mixture lost13hard questions versus Attention. No search was reopened after validation.
