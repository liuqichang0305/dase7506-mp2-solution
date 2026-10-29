# Actual train OOF research results

Generated UTC: 2026-10-29T09:00:15.126698+00:00

Common outer folds; percentages are fold mean ± sample SD followed by pooled correct/N. No official validation/test inputs or references were opened by this tool.

| Experiment / decoder | Status | Overall CV (%) | Hard CV (%) | Single-positive CV (%) | Parameters | Audited MiB (max fold) | Training seconds | Official validation |
|---|---|---|---|---|---:|---:|---:|---|
| attention_group_s0 | complete | 90.620 ± 0.823; 4309/4755 | 78.902 ± 1.853; 1668/2114 | 53.727 ± 7.177; 382/711 | 273025 | 1.0782 | 41.689 | 按协议未评估 |
| attention_group_s0::candidate_max | complete | 90.242 ± 0.514; 4291/4755 | 78.051 ± 1.149; 1650/2114 | 57.665 ± 5.276; 410/711 | 273025 | 1.0782 | 41.689 | 按协议未评估 |
| attention_regularized_s0 | complete | 91.020 ± 0.324; 4328/4755 | 79.801 ± 0.724; 1687/2114 | 54.430 ± 4.866; 387/711 | 273025 | 1.0782 | 133.297 | 按协议未评估 |
| attention_regularized_s0::candidate_max | complete | 90.410 ± 0.437; 4299/4755 | 78.430 ± 0.974; 1658/2114 | 58.790 ± 4.512; 418/711 | 273025 | 1.0782 | 133.297 | 按协议未评估 |
| deepsets_group_s0 | complete | 90.200 ± 0.506; 4289/4755 | 77.956 ± 1.139; 1648/2114 | 50.914 ± 8.293; 362/711 | 231937 | 0.9190 | 34.265 | 按协议未评估 |
| deepsets_group_s0::candidate_max | complete | 89.821 ± 0.311; 4271/4755 | 77.105 ± 0.689; 1630/2114 | 54.149 ± 6.390; 385/711 | 231937 | 0.9190 | 34.265 | 按协议未评估 |
| deepsets_regularized_s0 | complete | 90.326 ± 0.420; 4295/4755 | 78.240 ± 0.958; 1654/2114 | 50.633 ± 5.801; 360/711 | 231937 | 0.9190 | 117.764 | 按协议未评估 |
| deepsets_regularized_s0::candidate_max | complete | 90.116 ± 0.364; 4285/4755 | 77.767 ± 0.828; 1644/2114 | 55.274 ± 4.975; 393/711 | 231937 | 0.9190 | 117.764 | 按协议未评估 |
| linear_group_a0 | complete | 89.211 ± 0.539; 4242/4755 | 75.733 ± 1.210; 1601/2114 | 43.882 ± 0.731; 312/711 | 2065 | 0.0486 | 14.332 | 按协议未评估 |
| linear_group_a0p5 | complete | 89.085 ± 0.275; 4236/4755 | 75.450 ± 0.607; 1595/2114 | 45.288 ± 1.062; 322/711 | 2065 | 0.0486 | 14.540 | 按协议未评估 |
| linear_group_a1 | complete | 88.938 ± 0.222; 4229/4755 | 75.118 ± 0.482; 1588/2114 | 45.992 ± 0.844; 327/711 | 2065 | 0.0486 | 13.010 | 按协议未评估 |
| linear_group_hard | complete | 89.211 ± 0.539; 4242/4755 | 75.733 ± 1.210; 1601/2114 | 43.882 ± 0.731; 312/711 | 2065 | 0.0486 | 12.990 | 按协议未评估 |
| linear_original | complete | 88.601 ± 0.324; 4213/4755 | 74.551 ± 0.651; 1576/2114 | 49.226 ± 2.436; 350/711 | 2065 | 0.0486 | 11.718 | 按协议未评估 |
| linear_original::mask_max | complete | 88.728 ± 0.370; 4219/4755 | 74.646 ± 0.811; 1578/2114 | 49.789 ± 2.192; 354/711 | 2065 | 0.0486 | 11.718 | 按协议未评估 |
| linear_original::group_max | complete | 88.728 ± 0.370; 4219/4755 | 74.646 ± 0.811; 1578/2114 | 49.789 ± 2.192; 354/711 | 2065 | 0.0486 | 11.718 | 按协议未评估 |
| linear_original::group_lse_a0 | complete | 89.274 ± 0.167; 4245/4755 | 75.875 ± 0.369; 1604/2114 | 42.897 ± 2.399; 305/711 | 2065 | 0.0486 | 11.718 | 按协议未评估 |
| linear_original::group_lse_a0.5 | complete | 89.085 ± 0.501; 4236/4755 | 75.450 ± 1.107; 1595/2114 | 48.101 ± 1.934; 342/711 | 2065 | 0.0486 | 11.718 | 按协议未评估 |
| linear_original::group_lse_a1 | complete | 88.181 ± 0.284; 4193/4755 | 73.416 ± 0.622; 1552/2114 | 54.712 ± 2.472; 389/711 | 2065 | 0.0486 | 11.718 | 按协议未评估 |
| mlp_capacity_s0 | complete | 89.253 ± 0.380; 4244/4755 | 75.828 ± 0.838; 1603/2114 | 46.414 ± 2.233; 330/711 | 232190 | 0.9195 | 42.775 | 按协议未评估 |
| mlp_capacity_s0::candidate_max | complete | 88.580 ± 0.189; 4212/4755 | 74.314 ± 0.408; 1571/2114 | 51.336 ± 2.713; 365/711 | 232190 | 0.9195 | 42.775 | 按协议未评估 |
| mlp_group_s0 | complete | 89.169 ± 0.222; 4240/4755 | 75.638 ± 0.515; 1599/2114 | 48.101 ± 1.116; 342/711 | 272641 | 1.0738 | 32.114 | 按协议未评估 |
| mlp_group_s0::candidate_max | complete | 88.665 ± 0.222; 4216/4755 | 74.503 ± 0.516; 1575/2114 | 52.743 ± 0.422; 375/711 | 272641 | 1.0738 | 32.114 | 按协议未评估 |
| mlp_listwise_s0 | complete | 89.443 ± 0.565; 4253/4755 | 76.254 ± 1.253; 1612/2114 | 46.835 ± 2.954; 333/711 | 272641 | 1.0738 | 25.850 | 按协议未评估 |
| mlp_listwise_s0::candidate_max | complete | 88.875 ± 0.544; 4226/4755 | 74.977 ± 1.205; 1585/2114 | 50.070 ± 1.994; 356/711 | 272641 | 1.0738 | 25.850 | 按协议未评估 |
| resmlp_group_s0 | complete | 89.190 ± 0.311; 4241/4755 | 75.686 ± 0.692; 1600/2114 | 44.866 ± 2.809; 319/711 | 330625 | 1.2961 | 27.359 | 按协议未评估 |
| resmlp_group_s0::candidate_max | complete | 88.749 ± 0.410; 4220/4755 | 74.693 ± 0.906; 1579/2114 | 50.633 ± 2.532; 360/711 | 330625 | 1.2961 | 27.359 | 按协议未评估 |

Decoder-only rows reuse fitted models and cost; do not sum them as separate training jobs. Numeric payload and full audited bundle bytes are separately retained in analysis.json. Finalist validation and final artifact timing belong to the separately authorized validation report.

## attention_regularized_s0 versus linear_original

| Subset | Additional correct | Difference (pp) | Paired 95% bootstrap interval | First-only correct | Second-only correct | Both wrong |
|---|---:|---:|---|---:|---:|---:|
| overall | 115 | 2.419 | [1.703, 3.155] | 208 | 93 | 334 |
| hard | 111 | 5.251 | [3.642, 6.859] | 204 | 93 | 334 |

Error Jaccard: 0.5260. Complementarity is necessary evidence for considering an ensemble, not proof that one will improve.

## deepsets_regularized_s0 versus mlp_capacity_s0

| Subset | Additional correct | Difference (pp) | Paired 95% bootstrap interval | First-only correct | Second-only correct | Both wrong |
|---|---:|---:|---|---:|---:|---:|
| overall | 51 | 1.073 | [0.442, 1.703] | 145 | 94 | 366 |
| hard | 51 | 2.412 | [0.993, 3.832] | 145 | 94 | 366 |

Error Jaccard: 0.6050. Complementarity is necessary evidence for considering an ensemble, not proof that one will improve.

## attention_regularized_s0 versus attention_group_s0

| Subset | Additional correct | Difference (pp) | Paired 95% bootstrap interval | First-only correct | Second-only correct | Both wrong |
|---|---:|---:|---|---:|---:|---:|
| overall | 19 | 0.400 | [-0.021, 0.799] | 60 | 41 | 386 |
| hard | 19 | 0.899 | [0.000, 1.845] | 60 | 41 | 386 |

Error Jaccard: 0.7926. Complementarity is necessary evidence for considering an ensemble, not proof that one will improve.

## Incomplete runs

- attention_regularized_s1: 0/3 completed folds; no fabricated full-CV result.
