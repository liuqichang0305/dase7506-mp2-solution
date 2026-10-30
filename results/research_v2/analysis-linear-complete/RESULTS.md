# Actual train OOF research results

Generated UTC: 2026-10-29T08:44:48.069287+00:00

Common outer folds; percentages are fold mean ± sample SD followed by pooled correct/N. No official validation/test inputs or references were opened by this tool.

| Experiment / decoder | Status | Overall CV (%) | Hard CV (%) | Single-positive CV (%) | Parameters | Audited MiB (max fold) | Training seconds | Official validation |
|---|---|---|---|---|---:|---:|---:|---|
| linear_original | complete | 88.601 ± 0.324; 4213/4755 | 74.551 ± 0.651; 1576/2114 | 49.226 ± 2.436; 350/711 | 2065 | 0.0486 | 11.718 | 按协议未评估 |
| linear_original::mask_max | complete | 88.728 ± 0.370; 4219/4755 | 74.646 ± 0.811; 1578/2114 | 49.789 ± 2.192; 354/711 | 2065 | 0.0486 | 11.718 | 按协议未评估 |
| linear_original::group_max | complete | 88.728 ± 0.370; 4219/4755 | 74.646 ± 0.811; 1578/2114 | 49.789 ± 2.192; 354/711 | 2065 | 0.0486 | 11.718 | 按协议未评估 |
| linear_original::group_lse_a0 | complete | 89.274 ± 0.167; 4245/4755 | 75.875 ± 0.369; 1604/2114 | 42.897 ± 2.399; 305/711 | 2065 | 0.0486 | 11.718 | 按协议未评估 |
| linear_original::group_lse_a0.5 | complete | 89.085 ± 0.501; 4236/4755 | 75.450 ± 1.107; 1595/2114 | 48.101 ± 1.934; 342/711 | 2065 | 0.0486 | 11.718 | 按协议未评估 |
| linear_original::group_lse_a1 | complete | 88.181 ± 0.284; 4193/4755 | 73.416 ± 0.622; 1552/2114 | 54.712 ± 2.472; 389/711 | 2065 | 0.0486 | 11.718 | 按协议未评估 |
| linear_group_a0 | complete | 89.211 ± 0.539; 4242/4755 | 75.733 ± 1.210; 1601/2114 | 43.882 ± 0.731; 312/711 | 2065 | 0.0486 | 14.332 | 按协议未评估 |
| linear_group_a0p5 | complete | 89.085 ± 0.275; 4236/4755 | 75.450 ± 0.607; 1595/2114 | 45.288 ± 1.062; 322/711 | 2065 | 0.0486 | 14.540 | 按协议未评估 |
| linear_group_a1 | complete | 88.938 ± 0.222; 4229/4755 | 75.118 ± 0.482; 1588/2114 | 45.992 ± 0.844; 327/711 | 2065 | 0.0486 | 13.010 | 按协议未评估 |
| linear_group_hard | complete | 89.211 ± 0.539; 4242/4755 | 75.733 ± 1.210; 1601/2114 | 43.882 ± 0.731; 312/711 | 2065 | 0.0486 | 12.990 | 按协议未评估 |

Decoder-only rows reuse fitted models and cost; do not sum them as separate training jobs. Numeric payload and full audited bundle bytes are separately retained in analysis.json. Finalist validation and final artifact timing belong to the separately authorized validation report.

## linear_group_a0 versus linear_original::group_lse_a0

| Subset | Additional correct | Difference (pp) | Paired 95% bootstrap interval | First-only correct | Second-only correct | Both wrong |
|---|---:|---:|---|---:|---:|---:|
| overall | -3 | -0.063 | [-0.336, 0.210] | 20 | 23 | 490 |
| hard | -3 | -0.142 | [-0.757, 0.473] | 20 | 23 | 490 |

Error Jaccard: 0.9193. Complementarity is necessary evidence for considering an ensemble, not proof that one will improve.

## linear_group_a0p5 versus linear_group_a0

| Subset | Additional correct | Difference (pp) | Paired 95% bootstrap interval | First-only correct | Second-only correct | Both wrong |
|---|---:|---:|---|---:|---:|---:|
| overall | -6 | -0.126 | [-0.358, 0.084] | 12 | 18 | 501 |
| hard | -6 | -0.284 | [-0.804, 0.237] | 12 | 18 | 501 |

Error Jaccard: 0.9435. Complementarity is necessary evidence for considering an ensemble, not proof that one will improve.

## linear_group_a1 versus linear_group_a0

| Subset | Additional correct | Difference (pp) | Paired 95% bootstrap interval | First-only correct | Second-only correct | Both wrong |
|---|---:|---:|---|---:|---:|---:|
| overall | -13 | -0.273 | [-0.589, 0.042] | 25 | 38 | 488 |
| hard | -13 | -0.615 | [-1.325, 0.095] | 25 | 38 | 488 |

Error Jaccard: 0.8857. Complementarity is necessary evidence for considering an ensemble, not proof that one will improve.

## linear_group_hard versus linear_group_a0

| Subset | Additional correct | Difference (pp) | Paired 95% bootstrap interval | First-only correct | Second-only correct | Both wrong |
|---|---:|---:|---|---:|---:|---:|
| overall | 0 | 0.000 | [0.000, 0.000] | 0 | 0 | 513 |
| hard | 0 | 0.000 | [0.000, 0.000] | 0 | 0 | 513 |

Error Jaccard: 1.0000. Complementarity is necessary evidence for considering an ensemble, not proof that one will improve.

