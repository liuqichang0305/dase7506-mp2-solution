# Actual train OOF research results

Generated UTC: 2026-10-29T08:29:49.886658+00:00

Common outer folds; percentages are fold mean ± sample SD followed by pooled correct/N. No official validation/test inputs or references were opened by this tool.

| Experiment / decoder | Overall CV (%) | Hard CV (%) | Single-positive CV (%) | Parameters | Audited MiB (max fold) | Training seconds | Official validation |
|---|---|---|---|---:|---:|---:|---|
| linear_original | 88.601 ± 0.324; 4213/4755 | 74.551 ± 0.651; 1576/2114 | 49.226 ± 2.436; 350/711 | 2065 | 0.0486 | 11.718 | 按协议未评估 |
| linear_original::mask_max | 88.728 ± 0.370; 4219/4755 | 74.646 ± 0.811; 1578/2114 | 49.789 ± 2.192; 354/711 | 2065 | 0.0486 | 11.718 | 按协议未评估 |
| linear_original::group_max | 88.728 ± 0.370; 4219/4755 | 74.646 ± 0.811; 1578/2114 | 49.789 ± 2.192; 354/711 | 2065 | 0.0486 | 11.718 | 按协议未评估 |
| linear_original::group_lse_a0 | 89.274 ± 0.167; 4245/4755 | 75.875 ± 0.369; 1604/2114 | 42.897 ± 2.399; 305/711 | 2065 | 0.0486 | 11.718 | 按协议未评估 |
| linear_original::group_lse_a0.5 | 89.085 ± 0.501; 4236/4755 | 75.450 ± 1.107; 1595/2114 | 48.101 ± 1.934; 342/711 | 2065 | 0.0486 | 11.718 | 按协议未评估 |
| linear_original::group_lse_a1 | 88.181 ± 0.284; 4193/4755 | 73.416 ± 0.622; 1552/2114 | 54.712 ± 2.472; 389/711 | 2065 | 0.0486 | 11.718 | 按协议未评估 |

Decoder-only rows reuse fitted models and cost; do not sum them as separate training jobs. Numeric payload and full audited bundle bytes are separately retained in analysis.json. Finalist validation and final artifact timing belong to the separately authorized validation report.

## linear_original::mask_max versus linear_original

| Subset | Additional correct | Difference (pp) | Paired 95% bootstrap interval | First-only correct | Second-only correct | Both wrong |
|---|---:|---:|---|---:|---:|---:|
| overall | 6 | 0.126 | [0.042, 0.231] | 6 | 0 | 536 |
| hard | 2 | 0.095 | [0.000, 0.237] | 2 | 0 | 536 |

Error Jaccard: 0.9889. Complementarity is necessary evidence for considering an ensemble, not proof that one will improve.

## linear_original::group_lse_a0 versus linear_original::mask_max

| Subset | Additional correct | Difference (pp) | Paired 95% bootstrap interval | First-only correct | Second-only correct | Both wrong |
|---|---:|---:|---|---:|---:|---:|
| overall | 26 | 0.547 | [0.084, 1.009] | 79 | 53 | 457 |
| hard | 26 | 1.230 | [0.189, 2.271] | 79 | 53 | 457 |

Error Jaccard: 0.7759. Complementarity is necessary evidence for considering an ensemble, not proof that one will improve.

## linear_original::group_lse_a0.5 versus linear_original::mask_max

| Subset | Additional correct | Difference (pp) | Paired 95% bootstrap interval | First-only correct | Second-only correct | Both wrong |
|---|---:|---:|---|---:|---:|---:|
| overall | 17 | 0.358 | [0.042, 0.673] | 40 | 23 | 496 |
| hard | 17 | 0.804 | [0.095, 1.561] | 40 | 23 | 496 |

Error Jaccard: 0.8873. Complementarity is necessary evidence for considering an ensemble, not proof that one will improve.

## linear_original::group_lse_a1 versus linear_original::mask_max

| Subset | Additional correct | Difference (pp) | Paired 95% bootstrap interval | First-only correct | Second-only correct | Both wrong |
|---|---:|---:|---|---:|---:|---:|
| overall | -26 | -0.547 | [-0.967, -0.126] | 40 | 66 | 496 |
| hard | -26 | -1.230 | [-2.176, -0.331] | 40 | 66 | 496 |

Error Jaccard: 0.8239. Complementarity is necessary evidence for considering an ensemble, not proof that one will improve.

