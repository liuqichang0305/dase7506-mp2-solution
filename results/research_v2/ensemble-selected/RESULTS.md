# Equal group-probability ensemble: actual train OOF

Members: `attention_regularized_s0` and `deepsets_regularized_s0`; weights 0.5 / 0.5. No weight grid.

| Method | Overall correct/N | Overall fold mean ± SD (%) | Hard correct/N | Hard fold mean ± SD (%) |
|---|---:|---:|---:|---:|
| attention_regularized_s0 | 4328/4755 | 91.020 ± 0.324 | 1687/2114 | 79.801 ± 0.724 |
| deepsets_regularized_s0 | 4295/4755 | 90.326 ± 0.420 | 1654/2114 | 78.240 ± 0.958 |
| equal_probability_ensemble | 4315/4755 | 90.747 ± 0.324 | 1674/2114 | 79.186 ± 0.741 |

Complementarity was recorded before forming the mixture. Correctness uses the selected numeric answer group, with the highest-scoring candidate from member one used only to return an original index.

Reused measured member training: 251.061 seconds; additional training: zero. Maximum member asset sum: 1.9972 MiB. A final deployment wrapper is not included until one is created.

Validation/test were not evaluated. Paired bootstrap comparisons and raw group probabilities are saved alongside this table.
