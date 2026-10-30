# Common three-fold train OOF

Train OOF is used for selection. Intervals are paired descriptive uncertainty, not selection-unbiased claims. No validation or test inputs read.

| Version / configuration | Seed | Hard correct | Hard fold mean ± SD (%) | Full correct | Single positive | Params | Bytes | Fit seconds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| research_v2 / attention_regularized_s0 | 3036900957 | 1687/2114 | 79.801 ± 0.724 | 4328/4755 | 387/711 | 273025 | 1125124 | 133.29731520000132 |
| research_v3 / v3_attention_ema_s0 | 3036900957 | 1685/2114 | 79.707 ± 1.266 | 4326/4755 | 388/711 | 273025 | 1129252 | 266.25079879999976 |
| research_v3 / v3_attention_adamw_s0 | 3036900957 | 1682/2114 | 79.565 ± 0.745 | 4323/4755 | 382/711 | 273025 | 1129252 | 181.65113519999431 |
| research_v2 / attention_hard_ablation | 3036900957 | 1680/2114 | 79.470 ± 0.893 | 4321/4755 | 386/711 | 273025 | 1125124 | 133.29358649999813 |
| research_v2 / attention_regularized_s1 | 3036900958 | 1678/2114 | 79.376 ± 0.716 | 4319/4755 | 371/711 | 273025 | 1125124 | 1098.3893508 |
| research_v2 / deepsets_regularized_s1 | 3036900958 | 1672/2114 | 79.092 ± 1.034 | 4313/4755 | 372/711 | 231937 | 960772 | 59.90083009999944 |
| research_v2 / attention_regularized_s2 | 3036900959 | 1671/2114 | 79.045 ± 1.390 | 4312/4755 | 372/711 | 273025 | 1125124 | 60.207461500000136 |
| research_v3 / v3_attention_small_s0 | 3036900957 | 1670/2114 | 78.997 ± 0.647 | 4311/4755 | 367/711 | 104321 | 454436 | 111.08153440001479 |
| research_v2 / attention_group_s0 | 3036900957 | 1668/2114 | 78.902 ± 1.853 | 4309/4755 | 382/711 | 273025 | 1125124 | 41.68935790000069 |
| research_v2 / deepsets_regularized_s2 | 3036900959 | 1667/2114 | 78.855 ± 1.086 | 4308/4755 | 375/711 | 231937 | 960772 | 59.6540313000005 |
| research_v3 / v3_attention_muon_s0 | 3036900957 | 1666/2114 | 78.808 ± 0.544 | 4307/4755 | 367/711 | 273025 | 1129252 | 2288.8305933999945 |
| research_v2 / deepsets_no_context | 3036900957 | 1655/2114 | 78.287 ± 0.873 | 4296/4755 | 373/711 | 166401 | 698628 | 86.94164140000248 |
| research_v2 / deepsets_regularized_s0 | 3036900957 | 1654/2114 | 78.240 ± 0.958 | 4295/4755 | 360/711 | 231937 | 960772 | 117.76401540000006 |
| research_v2 / deepsets_group_s0 | 3036900957 | 1648/2114 | 77.956 ± 1.139 | 4289/4755 | 362/711 | 231937 | 960772 | 34.265074100001584 |
| research_v3 / v3_tabm_k8_s0 | 3036900957 | 1619/2114 | 76.585 ± 0.728 | 4260/4755 | 319/711 | 322696 | 1327936 | 321.9080899999972 |
| research_v2 / mlp_listwise_s0 | 3036900957 | 1612/2114 | 76.254 ± 1.253 | 4253/4755 | 333/711 | 272641 | 1123588 | 25.850140100003046 |
| research_v3 / v3_realmlp_robust_all_s0 | 3036900957 | 1611/2114 | 76.206 ± 0.712 | 4252/4755 | 340/711 | 299537 | 1235300 | 73.93353800000477 |
| research_v3 / v3_tabm_k4_s0 | 3036900957 | 1606/2114 | 75.970 ± 0.934 | 4247/4755 | 322/711 | 309828 | 1276464 | 156.9972017000182 |
| research_v3 / v3_tabm_k4_drop03_s0 | 3036900957 | 1605/2114 | 75.922 ± 0.733 | 4246/4755 | 324/711 | 309828 | 1276464 | 184.95588179999322 |
| research_v2 / mlp_capacity_s0 | 3036900957 | 1603/2114 | 75.828 ± 0.838 | 4244/4755 | 330/711 | 232190 | 961784 | 42.775077899999815 |
| research_v2 / linear_group_a0 | 3036900957 | 1601/2114 | 75.733 ± 1.210 | 4242/4755 | 312/711 | 2065 | 49544 | 14.332354699998177 |
| research_v2 / linear_group_hard | 3036900957 | 1601/2114 | 75.733 ± 1.210 | 4242/4755 | 312/711 | 2065 | 49544 | 12.990395599999829 |
| research_v3 / v3_mlp_ema_s0 | 3036900957 | 1601/2114 | 75.733 ± 1.153 | 4242/4755 | 316/711 | 272641 | 1127716 | 73.267196899993 |
| research_v2 / resmlp_group_s0 | 3036900957 | 1600/2114 | 75.686 ± 0.692 | 4241/4755 | 319/711 | 330625 | 1355524 | 27.359151000002385 |
| research_v3 / v3_realmlp_robust_stats_s0 | 3036900957 | 1599/2114 | 75.639 ± 1.136 | 4240/4755 | 336/711 | 299537 | 1235300 | 50.11985230000573 |
| research_v2 / mlp_group_s0 | 3036900957 | 1599/2114 | 75.638 ± 0.515 | 4240/4755 | 342/711 | 272641 | 1123588 | 32.1143697000025 |
| research_v3 / v3_mlp_muon001_s0 | 3036900957 | 1598/2114 | 75.591 ± 0.482 | 4239/4755 | 310/711 | 272641 | 1127716 | 1078.9769142000005 |
| research_v3 / v3_realmlp_standard_s0 | 3036900957 | 1597/2114 | 75.544 ± 1.348 | 4238/4755 | 327/711 | 299537 | 1235300 | 50.186106400004064 |
| research_v2 / linear_group_a0p5 | 3036900957 | 1595/2114 | 75.450 ± 0.607 | 4236/4755 | 322/711 | 2065 | 49544 | 14.54017279999971 |
| research_v3 / v3_mlp_adamw_s0 | 3036900957 | 1592/2114 | 75.307 ± 1.038 | 4233/4755 | 317/711 | 272641 | 1127716 | 49.46507669998391 |
| research_v3 / v3_mlp_muon002_s0 | 3036900957 | 1589/2114 | 75.166 ± 0.143 | 4230/4755 | 304/711 | 272641 | 1127716 | 789.9752761999989 |
| research_v2 / linear_group_a1 | 3036900957 | 1588/2114 | 75.118 ± 0.482 | 4229/4755 | 327/711 | 2065 | 49544 | 13.01044759999968 |
| research_v2 / linear_original | 3036900957 | 1576/2114 | 74.551 ± 0.651 | 4213/4755 | 350/711 | 2065 | 49544 | 11.718261099998927 |

Incomplete jobs are excluded from rankings:

```json
[
  {
    "experiment_id": "v3_tabm_k4_lr0003_s0",
    "round": "r05_tabm_optimization_features",
    "status": "running",
    "repair_retries": 0
  },
  {
    "experiment_id": "v3_tabm_k4_ema_s0",
    "round": "r05_tabm_optimization_features",
    "status": "pending",
    "repair_retries": 0
  },
  {
    "experiment_id": "v3_tabm_k4_embedding_stats_s0",
    "round": "r05_tabm_optimization_features",
    "status": "pending",
    "repair_retries": 0
  },
  {
    "experiment_id": "v3_attention_listwise_s0",
    "round": "r06_attention_objectives",
    "status": "pending",
    "repair_retries": 0
  },
  {
    "experiment_id": "v3_attention_bce_s0",
    "round": "r06_attention_objectives",
    "status": "pending",
    "repair_retries": 0
  },
  {
    "experiment_id": "v3_attention_pairwise_s0",
    "round": "r06_attention_objectives",
    "status": "pending",
    "repair_retries": 0
  },
  {
    "experiment_id": "v3_attention_group_alpha05_s0",
    "round": "r06_attention_objectives",
    "status": "pending",
    "repair_retries": 0
  },
  {
    "experiment_id": "v3_deepsets_adamw_s0",
    "round": "r07_regularized_interactions",
    "status": "pending",
    "repair_retries": 0
  },
  {
    "experiment_id": "v3_deepsets_ema_s0",
    "round": "r07_regularized_interactions",
    "status": "pending",
    "repair_retries": 0
  },
  {
    "experiment_id": "v3_resmlp_adamw_s0",
    "round": "r07_regularized_interactions",
    "status": "pending",
    "repair_retries": 0
  },
  {
    "experiment_id": "v3_resmlp_ema_s0",
    "round": "r07_regularized_interactions",
    "status": "pending",
    "repair_retries": 0
  },
  {
    "experiment_id": "v3_attention_wd01_s0",
    "round": "r09_attention_regularization_capacity",
    "status": "pending",
    "repair_retries": 0
  },
  {
    "experiment_id": "v3_attention_drop05_s0",
    "round": "r09_attention_regularization_capacity",
    "status": "pending",
    "repair_retries": 0
  },
  {
    "experiment_id": "v3_attention_width256_s0",
    "round": "r09_attention_regularization_capacity",
    "status": "pending",
    "repair_retries": 0
  },
  {
    "experiment_id": "v3_realmlp_robust_lr0003_s0",
    "round": "r10_realmlp_resolution",
    "status": "pending",
    "repair_retries": 0
  },
  {
    "experiment_id": "v3_realmlp_robust_width256_s0",
    "round": "r10_realmlp_resolution",
    "status": "pending",
    "repair_retries": 0
  },
  {
    "experiment_id": "v3_realmlp_robust_embedding_stats_s0",
    "round": "r10_realmlp_resolution",
    "status": "pending",
    "repair_retries": 0
  }
]
```
