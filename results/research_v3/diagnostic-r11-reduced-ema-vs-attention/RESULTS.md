# Fixed train-OOF diagnostic

Complete aligned train OOF only; official train text used only for predefined lengths and parser alignment.

| Method | Overall correct/N | Hard correct/N | Single-positive correct/N |
|---|---:|---:|---:|
| v3:v3_tabm_embedding_stats_ema_s0 | 4328/4755 | 1687/2114 | 376/711 |
| v3:v3_attention_ema_s0 | 4326/4755 | 1685/2114 | 388/711 |
| v3:v3_tabm_k4_embedding_stats_s0 | 4309/4755 | 1668/2114 | 364/711 |

Oracle/reciprocal results are diagnostic; no fusion is implied.

| Explicit methods | Unique corrections per member | Hard oracle gain over strongest (pp) | Gate |
|---|---:|---:|---|
| v3:v3_tabm_embedding_stats_ema_s0 + v3:v3_attention_ema_s0 | [51, 49] | 2.3179 | True |

All fixed slices, provenance and checks are in `diagnostic.json`. No question text or answer is emitted.

Fusion status: not_run.

Descriptive OOF slices overlap and are not confirmatory significance tests.
Historical architecture comparison includes optimizer/regularization recipe differences; it is not a clean causal architecture ablation.
Oracle gains are not expected ensemble gains. OOF-guided future hypotheses require registration and fresh fitting within the same bounded campaign.
