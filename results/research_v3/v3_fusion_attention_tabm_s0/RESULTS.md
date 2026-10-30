# Fixed train-OOF diagnostic

Complete aligned train OOF only; official train text used only for predefined lengths and parser alignment.

| Method | Overall correct/N | Hard correct/N | Single-positive correct/N |
|---|---:|---:|---:|
| v3:v3_attention_adamw_s0 | 4323/4755 | 1682/2114 | 382/711 |
| v3:v3_tabm_k8_s0 | 4260/4755 | 1619/2114 | 319/711 |

Oracle/reciprocal results are diagnostic; no fusion is implied.

| Explicit methods | Unique corrections per member | Hard oracle gain over strongest (pp) | Gate |
|---|---:|---:|---|
| v3:v3_attention_adamw_s0 + v3:v3_tabm_k8_s0 | [146, 83] | 3.9262 | True |

All fixed slices, provenance and checks are in `diagnostic.json`. No question text or answer is emitted.

Fusion status: complete.

Descriptive OOF slices overlap and are not confirmatory significance tests.
Historical architecture comparison includes optimizer/regularization recipe differences; it is not a clean causal architecture ablation.
Oracle gains are not expected ensemble gains. OOF-guided future hypotheses require registration and fresh fitting within the same bounded campaign.
