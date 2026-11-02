# Fixed train-OOF diagnostic

Complete aligned train OOF only; official train text used only for predefined lengths and parser alignment.

| Method | Overall correct/N | Hard correct/N | Single-positive correct/N |
|---|---:|---:|---:|
| v3:v3_mlp_adamw_s0 | 4233/4755 | 1592/2114 | 317/711 |
| v3:v3_mlp_ema_s0 | 4242/4755 | 1601/2114 | 316/711 |
| v2:attention_regularized_s0 | 4328/4755 | 1687/2114 | 387/711 |

Oracle/reciprocal results are diagnostic; no fusion is implied.

| Explicit methods | Unique corrections per member | Hard oracle gain over strongest (pp) | Gate |
|---|---:|---:|---|
| v3:v3_mlp_adamw_s0 + v3:v3_mlp_ema_s0 | [26, 35] | 1.2299 | True |
| v3:v3_mlp_adamw_s0 + v2:attention_regularized_s0 | [85, 180] | 4.0208 | True |
| v3:v3_mlp_ema_s0 + v2:attention_regularized_s0 | [88, 174] | 4.1627 | True |

All fixed slices, provenance and checks are in `diagnostic.json`. No question text or answer is emitted.

Fusion status: not_run.

Descriptive OOF slices overlap and are not confirmatory significance tests.
Historical architecture comparison includes optimizer/regularization recipe differences; it is not a clean causal architecture ablation.
Oracle gains are not expected ensemble gains. OOF-guided future hypotheses require registration and fresh fitting within the same bounded campaign.
