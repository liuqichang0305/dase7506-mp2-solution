# R11 reduced TabM EMA train-OOF diagnostic

Exactly one prior-specified reciprocal gate; no probability mixture, fitting, calibration, validation, test or ranking access.

| Method | Overall | Hard | Single positive |
|---|---:|---:|---:|
| v3:v3_tabm_embedding_stats_ema_s0 | 4328/4755 | 1687/2114 | 376/711 |
| v3:v3_attention_ema_s0 | 4326/4755 | 1685/2114 | 388/711 |
| v3:v3_tabm_k4_embedding_stats_s0 | 4309/4755 | 1668/2114 | 364/711 |

Reciprocal unique corrections: [51, 49]; hard oracle 1736/2114, gain above stronger member 2.31788079 pp; gate passed: True.

The oracle is unattainable and does not establish any actual fusion improvement. This does not change the already preregistered R12 live reduced-TabM mixture. A future mixture would require its own bounded independent registration.

Actual diagnostic child wall: 1.374983200 s (direct base Python -S; includes startup and guarded train-OOF diagnostic/output; not inference latency or CPU time).
