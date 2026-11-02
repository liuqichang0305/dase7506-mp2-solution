# Initial V3 train-OOF diagnostic and bounded hypotheses

Evidence: `results/research_v3/diagnostic-r1-verified/diagnostic.json`, with the exact diagnostic source snapshot beside it. Only the completed `v3_mlp_adamw_s0`, `v3_mlp_ema_s0`, and historical `attention_regularized_s0` were analyzed. The tool verified common question-level folds, identical labels/groups/order, complete three-fold reports, registered source hashes, registered official train hashes and saved per-fold/OOF score hashes. Official train candidate text was used only for parser alignment and predetermined length counts. No question text or generated answer is emitted. Validation/test/leaderboard inputs were not opened.

The initial diagnostic is preserved in `diagnostic-r1-initial`. The verified run adds explicit wholly-correct answer-group label validation, per-fold completion checks and stronger future fusion-registration checks; it yields the same metrics. Four synthetic diagnostic tests passed in 0.166 seconds, including fixed-bin coverage, exact gate thresholds, three-member unique contributions, and rejecting unregistered fusion/path traversal. Evidence: `results/research_v3/diagnostic-synthetic-tests.log`.

## Completed OOF results

All figures below are actual pooled train-OOF counts, not new validation or test results.

| Method | Overall correct/N | Hard correct/N | Single-positive correct/N |
|---|---:|---:|---:|
| V3 MLP AdamW | 4,233/4,755 | 1,592/2,114 | 317/711 |
| V3 MLP EMA | 4,242/4,755 | 1,601/2,114 | 316/711 |
| Historical V2 Attention | 4,328/4,755 | 1,687/2,114 | 387/711 |

EMA adds nine hard questions to its matched live-weight MLP control: approximately 0.426 percentage points, below the campaign's 0.5-point continuation threshold by itself. Attention is a historical architecture-and-recipe comparator, not a clean one-variable causal ablation. Its hard advantage over EMA is 86 questions. All methods correctly handle the 2,641 questions with only one parseable answer group; current errors are concentrated in the genuinely ambiguous group decisions.

| Number of parseable answer groups | N | AdamW correct | EMA correct | Attention correct |
|---|---:|---:|---:|---:|
| 2 | 1,338 | 1,108 | 1,107 | 1,155 |
| 3 | 608 | 400 | 406 | 428 |
| 4 | 168 | 84 | 88 | 104 |

Within hard questions with exactly one positive candidate, AdamW scores 290/684, EMA 289/684, and Attention 360/684. With three positives, the hard results are 717/742, 719/742 and 718/742 respectively. Thus the large residual weakness is not a uniform difficulty increase across all label multiplicities.

All 37 fixed descriptive bins are retained in the JSON, including empty bins: answer-group counts 1–4, positive counts 1–4, unparseable counts 0–4, question word/character counts, mean/max candidate word counts, candidate word-count spread, and mean candidate character counts. Bins are question-level, overlap across dimensions and were not adapted to errors. For example, the candidate-max-length 200–400-word bin has 182 hard questions, with 125/182 AdamW, 127/182 EMA and 139/182 Attention, versus 506/656, 506/656 and 533/656 in the under-100-word bin. Length is confounded with reasoning and answer-group difficulty; these figures do not establish a causal length penalty. Very sparse tail bins are not grounds for a new specialization.

## Reciprocal errors and oracle ceiling

| Explicit pair (order shown) | Only first correct | Only second correct | Hard oracle correct/N | Oracle gain over stronger member |
|---|---:|---:|---:|---:|
| AdamW + EMA | 26 | 35 | 1,627/2,114 | 1.2299 pp |
| AdamW + Attention | 85 | 180 | 1,772/2,114 | 4.0208 pp |
| EMA + Attention | 88 | 174 | 1,775/2,114 | 4.1627 pp |

Each pair meets the necessary threshold of at least ten reciprocal corrections and at least one hard-accuracy point of oracle potential. An oracle chooses the correct member using labels and is unavailable at inference. No probability fusion was evaluated, no pair was selected, and no gain from real fusion is claimed. The relatively large Attention/MLP complementarity supports retaining structurally different candidates for later evidence review, rather than assuming the weaker overall method is redundant.

## At most three future hypotheses

1. **Candidate interaction should remain a priority.** The historical Attention advantage is present at two, three and four answer groups and is largest proportionally at four groups (104/168 versus EMA 88/168). The already planned width-64 Attention and matched EMA control can test whether a compact interaction model retains that advantage under the V3 optimizer/stopping protocol. Register each exact configuration first; this diagnostic does not authorize changing a running model or prove that attention alone caused the difference.

2. **Multiplicity-sensitive objectives/decoding may address the single-positive weakness.** The EMA model has 289/684 hard single-positive correctness but 719/742 hard three-positive correctness. A bounded group-size penalty decoder comparison (`alpha=0,0.5,1`, fixed `tau=1`) followed, if justified, by one registered group-CE versus question-averaged pairwise fit on the same eligible family can test reliance on answer frequency. Keep every correct candidate positive, use the same common folds and avoid selecting new thresholds from validation/test. This is a hypothesis about the training objective, not evidence of an achieved improvement.

3. **Robust treatment of text/statistic coordinates merits its planned control.** Performance is lower in longer or more dispersed candidate-text bins, but these bins are confounded. The already planned RealMLP-inspired standard versus `robust_stats` comparison can test whether fitting median/IQR plus smooth clipping only on the 16 statistic coordinates helps without altering the embedding dimensions. Compare identical architecture/optimizer/seed and training partitions; report the full OOF and hard results alongside the same fixed length slices. Do not add length-specific routing or widen the search because of small tail-bin noise.

These are exploratory OOF-derived priorities within the existing budget, not additional completed experiments. No training or model selection was started by this diagnostic task.

## Tool usage and fusion guard

```text
.venv/Scripts/python.exe code/tools/diagnose_oof_v3.py --output <new_diagnostic_directory>
```

With no `--method`, the tool uses exactly the three methods above and their three explicitly listed diagnostic pairs. To use other completed results, repeat `--method v3:<name>` or `--method v2:<name>`, then explicitly name each `--pair <first> <second>` or `--triple <first> <second> <third>`. There is no pair/weight grid or automatic mixture selection.

An optional `--fusion-registration <file>` requires a separate registration with `status="registered"`, `purpose="single_fixed_equal_probability_oof"`, ordered `methods`, exact equal `weights`, each member's locked `oof_sha256`, an ISO `registered_at_utc`, and `gate_evidence` containing the repository-relative path and SHA256 of an earlier diagnostic JSON. The earlier evidence must contain the same ordered pair/triple, the same OOF hashes, and a passing gate. All members must use the fixed deployment seed and masked group-LSE decoder. The registration must follow that evidence in time. For three members, the conservative gate requires each member to uniquely correct at least ten errors shared by the other two, plus total hard oracle potential of at least one point over the strongest individual. Only then does the tool calculate the one equal group-probability average and sum actual member asset sizes. This task supplied no fusion registration, so fusion remains `not_run`.

Future completed fusion output includes an exact byte copy `fusion-registration.json`, its SHA256, and `fusion-oof.npz` with probabilities, choices, correctness, fold, groups and labels. `diagnostic.fusion` binds the copied registration and NPZ hashes. The separate `reliability_fusion_v3.load_fusion_oof` reader validates those artifacts, every member's complete OOF/source hashes and shared folds, the prior gate, equal probability reconstruction and the actual chosen-group confidence event. It preserves the complete registration object as the selector identity and refuses `not_run` outputs. Its five synthetic tests passed in 0.584 seconds (`reliability-fusion-synthetic-tests.log`); these tests used temporary artificial OOF arrays, with no actual fusion experiment, calibration fit or holdout access.
