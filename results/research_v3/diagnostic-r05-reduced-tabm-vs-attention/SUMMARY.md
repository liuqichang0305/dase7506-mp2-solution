# R05 reduced TabM: actual train-OOF diagnostic

The reduced-feature TabM improves on its matched full-feature k4 TabM by 62
questions, all in the hard subset. This is complete three-fold training OOF,
not validation or test evidence. No calibration or cross-model fusion was run.

| Method | Overall / 4755 | Hard / 2114 | Single-positive / 711 |
|---|---:|---:|---:|
| k4 TabM, embedding + 16 statistics | 4309 (90.6204%) | 1668 (78.9026%) | 364 (51.1955%) |
| Matched k4 TabM, full relation features | 4247 (89.3165%) | 1606 (75.9697%) | 322 (45.2883%) |
| Attention + EMA | 4326 (90.9779%) | 1685 (79.7067%) | 388 (54.5710%) |

The two TabM configurations differ only in experiment ID and `input_features`:
`embedding_stats` versus the default `relation`. They retain the same fixed
seed 3036900957, folds, preprocessing recipe, objective, decoder, k=4, layer
width/depth, dropout, optimizer, learning rate and weight decay. The selected
input dimension changes from 2064 to 1040, so this comparison also changes the
first-layer parameter count. It cannot distinguish information removal from
capacity/regularization effects.

Using the existing `analyze_oof_v2.paired_comparison` with 5000 resamples and
base seed 3036901059:

| Subset | Reduced minus full | Difference | Paired percentile 95% interval |
|---|---:|---:|---:|
| Overall | +62 | +1.3039 pp | [0.6940, 1.9353] pp |
| Hard | +62 | +2.9328 pp | [1.5137, 4.3992] pp |

The function uses seed 3036901059 for overall and its fixed `+1` offset,
3036901060, for hard. It resamples whole question pairs. Reduced TabM uniquely
corrects 146 full-TabM errors and loses 84 questions that full TabM got right.
Both miss 362. These intervals are conditional on the completed OOF predictions;
the search that identified this configuration adds selection optimism.

The improvement is uneven across folds: overall/hard gains are +45, +11 and +6.
Single-positive gains are +41, +3 and -2. Thus the positive pooled result does
not establish seed stability, and much of the single-positive gain is in fold 0.

Only one reciprocal gate was tested: reduced TabM versus Attention + EMA.
Reduced TabM uniquely gets 43 questions right; Attention uniquely gets 60 right.
All these disagreements are in the hard subset. The unattainable oracle gets
1728/2114 hard questions correct, versus Attention's 1685/2114, a +2.0341 pp
upper-bound gain. Both directions exceed the fixed ten-question minimum and
the hard oracle gain exceeds 1 pp, so this diagnostic gate passes. This is not
an observed fusion result or a promise of improvement. The two previously
failed full-relation mixtures are different methods; this run neither repeats
nor replaces their results. Fusion remains `not_run`.

Fixed descriptive slices provide a limited, interpretable complementarity
pattern:

| Fixed slice | N | Reduced | Full k4 | Attention | Reduced-only / Attention-only |
|---|---:|---:|---:|---:|---:|
| 2 answer groups | 1338 | 1143 | 1113 | 1151 | 20 / 28 |
| 3 answer groups | 608 | 428 | 403 | 431 | 17 / 20 |
| 4 answer groups | 168 | 97 | 90 | 103 | 6 / 12 |
| 1 positive candidate | 711 | 364 | 322 | 388 | 20 / 44 |
| 2 positive candidates | 732 | 657 | 637 | 652 | 14 / 9 |
| 3 positive candidates | 1070 | 1046 | 1046 | 1044 | 9 / 7 |
| 4 positive candidates | 2242 | 2242 | 2242 | 2242 | 0 / 0 |

All 2641 one-answer-group questions are correct for all three methods; the
feature change's gain is not an improvement on these already solved cases.
The +62 matched-baseline gain splits into +30/+25/+7 across 2/3/4 answer groups,
and +42/+20/+0/+0 across 1/2/3/4 positive candidates. It also occurs with no
unparseable candidates (+50), one (+9), and two (+3), so it is not confined to
the parseability edge cases.

Attention remains better on single-positive questions by 24, while reduced
TabM is ahead by 5 on two-positive and by 2 on three-positive questions. This
supports heterogeneous error evidence rather than dominance by reduced TabM.
These positive-count slices use saved training labels and are explanatory;
they cannot be used as a deployed routing feature. Question-length slices
show no clean routing rule: reduced versus Attention is +1 for fewer than 40
words, -20 for 40--79 words, +2 for 80--119 words (N=144), and tied at N=8 for
120+ words. Candidate-mean-word bins show matched-baseline gains +31/+30/+1/0
for [0,100), [100,200), [200,400), and [400,infinity); the last two bins have
only 44 and 0 questions. No new thresholds, grid or mixture weights were fitted.

All 37 predefined slices, including empty bins, are in `diagnostic.json` and
`paired-and-slices.json`; `FIXED_SLICES.md` is a complete count table. Slices
overlap and were not separately significance-tested. Common folds, labels,
groups, candidate ordering, OOF hashes and source provenance were verified.
The input guard recorded zero rejected accesses. Official raw reads were
limited to the train candidate text and manifest for parser alignment and the
predefined length bins; no raw gold or embeddings were needed.

Actual command (refuses to overwrite this completed diagnostic directory):

```powershell
.venv/Scripts/python.exe code/tools/diagnose_reduced_tabm_v3.py
```

`paired-and-slices.json` contains the complete input/source SHA-256 inventories,
configuration differences, exact bootstrap values, scope flags and elapsed
wall time. Exact diagnostic/wrapper/paired-comparison source snapshots and the
actual execution log are retained. No question text or generated answer is
included in the outputs.
