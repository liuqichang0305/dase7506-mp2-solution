# Measured budget decision

The MLP fold-0 pilot completed at 2026-10-29T08:45:24Z. Inner early stopping
ran 10 epochs and selected epoch 2 (7.245 seconds); outer refit took 1.676
seconds. Thus actual optimization cost was 8.920 seconds, with additional data
loading/feature/initial-import overhead in the process log. Fold-0 hard accuracy
was 533/705; this is a pilot, not the complete comparison.

Use a practical 60-minute additional sequential CPU experiment budget, with
15-minute wall caps per primary/stability/ablation batch. This is far below the
absolute six-hour cap. Retain and resume the pilot fold rather than repeat it.
The measured cost permits all registered MLP, ResMLP, Deep Sets and attention
families, the listwise loss control, and a matched-capacity MLP (width109,
232,190 parameters vs Deep Sets231,937). The matched-capacity control is registered
here before its first run. No TabM run is planned; it will be marked unrun.

Pilot training loss fell from 0.3003 to 0.0024 while stopping loss rose from
0.2667 to 0.5465. This is direct evidence of rapid overfitting. Before further
OOF comparisons, register one stronger-regularization variant for each of the
two strongest nonlinear families after the primary batch: dropout0.3,
learning_rate0.0003, weight_decay0.01, otherwise identical. This is a bounded
extension to the initial plan, motivated only by train-internal pilot curves.
Each family remains within four structural/hyperparameter configurations.
Then select the better registered variant within each family by OOF hard mean,
and add the two fixed extra seeds for the strongest two families. Hard-only
normalization and (if Deep Sets is a selected family) context removal remain
key ablations. No further architecture/hyperparameter round follows these.

If interrupted, read the plan, this budget amendment, events.jsonl, each
registration/summary, finalist lock, freeze and Git status. Rerun a named batch
only while its train-only stage remains open; completed fold hashes are checked.
After finalist locking/validation, do not restart search. Formal test remains off.
