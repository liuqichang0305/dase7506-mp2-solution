# One reduced-TabM / Attention-EMA mixture

This is a separate, bounded follow-up to the saved train-OOF diagnostic in
`results/research_v3/diagnostic-r05-reduced-tabm-vs-attention/`. That diagnostic
reported 43 reduced-TabM-only and 60 Attention-EMA-only correct hard questions,
with an unattainable oracle of 1728/2114 versus the stronger member's 1685/2114
(+2.0341 percentage points). Passing this gate does not establish a real mixture
improvement. The two previously evaluated full-feature mixtures remain rejected.

The ordered members are permanently fixed:

1. `v3:v3_tabm_k4_embedding_stats_s0`
2. `v3:v3_attention_ema_s0`

Use `[0.5, 0.5]` model-level answer-group probabilities. TabM first averages its
four internal members' probabilities. A winning group is represented by the
highest raw score from the first TabM internal member; exact group or candidate
ties use the earliest original candidate, and all-None returns index 0. This
matches the existing diagnostic and deployment contract. No learned weights,
temperature grid, seed selection, conditional routing, calibration or member
replacement is permitted. The R11 reduced-feature k8/EMA configurations cannot
be substituted after their results arrive.

Reject if hard correctness does not strictly exceed Attention EMA's 1685/2114,
or overall correctness falls below its 4326/4755. Always disclose the change
from 388/711 single-positive correct. A local pass only retains the mixture for
the unchanged final comparison and resource rules; it does not trigger deployment
or holdout access. Separately report the pooled hard increment above the fixed
historical reference 1687/2114. The campaign continuation threshold is different:
at registration, lock the current best completed BASE957 global incumbent's ID,
summary/diagnostic hash and hard three-fold mean (including R11 if completed).
Compare this mixture's hard three-fold mean with that frozen value, using the
unchanged strictly-greater-than-0.5-percentage-point threshold. Refuse evaluation
if the queue changes between registration and evaluation.

The reduced feature result is not a pure information ablation: removing 1024
absolute-deviation coordinates also reduces first-layer capacity; 1024 embedding
coordinates and 16 statistics remain. The matched +62 hard-question improvement
is concentrated across folds (+45/+11/+6). The diagnostic does not establish
seed stability or a causal architecture advantage.

Run these commands from the repository root, in order. Only `prepare` is safe
while the current scheduler runs; it writes a separate immutable preregistration
and does not alter the shared registry or perform any mixture computation.

```powershell
.venv/Scripts/python.exe scripts/register_reduced_tabm_attention_fusion_v3.py prepare
# The root operator runs the following only after the current queue is terminal.
.venv/Scripts/python.exe scripts/register_reduced_tabm_attention_fusion_v3.py register
.venv/Scripts/python.exe scripts/register_reduced_tabm_attention_fusion_v3.py evaluate
.venv/Scripts/python.exe scripts/register_reduced_tabm_attention_fusion_v3.py summarize
```

The last three stages require an absent scheduler lock, no trainer lock, no
active project trainer and terminal queue states (`completed`, `failed`, or
`pruned`). They acquire the existing scheduler lock using this helper's actual
command identity. Existing locks are never removed. Registration/evaluation
also require the original absolute research deadline and the shared configuration
cap to remain open. The timing pilot and all prior mixtures count toward the cap;
the new mix adds exactly one configuration and zero training fits. The helper
does not modify the queue, core training sources, scheduler or old aggregate.

The evaluation invokes only the existing guarded `diagnose_oof_v3.py` with this
exact pair and the separately saved formal registration. Direct base Python
`-S` and only the fixed environment's site-packages avoid the Windows venv launcher;
the subprocess timeout therefore terminates the actual non-spawning worker.
Child wall time includes
Python startup, train-OOF checks, deterministic mixing and output generation.
Parent postcheck elapsed is recorded separately. Neither value is inference
latency or CPU time. Raw stdout/stderr, UTC start/end, source and input/output
hashes are retained. A failed or interrupted attempt is preserved and never
automatically replayed; engineering repairs need a separate review. A completed
stage can verify its immutable receipt without rerunning the diagnostic.

Outputs, once actually run:

- Preregistration: `docs/research_v3/rounds/r12_reduced_tabm_attention_equal_fusion_preregistration.json`.
- Formal registration: `docs/research_v3/rounds/v3_fusion_reduced_tabm_attention_ema_s0.json`.
- Evaluation: `results/research_v3/v3_fusion_reduced_tabm_attention_ema_s0/`.
- Execution receipt: `results/research_v3/fusion_r12_execution/execution.json`.
- New three-result aggregate: `results/research_v3/fixed_fusion_analysis_final.json`.
- New final hash ledger: `results/research_v3/fusion_r12_final_ledger.json`.

The new aggregate preserves both old result objects and their measured wall
values exactly, appends the true third outcome, and binds the full three-entry
registry. It appends one new round decision with `configurations: 1`; it does not
count the previous mixtures twice. Its receipt binds an immutable snapshot of
the round ledger and the exact R12 record. Later stability rounds can append to
the live ledger without invalidating this historical completion receipt.
The report generator accepts the new aggregate without parser changes:

```powershell
# Only at the later authorized report stage, after all required final evidence exists:
.venv/Scripts/python.exe reports/research_v3/prepare_report_inputs.py --fusion-analysis results/research_v3/fixed_fusion_analysis_final.json
```

Synthetic helper tests are in `code/tests/test_reduced_tabm_fusion_registration_v3.py`.
They use temporary fabricated JSON/files and a mocked diagnostic process; they
do not fit models or open any real train, validation, test or ranking data. The
test receipt is not evidence that the real mixture was evaluated.
