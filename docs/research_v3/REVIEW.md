# V3 independent review

Status: **startup checks passed; implementation and final delivery review pending**.

This is AI internal review, not human peer review, instructor approval or reference-machine certification.

Completed at startup:
- Full adopted user specification read; source and archived prompt hashes match.
- Current official rules and public notices checked; precise V3 final-test authority documented in [RULES.md](RULES.md).
- Personal account/student identity independently confirmed against the existing MP2 issue.
- V1/V2 frozen sources/assets and old search protection preserved; 92 previous preservation checks pass.
- Historical inventory and pre-existing untracked ranking logs recorded.
- Ranking background isolated from experiment configuration; no new test run or test data inspection.

The auditor will review each new implementation and the scheduler before signing its scientific/provenance claims. Final approval requires actual experiment evidence, unique-finalist selection/freeze sequencing, export and contract tests, fresh-environment reproduction, resource measurements, report QA and public code/asset/score consistency. No final V3 candidate has been signed off.

## First implementation checks

The persisted V3 folds are byte-identical to V2's common folds (SHA256 `fc2e962ccb38e8def416bf03e519a720c021822f025021fcc2bb950de4a8c4c9`). All three outer partitions contain 3,170 fit and 1,585 holdout questions, with disjoint 2,536/634 inner fit/stop partitions. Every question appears in exactly one outer holdout. The timing pilot uses inner stopping and a fresh outer refit through unchanged V2 primitives; it is correctly labeled one-fold timing evidence. See `results/research_v3/audit_pilot_review.json`.

`optimizers.py` and its eight synthetic tests were read in full. No blocking module issue was found: eligible hidden matrices and auxiliary AdamW groups are disjoint/complete; tied output aliases are excluded from Muon; EMA updates only one live trajectory and preserves buffer semantics; exact optimizer recovery and comparison with PyTorch EMA are tested. Trainer integration, RNG/scheduler restoration and stop/refit EMA consistency remain pending. See `results/research_v3/audit_optimizer_review.json`.

The preregistered plan was read. Two clarifications were sent to the coordinator before new validation: specify the exact validation fallback/tie rule against historical V2 and state explicitly that deployment uses fixed seed 3036900957 while extra seeds measure family stability. Incomplete or pruned fold sets must remain labeled incomplete, not be treated as full CV. The review does not reopen a separate-email final-test gate.

The new model/preprocessing modules and all six synthetic tests were read in full. TabM applies distinct member scaling before its first shared mixing; the tests check genuine member outputs/gradients, not copied predictions. RealMLP export parity uses a nonzero test head, avoiding trivial zero-output agreement. Fold-fitted median/IQR fallback and constant handling, candidate/question permutations, 137-row chunking, linear float64 preservation and NumPy-only loading are covered. Maximum logged synthetic score error is 5.9605e-7. No blocking module issue was found; actual finalist export/contract and resource checks remain pending. The coordinator/model author were asked to handle an interrupted weights-only bundle explicitly without overwriting a complete bundle. See `results/research_v3/audit_model_review.json`.

