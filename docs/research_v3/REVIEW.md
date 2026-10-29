# V3 independent review

Status: **startup and first implementation/scheduler checks passed; experimental and final delivery review pending**.

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

## Scheduler findings and verified repairs

The independent reviewer added `code/tests/test_scheduler_v3.py`, using temporary synthetic files and mocked process launch; no task training or evaluation was run by these checks. The first 14 checks exposed three real boundary failures: jointly changed queue/config values could bypass their stored digest, an unrecorded live project trainer could be launched twice, and a complete-looking summary could be accepted with stale source/data identity. The coordinator repaired these. Expanded testing found a fourth boundary: three fold IDs did not require each fold's completion status. This was also repaired, with per-fold question counts checked.

All 18 final tests now pass, including a positive authentic-schema three-fold reuse that cannot launch training, absolute clocks and delivery buffer, process identity, active/stale locks, retry cap, log preservation, changed source/data rejection, output/asset/OOF hashes and path containment. The test took 0.865 seconds (1.298 seconds including process startup), and the scheduler source hash was identical before and after. Earlier failures remain in `audit_scheduler_tests_initial.*` and `audit_scheduler_tests_after_first_repairs.*`; final evidence is `audit_scheduler_tests_passed.*`.

## Trainer and inference integration

The trainer was read in full. Inner-fit preprocessing stays paired with its model; inner stopping determines epochs, then outer and final models are freshly fitted on their allowed training partitions. EMA updates after the optimizer and is used consistently for stopping/export. Epoch recovery restores model, optimizer, EMA, NumPy/Torch RNG, permutation state and constant-schedule epoch. Seven actual synthetic tests passed, including exact Muon+EMA+dropout+permutation recovery, partition-fitted scaling, completed three-fold reuse without training, stale lock handling, read denial and incomplete-bundle quarantine. Safe experiment names and full bundle architecture/decoder/preprocessing consistency are enforced. Function-level CLOSED/finalist/FREEZE guards prevent bypass through direct training calls.

The scheduler and trainer now share the same fixed 11-file training source set, independently compared using AST literals. This keeps later inference/report-only additions from invalidating completed CV while preserving binding hashes for actual training dependencies. See `audit_training_scheduler_review.json`.

The actual V3 inference entry and both contract/export tools were also read. They use the average answer-group probabilities, then a documented first-model/first-member representative; ties are audited using that actual decision. Twelve isolated synthetic export/contract checks passed across MLP, TabM, RealMLP, V2 linear compatibility and two-/three-member manifests. Test inputs are unsupported, references are denied, and inference imports NumPy only. These synthetic manifests establish interface support, not evidence that a task ensemble was selected or improved. Real finalist full-validation export equality, contract, resource timing and fresh reproduction remain required. See `audit_inference_review.json`.

The coordinator clarified the preregistered validation rule before new validation: fixed deployment seed 3036900957; resource/contract eligibility; both hard correct >=297 and full correct >=740, with at least one strict improvement to replace historical V2. Eligible new candidates rank by hard correct, full correct, asset bytes, then fresh latency. Exact ties retain the historical method. Additional seeds measure family stability and cannot be cherry-picked. The earlier plan clarification is resolved.

