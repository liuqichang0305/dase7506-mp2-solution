# Project state

## Current post-deadline research archive — 2026-11-02 UTC+8

- Actual resumed time is November2. The original research budget and October31
  score deadline have expired. SEARCH_CLOSED.json is now immutable; no search
  retry or clock reset is permitted. Exact closure-time log snapshots are in
  results/research_v3/closure_evidence/SNAPSHOTS.json.
- 35 new base configurations/105 outer folds completed. Four extra-seed runs
  were pruned, with no complete new seed result. Long monitor gaps and actual
  wall-clock overshoots are preserved; continuous CPU work and a hard real-time
  cutoff are not claimed. All four fixed mixtures were evaluated and rejected.
- The two locked base957 final fits and research-only validation completed.
  Reduced-input TabM EMA scored735/798, hard292/355; Attention EMA736/798,
  hard293/355. Both failed the fixed replacement rule against historical V2
  Deep Sets740/798, hard297/355. No new validation improvement is claimed.
- The unique SELECTED_MODEL is historical deepsets_regularized_s0, copied
  byte-for-byte to artifacts/research_v3/final_selected. Actual798-question
  export agreement and contract checks passed in selected-export-validation.json
  and selected-contract-validation.json. Independent final review subsequently
  passed; see submission/v3/REVIEW_RESULT.json.
- Fresh public-checkout reproduction completed at exact source commit
  eee478b4959e555b6ec8c902b6db8b69f31df508 with a new NumPy-only environment:
  740/798, all 798 choices identical, fresh prediction plus independent scoring
  2.2803907999768853 seconds on local hardware. See
  submission/v3/REPRODUCTION_RESULT.json and docs/research_v3/REPRODUCTION.md.
  The actual research-only submission/v3/FREEZE.json is complete and records
  research_only=true/formal_test_allowed=false. Do not rerun the reproducer
  inside this now-frozen checkout or remove its guards.
- FORMAL_SUBMISSION_BLOCKED.json records the actual deadline/review-phase rule.
  No new V3 official test, score update, or replacement of historical submitted
  assets will be made. The immutable V1 Issue294 and its348/467 assets remain.
- Official MP2 peer review opened2026-11-01T04:42:52.971Z and closes
  2026-11-09T00:00:00+08:00. The separate published-submission240 workflow
  completed its exact-checkpoint reproduction and official Peer Review Report:
  https://github.com/xudongwu-0/xudongwu-0.github.io/issues/383.
  Creation and public readback are saved in
  peer_review/submission_240/report_form/REPORT_SUCCESS.json. No duplicate
  evaluation or POST is needed. It cannot influence our closed research.
  See RULES_RESUMPTION_20261101.md for current public rules and
  the recorded incidental public-score exposure during the rules audit.
- Saved confidence and paired validation diagnostics are complete; choices are
  unchanged and calibration worsened validation ECE. Their existence is not a
  new model score or an unqualified confidence-improvement claim.
- The 10-page English report and Chinese guide are complete. Delivered PDF:
  reports/research_v3/research_closeout_delivered/research_closeout_v3.pdf.
  All rendered pages passed visual QA; earlier attempts are preserved.
  Portable official validation commands were also checked in the existing clean
  environment without train data/states, retaining all 798 choices and 740/798.
  The original guarded 2.2803908-second result remains the freeze evidence.
- Remaining: own PR #3 review/merge and final delivery receipt. Do not rerun
  completed selection, reproduction or peer evaluation. Original formal tools
  remain TEST_COMPLETE-gated; no V3 test result or score update is claimed.
- The post-freeze ranking check is already saved in
  results/leaderboard-review-2026-11-02-final/README.md. Do not use it to reopen
  research or imply a new V3 official score.

## V3 research history — 2026-10-30 UTC+8

Historical checkpoint only. Earlier plans to continue research or perform a
new formal test/submission are superseded by the expired budget, closed search
and current post-deadline state above. Historical numbers and logs are retained.

The user adopted `docs/research_v3/AUTONOMOUS_RESEARCH_SPEC.txt` and explicitly
authorized a new bounded train/CV campaign, review, reproduction and one uniquely
frozen final test/submission. This new authority supersedes the historical stop
instructions below for V3; all historical method/assets/logs remain preserved.

- Branch: `research/autonomous-v3`, based on V2 `cb27a2c2ebce501c7266ea894a2187fa128d0102`.
- Budget: original start `2026-10-29T16:22:42Z`, absolute research end
  `2026-10-30T16:22:42Z`, <=96 configurations, <=2 justified repair retries each.
- Actual timing pilot completed: 26.866 seconds, one outer fold only,
  hard530/705. No new validation/test evaluation has occurred.
- V3 TabM/RealMLP/preprocessing and AdamW/Muon/EMA modules passed synthetic
  tests. NumPy-only inference/contract, exact epoch resume and scheduler
  integration checks passed. The original nine training rounds completed31
  configurations/93 outer fits at2026-10-30T04:20Z (R08 is separate fusion).
  Best new TabM k8:1619/2114 hard; robust-all RealMLP:1611/2114. Both remain
  below historical Attention1687/2114. V3 Attention AdamW1682, EMA1685,
  Muon1666 and width64 1670. Evidence: analysis/r04_complete.
  Muon attempt1 hit1800s; attempt2 resumed only fold2 from epoch8 and completed.
  Reduced-feature TabM k4 reached1668 hard (+62 over full-input k4), with gains
  +45/+11/+6 by fold. This changes input width and first-layer capacity, so it
  does not isolate a causal feature effect. R06 pairwise1684, listwise1672,
  BCE1668 and group/decoder alpha0.5 1663 do not beat historical Attention.
  R07 Deep Sets AdamW1657 / EMA1665 and ResMLP AdamW1621 / EMA1625 are
  complete; neither exceeds the existing Attention controls. All four rows
  remain in the main table, with actual costs and common-fold evidence.
  R09 stronger weight decay1683/dropout1673/width256 1677 did not improve
  Attention. R10 RealMLP reduced input1665 (+54 versus matched full-input1611)
  warrants two bounded variance controls; lower LR1610/width256 1607 did not help.
  R11 and R14 completed at04:37Z: reduced TabM k8=1663, EMA=1687;
  reduced RealMLP dropout0.1=1682, EMA=1658 hard. All35 base recipes/105 outer
  fits are complete. Scheduler PID44908 exited; no current trainer is active.
  Two separately registered equal-probability fusions are complete and rejected:
  Attention+TabM1667 hard; Attention+RealMLP1670, both below their Attention member.
- Train-only table recomputed all19 historical full-CV configurations with
  source/data/common-fold verification. No leaderboard feedback enters selection.
- Current queue/state is authoritative in `results/research_v3/campaign_queue.json`,
  registry and heartbeat files. Scheduler restart retained the live trainer;
  Windows venv child-tree timeout cleanup passed a real no-data process test.
- The R12 reduced-live TabM/Attention-EMA fixed mixture actually completed:
  1680 hard/4321 full, rejected. Its new aggregate preserves both prior mixtures.
  A separate fresh gate supports exactly one reduced-EMA TabM/Attention-EMA
  equal mixture (R15 tool preparation/review; not yet evaluated).
- Next: execute R15 after its immutable preparation/review. R13 stability
  recipes are now explicitly prepared (not registered): reduced TabM EMA and
  Attention EMA, each with the original fixed extra seeds958/959. Deployment
  remains seed957. No extra-seed outcomes have been observed. Read
  `docs/research_v3/SEARCH_PLAN_V3.md` and `RESUME.txt`.
- At most two NEW validation finalists, fixed deployment seed3036900957. New
  selection must satisfy the exact historical-fallback criterion in the plan.
- Final submission and actual peer review are not completed for V3. Peer review
  awaits the instructor's actual MP2 release. Historical Issue#294 remains valid.
- V2 PR#2 was actually merged2026-10-30T02:50:29Z (merge fafa742443926920298b82c2fbe022cef6d29632).
  V3 PR#3 now targets main and remains a draft. Reviewed delivery-boundary,
  clean-reproduction and anonymous-publication tools are implemented; their
  synthetic tests are not actual final-model reproduction or submission.

## Historical V1 state (preserved)

Updated: 2026-10-29 (Asia/Shanghai). Student: 3036900957. GitHub: liuqichang0305.

- Status: **submitted_waiting_peer_release**. Stages 0-11 completed. Stage 12 preparation complete; actual peer review NOT completed because the instructor has not released MP2 links.
- Upstream main commit verified through GitHub API: bb801f4f2e7629d270df765ab12c333b2dc616bb (unchanged from supplied workflow).
- Workspace: C:/AAmy work information/7503vision/dase7506-mp2-solution. Isolated under the granted writable root.
- Python 3.12.0 virtual environment created at .venv.
- Host: Windows 11, Intel Core i9-12900H, 20 logical CPUs; RTX 3070 Ti Laptop detected. Default computation: CPU.
- GitHub connector authenticated as liuqichang0305. Proposed solution repository returned 404; existence/creation still to verify.
- Frozen source commit: 8602992842d444524ab4148991da2f1dc88f316f; see submission/PRE_TEST_FREEZE.json.
- Official full test scored after freeze: 348/467 = 74.51820128479658%, full local wall time 0.705959s, results/test-score.json and test-timing.json.
- DO NOT change model/weights/selection using this test result. No test question/reference inspection is permitted for development. Only same-frozen-method reproduction remains allowed.
- Formal course Issue: https://github.com/xudongwu-0/xudongwu-0.github.io/issues/294, author liuqichang0305 (User), student 3036900957, created 2026-10-29 15:04:45 UTC+8. Exact official generated body read back and verified.
- Course snapshot 2026-10-29T07:04:57.955Z recognizes #294 as self-reported with artifact_status=received. This is not instructor verification.
- Submitted immutable code commit: 419e1158ba2f83b8c08796429e635f1bf8ab2977; matching checkpoint under artifacts/final_bundle at that commit. Public repository: https://github.com/liuqichang0305/dase7506-mp2-solution.
- Parallel task: official-rule and primary-literature review; no test access permitted.

Completed: official checker passed; 6 inner-train screens, 3 full fits, 3 train-only OOF fits; validation comparisons and calibration; independent AI internal review, 8 unit tests, isolated synthetic/validation interfaces, original official hashes, 51,923-byte asset audit.

Report complete: reports/report.pdf, 7 pages, all rendered and AI visually checked. Independent anonymous public clone at 92c7240c5a4710054de1363d23e2f11f81d33b72 into a new Python 3.12 NumPy-only environment reproduced 348/467 with all 467 choices identical, no retraining, 1.1398808s local wall time. Code, PDF and both model assets were anonymously downloaded and hash-verified. See docs/REPRODUCTION.md.

Merged PR: https://github.com/liuqichang0305/dase7506-mp2-solution/pull/1 (personal account; internal AI review only). No branch protection was bypassed.

Next: when resumed, read submission/SUBMISSION_RECORD.md and submission/PEER_REVIEW_RESUME.txt, then check the real publication.projects.mp2 published_at/review_ends_at. If absent, keep waiting. After release, perform a genuine independent peer reproduction following the saved checklist. Do not alter the frozen submitted method or run new test-driven model selection. No automatic future task was established.

Resume: read this file, DECISIONS.md, BLOCKERS.md, EXPERIMENT_LOG.jsonl, requirements_matrix.md and submission/ before running anything. Never repeat model selection after test is frozen/evaluated.
