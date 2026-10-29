# Project state

## Active V3 research — 2026-10-30 UTC+8

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
  tests. NumPy-only inference/contract checks passed; train-only resume and
  scheduler integration checks are being completed before the first full round.
- Train-only table recomputed all19 historical full-CV configurations with
  source/data/common-fold verification. No leaderboard feedback enters selection.
- Current queue/state is authoritative in `results/research_v3/campaign_queue.json`
  (created when the first round is registered), registry and heartbeat files.
- Next: run four registered MLP optimizer controls, then TabM, RealMLP and
  interaction controls. Read `docs/research_v3/SEARCH_PLAN_V3.md` and `RESUME.txt`.
- At most two NEW validation finalists, fixed deployment seed3036900957. New
  selection must satisfy the exact historical-fallback criterion in the plan.
- Final submission and actual peer review are not completed for V3. Peer review
  awaits the instructor's actual MP2 release. Historical Issue#294 remains valid.

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
