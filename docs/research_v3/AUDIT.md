# V3 startup audit

Independent AI audit at 2026-10-30 00:28 UTC+8. This document records completed checks, not completion of the V3 experiment campaign.

The full 288-line user specification was read in UTF-8. Its archived bytes match the source SHA256 `89ca6ae426de59b42bfcd7671ea77b04904a4416437ef97df0a9e8220516fcce`. The initially attempted system-default text decoding was garbled; all three UTF-8 chunks were subsequently read in full.

The actual working tree is on `research/autonomous-v3`, based on commit `cb27a2c2ebce501c7266ea894a2187fa128d0102`. No existing tracked diff was present at audit startup. The two pre-existing untracked ranking logs, `results/leaderboard-review-2026-10-29.json` and `results/leaderboard-review-2026-10-30.json`, were inventoried and left unchanged.

Integrity checks passed:
- All 15 V1 frozen source/asset hashes and all 20 V2 frozen source/asset hashes match.
- All 92 checks in the prior V1 preservation manifest still match, including original search code, guard-bearing training code, logs, reports and assets.
- A hash/byte inventory of 829 historical tracked code, assets, results, reports, submission records and documentation is saved for final preservation comparison.
- Historical result files were hashed only, not parsed for test errors. Official test data/reference files were not read. No prediction, scoring or training was run by this audit.
- The manifest protocol/counts were inspected; official parser, prediction and scoring source hashes are preserved.

Re-run evidence check with a new output: `.venv/Scripts/python.exe results/research_v3/audit_startup_check.py --output results/research_v3/audit_preservation_recheck.json`. The checker refuses to overwrite evidence. The authoritative startup evidence is `results/research_v3/audit_startup_preservation.json`; preserve it at final review, because it is the campaign's historical baseline.

Official current heads, notices, submission flow and identity were checked independently through the GitHub connector. See [RULES.md](RULES.md). The new user-authorized final-test path has no concrete conflict in the checked current official sources. It is conditional on completed development, a unique frozen candidate, review and reproduction; it is not permission for test-feedback optimization.

The existing V2 research is valid historical development evidence and need not be mechanically repeated. It includes common nested folds, losses/decoders and nonlinear families; the V3 scheduler must verify fold/early-stop/preprocessing compatibility before reusing any OOF. Historical validation has already been examined, and the old test aggregate is known; neither is a pristine unexposed holdout.

Pending review: V3 question grouping and duplicate handling, scaler/weight pairing, scheduler deadlines and process identity, budget persistence, optimizer/EMA state, member diversity and loss reduction for TabM, validation lock, export parity, whole-bundle resources, clean reproduction, final freeze and submission consistency. These require actual new code/results and are not marked complete here.

