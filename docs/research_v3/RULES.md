# V3 rules and authority audit

Checked on 2026-10-30 at 00:28 UTC+8 (2026-10-29 16:28 UTC). This is an independent AI audit, not instructor approval. Complete retrieved source evidence is in `results/research_v3/audit_rules_startup.json`.

## Applicable authority and the final-test gate

The user adopted the full 288-line `DASE7506_MP2_Autonomous_Research_Prompt.txt`; the archived copy is `AUTONOMOUS_RESEARCH_SPEC.txt`, SHA256 `89ca6ae426de59b42bfcd7671ea77b04904a4416437ef97df0a9e8220516fcce`. Section XII explicitly authorizes one final evaluation and course submission of the unique candidate after development, independent review, clean-environment validation reproduction and freezing. It supersedes the V2 user-workflow restriction that waited for separate course-process confirmation. Historical V1/V2 freeze records remain unchanged as records of their original state.

The current [official Guide](https://github.com/xudongwu-0/DASE7506-MP2/blob/bb801f4f2e7629d270df765ab12c333b2dc616bb/guide/GUIDE.md), section 3, permits fitting only supplied train data and using validation for development/selection; test references are restricted to final scoring and reproduction. The [official README](https://github.com/xudongwu-0/DASE7506-MP2/blob/bb801f4f2e7629d270df765ab12c333b2dc616bb/code/README.md), section 2, requires freezing before final testing. Guide section 4 permits score updates with the same account and student ID, retaining historical entries and the best eligible score. The [website workflow](https://github.com/xudongwu-0/xudongwu-0.github.io/blob/63ee1cb984b9d3229196a14fc0d2312ba7704dd6/scripts/dase7506/README.md) requests a new issue for an improved result.

No explicit applicable ban on redevelopment after seeing an older aggregate score, or concrete conflict with this unique frozen-candidate procedure, was found in the checked official Guide, README, website and current public instructor MP2 issues. The official code repository issue search returned no issues; the website instructor MP2 search returned only two explicitly named `test` demonstration submissions. This is a bounded source finding, not a claim about unseen private notices. Lack of a separate permission email is not an added gate under the new user instruction.

**Current state: development only; no V3 test evaluation has been performed or authorized to run before its prerequisites.** The normal final gate is:
1. Finish train-only common-CV research, lock at most two finalists, use validation only for final development confirmation, select one.
2. Complete implementation/resource review and fresh-environment validation reproduction.
3. Freeze a unique source commit and every inference asset; disable any path back to research.
4. Run official prediction on the complete test bank and official independent scoring once. Save the exact score, timing and evidence.
5. Do not change features, weights, seeds, fusion, selector or candidate choice in response to that result, even if the ranking goal is unmet.
6. Publish matching immutable code/assets and create the course issue using the real website payload and personal account; preserve old submissions/assets. A same-frozen-model reproduction is separately documented and cannot select a model.

A concrete new applicable course restriction, if encountered, takes precedence and must be quoted with its source. It would block only the affected final action while the independent development and review work continues. No email, private message or unrelated comment is authorized.

## Versioned official sources

| Repository/file | Commit or blob |
|---|---|
| Official MP2 main commit | `bb801f4f2e7629d270df765ab12c333b2dc616bb` |
| Guide blob | `6e3df3ee0542a87c098076f1664e9bc4b394c3b1` |
| Code README blob | `6219d660006faecd648d4f318b69f7ce9331d6ca` |
| Website main commit | `63ee1cb984b9d3229196a14fc0d2312ba7704dd6` |
| Website workflow README blob | `1093195b296f56ed8771b93494556b831a96620b` |
| Publication metadata blob | `26bb40ff785d455a9ed6282348f5ddd5ba5fc1c5` |

Repository commit IDs above are distinct from Git blob IDs. Both current heads match the most recent V2 rule check. No new applicable course notice was found, so the historical root requirements matrix is not silently rewritten as though a new rule had appeared.

## Required constraints and V3 acceptance

| Constraint | Required evidence |
|---|---|
| Official train 4,755 / validation 798 / test 467; protocol v5 | Original manifest, official data/hash checker; no test content transferred to development |
| Official train only for weights and all learned preprocessing/calibration/fusion | Saved question-level common outer folds; inner early stopping; fold-local fits; matching preprocessing/weights |
| Full four-candidate labels and deterministic answer grouping | Multi-positive loss/gradient tests, None handling, exact parser/scorer preserved |
| At most two new validation finalists; unique test candidate | Preregistered selection rule, immutable lock and timestamps, no per-round validation tuning |
| Offline CPU inference, arbitrary N/IDs/order; Python int indices | Reference/network-denial contract, order and candidate-permutation tests with actual decision ties |
| <=16 MiB uncompressed all inference assets | Whole-bundle and embedded-parameter audit, including every ensemble member; target headroom below 12 MiB |
| <=30 seconds complete fresh prediction plus independent score; <=4 total threads | External complete-process timing and thread evidence; local hardware does not certify reference Xeon Platinum 8457C |
| CPU or one GPU no stronger than RTX4090 for training | Actual device/process records; one coordinated training scheduler |
| <=10-page report, precise reproducibility and AI/reuse disclosure | Rendered-page QA, real completed experiment table, negative results and total costs |
| Deadline 31 October 2026 23:59 UTC+8 | Form cutoff 1 November 00:00 UTC+8; reserve the user-required delivery buffer |
| Peer review begins on actual instructor release | Publication still has MP1 only; MP2 not open. Prepare recovery materials, do not mark review done |

## Ranking and identity boundary

The one startup ranking context reuses the previously read snapshot because the current website head is unchanged. The requested strict target is `rank <= floor(0.35 * N)`, currently 19 among 55 eligible student entries. The official display uses competition ranking for ties; best entries are grouped by exact student ID and case-insensitive account, with only `self-reported` and `verified` statuses eligible. The clearly identified teacher `test` demonstration is excluded from the student count.

The snapshot-specific threshold is recorded only in `RANKING_BACKGROUND.json`. It must not be read by the scheduler, training/configuration generator or model selection, and is not sent to model research agents. Do not refresh the leaderboard during development. One final post-submission check is allowed for reporting. Development hard accuracy and full-test leaderboard accuracy are not interchangeable.

The existing [MP2 Issue #294](https://github.com/xudongwu-0/xudongwu-0.github.io/issues/294) independently confirms personal account `liuqichang0305` (GitHub type User) and student ID `3036900957`; it is unchanged and self-reported. No new course issue has been created by this audit. Existing [research PR #2](https://github.com/liuqichang0305/dase7506-mp2-solution/pull/2) was open, unmerged and mergeable at the startup check. Merging one's reviewed PR is now within the user's explicit authority, subject to actual identity/tool permissions and branch protection.

