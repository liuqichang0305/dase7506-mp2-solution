# Official-source review

Reviewed on 29 October 2026 (Asia/Shanghai), by an AI research sub-agent. This is an internal evidence record, not instructor certification. No test input, test reference, or test prediction was used by this reviewer.

## Versions and sources

- Course repository main commit: `bb801f4f2e7629d270df765ab12c333b2dc616bb`, matching the workflow's recorded commit. [Guide](https://github.com/xudongwu-0/DASE7506-MP2/blob/bb801f4f2e7629d270df765ab12c333b2dc616bb/guide/GUIDE.md) and [code README](https://github.com/xudongwu-0/DASE7506-MP2/blob/bb801f4f2e7629d270df765ab12c333b2dc616bb/code/README.md) were read in full.
- Website main commit observed: `383e61ac4158d6a1d66d632c4fdfa0795d9d379e`, dated `2026-10-29T02:18:04Z`. [Commit](https://github.com/xudongwu-0/xudongwu-0.github.io/commit/383e61ac4158d6a1d66d632c4fdfa0795d9d379e).
- Read the submission and peer-review logic in [mp2.mjs](https://github.com/xudongwu-0/xudongwu-0.github.io/blob/383e61ac4158d6a1d66d632c4fdfa0795d9d379e/courses/dase7506/mp2.mjs), the validators and phase function in [arena.mjs](https://github.com/xudongwu-0/xudongwu-0.github.io/blob/383e61ac4158d6a1d66d632c4fdfa0795d9d379e/courses/dase7506/arena.mjs), and [projects.mjs](https://github.com/xudongwu-0/xudongwu-0.github.io/blob/383e61ac4158d6a1d66d632c4fdfa0795d9d379e/courses/dase7506/projects.mjs).
- The [leaderboard snapshot](https://github.com/xudongwu-0/xudongwu-0.github.io/blob/383e61ac4158d6a1d66d632c4fdfa0795d9d379e/courses/dase7506/data/leaderboard.json) was fetched to inspect only `publication` and `generated_at`. No peer scores or solutions were used for model development. Snapshot generation: `2026-10-29T02:18:04.537Z`.

Git blob identifiers for the fetched files are recorded separately from commits: GUIDE `6e3df3ee0542a87c098076f1664e9bc4b394c3b1`; code README `6219d660006faecd648d4f318b69f7ce9331d6ca`; mp2.mjs `681238abc53d023e1ac049a9ea086c7d12aee478`; projects.mjs `bc0ea31136171598c902c7ad30c84636b162ead2`; leaderboard.json `9a9c08f8c0ad2af86d78bee3d0ca6f0f8626e008`.

## Confirmed course requirements

The course requires training-only fitting, validation development, frozen final-test scoring, unchanged supplied data/parser/scorer, no external datasets or pretrained weights, and offline selection of one supplied candidate. Inputs use four candidates and `[N,4,1024]` float32 representations; the output is one Python integer in `0..3` per question. IDs are for alignment; a new audit bank must work without retraining.

The limits are 30 seconds for a fresh CPU prediction process plus independent scoring, at most four threads across libraries/processes, and 16 MiB of uncompressed additional inference assets. Reference eligibility is decided on the Xeon Platinum 8457C; local measurements must identify their hardware. Training may use CPU or one GPU no stronger than RTX 4090. Assets include configuration and learned statistics. Ordinary source and supplied data are excluded, but arrays hidden in source are included.

Final delivery needs publicly accessible frozen code and matching checkpoint/configuration, a report of at most 10 pages, exact setup/training/evaluation commands, official score JSON, experiment/resource records, seeds, hashes, and AI/reuse disclosure. Evaluation must work without retraining. Python 3.12 is specified by the code README.

## Submission and publication state

The Guide states **31 October 2026, 23:59 (UTC+8)**. The site's exclusive cutoff is `2026-11-01T00:00:00+08:00`. Both final code and checkpoint links are due before that cutoff. A later link-collection interface does not relax the Guide's deadline.

The [official form](https://xudongwu-0.github.io/courses/dase7506/mp2/) prepares an Issue for `xudongwu-0/xudongwu-0.github.io`. Create that Issue to finish submission; merely preparing a link is insufficient. Its schema is `dase7506/submission-v1`, website protocol `7506-submissions-v1`, project `mp2`. This differs from the benchmark protocol. The form encrypts artifact links; student ID, score, and submitting account are public. Preserve the generated payload. `parseIssue` accepts only authors with GitHub type `User`.

The publication snapshot has only `mp1` in `publication.projects`; **there is no MP2 publication entry**, so no MP2 `published_at` or `review_ends_at` has been announced in this snapshot. Peer review is **not open**. The seven-day period starts at actual instructor publication; do not assume 1–7 November. MP2's `review_threshold` is `null`; no fixed discrepancy threshold or guaranteed reward is asserted. Reports require an actual reproduced score and target submission number, and use `dase7506/peer-review-v1`. Unavailable installation is not a measured zero score.

Recheck the live publication state when resuming and before submission. No course Issue or peer-review Issue was created in this review task.

## Workflow checks versus course requirements

No rule conflict was found between the supplied workflow and the current Guide/README. The workflow additionally requires grouped OOF/CV screening, detailed calibration/error analysis, internal AI review, a PR in the student's repository, clean-environment reproduction, and peer-review preparation. These are project engineering practices; the Guide does not mandate a particular search count, loss, model family, PR, or a fixed number of peer reviews. Do not present the workflow's suggested approximately 12 configurations as an official cap.
