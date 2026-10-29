# Peer-review preparation checklist

Status on 29 October 2026: preparation only. MP2 links have not been published in the checked official snapshot. No peer reproduction or report has been completed. This is separate from internal AI code review.

- [ ] Read `submission/SUBMISSION_RECORD.md` and confirm our own actual course Issue, account `liuqichang0305`, and student ID `3036900957`. Missing record means submission is unconfirmed.
- [ ] Refresh the [official MP2 page](https://xudongwu-0.github.io/courses/dase7506/mp2/) and its [publication snapshot](https://raw.githubusercontent.com/xudongwu-0/xudongwu-0.github.io/main/courses/dase7506/data/leaderboard.json). Record snapshot generation time, `publication.projects.mp2.published_at`, and `review_ends_at`.
- [ ] Verify that the current UTC+8 instant is in the instructor's actual review window. Missing publication means wait, not complete.
- [ ] Choose another student's eligible frozen submission. Record its Issue number, claimed score, full code commit, checkpoint URL, and downloaded hashes. Do not use a newer main branch or retrain a substitute.
- [ ] Human-review third-party source, installation instructions, dependencies, serialized assets, and network/file/process behavior. Record the reviewed manifest; no unsafe deserialization or unreviewed installation.
- [ ] Provision a disposable sandbox or separate account/container with no personal files, GitHub token, SSH keys, or other credentials. Disable network for evaluation. A virtual environment alone does not isolate file or credential access.
- [ ] Install only reviewed dependencies in that isolated environment. Preserve setup commands, full stdout/stderr, exit codes, package versions, OS, CPU, RAM, and Python version.
- [ ] Verify official evaluator/data hashes against the recorded course release. Use the exact submitted method/checkpoint, not a rewritten scorer.
- [ ] Run `peer_review/run_reproduction.py` first in its default prepare-only mode. Inspect the generated commands and manifests. Execution requires separate human-review evidence and an explicit execution flag inside the isolation boundary.
- [ ] Run one complete fresh prediction process followed by independent scoring. Keep every result and failure; do not cherry-pick. Enforce at most four threads in total and measure full wall time. Record uncompressed inference-asset size.
- [ ] Compare the scorer's actual percentage and denominator with the claim; distinguish rounding/numerical differences, environment problems, and substantive mismatches. An install failure supplies no reproduced score.
- [ ] Complete `peer_review/report_template.md` with facts, evidence and unresolved uncertainty. Do not infer misconduct from failure alone.
- [ ] Only with a valid reproduced score and an open review window, use the official **Peer Review Report** form. Preserve its generated JSON, create the actual Issue with the personal account, and save the Issue URL/author/time. Do not send email unless separately requested.
- [ ] Mark review completed only when real reproduction and the actual report Issue are evidenced. No scheduling task has been established by this preparation.

The workflow requests at least one real peer reproduction when available. This is not a claim that the course mandates a review count or guarantees a reward.
