# Evidence-bound official Peer Review Report

This pipeline is separate from the project's score submission and artifact supplement. It cannot update our MP2 score or replace the assets submitted with Issue 294. It prepares a public peer report about exactly one independently selected submission, using that peer's actual completed checkpoint reproduction.

The live site's main commit was checked on 2026-11-02 at 07:12:56 UTC and remains `65c16a3eef424e30b254ff58804dffe0fbda8793`. The actual published graph is:

`courses/dase7506/mp2/index.html` → `assets/mp2.9a439c7af66b.mjs` → `assets/arena.4f1cb17a638d.mjs` → `assets/projects.ae4b9dc4b02c.mjs`.

The four exact Git blobs, SHA256 hashes and bytes are in `official_form/65c16a3eef424e30b254ff58804dffe0fbda8793/FORM_SOURCE_SNAPSHOT.json`. The offline adapter executes these unchanged files, clicks the official target-row report button, fills the visible reproduced-score/Markdown/evidence fields, invokes the actual `challenge-form` submit handler and captures its generated issue URL. It supplies only the selected peer row and publication dates through a local fetch substitute; it downloads no leaderboard. Timers are suppressed and all global descriptors are restored. Source graph or input changes cause refusal.

The official payload is `dase7506/peer-review-v1`, protocol `7506-submissions-v1`, project `mp2`, with target `submission`, required `reproduced_score`, trimmed optional `reason` (at most 2,000 characters) and optional HTTPS `evidence_url`. The report author is the authenticated personal GitHub account; no fabricated student-ID or author field is added. The official threshold is currently null, so a matching reproduction can be reported. The official destination is `xudongwu-0/xudongwu-0.github.io`, with title/body produced by its original formatter.

Preparation requires `peer_review/submission_<number>/target.json` and an actual `COMPLETE.json`. Completion must bind the target hash and peer commit, successful prediction/scoring, protocol/split/full 467-question count, original official score and timing filenames, and the complete saved output hash mapping. Score arithmetic is checked against the original scorer's `correct/n`. Resource eligibility is reported by the peer reproduction evidence and is not fabricated or substituted for numerical success. The pipeline does not rerun prediction or scoring. Source/assets, predictions and logs are hash-bound without decoding peer per-question predictions.

Create a concise, factual Markdown report inside that peer directory after reading its actual completion evidence. For the selected peer directory:

```powershell
.venv/Scripts/python.exe peer_review/tools/peer_report.py prepare --peer peer_review/submission_240 --reason-file peer_review/submission_240/report.md
.venv/Scripts/python.exe peer_review/tools/peer_report.py check --peer peer_review/submission_240
```

An optional `--evidence-url https://...` can point to a published reproduction record. The preparation directory `report_form/` contains exact issue text, URL, generated payload, input hashes, adapter logs and an execution receipt. Existing or partial output is never silently overwritten. Preparation/check are offline and do not access credentials.

Before any publication, the coordinator must inspect the actual reproduction and generated public text, then create `report_form/REVIEW_APPROVAL.json` with these exact bindings:

```json
{
  "schema": "mp2-peer-review-approval-v1",
  "approved": true,
  "reviewer": "root",
  "approved_at_utc": "<actual review time after preparation>",
  "generated_form_sha256": "<GENERATED_FORM.json hash>",
  "report_input_sha256": "<FORM_INPUT.json hash>",
  "complete_sha256": "<peer COMPLETE.json hash>",
  "target_sha256": "<peer target.json hash>",
  "tool_sources": {
    "peer_review/tools/peer_report.py": "<current hash>",
    "peer_review/tools/prepare_official_peer_form.mjs": "<current hash>"
  }
}
```

This is the specifically required coordinator review record, not a claim of manual student verification or instructor approval. The implementation does not create it automatically.

```powershell
# Read-only remote reconciliation, including after the review window:
.venv/Scripts/python.exe peer_review/tools/peer_report.py reconcile --peer peer_review/submission_240
# Only after actual reproduction and coordinator approval:
.venv/Scripts/python.exe peer_review/tools/peer_report.py publish --peer peer_review/submission_240
```

The transport verifies the existing personal account `liuqichang0305` and only reads that account's course issues for duplicate detection. An existing different report for the same peer blocks a new issue. Immediately before a new POST it rechecks the four current official source blobs, publication dates and the one selected published peer entry. Review must still be open. Post-release score/artifact closure is irrelevant to this distinct authorized peer-review endpoint, whose own window ends 2026-11-09 00:00 UTC+8.

A durable pre-POST record is written before the sole request. Successful response and independent GET readback must exactly match the official title/body and personal identity. Ambiguous errors preserve the attempt; a rerun only reconciles an exact remote report and never blindly posts again. Existing success is verified by local hashes plus remote readback. A process lock prevents concurrent publication; an abandoned lock is preserved for explicit inspection. Tokens are obtained from existing Git credentials only at remote-operation time, never printed or written, and redirects/out-of-scope endpoints are rejected.

Initial 13 synthetic cases passed in 8.870 seconds; a stricter complete-preparation receipt check added one case. The captured final 14/14 result is `audit/report-pipeline-synthetic/RESULT.json` with raw logs and unchanged-source hashes. Tests use temporary generated scores and fake HTTP; the published JavaScript handler itself runs offline. Neither these tests nor this document establish that a real report was prepared, posted or numerically reproduced. Actual peer reproduction belongs to its own evidence directory and independent review remains required.
