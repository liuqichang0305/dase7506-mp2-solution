# Final V3 submission transport

`submission/v3/submit_final.py` prepares no model, reads no test questions or
references, and is not imported by research. Importing it performs no I/O. Its
only live CLI action is:

```powershell
.venv\Scripts\python.exe submission/v3/submit_final.py --submit
```

The root agent runs this only after the unique final freeze, the authorized
official full-test scoring, public artifact verification, and the latest bounded
public-rule check. The user's authorization covers the final course issue;
there is no additional teacher-email approval requirement. Development and
testing of this transport use generated fixtures and fake API responses only.

The existing V1 transporter and records remain unchanged. The only permitted
POST endpoint is `/repos/xudongwu-0/xudongwu-0.github.io/issues`. The existing Git
credential helper supplies authentication in memory, with interactive prompts
disabled. `/user` must report personal account `liuqichang0305`, type `User`.
Credentials and remote error bodies are never printed or persisted. HTTP
redirects are rejected. Only the fixed course issue collection, its individual
issues, and `/user` are available to the transport.

## Required immutable local inputs

All form files are in `submission/v3`: `GENERATED_FORM.json`, `issue-title.txt`,
`issue-body.txt`, `issue-url.txt`, `FREEZE.json`, `test-score.json`, and
`FORM_SOURCE_SNAPSHOT.json`. Generated title/body bytes and hashes must match
exactly. The embedded payload, generated URL, student identity, aggregate score
hash/count/percentage, frozen source commit, local source/asset hashes and
published-commit bytes are checked. The script uses only the aggregate score
JSON; no score information is sent into a research process.

The official scoring protocol is `dase7506-mp2-open-hard-verifier-v5`; the form's
submission protocol is separately validated by the current official module
(currently `7506-submissions-v1`). They are intentionally not compared for
equality. Before POST, the unchanged official `validateSubmission`,
`validSealedArtifacts` and `makeIssueURL` must accept the exact payload and
reproduce the exact generated URL.

The V3 local adapter additionally observes, without modifying, the official
WebCrypto encryption calls. `ARTIFACT_BINDING.json` and the matching embedded
`GENERATED_FORM.artifact_binding` bind the exact public-link plaintext, IV,
ciphertext and wrapped-key output hashes to the adapter, observer helper and
official entry-source hashes. The submitter checks plaintext links against the
publicly verified immutable URLs and hashes the decoded envelope bytes. No AES
key or RSA input key material is inspected, copied or saved by the observer.
This V3 observer is a substantive reviewed local-adapter addition; the earlier
path-only comparison is retained as historical evidence, and V1 is unchanged.

`TEST_COMPLETE.json` must record a passing, resource-eligible once-only evaluation
of the same freeze, method/checkpoint and all 467 examples. Every listed output
hash is rechecked, including the exact aggregate score being submitted. See
`FINAL_FREEZE_AND_TEST.md` for the evaluation workflow.

`results/research_v3/SEARCH_CLOSED.json` must exist. The freeze requires
`frozen: true`, `test_evaluated_before_freeze: false`, `created_at_utc`, full
40-character `source_commit`, and nonempty distinct `source_files` and `assets`
SHA256 maps. Every asset must be under `artifacts/research_v3/final_selected/`.
The final publication commit must descend from the freeze commit. Its frozen
source and asset bytes must match the local freeze. The source files must also
match the freeze source commit itself.

The code URL is exactly
`https://github.com/liuqichang0305/dase7506-mp2-solution/tree/FULL_COMMIT`; the
checkpoint URL appends `/artifacts/research_v3/final_selected`. The official
module snapshot is fetched after freeze, matched to the generated form, and
rehashes each local module/source before its phase function is used.

## Public verification schema

`submission/v3/PUBLICATION_VERIFICATION.json` has this structure:

```json
{
  "schema": "mp2-v3-publication-verification-v1",
  "passed": true,
  "verified_at_utc": "ISO-8601 time after freeze",
  "source_commit": "freeze source commit",
  "final_code_commit": "same full commit as GENERATED_FORM.json",
  "code_url": "same immutable code URL as GENERATED_FORM.json",
  "checkpoint_url": "same immutable asset URL as GENERATED_FORM.json",
  "freeze_file_sha256": "SHA256 of FREEZE.json",
  "source_files": {"code/example.py": "SHA256; exact FREEZE source map"},
  "assets": {"artifacts/research_v3/final_selected/config.json": "SHA256; exact FREEZE asset map"},
  "anonymous_file_checks": [
    {"path": "repository-relative frozen file", "sha256": "SHA256", "url": "https://raw.githubusercontent.com/liuqichang0305/dase7506-mp2-solution/FULL_COMMIT/repository-relative-file", "http_status": 200, "bytes": 123}
  ],
  "report_check": {"path": "reports/report-v3.pdf", "url": "https://raw.githubusercontent.com/liuqichang0305/dase7506-mp2-solution/FULL_COMMIT/reports/report-v3.pdf", "sha256": "SHA256", "http_status": 200, "bytes": 123, "pages": 7}
}
```

The anonymous checks must cover every frozen source and asset exactly once, at
the same publication commit. The report is separate from method-frozen sources,
must have 1–10 pages, and must have a public raw URL at that same commit. The
producer of this record performs the actual anonymous downloads/page check;
the transport validates the record against local bytes and fixed URLs. It does
not claim to perform another anonymous clone or reproduction itself.

## Final bounded rule check

`submission/v3/RULES_RECHECK.json` is required before a new POST:

```json
{
  "checked_at_utc": "ISO-8601 time",
  "retrieved_at_utc": "earliest retrieval time in this check",
  "no_conflicting_notice_found": true,
  "phase": "scores",
  "deadline_utc": "2026-10-31T16:00:00Z",
  "official_site_commit": "same official commit as GENERATED_FORM.json",
  "official_source_snapshot_sha256": "same snapshot hash as GENERATED_FORM.json",
  "evidence_sources": [
    {"url": "https://xudongwu-0.github.io/courses/dase7506/mp2/", "retrieved_at_utc": "ISO-8601 retrieval time"}
  ]
}
```

The retrievals must be after freeze and within one hour of the POST. This says
only that no conflicting notice was found in the named public sources; it is
not a claim about private or inaccessible notifications. If sources changed,
the root agent reviews the change and explicitly preserves/replaces earlier
form evidence; the transporter will not silently rewrite it.

Immediately before POST the script revalidates all local inputs, runs the
unchanged hashed official arena `submissionPhase` with the current time and
captured publication metadata, and verifies the official deadline/repository.
It checks the hard UTC deadline again just before sending. Existing issues can
still be read back after that deadline, but a new issue cannot be created.

## Duplicate prevention and recovery

A process-identity lock prevents concurrent submissions. Stale locks are renamed
and preserved. The script validates an existing local success receipt and reads
back that issue, without POST. Otherwise it queries all course issues by the
verified creator using `state=all`, 100 per page. An exact title/body-hash match
is read back and saved as the existing submission. Pull requests, wrong authors,
different bodies and post-deadline creations are rejected.

Before a new POST, a unique `submission-attempts/TIMESTAMP-ID/before-post.json`
is flushed to disk. The exact generated title/body are then sent. A successful
response must be followed by a matching GET readback before
`SUBMISSION_SUCCESS.json` is written. The record binds generated-form, body,
freeze, public-verification, score and readback hashes, plus the issue identity.

After any ambiguous POST/readback error, the attempt evidence is preserved. A
retry first performs the identity/list/readback sequence. If the exact remote
issue is still absent, it records the unresolved attempt and refuses another
POST. This conservative behavior avoids duplicate issues when creation succeeded
but the response or readback was lost. Reconciliation requires actual evidence;
do not delete an unresolved attempt merely to bypass the gate. All previous
attempts, successful receipts and original V1 records are retained.
