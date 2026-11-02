# Anonymous final artifact verification

Implementation status: synthetic-tested only. No actual HTTP request, course
score/holdout read, GitHub issue request or formal submission was performed by
this implementation task.

Once the real `FREEZE.json`, `TEST_COMPLETE.json`, and `GENERATED_FORM.json`
exist, and the final immutable code/assets/report commit is publicly available,
run from repository root:

```powershell
.venv/Scripts/python.exe submission/v3/verify_publication.py --attempt public-final-01
```

The tool takes the full commit and both artifact URLs from `GENERATED_FORM.json`.
There is no repository, URL, token, model, score, split or destination override.
The only repository is `liuqichang0305/dase7506-mp2-solution`; the checkpoint is
`artifacts/research_v3/final_selected`, and the report is exactly
`reports/research_v3/report_v3.pdf`.

Before HTTP, it checks the typed freeze/test-completion/form linkage, current
source/asset hashes, Git ancestry, and source/asset bytes at both the freeze and
final publication commits. It checks the report against its committed Git blob
and measures one to ten pages using `pypdf.PdfReader(strict=True)`; encrypted
reports are rejected. It inspects the test-completion record and score-file
hash reference only. It does not open `test-score.json`, raw course data, or
predictions, and does not run evaluation.

Every frozen source and actual selected asset receives a separate anonymous
HTTPS GET from `raw.githubusercontent.com` at the exact final commit. Path
segments are URL-encoded. The transport has no authentication/cookie handler,
credential helper, inherited proxy, redirect following, token or issue API.
HTTP 200, the unchanged final URL, byte count and SHA-256 must all match the
local and committed artifact. Download size is bounded by the expected bytes
plus one byte. The downloaded report also receives its own real PDF page count.

All request outcomes and downloaded bytes are retained under
`submission/v3/publication-attempts/<attempt>`, including errors. Failure writes
`FAILED.json` with `passed: false`; it cannot create the final passing record.
After a transient timeout, 404, or public propagation delay, explicitly repeat
the read-only check under a new name:

```powershell
.venv/Scripts/python.exe submission/v3/verify_publication.py --attempt public-final-02
```

Old attempts cannot be overwritten. An existing final
`PUBLICATION_VERIFICATION.json` also cannot be overwritten or silently replaced.
No automatic upload, commit, publish, score update or issue creation occurs.

A success creates the exact schema consumed by `submit_final.py`:
`mp2-v3-publication-verification-v1`, `passed: true`, `verified_at_utc`, immutable
commit/URLs, full frozen source and asset maps, freeze SHA-256,
`anonymous_file_checks`, and `report_check`. Each check records `path`, encoded
`url`, `http_status`, `sha256`, `bytes`, anonymous-request metadata and cached
download path; the report adds `pages`. The final record also binds its input
form/completion hashes and all attempt evidence hashes. No remote access claim
is manufactured from local Git availability alone.

Synthetic verification: `code/tests/test_verify_publication_v3.py`, 12/12 tests
passed in 1.485 seconds (exit zero). Tests use mocked HTTP only and temporary
synthetic repositories/files; real pypdf parsed synthetic two-page and eleven-page
PDFs. They cover exact typed output, forbidden destinations, no auth headers,
encoded Unicode/spaces, corrupted bytes, redirected URLs, HTTP 404/timeouts,
Git ancestry/blob command coverage, retry evidence, no overwrite and an explicit
guard proving the score path is never opened. Log:
`results/research_v3/publication-verification-synthetic-tests.log`.
