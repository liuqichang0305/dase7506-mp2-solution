# DASE7506 website maintenance

MP1 is archived at `/courses/dase7506/`. Its page reads only the fixed snapshot in `courses/dase7506/archive/mp1-2026-fall/`; submissions and review are closed. The archive includes scores, artifact links, review records, publication and identity metadata with SHA-256 checksums. The sync job verifies these checksums and copies MP1 records unchanged into the shared snapshot; issue edits, new reports and MP2 updates cannot change archived MP1 results. Do not rewrite archive files or MP1 publication entries.

MP2 is open at `/courses/dase7506/mp2/`, using self-reported full-test accuracy (percentage; higher is better). Its deadline is **31 October 2026, 23:59 Beijing time (UTC+8)**. The cutoff is implemented as `2026-11-01T00:00:00+08:00`. Early score-only entries are allowed; the final entry must include code and checkpoint/configuration links before the deadline. After cutoff, new scores are marked late, score edits are ignored and the form accepts only artifact supplements for an existing issue. The instructor opens a seven-day peer review by releasing MP2 links manually. Use `release-links.mjs --project mp2` with the existing private key, first with `--output` for a private preview, then with `--publish` after inspection. MP3 remains closed.

Both student pages are English-only, with no links to the personal homepage. MP2 materials are maintained in [DASE7506-MP2](https://github.com/xudongwu-0/DASE7506-MP2); no reference method is supplied. After changing browser sources, run `node scripts/dase7506/build-assets.mjs` and `node --test scripts/dase7506/*.test.mjs` before publishing.

MP2 links were released on **1 November 2026**. Peer review closes at **8 November 2026, 24:00 Beijing time**, stored as `2026-11-09T00:00:00+08:00`. The instructor-controlled `publication.projects.mp2.review_ends_at` is respected by the page, report validation and report statistics; older releases without an explicit end retain the seven-day default. During review, the score/link form is hidden and the leaderboard occupies the full width. Missing artifact links are shown explicitly. Numerical scores remain self-reported until reproduced.

## Storage and identity

The form creates a prefilled GitHub Issue. The student must log in to GitHub and
confirm creation. Generating the link alone is **not** a successful submission.
Student IDs and scores are displayed publicly. GitHub accounts remain the backend
identity for submissions and review requests, and are visible in GitHub records.
The current form encrypts artifact links with AES-256-GCM, wrapping each AES key
with the course RSA public key. Links are published together by the instructor.
Legacy test issues already containing plaintext links remain public records.
Student IDs are now written in plaintext issue payloads, with consent on the form.
They are not retained in browser localStorage. GitHub usernames are not displayed
in the student leaderboard's identity column.
The Student ID field also accepts result labels such as `[Baseline] GPT-128`:
1–64 characters, including spaces, punctuation and other visible Unicode symbols.
Leading/trailing whitespace is trimmed by the form; control characters are rejected.
Labels are displayed as text. Within each project, the leaderboard keeps the best
eligible score for each exact Student ID and GitHub account pair (account names
ignore case). One instructor account can therefore display several named baselines.
Repeating the same ID with the same account updates that entry only when the score
improves; tied scores show the submission with more available artifact links. All submissions remain in history. Unresolved legacy IDs retain account
grouping until resolved. The GitHub account determines artifact ownership.

The private key is kept outside this repository by the instructor. Back it up
privately; losing it prevents decryption of sealed links and legacy student IDs. To export a
private roster (Python 3.10+ and OpenSSL, outside this repository):

```bash
python3 scripts/dase7506/decrypt-ids.py \
  --key /private/path/submission-private.pem \
  --output /private/path/7506-identities.csv
```

Student IDs and GitHub accounts together determine leaderboard entries.
Enforce one account per student by comparing submitted IDs with the course roster before grading.
Possession of an account alone does not establish a student's enrolment. There
is no student roster, teacher key, server token or database in this repository.

To migrate legacy encrypted IDs after the instructor's decision to show IDs:

```bash
node scripts/dase7506/publish-student-ids.mjs --key /private/path/submission-private.pem
node scripts/dase7506/sync.mjs
```

Commit the updated `data/student-identities.json` and snapshot. Each migrated
ID is bound to its original ciphertext; the private key is never published.

## Automatic leaderboard

`.github/workflows/dase7506-leaderboard.yml` reads all issue pages on submissions,
edits and review-label changes, plus manual runs and a six-hour recovery schedule.
It validates structured JSON and commits `data/leaderboard.json` using the
repository-scoped `GITHUB_TOKEN`. It never downloads or executes student code.
The page first reads that snapshot from raw.githubusercontent.com. If that host
is unavailable, it tries the GitHub Contents API, then a deployed snapshot.
The primary path avoids GitHub's shared anonymous REST API rate limit.
Issue creation is immediate; leaderboard refresh may take several minutes.

Workflow commits made with `GITHUB_TOKEN` do not trigger a Pages rebuild. This is
intentional: the live page reads the updated raw snapshot directly. Changes to
HTML, CSS and scripts pushed normally continue to use the existing Pages build.
If Actions is disabled, enable it in the repository settings and run
**Actions → DASE7506 leaderboard → Run workflow**. The GitHub Issue remains the
source record even while a workflow or CDN update is delayed.

## Score deadline, link collection and manual release

Before 1 October 2026, 00:00 (UTC+8), a score submission needs only a public
student ID and BPB. Code and checkpoint links are optional. After that cutoff,
new scores are excluded and edits cannot replace the recorded deadline score.
If a score is first seen after the cutoff and its issue was edited after the
cutoff, it requires instructor review of the deadline evidence.

After the deadline the form requires both links and the original score issue
number. The GitHub account must match that original submission. Links remain
encrypted during collection. A “Links received” badge acknowledges the sealed
payload; it does not establish that the files or score are valid. Students must
preserve the method and checkpoint that produced their deadline score.

After collecting the links, the instructor runs this command locally with the
private key, outside the public repository, to inspect a private release preview:

```bash
node scripts/dase7506/release-links.mjs \
  --key /private/path/submission-private.pem \
  --output /private/path/7506-release-preview.json
```

The tool refreshes issue records, selects each Student ID/account pair's best eligible score and
requires two decryptable HTTPS links for every selected score. It refuses to
release before the deadline or replace an existing release. Resolve missing
links and reviewed/withdrawn entries before publishing. To open public review:

```bash
node scripts/dase7506/release-links.mjs \
  --key /private/path/submission-private.pem --publish
git add courses/dase7506/data/publication.json
git commit -m 'Open MP1 seven-day public review'
git push origin main
```

For an instructor-authorized release of all active submission history, including
records with absent links, add `--all-submissions --allow-incomplete`. Existing
Code and Checkpoint URLs are published independently; absent URLs display “Not
provided”. Withdrawn and invalidated submissions are excluded. Frozen scores,
review status and best-per-student ranking are preserved. Invalid URLs or
undecryptable payloads still stop publication.

Run the publication command immediately before committing and pushing: the
seven-day window starts at its recorded `published_at` time. This teacher action
releases the links together; the leaderboard displays the review deadline.
Only reproduction claims created within that seven-day window are eligible for
the peer-reproduction reward. General review requests can be filed earlier. Scores
remain frozen, and the published links remain visible after the window closes.

## Automated checking agent and score spot checks

`.github/workflows/dase7506-artifact-checks.yml` runs the file-checking agent when
links are released, and on manual dispatch. The agent downloads code at immutable
GitHub commits and checkpoint files, checks Python syntax and expected interfaces,
inspects checkpoint structure/protocol and records hashes and sizes. It selects
a random sample for instructor reproduction and uploads `artifact-checks.json`
under the workflow run's artifacts.

The agent never executes submitted Python or unpickles checkpoint objects. Its
output always distinguishes file inspection from numerical score reproduction
(`score_reproduced: false`). Inaccessible or unrecognized files require instructor
review; they do not automatically invalidate a score. Complete the selected score
spot checks in an isolated evaluation environment before applying `7506:verified`.

Malformed records are omitted from the board; the original issue remains for
correction. Student results are self-reported unless the instructor verifies them.
For an improved score, create a new issue instead of editing a reviewed result.
Closing an ordinary submission withdraws it; history still shows it. An invalid
submission stays invalid and recorded even if its author edits or closes it.

## Instructor review overview

Open `/courses/dase7506/instructor.html` directly to see one entry per
**project + exact Student ID + GitHub account** (account names ignore case).
Each entry shows its **latest submission**, even if its score is worse, withdrawn,
late or invalid; the latest score status remains visible. Creation time determines
recency, with issue number breaking ties or handling missing timestamps. An edit
to an older issue does not make it the latest submission. Earlier submissions
remain available in expandable history and their reports accumulate in the entry.
Legacy records without a resolved Student ID remain separate until resolved.
The student page has no link to this overview and displays neither counts nor
flags. This is a separate view, not an access-controlled backend: its source
issues and snapshot can still be inspected through GitHub.

“Total reports” includes duplicate, closed, rejected, late and self reports.
“Active reporters” counts distinct GitHub accounts across **all submissions in
the entry**, with pending or upheld reports, excluding self, closed, rejected,
late and scoreless reports. Repeat reports by the same account on different
submissions count once. **More than 3** active reporters means **4 or more** and
sets a flag. If the **same exact Student ID is used by multiple GitHub accounts**,
all related entries across projects are flagged, including entries with no reports.
The flag column states each reason and lists conflicting accounts. The report
selector includes the entry's entire history and retains each original target
issue number. Re-submitting cannot reset an entry's accumulated report counts.

These rules apply to both the online instructor page and the local HTML export.
Raw submission records and the student leaderboard's best-score rule are unchanged.
Flags do not invalidate a score, remove it from ranking, apply an instructor label
or change grades.

The **Peer Review Report** form requires the reproduced score. The Markdown
report and evidence link are optional. Current submissions use
`dase7506/peer-review-v1`, which validates a numeric score even when both optional
fields are omitted. Reports during the seven-day review enter the existing
instructor adjudication flow; earlier reports remain inspection leads. Historical
requests without scores remain visible in the overview as `score-missing` and do
not contribute to active counts. Reports alone do not affect grades.

Students may also email **wu.xudong@connect.hku.hk**. The email link checks the
required score and opens a draft containing the submission number, reproduced
score and any optional details. The student must send the email. Email reports
are handled directly by the instructor and are not automatically imported into
the GitHub counts. GitHub reports and their author accounts remain public.

For a local instructor copy, including all request records:

```bash
node scripts/dase7506/review-summary.mjs --refresh \
  --output /private/path/review-overview.html
```

## Reproduction and adjudication

The workflow creates these five instructor labels automatically. Students cannot
assign them. Check both parties' evidence and explain the decision in the issue
before applying the corresponding label:

| Label | Where | Effect |
|---|---|---|
| `7506:verified` | Submission | Exact submitted checkpoint reproduced |
| `7506:review` | Submission | Excluded from main ranking while reviewed |
| `7506:invalid` | Submission | Invalidated after instructor review |
| `7506:upheld` | Reproduction issue | Instructor confirms the report |
| `7506:rejected` | Reproduction issue | No penalty or reward |

A pending report has no grading effect. An invalid submission **and** an upheld
report are both required for an adjustment. If a numeric difference threshold is
configured, the report must exceed it; otherwise the instructor judges the evidence.
The first upheld reporter and submission author receive the adjustments configured for that project. Each reward and
penalty is capped independently per student/project. Repeated syncs and duplicate
reports cannot multiply them. The snapshot records adjustments separately; apply them
to project marks with a final 0–100 clamp. A withdrawn result does not imply
misconduct. If a decision is reversed, remove the corresponding instructor label.

Edits of a verified payload return it to review. To verify the new payload, remove
`7506:verified`, let the snapshot sync, then add it again. Apply only one submission
review label at a time. Adjudicated submissions/reports retain the reviewed payload
so their authors cannot redirect rewards or erase penalties by editing JSON.

## Checks

```bash
node scripts/dase7506/build-assets.mjs
node --test scripts/dase7506/*.test.mjs
python3 -m unittest discover -s scripts/dase7506 -p test_artifact_agent.py -v
node scripts/dase7506/sync.mjs
```

The sync command uses optional `GITHUB_TOKEN` for API rate limits; never commit it.
The workflow only processes JSON data and instructor labels, not issue-body shell
expressions or code from forks. GitHub Issues / Actions must remain enabled.

Run `build-assets.mjs` after changing browser code, project settings or CSS, and
commit the generated `assets/` files together with `index.html` and `instructor.html`. Each module
references fingerprinted dependencies, including the project configuration.
This prevents a new English page from loading an old cached Chinese script or
closed-project configuration. Asset tests check that the published graph matches
the source. Older fingerprinted files remain available for cached HTML.

Implementation references: [GitHub Pages static hosting](https://docs.github.com/en/pages/getting-started-with-github-pages/what-is-github-pages),
[prefilled issue URLs](https://docs.github.com/en/issues/tracking-your-work-with-issues/using-issues/creating-an-issue#creating-an-issue-from-a-url-query),
[issue events](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#issues),
[GITHUB_TOKEN](https://docs.github.com/en/actions/tutorials/authenticate-with-github_token).

## Open projects after the instructor has confirmed them

Edit `courses/dase7506/projects.mjs`: set the project name, score unit, direction
(`lower`), optional `min` / `max`, reproduction threshold and any reward / penalty.
MP1 currently has `open: true` for instructor submission testing. Set other projects to `open: true` only after instructor confirmation.
The same configuration gates both the browser form and the trusted snapshot
builder. Closed projects reject manually crafted GitHub submissions too.

Publish the confirmed task instructions separately at that time. For a new course
cohort or incompatible evaluation protocol, change `PROTOCOL` and archive the old
leaderboard before accepting results; check for existing records before changing the protocol. A test submission can be withdrawn by closing its GitHub issue; it remains in history.

Run the rule tests before pushing. Their synthetic score ranges and reward values
exist only in the test process and are not published course policies.

## Public link warnings and instructor-entered results

`publication.json` may contain `link_checks`, indexed by the exact artifact URL.
An entry with `status: "unavailable"`, a reason and a check timestamp displays
**Unable to open** beside that link. Browser-confirmed access and automated-request
rate limits do not produce this warning. Replacing the URL removes the old warning.

The instructor can supplement a published entry with student-provided links
without changing its score or restarting the public review window. To record an
existing malformed issue, the trusted publication entry must match its issue number
and actual GitHub author and include `student_id`, `score`, both links,
`instructor_accepted: true` and `instructor_recorded_score: true`. This explicit
instructor entry supports leaderboard and peer review while retaining the source
issue and account. It remains **Self-reported**, not Verified. A student cannot
accept a late score by including these fields in an issue payload.

The instructor may mark a score issue in `publication.json` under
`superseded_submissions.mp1` when a student identifies a later result as final.
The earlier score remains visible in the All submissions history but is omitted
from the default ranking; the later self-reported score is not verified by this
change.
