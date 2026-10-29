# Official form adapter: independent AI review

Reviewed on 29 October 2026, approximately 14:51-14:54 Asia/Shanghai, by a separate Codex AI agent. This is an internal AI review, not human approval or instructor certification.

**Conclusion:** no blocking defect was found in the reviewed adapter for preparing this project's official payload. Preparation still needs to run successfully, and the actual GitHub Issue must subsequently be verified. This review did **not** generate a payload, call the official submit handler, create an Issue, or claim that a browser was clicked.

## Scope and exact reviewed bytes

| File | SHA-256 |
|---|---|
| `submission/prepare_official_form.mjs` | `ae632b1a3eb5e3b9874e1a7ba863ddb992ca6fb42f58f7336e26fc5fb6df0837` |
| `submission/fetch_form_sources.py` | `b7a6709d9cef26b12a9bf91735e7980cb70cb29ce70e03640ef856c9dcd66a9b` |
| `submission/FORM_SOURCE_SNAPSHOT.json` | `f9e1c7053bb6cd9d6e3c12a4debe1f3c5e9c514311e8904181fb8328d5ef148f` |
| `submission/PRE_TEST_FREEZE.json` | `2734218beef0a53d63015e7483d374772c9fcebe99c1226785f40cd0cf792899` |
| `results/test-score.json` | `5129103d977984a174e71a5d89697a9fee50a4dd53fff580c271e48e8ea913d7` |

The authored and actually served official modules, the HTML form and public encryption key were inspected. All nine snapshot files matched their recorded SHA-256 and byte count. No test question, candidate, prediction or reference was accessed; only the separately authorized aggregate score was read.

## Official source and current phase

The source snapshot fixes website commit `bbec4921a0646abe127588cb121b4aba48c8fd3b`. The actual HTML module entry is `courses/dase7506/assets/mp2.834d33d4aea3.mjs`, importing `arena.16ac78fce8b4.mjs`, which imports `projects.ae4b9dc4b02c.mjs`. The adapter imports this served graph, not a hand-written replacement for the payload schema. The snapshot's comparison with authored modules changes only relative import spelling; those comparisons all match.

An independent live GitHub read during review returned matching blob identifiers:

| Official live file | Git blob SHA (not a commit) |
|---|---|
| `mp2/index.html` | `d08606fef97c6ad27f16437811fd689c8dee86c3` |
| `assets/mp2.834d33d4aea3.mjs` | `77d3de5efc009be84804babcf7c01ac03db6df9b` |
| `assets/arena.16ac78fce8b4.mjs` | `d16c7cf6c905efa38960daf75b77aa30dbf38e69` |
| `assets/projects.ae4b9dc4b02c.mjs` | `bc0ea31136171598c902c7ad30c84636b162ead2` |
| `data/student-id-public.pem` | `d590a72630074fca223a76609df022389c7fd9fb` |

Sources: [live official HTML](https://github.com/xudongwu-0/xudongwu-0.github.io/blob/main/courses/dase7506/mp2/index.html), [published handler](https://github.com/xudongwu-0/xudongwu-0.github.io/blob/bbec4921a0646abe127588cb121b4aba48c8fd3b/courses/dase7506/assets/mp2.834d33d4aea3.mjs), [published validators](https://github.com/xudongwu-0/xudongwu-0.github.io/blob/bbec4921a0646abe127588cb121b4aba48c8fd3b/courses/dase7506/assets/arena.16ac78fce8b4.mjs), [public key](https://github.com/xudongwu-0/xudongwu-0.github.io/blob/bbec4921a0646abe127588cb121b4aba48c8fd3b/courses/dase7506/data/student-id-public.pem).

The pure official `submissionPhase` function was evaluated without invoking the form handler. At `2026-10-29T06:53:08.216Z`, it returned `scores`, with repository `xudongwu-0/xudongwu-0.github.io`, protocol `7506-submissions-v1`, key ID `7506-2026-v1`, and exclusive score cutoff `2026-11-01T00:00:00+08:00`. The captured board was generated at `2026-10-29T06:25:49.820Z` and has no MP2 publication entry. This snapshot is not a promise of future availability; the actual Issue creation must still check the time and account.

## Findings

1. **Payload provenance is preserved.** Adapter lines 99-116 hash the supplied official files and locate the published graph. Lines 176-217 import the unchanged page module, obtain its registered submit listener, and extract the Issue link it creates. The official `encryptArtifactLinks` generates AES-GCM key/IV, wraps the key with the official RSA-OAEP public key, and the official `makeIssueURL` formats the body. The adapter validates a round trip but writes the original extracted title/body, not a reserialized replacement. The JSON text and body hashes are preserved in the generated evidence.

2. **HTML constraints are covered for these inputs.** The current official score input is numeric, `min=0`, `max=100`, `step=any`; student ID is required and at most 64 characters; HTTPS links are at most 500 characters; consent is a required checkbox. The adapter's explicit required arguments, fixed student ID, official validators, immutable URL constraints and consent precondition cover these constraints for the intended invocation. It is a limited DOM adapter, not a general browser implementation, and fails on unknown selectors or unexpected network requests.

3. **Score and identity are consistent.** The inspected official aggregate is 348 correct out of 467, accuracy `74.51820128479658`, split `test`, protocol `dase7506-mp2-open-hard-verifier-v5`. The adapter checks the full-bank count against the official manifest and arithmetic consistency, then uses the exact score field. The supplied student ID is fixed to `3036900957`. A fresh GitHub profile call independently confirmed connected nickname `liuqichang0305`. This does not substitute for checking the eventual Issue's actual author and `User` type.

4. **Frozen source and assets are checked.** All 15 recorded source/asset files matched both working-tree bytes and the inspected HEAD `92c7240c5a4710054de1363d23e2f11f81d33b72`. The pre-test source commit is `8602992842d444524ab4148991da2f1dc88f316f`. Adapter lines 119-138 require code/checkpoint links to the same full commit in `liuqichang0305/dase7506-mp2-solution`, require that commit to descend from the freeze, and hash its frozen files. The checkpoint paths are fixed to `artifacts/final_bundle`. Configuration SHA-256 is `24473c3b95b9472b5d844b795edf86c7902d21fe7351604cc680d93d7ead0703`; weights SHA-256 is `a785f4a48e2d5708aec9fec29c776e9ce1b7fea1832d064e95b25c926626d462`. The code does not certify that the commit is publicly downloadable; the separate public-download evidence must cover the final links.

5. **The declaration is limited and disclosed.** The exact official text is “I understand and confirm that this is my result.” The explicit `--confirm-result` gate is required. Its recorded basis is the user's authorization to submit this user-owned coursework result with substantive AI help disclosed. It does not assert an unperformed separate human understanding exercise, manual code audit, or personal reproduction. The official declaration remains unchanged; neither this review nor the adapter claims to have measured the student's mental understanding.

6. **No browser or submission claim is fabricated.** `window.isSecureContext` is expressly emulated for local WebCrypto, fixed official snapshots supply the GET responses, and the output evidence says `prepared_not_submitted`, `publication_complete=false`, with an explicit local-adapter description. It never says a browser click occurred. The downloader retrieves public immutable sources without account credentials. The adapter itself makes no Issue request. Output files use exclusive creation and cannot silently replace an earlier encrypted payload.

7. **Tests performed and limits.** `node --check submission/prepare_official_form.mjs` returned exit 0. Source and freeze hashes were checked independently in Python. The pure official phase function was evaluated successfully. No end-to-end handler execution or encryption output was produced in this review, so runtime acceptance remains to be evidenced by the actual preparation run. The downloader's reviewed `--restore` mode verifies every file against the recorded hash and refuses different pre-existing bytes.

## Remaining checks at the authorized final action

- Preserve the exact generated Issue body and compare its SHA-256 immediately before sending; verify score, student ID, target repository, frozen links and public accessibility in the generated record.
- Recheck current UTC+8 time and actual connected GitHub identity at creation. Do not create a duplicate if an earlier attempt's Issue already exists.
- After creation, retrieve the Issue and verify actual author `liuqichang0305`, author type `User`, body equality, creation time and URL. Save that evidence before marking submitted; leaderboard synchronization may lag.

These are completion checks for the already authorized submission, not a new permission requirement. No code/model change or additional evaluation is recommended by this review.
