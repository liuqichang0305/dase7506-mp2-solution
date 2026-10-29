# Decisions

- 2026-10-30 V3: Adopt the new explicit autonomous-research authorization. Preserve
  all V1/V2 frozen methods and evidence. New research cannot consume test feedback.
- V3 budget is absolute and persisted, not renewed by a restart. All main methods
  use the same nested three folds. Deploy the predetermined seed; extra seeds
  measure stability. Reuse verified historical OOF rather than fabricate reruns.
- V3 exact replacement rule: resource-compliant new finalist hard>=297/355 and
  full>=740/798, at least one strict increase; otherwise preserve historical
  Deep Sets. Finalist lock closes search; one final test cannot reopen it.
- V3 scheduler review found three recovery/identity defects before any full run:
  changed configuration identity, orphan trainer duplication, stale source result
  reuse. Fixes and both failing/passing test evidence are retained.

- 2026-10-29: Adopt user-supplied docs/WORKFLOW.txt as execution specification; official latest rules prevail on course constraints.
- Work within the granted workspace, in an isolated subdirectory. Preserve unrelated files.
- Use CPU and Python 3.12.0. Inference scientific libraries will be limited to one thread initially.
- Only official training data may fit any parameters or transformations. Validation is selection/diagnosis only. Test references may be read only by integrity checker and, after freeze, independent scorer.
- No external model API, new pretrained weights, outside datasets, lookup answers, or ID-derived features.
- Report AI assistance honestly, including internal AI review; do not claim human understanding checks have occurred.
- Windows global core.autocrlf=true changed official checkout bytes. Recovered exact raw git-object bytes, retained failure evidence, and passed unchanged official checker. No manifest/data edits to weaken validation.
- Finite search finished; relation-listwise L2=.1 selected by full validation (724/798), above majority (688) and embedding-BCE (715). Paired CI vs embedding crosses zero. Relation-BCE was worse; listwise L2=.01 inner fit reached 200-iteration limit. Do not conceal these negative outcomes.
- Freeze uses commit 8602992842d444524ab4148991da2f1dc88f316f and original model/config hashes. Initial freeze attempt rejected CRLF differences in config/dependency files before any test; restored committed LF bytes without semantic changes and then froze successfully.
- Final test was 348/467. Algorithm/weights/thresholds will not change in response. Only publication, documents, and identical frozen-method reproduction follow.
