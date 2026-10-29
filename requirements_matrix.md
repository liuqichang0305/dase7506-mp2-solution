# Requirements matrix

Source commit: bb801f4f2e7629d270df765ab12c333b2dc616bb. No difference from workflow's recorded upstream commit. Full official files independently reviewed; website snapshots/actual form assets recorded separately. Checked 2026-10-29 UTC+8.

| Class | Requirement | Source | Implementation / acceptance | Status |
|---|---|---|---|---|
| Course | Fit only official train; validation for selection; test final only | guide/GUIDE.md sec 3 | Split guards, provenance, submission/PRE_TEST_FREEZE.json | Passed internal AI audit; test after freeze only |
| Course | Keep official data/parser/scorer unchanged | Guide sec 3 | results/official-hashes.json and original checker logs | Passed; CRLF acquisition issue resolved from original git bytes |
| Course | Offline selection, 4 candidates, arbitrary new bank/IDs | Guide sec 3; code/submission.py | solution.py; results/contract-validation.json and contract-synthetic.json | Passed isolated synthetic/validation checks |
| Course | Total CPU prediction + scoring <=30s; <=4 threads | Guide sec 3 | results/test-timing.json and contract threadpool data | Local 0.705959s, one OpenBLAS thread; reference-machine eligibility unverified |
| Course | Additional assets <=16 MiB uncompressed | Guide sec 3 | results/asset-audit.json | Passed: 51,923 bytes, including calibration/config/container |
| Course | CPU or <=RTX4090 single GPU training | Guide sec 3 | results/environment.json and development logs | CPU only |
| Course | Final score from complete official test | Guide sec 2 | results/test-score.json; freeze source 8602992842d444524ab4148991da2f1dc88f316f | 348/467, 74.51820128479658% |
| Course | Report <=10 pages; initial/improved and ablation evidence | Guide sec 2,4 | reports/report.pdf, VISUAL_QA.md and development results | 7 pages; all rendered/AI visually checked |
| Course | Public code and matching checkpoint before deadline | Guide sec 4 | Frozen commit URLs and results/reproduction/public-downloads.json | Public prepared commit 92c7240c5a4710054de1363d23e2f11f81d33b72 verified; final submission links pending |
| Course | Finish generated GitHub submission issue | Guide sec 4 | Actual issue author/time/payload evidence | NOT SUBMITTED |
| Course | Disclose AI and reuse; retain licenses | Guide sec 3, attribution | docs/AI_DISCLOSURE.md, code/data/LICENSE.txt and NOTICE.txt | Complete; human understanding checks not claimed |
| Course + workflow | Python 3.12, isolated environment | code/README.md sec 1; workflow stage 0 | .venv Python version | Python 3.12.0 created and dependencies installed |
| Workflow | Train-only calibration, reliability and risk curves | Workflow stage 6 | Train OOF + results/reliability | Complete: validation Brier 0.067713, 10-bin ECE 0.021821 |
| Workflow | Internal review and clean-environment reproduction | Workflow stage 10 | docs/CODE_REVIEW.md, REPRODUCTION.md | Passed: anonymous fixed-commit clone, new venv, no retraining, identical 348/467 and all choices; 1.1398808s local |
| Workflow | Peer-review preparation and future recovery | Workflow stage 12 | Checklist, template, resume commands | Preparation complete; actual review awaits instructor release |
