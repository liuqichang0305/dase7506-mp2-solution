# Requirements matrix

Source commit: bb801f4f2e7629d270df765ab12c333b2dc616bb. No difference from workflow's recorded upstream commit. Rows pending independent review of full official files.

| Class | Requirement | Source | Implementation / acceptance | Status |
|---|---|---|---|---|
| Course | Fit only official train; validation for selection; test final only | guide/GUIDE.md sec 3 | Split guards, provenance, freeze manifest | Enforced in work allocation; implementation pending |
| Course | Keep official data/parser/scorer unchanged | Guide sec 3 | Original SHA-256 manifest and recheck | Download pending |
| Course | Offline selection, 4 candidates, arbitrary new bank/IDs | Guide sec 3; code/submission.py | solution.py and interface tests | Pending |
| Course | Total CPU prediction + scoring <=30s; <=4 threads | Guide sec 3 | Fresh sequential process timing, threadpool inspection | Pending; local host differs from reference |
| Course | Additional assets <=16 MiB uncompressed | Guide sec 3 | Recursive asset/hash audit | Pending |
| Course | CPU or <=RTX4090 single GPU training | Guide sec 3 | CPU-only training logs | CPU selected |
| Course | Final score from complete official test | Guide sec 2 | Freeze then official scorer JSON | NOT RUN |
| Course | Report <=10 pages; initial/improved and ablation evidence | Guide sec 2,4 | Generated PDF, page/render inspection | Pending |
| Course | Public code and matching checkpoint before deadline | Guide sec 4 | Frozen commit URLs and anonymous download | Pending |
| Course | Finish generated GitHub submission issue | Guide sec 4 | Actual issue author/time/payload evidence | NOT SUBMITTED |
| Course | Disclose AI and reuse; retain licenses | Guide sec 3, attribution | AI_DISCLOSURE.md, licenses | Pending |
| Workflow | Python 3.12 isolated environment | docs/WORKFLOW.txt stage 0 | .venv Python version | Created |
| Workflow | Train-only calibration, reliability and risk curves | Workflow stage 6 | Train OOF/holdout + validation analysis | Pending |
| Workflow | Internal review and clean-environment reproduction | Workflow stage 10 | CODE_REVIEW.md, REPRODUCTION.md | Pending |
| Workflow | Peer-review preparation and future recovery | Workflow stage 12 | Checklist, template, resume commands | In progress; actual review awaits release |
