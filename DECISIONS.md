# Decisions

- 2026-10-29: Adopt user-supplied docs/WORKFLOW.txt as execution specification; official latest rules prevail on course constraints.
- Work within the granted workspace, in an isolated subdirectory. Preserve unrelated files.
- Use CPU and Python 3.12.0. Inference scientific libraries will be limited to one thread initially.
- Only official training data may fit any parameters or transformations. Validation is selection/diagnosis only. Test references may be read only by integrity checker and, after freeze, independent scorer.
- No external model API, new pretrained weights, outside datasets, lookup answers, or ID-derived features.
- Report AI assistance honestly, including internal AI review; do not claim human understanding checks have occurred.
