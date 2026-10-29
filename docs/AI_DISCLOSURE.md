# AI and reuse disclosure

Work date: 29 October 2026, Asia/Shanghai. Tool: Codex in the user's Windows workspace, with collaborating AI agents, terminal execution, GitHub connector, and primary-source web retrieval.

Codex provided substantive assistance in reading course rules and literature, designing and implementing the lightweight selector/training/analysis tools, running the recorded experiments, preparing asset and interface checks, writing documentation, and generating the report. AI agents divided implementation, source review, internal code review, and documentation tasks. Their work is **AI internal review**, not human peer review, teacher certification, or another GitHub account's approval.

The project uses supplied whole-trace embeddings, training inputs and training references. No external answer teacher, model API, newly generated reasoning trace, external dataset, additional pretrained weight, or test-based parameter selection was used in the development recorded here. Validation examples are used for explanation and diagnosis, not to generate additional fitting labels. The report generator only accepts a separately supplied aggregate final score after freezing.

Official `common.py`, `predict.py`, `score_predictions.py`, `check_data.py`, data and attribution/license files are reused from `xudongwu-0/DASE7506-MP2` at commit `bb801f4f2e7629d270df765ab12c333b2dc616bb`; the project retains their original bytes and checks hashes. Data attribution to GSM8K and candidate/representation attribution to Qwen3-0.6B follow the official NOTICE. Published verifier, self-consistency and calibration ideas are cited in the report and `references.bib`; their large models or training data were not imported.

NumPy provides numerical inference, SciPy provides training optimization, matplotlib creates diagnostic figures, and ReportLab creates the PDF. Exact installed/pinned versions and actual resource records are kept in the project. Original project implementation and prose were AI-assisted; the student must review and be able to explain the submitted work.

**Human understanding status:** this record does not claim that the student has already checked every line, understood every formula, manually certified reasoning examples, or reproduced the project personally. The Chinese guide supplies questions for that review. Actual execution/check results are documented separately; no unperformed human check is implied.

**Future peer review:** preparation is complete, but another student's reproduction and a course peer-review Issue are not complete until the teacher publishes the review window and actual evidence is saved. No automatic future task is promised by this disclosure.
