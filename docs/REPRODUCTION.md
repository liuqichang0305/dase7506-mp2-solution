# Independent clean-environment reproduction

Reviewer: Codex `audit_tools` sub-agent (internal AI reproduction, not a human peer review or instructor certification).

Status: preparation only. No reproduction evaluation has run yet. Awaiting the exact publicly downloadable release commit from the primary agent.

The original frozen source commit supplied for this check is `8602992842d444524ab4148991da2f1dc88f316f`. The published packaging commit may add reports and evidence, but its method source and checkpoint must match the pre-test freeze.

## Preparation

- Created a new Python 3.12 virtual environment at `tmp/reproduction/venv`, independently of the development environment.
- Installed only `requirements-inference.txt` (`numpy==2.5.3`), with pip's package cache disabled. The fresh environment contains NumPy 2.5.3 and pip 23.2.1; installation returned 0.
- Complete setup stdout, stderr, command, return code and timing are being saved under `results/reproduction/`.
- The reproduction source will be a fresh anonymous/public clone of the published commit. It will fetch official data using the committed download instructions and evaluate the committed checkpoint without training. It will not use local development model/data files.
- Only the identical frozen method may be evaluated on test. No test examples or per-question references will be inspected for development.

## Pending

1. Receive and anonymously clone the published full commit SHA.
2. Execute the final README setup/data-integrity commands with the clean interpreter.
3. Verify the frozen source files, checkpoint SHA-256 and complete asset list.
4. Run the same frozen checkpoint once with official complete-test prediction/scoring, saving full process logs and local wall time.
5. Compare the official aggregate score and prediction-file selected indices mechanically, without inspecting test examples/references or selecting another model.
6. Replace this preparation status with actual results and any blockers.
