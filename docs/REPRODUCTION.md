# Independent clean-environment reproduction

Reviewer: Codex `audit_tools` sub-agent (internal AI reproduction, not a human peer review or instructor certification).

Status: **passed**, 2026-10-29. One complete official test reproduction was executed, without training or changing the method. Reproduced **348/467 = 74.51820128479658%**. Every selected candidate index and ID matched the originally published predictions exactly (zero mismatches).

The anonymously cloned and checked-out public packaging commit was [`92c7240c5a4710054de1363d23e2f11f81d33b72`](https://github.com/liuqichang0305/dase7506-mp2-solution/tree/92c7240c5a4710054de1363d23e2f11f81d33b72). Its 13 frozen source files and two checkpoint files match `submission/PRE_TEST_FREEZE.json`; every source hash also matches the Git object at the original source commit `8602992842d444524ab4148991da2f1dc88f316f`. Later documentation/submission commits are outside this exact reproduction; their source/asset equivalence must be checked separately.

## Preparation

- Created a new Python 3.12 virtual environment at `tmp/reproduction/venv`, independently of the development environment.
- Installed only `requirements-inference.txt` (`numpy==2.5.3`), with pip's package cache disabled. The fresh environment contains NumPy 2.5.3 and pip 23.2.1; installation returned 0.
- Complete setup stdout, stderr, commands, return codes and timing are saved under `results/reproduction/`.
- Source was freshly cloned into `tmp/reproduction/repo`. Git credential helpers, askpass, extra Authorization headers and interactive prompts were disabled for the public clone. No development model or data file was copied into this checkout.
- The documented fetcher downloaded official coursework data at the fixed upstream commit; each downloaded file was verified before writing. Official package hash verification and `check_data.py` both returned 0.
- Only the identical frozen method was evaluated on test. Test examples and per-question references were not inspected for development. Predictions were compared mechanically, without displaying their individual indices or references to the reviewer.

## Executed reproduction commands

The README was followed using the fresh environment in the sibling `venv` directory instead of a `.venv` inside the clone. This interpreter-path substitution is the only setup adaptation. From `tmp/reproduction/repo`:

```powershell
git checkout --detach 92c7240c5a4710054de1363d23e2f11f81d33b72
..\venv\Scripts\python.exe -m pip install --no-cache-dir --disable-pip-version-check -r requirements-inference.txt
..\venv\Scripts\python.exe code\tools\fetch_official.py
..\venv\Scripts\python.exe code\tools\verify_official.py --output results\reproduced-official-hashes.json
..\venv\Scripts\python.exe code\check_data.py
..\venv\Scripts\python.exe code\tools\evaluate.py --checkpoint artifacts\final_bundle --split test --output-dir results\reproduced --freeze submission\PRE_TEST_FREEZE.json
```

The clean environment contains only Python 3.12.0, pip 23.2.1 and NumPy 2.5.3. SciPy and training dependencies were absent. Evaluation uses no network operation or retraining. The exact original freeze is checked before prediction. `git diff --exit-code` confirmed that tracked files in the cloned checkout remained unchanged.

## Measured outcome

| Check | Actual result |
|---|---|
| Prediction process | Exit 0, 0.7725343000001885 seconds |
| Independent scoring process | Exit 0, 0.36406970000007277 seconds |
| Complete fresh prediction + scoring | **1.139880799999446 seconds**, local wall time |
| Official complete-test result | **348/467; 74.51820128479658%** |
| Original vs reproduced IDs/choices | Exactly equal; zero candidate-index mismatches |
| Frozen files | All 13 source and 2 asset hashes passed |
| Additional inference assets | 51,923 bytes uncompressed/conservatively counted |
| Retraining | None |
| Test runs in this reproduction | Exactly one, identical frozen method |

The host was Windows 11, Intel Core i9-12900H, 20 logical processors, Python 3.12.0. Evaluation ran sequential processes with all configured native-library thread limits set to one. The original run's 0.705959-second local result and this 1.139881-second result are both retained. Neither certifies eligibility on the instructor's Xeon reference CPU; no fastest-run selection was used.

Full official process stdout/stderr, commands, return codes, component timing and official JSON are in `results/reproduction/official-evaluation/`. `comparison.json` records the mechanical equality checks. `environment.json`, `official-hashes.json` and `asset-audit.json` record the clean interpreter and actual package/asset checks.

## Freeze-manifest newline discrepancy

The manifest JSON itself was converted from CRLF in the original Windows workspace to LF in the published Git object. Its SHA-256 therefore differs:

- Original evaluation's CRLF manifest: `2734218beef0a53d63015e7483d374772c9fcebe99c1226785f40cd0cf792899`.
- Published/reproduced LF manifest: `bf8ece6e5a7ab1ed7f6803eec3c48e28fef0d6566ceddb65edd8965414a9e572`.

This was investigated before reproduction. Parsed JSON is identical; normalizing line endings makes the entire byte sequences equal; converting the published LF file back to CRLF produces exactly the original recorded SHA-256. All 13 source and both asset hashes inside the manifest remain unchanged and pass against the original source commit. No frozen code, weights, configuration or selection logic was changed. The full comparison is saved in `freeze-verification.json`; the published manifest was not edited to conceal the difference.

## Anonymous download verification

Unauthenticated HTTPS GETs returned HTTP 200 for all four immutable raw resources below. No Authorization header or cookies were supplied; downloaded bytes matched the public clone. Status codes, sizes and SHA-256 are saved in `public-downloads.json`.

- [Inference code](https://raw.githubusercontent.com/liuqichang0305/dase7506-mp2-solution/92c7240c5a4710054de1363d23e2f11f81d33b72/code/solution.py): SHA-256 `cb40cd5a2912343dfa5eab8e9865b37c0032e1520d5a2d036d415b9821cbb2a6`.
- [Report PDF](https://raw.githubusercontent.com/liuqichang0305/dase7506-mp2-solution/92c7240c5a4710054de1363d23e2f11f81d33b72/reports/report.pdf): 126,314 bytes; SHA-256 `fb4a0253f227a944363bf84609c95479d5f64d85911d0236d5c5435f0227cf88`.
- [Checkpoint configuration](https://raw.githubusercontent.com/liuqichang0305/dase7506-mp2-solution/92c7240c5a4710054de1363d23e2f11f81d33b72/artifacts/final_bundle/config.json): 249 bytes; SHA-256 `24473c3b95b9472b5d844b795edf86c7902d21fe7351604cc680d93d7ead0703`.
- [Checkpoint weights](https://raw.githubusercontent.com/liuqichang0305/dase7506-mp2-solution/92c7240c5a4710054de1363d23e2f11f81d33b72/artifacts/final_bundle/weights.npz): 51,674 bytes; SHA-256 `a785f4a48e2d5708aec9fec29c776e9ce1b7fea1832d064e95b25c926626d462`.

No reproduction blocker remains for this exact published commit. Actual course submission and the instructor's later public peer-review window are separate workflow steps, not established by this internal reproduction.
