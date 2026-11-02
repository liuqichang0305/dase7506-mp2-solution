# Verified research-archive reproduction

The real `final-selected-01` reproduction passed on 2 November 2026, before
the research-only freeze. The selected historical Deep Sets checkpoint at
`artifacts/research_v3/final_selected`, loaded by `code/solution_v3.py`, gave
**740/798 = 92.73182957393483%** on official validation. All **798 ordered
candidate choices matched** the locked historical outputs. No weights were
retrained and no new official test was opened.

The exact public source/asset commit was
`eee478b4959e555b6ec8c902b6db8b69f31df508`. The actual new environment contained
NumPy 2.5.3, threadpoolctl 3.7.0, psutil 7.2.2 and pip 23.2.1; Torch and
TensorFlow were unavailable. The inference implementation uses NumPy only.
Fresh prediction and a separate official scorer took **2.2803907999768853 s**
under one 30-second deadline. Task-process peak sampled OS threads were 4;
scientific thread pools were capped at 1. Setup time is separate. These are
local measurements, not certification on the reference Xeon hardware.

## Evidence and the completed command

The authoritative wrapper is
[`submission/v3/REPRODUCTION_RESULT.json`](../../submission/v3/REPRODUCTION_RESULT.json).
It binds the source/assets, original comparison outputs and the actual
[`final-selected-01` logs](../../results/research_v3/clean_reproduction/final-selected-01).
The saved `setup-*.command.json`/`setup-*.result.json` files record exact Git,
venv, package-install and probe commands. `validation-data-files.json` records
every copied data path/hash; `environment.json` records actual package and
checkout locations. `validation-timing.json` retains the full direct-Python
guard wrappers, both official commands, elapsed time and resource observations.

The actual coordinator command was:

```powershell
.venv\Scripts\python.exe scripts\reproduce_selected_v3.py --attempt final-selected-01
```

The first preflight, at earlier commit `ed7fbc6b6b90d547733b61aa823d59295abc4152`,
refused a Git/local line-ending difference in `requirements-lock.txt` before
cloning or model execution. Its saved incident and byte comparison are retained.
The subsequent exact-byte commit above passed; dependency versions, model and
checkpoint were unchanged. The sole real reproduction attempt has no earlier model execution
or alternative runtime to select from.

## Independent checkout instructions

**The completed coordinator is evidence, not a portable fresh-clone entry
point.** Its preflight checks require the original training data and absolute
training-state provenance paths. It also correctly rejects the current
`FREEZE.json` or an existing completed reproduction. Do not remove guards or
attempt to recreate those local paths. The command above describes the actual
completed run in its original provenance environment.

An independent verifier can instead use the unchanged official predictor and
separate scorer on the exact source and checkpoint. The following Windows
PowerShell commands obtain only validation data, create a NumPy environment,
and run direct base Python `-S` processes. The environment setup uses a venv;
the prediction/scoring commands avoid the Windows venv launcher.
Choose new directories and preserve any existing outputs.

```powershell
git clone --no-checkout https://github.com/liuqichang0305/dase7506-mp2-solution.git v3-validation-reproduction
Set-Location v3-validation-reproduction
git checkout --detach eee478b4959e555b6ec8c902b6db8b69f31df508
python -I -m venv .venv
.venv\Scripts\python.exe -I -m pip --isolated --disable-pip-version-check install --no-input --only-binary=:all: --no-deps --index-url https://pypi.org/simple numpy==2.5.3 threadpoolctl==3.7.0 psutil==7.2.2
.venv\Scripts\python.exe code\tools\fetch_development_v2.py --split validation

$reproBasePython = (& .\.venv\Scripts\python.exe -I -c "import sys; print(sys._base_executable)").Trim()
$reproSitePackages = (Resolve-Path '.\.venv\Lib\site-packages').Path
$env:PYTHONPATH = $reproSitePackages
$env:PYTHONNOUSERSITE = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONUTF8 = '1'
foreach ($reproThreadVariable in @('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS', 'OMP_THREAD_LIMIT')) {
    Set-Item -Path "Env:$reproThreadVariable" -Value '1'
}
$reproOutput = 'results\portable-validation-01'
if (Test-Path -LiteralPath $reproOutput) { throw 'Use a new output directory; retain earlier evidence.' }
New-Item -ItemType Directory -Path $reproOutput | Out-Null
& $reproBasePython -S code\predict.py --method code\solution_v3.py --checkpoint artifacts\research_v3\final_selected --split validation --output "$reproOutput\validation-predictions.json"
if ($LASTEXITCODE -ne 0) { throw 'Official prediction failed; do not score partial output.' }
& $reproBasePython -S code\score_predictions.py --split validation --predictions "$reproOutput\validation-predictions.json" --output "$reproOutput\validation-score.json"
if ($LASTEXITCODE -ne 0) { throw 'Official scoring failed; retain the failure evidence.' }
```

The downloader verifies the existing official package manifest and fixed
official commit `bb801f4f2e7629d270df765ab12c333b2dc616bb`. With the explicit
`--split validation`, it fetches only the validation references, candidate
JSONL/NPZ, manifest and attribution files. No train/test download is needed.
In the completed real run, the reproducer hash-verified and copied those
already available official validation files into its fresh checkout; it did
not itself download data or parse raw examples for model selection.

The independent commands use the fixed research checkpoint without fitting. Expected
official aggregate: **740/798**. The selected assets are `config.json` SHA256
`e2c210a3717c5b48ee312226491b4b9c444cb6642d94fb3afd1fa39226bd89cf` and
`weights.npz` SHA256
`2bb9002701b0bcb6105b30f88863db20d7fd7267bc530ce85048168d47a60ab1`.
The committed source and checkpoint must remain unchanged. No Torch package
or training data is needed for these underlying inference commands.

The completed `final-selected-01` run used additional audit wrappers and resource
monitoring: sequential direct base Python `-S`, only its new venv site-packages
on `PYTHONPATH`, denied networking/child processes, and phase-specific data
access. Prediction could not open gold or old results; only the independent
scorer could open gold. Its coordinator compared all saved choices and official
aggregate fields. Its complete measured interval included startup, imports,
input/model loading, prediction, independent scoring and wrapper overhead;
network/setup occurred beforehand.

The portable commands above reproduce inference/scoring without invoking that
provenance-dependent coordinator. They set the same scientific thread limits,
but do not themselves implement its audit hooks, process-tree monitoring or
shared 30-second deadline. Do not label an unmeasured portable run as a new
resource certification. This documentation change did not run another clone,
installation or evaluation; any subsequent smoke-check evidence must be saved
and described separately from the completed fresh-environment run.

The original search, selection and submitted V1 artifacts remain unchanged. The later
[`FREEZE.json`](../../submission/v3/FREEZE.json) records `research_only=true`
and `formal_test_allowed=false`; independent validation reproduction does not
authorize a new V3 test, score update, retraining or replacement submission.
