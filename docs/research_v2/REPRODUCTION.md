# Frozen v2 validation reproduction

Selected model: `artifacts/research_v2/final_deepsets`, using `code/solution_v2.py`.
Immutable source and asset commit: `36f9617175321a2861e78bb2f4ffcde643c4fc03`.
The later `results/research_v2/V2_FREEZE.json` records 18 source-file hashes,
the two asset hashes, the selection evidence and the explicitly closed test gate.
The original v1 release, report, checkpoint and course Issue294 remain unchanged.

## Fresh Python 3.12 environment, without retraining

From a new directory (Windows PowerShell):

```powershell
git clone --branch research/performance-v2 --single-branch https://github.com/liuqichang0305/dase7506-mp2-solution.git v2-reproduction
cd v2-reproduction
git checkout --detach 36f9617175321a2861e78bb2f4ffcde643c4fc03
python -m venv .venv-repro
.\.venv-repro\Scripts\python.exe -m pip install -r requirements-inference.txt
.\.venv-repro\Scripts\python.exe code\tools\fetch_development_v2.py --split validation
.\.venv-repro\Scripts\python.exe code\tools\evaluate.py --method code\solution_v2.py --checkpoint artifacts\research_v2\final_deepsets --split validation --output-dir results\research_v2\fresh-user-reproduction
```

Expected official result: **740/798 = 92.73182957393483%**. The actual original
local complete fresh prediction plus independent scoring measurement was
1.7397279 seconds. This is not certification on the reference Xeon 8457C.
The evaluation tool explicitly sets all compute libraries to one thread.
It refuses to overwrite a previous evaluation directory. Inference requires
only NumPy, loads no references, downloads nothing and calls no model/API.

Check uncompressed assets and isolated inference with the same NumPy environment:

```powershell
.\.venv-repro\Scripts\python.exe code\tools\audit_assets.py artifacts\research_v2\final_deepsets --output results\research_v2\fresh-user-assets.json
.\.venv-repro\Scripts\python.exe code\tools\check_contract_v2.py --checkpoint artifacts\research_v2\final_deepsets --input-source validation --output results\research_v2\fresh-user-contract.json
```

The bundle is 963,670 bytes including configuration and NPZ container overhead,
below 16,777,216 bytes. Final inference chooses one original candidate index.
All-unparseable synthetic rows return index 0; the course guarantee of at least
one correct candidate is what justifies valid-mask/unique-group reasoning.

## Training record and boundaries

Training used Python3.12, NumPy2.5.3, SciPy1.18.1, scikit-learn1.9.1 and
PyTorch2.15.0+cpu. Current-branch `requirements-training-v2.txt` supplies pinned
training dependencies; PyTorch was downloaded from its official CPU index.
All weights were random-initialized and fit only on the official train data.
No GPU, external labels, extra data, pretrained weights or external LLM were used.

Exact historical commands are recorded in the batch JSONL/process logs, e.g.:

```text
python code/train_v2.py --config code/configs/research_v2/deepsets_regularized_s0.json
python code/train_v2.py --config code/configs/research_v2/deepsets_regularized_s0.json --final-output artifacts/research_v2/final_deepsets
```

These are historical fitting commands, not prerequisites for reproducing the
frozen checkpoint. The current entry point deliberately refuses additional CV
after FINALISTS_SELECTION/FINALISTS_LOCK. Do not delete locks or old search
summaries. Full configs, fixed common folds, inner epoch histories, OOF scores,
all fold weights, data/source hashes and final fitting records are retained.
The final all-train Deep Sets fit used the predetermined seed3036900957 and
9 epochs (median of inner stopping epochs11,7,9). It did not use validation
for epoch selection. Old test aggregate348/467 was known before this research.

The report and consolidated table separate original/replayed baseline results,
train-only OOF, two prelocked new validation candidates and unrun methods.
Fresh reproduction must match all798 frozen selected indices, not merely total
accuracy. Reproduction never replaces the checkpoint with a retrained model.
New test scoring and course score updates remain closed pending an explicit
applicable course process. No new ranking is claimed. Recovery instructions:
`docs/research_v2/RESUME.txt` and `results/research_v2/STATE.json`.
