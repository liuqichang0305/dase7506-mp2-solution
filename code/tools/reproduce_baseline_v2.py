"""A0: reproduce only the already-known v1 train-inner and validation results."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS', 'OMP_THREAD_LIMIT'):
    os.environ[key] = '1'
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code'))
from mp2.training import load_development, fit_features, metrics
from mp2.features import build_features, hard_mask

destination = ROOT / 'results/research_v2/baseline_reproduction'
destination.mkdir(parents=True, exist_ok=True)
if (destination / 'summary.json').exists():
    raise FileExistsError('A0 complete; preserve original reproduction')
bank, embeddings, labels = load_development('train')
features = build_features(bank, embeddings, 'relation')
split = json.loads((ROOT / 'results/development/train_inner_split.json').read_text())
fit, hold = split['fitting_rows'], split['holdout_rows']
config = {'features': 'relation', 'loss': 'listwise', 'l2': .1, 'maxiter': 200}
arrays, info = fit_features(features[fit], labels[fit], config)
scores = ((features[hold] - arrays['mean']) / arrays['scale']) @ arrays['weight'] + arrays['bias']
observed = metrics(labels[hold], scores.argmax(1), hard_mask([bank[i] for i in hold]))
old = json.loads((ROOT / 'results/development/search_summary.json').read_text())
expected = next(row for row in old['screens'] if row['experiment_id'] == 'relation-listwise-l2-0.1')
assert observed['correct'] == expected['correct'] and observed['hard_correct'] == expected['hard_correct']
np.savez(destination / 'inner_refit.npz', **arrays)
np.save(destination / 'inner_scores.npy', scores)
command = [sys.executable, 'code/tools/evaluate.py', '--checkpoint', 'artifacts/final_bundle',
           '--split', 'validation', '--output-dir', str(destination / 'validation')]
proc = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, encoding='utf-8')
(destination / 'official.stdout.log').write_text(proc.stdout, encoding='utf-8')
(destination / 'official.stderr.log').write_text(proc.stderr, encoding='utf-8')
if proc.returncode:
    raise RuntimeError(proc.stderr)
new = json.loads((destination / 'validation/validation-predictions.json').read_text())
prior = json.loads((ROOT / 'results/development/relation-listwise-l2-0.1/predictions.json').read_text())
assert new['selected'] == prior['selected']
report = {'status': 'passed', 'train_inner': observed, 'fit': info, 'same_original_inner_rows': True,
          'validation_all_choices_identical': True, 'test_access': False,
          'official_validation': json.loads((destination / 'validation/validation-score.json').read_text()),
          'command': command}
(destination / 'summary.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
print(json.dumps(report))
