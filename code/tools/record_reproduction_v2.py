"""Verify a fresh public checkout and record its frozen validation reproduction."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/research_v2'
CLONE = ROOT / 'tmp/public-v2-reproduction'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


freeze = read(OUT / 'V2_FREEZE.json')
commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=CLONE, text=True).strip()
assert commit == freeze['source_commit']
for relative, expected in {**freeze['source_files'], **freeze['assets']}.items():
    assert hashlib.sha256((CLONE / relative).read_bytes()).hexdigest() == expected, relative
fresh = CLONE / 'results/research_v2/clean-reproduction'
prediction = read(fresh / 'validation-predictions.json')
original = read(OUT / 'validation/deepsets_regularized_s0/validation-predictions.json')
assert prediction['ids'] == original['ids'] and prediction['selected'] == original['selected']
timing = read(fresh / 'validation-timing.json')
assert timing['status'] == 'passed' and timing['local_under_30_seconds']
assert read(OUT / 'fresh-reproduction-contract.json')['status'] == 'passed'
packages = subprocess.check_output([str(ROOT / 'tmp/v2-repro-env/Scripts/python.exe'), '-m', 'pip', 'freeze'], text=True).strip().splitlines()
assert packages == ['numpy==2.5.3'], packages
files = [p.relative_to(CLONE).as_posix() for p in (CLONE / 'code/data').rglob('*') if p.is_file()]
assert not any('test' in Path(p).name or 'train' in Path(p).name for p in files)
target = OUT / 'clean-reproduction'
if target.exists():
    raise FileExistsError('Preserve existing clean reproduction evidence')
shutil.copytree(fresh, target)
reproduction = {'status': 'passed', 'completed_at_utc': datetime.now(timezone.utc).isoformat(),
    'source': 'Fresh public GitHub clone, exact frozen commit; independently downloaded official validation only',
    'source_commit': commit, 'source_and_asset_hashes_verified': len(freeze['source_files']) + len(freeze['assets']),
    'all_predictions_identical': True, 'matching_choices': len(prediction['selected']),
    'correct': timing['official_score']['correct'], 'n': timing['official_score']['n'],
    'prediction_plus_scoring_seconds': timing['total_wall_seconds'], 'packages': packages,
    'numpy_only': True, 'checkpoint_retrained': False, 'no_train_or_test_data_present': True,
    'isolated_inference_passed': True, 'threadpool_introspection': 'Not installed in NumPy-only environment; all thread environment settings are1. Original environment actual pool count1.',
    'evidence': 'results/research_v2/clean-reproduction', 'new_test_evaluated': False}
(OUT / 'REPRODUCTION_RESULT.json').write_text(json.dumps(reproduction, indent=2) + '\n', encoding='utf-8')
confirmation = read(OUT / 'validation_summary.json')
summary = {**confirmation, 'status': 'development_validated_pending_course_test',
    'analysis_path': 'results/research_v2/analysis-final/analysis.json',
    'ensemble_path': 'results/research_v2/ensemble-selected/summary.json',
    'freeze_path': 'results/research_v2/V2_FREEZE.json', 'reproduction': reproduction,
    'narrative': [
        'Nineteen configuration/seed runs completed the same nested train-only three-fold protocol; the old test aggregate was already known.',
        'Attention has stronger seed-averaged train OOF evidence. Deep Sets was selected by the preregistered two-finalist validation rule; its two-question lead over Attention does not establish stable family superiority.',
        'Selected Deep Sets improved observed validation from724/798 to740/798, hard282/355 to297/355, and single-positive59/117 to68/117.',
        'Mask/group decoding helped old linear OOF, but MLP group loss, hard normalization, Deep Sets learned pooling and the unique equal mixture did not show the hypothesized additional gains.',
        'All model choices were locked before new validation. No search reopened after confirmation. Old assets, logs, guards and course Issue294 are preserved.',
        'No new test evaluation or grade/rank update was performed. An explicit applicable course reevaluation process remains required.']}
(OUT / 'final_summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
print(json.dumps(reproduction))
