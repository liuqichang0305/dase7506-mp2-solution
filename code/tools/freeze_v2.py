"""Freeze selected development artifacts; explicitly does not unlock test."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/research_v2'
sys.path.insert(0, str(ROOT / 'code/tools'))
from audit_assets import audit


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


if __name__ == '__main__':
    destination = OUT / 'V2_FREEZE.json'
    if destination.exists():
        raise FileExistsError('V2 is already frozen; do not replace this record')
    confirmed = json.loads((OUT / 'validation_summary.json').read_text())
    lock = json.loads((OUT / 'FINALISTS_LOCK.json').read_text())
    chosen = next(r for r in confirmed['finalists'] if r['experiment_id'] == confirmed['selected_experiment_id'])
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    source_paths = sorted(set(lock['inference_source_sha256']) | {
        'code/mp2/training_v2.py', 'code/train_v2.py', 'code/run_research_v2.py',
        'code/tools/confirm_validation_v2.py', 'code/tools/lock_finalists_v2.py',
        'code/tools/check_contract_v2.py', 'code/tools/check_export_v2.py',
        'code/tools/fetch_development_v2.py', 'code/tools/freeze_v2.py',
        'code/configs/research_v2/' + chosen['experiment_id'] + '.json'})
    sources = {name: sha(ROOT / name) for name in source_paths}
    assets = {p.relative_to(ROOT).as_posix(): sha(p) for p in (ROOT / chosen['checkpoint']).rglob('*') if p.is_file()}
    for relative, expected in {**sources, **assets}.items():
        blob = subprocess.run(['git', 'show', commit + ':' + relative], cwd=ROOT, capture_output=True)
        if blob.returncode or hashlib.sha256(blob.stdout).hexdigest() != expected:
            raise ValueError(f'Frozen file does not exactly match source commit: {relative}')
    for relative, expected in {**lock['asset_sha256'], **lock['inference_source_sha256']}.items():
        if sha(ROOT / relative) != expected:
            raise ValueError('A finalist changed after confirmation')
    record = {'schema': 'dase7506-mp2-research-freeze-v2', 'frozen': True,
              'created_at_utc': datetime.now(timezone.utc).isoformat(), 'source_commit': commit,
              'upstream_commit': 'bb801f4f2e7629d270df765ab12c333b2dc616bb',
              'selected_experiment_id': chosen['experiment_id'], 'method': chosen['method'],
              'checkpoint': chosen['checkpoint'], 'source_files': sources, 'assets': assets,
              'config_sha256': sha(ROOT / 'code/configs/research_v2' / (chosen['experiment_id'] + '.json')),
              'validation': chosen['validation'], 'resource': audit(ROOT / chosen['checkpoint']),
              'selection_reason': confirmed['selection_rule'],
              'finalists_lock_sha256': sha(OUT / 'FINALISTS_LOCK.json'),
              'common_folds_sha256': sha(OUT / 'common_folds.json'),
              'known_old_test_aggregate': {'correct': 348, 'n': 467, 'known_before_v2_development': True},
              'new_test_evaluated': False, 'formal_test_allowed': False,
              'allowed_evaluation_state': 'Frozen validation reproduction and audit only; course-specific new test/score update process pending confirmation',
              'old_submission_preserved': 'https://github.com/xudongwu-0/xudongwu-0.github.io/issues/294'}
    destination.write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'source_commit': commit, 'selected': chosen['experiment_id'], 'formal_test_allowed': False}))
