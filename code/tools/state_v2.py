"""Refresh a recoverable v2 status snapshot while retaining append-only history."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/research_v2'

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--phase', required=True)
    p.add_argument('--next-action', required=True)
    a = p.parse_args()
    completed = []
    for path in OUT.glob('*/summary.json'):
        data = json.loads(path.read_text(encoding='utf-8'))
        if data.get('status') == 'complete' and 'config' in data:
            completed.append(data['experiment_id'])
    state = {'updated_utc': datetime.now(timezone.utc).isoformat(), 'phase': a.phase,
             'completed_cv_experiments': sorted(completed), 'next_action': a.next_action,
             'branch': subprocess.check_output(['git', 'branch', '--show-current'], cwd=ROOT, text=True).strip(),
             'new_test_enabled': False, 'course_score_updated': False,
             'formal_test_blocker': 'No explicit applicable post-evaluation reevaluation process located in latest official sources',
             'old_official_submission': 'https://github.com/xudongwu-0/xudongwu-0.github.io/issues/294',
             'finalists_locked': (OUT / 'FINALISTS_LOCK.json').exists(),
             'validation_confirmed': (OUT / 'validation_summary.json').exists(),
             'freeze_exists': (OUT / 'V2_FREEZE.json').exists()}
    (OUT / 'STATE.json').write_text(json.dumps(state, indent=2) + '\n', encoding='utf-8')
    with (OUT / 'state_history.jsonl').open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(state) + '\n')
    print(json.dumps(state))
