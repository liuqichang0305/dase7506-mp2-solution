"""One documented-command smoke check; reuse the preserved clean environment.

This is not a second installation, model selection, or a new formal test.
The original clean reproduction evidence and checkout remain unchanged.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code/tools'))
import confirm_validation_v3 as timing


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


def main():
    out = ROOT / 'results/research_v3/portable_validation_commands'
    original = ROOT / 'results/research_v3/clean_reproduction/final-selected-01'
    environment = read(original / 'environment.json')
    frozen = read(ROOT / 'submission/v3/FREEZE.json')
    checkout = Path(environment['checkout'])
    assert frozen['research_only'] is True and frozen['formal_test_allowed'] is False
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=checkout, text=True).strip() == frozen['source_commit']
    for name, expected in {**frozen['source_files'], **frozen['assets']}.items():
        assert sha(checkout / name) == expected, name
    assert not (checkout / 'tmp/research_v3/training_states').exists()
    data = sorted(p.relative_to(checkout / 'code/data').as_posix() for p in (checkout / 'code/data').rglob('*') if p.is_file())
    assert data == environment['official_data_files']
    assert not any('train' in name or 'test' in name for name in data)
    out.mkdir(exist_ok=False)
    identity = {'source_commit': frozen['source_commit'], 'freeze_sha256': sha(ROOT / 'submission/v3/FREEZE.json'),
                'reused_environment_sha256': sha(original / 'environment.json'),
                'script_sha256': sha(__file__), 'documentation_sha256': sha(ROOT / 'docs/research_v3/REPRODUCTION.md'),
                'fresh_installation': False, 'model_or_selection_changed': False, 'split': 'validation',
                'reason': 'Verify portable official commands without coordinator preflight, train data or original training-state files.'}
    write(out / 'STARTED.json', {**identity, 'started_at_utc': datetime.now(timezone.utc).isoformat()})
    env = {**timing.THREAD_ENV, 'PYTHONPATH': os.pathsep.join(environment['site_packages']),
           'PYTHONNOUSERSITE': '1', 'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONUTF8': '1', 'PYTHONHOME': ''}
    commands = [
        [environment['base_executable'], '-S', 'code/predict.py', '--method', 'code/solution_v3.py',
         '--checkpoint', 'artifacts/research_v3/final_selected', '--split', 'validation',
         '--output', str(out / 'validation-predictions.json')],
        [environment['base_executable'], '-S', 'code/score_predictions.py', '--split', 'validation',
         '--predictions', str(out / 'validation-predictions.json'), '--output', str(out / 'validation-score.json')]]
    processes = []
    with timing.exclusive_evaluation():
        previous_root = timing.ROOT
        try:
            timing.ROOT = checkout
            started = time.perf_counter()
            deadline = started + 30
            for command, phase in zip(commands, ('prediction', 'scoring')):
                result = timing.run_process(command, out, phase, deadline, environment=env)
                processes.append(result)
                assert result['returncode'] == 0 and not result['timed_out'], phase
            elapsed = time.perf_counter() - started
        finally:
            timing.ROOT = previous_root
    a, b = read(out / 'validation-predictions.json'), read(original / 'validation-predictions.json')
    score, old = read(out / 'validation-score.json'), read(original / 'validation-score.json')
    assert a['ids'] == b['ids'] and a['selected'] == b['selected'] and len(a['ids']) == 798
    assert all(score[key] == old[key] for key in ('protocol', 'split', 'correct', 'n', 'accuracy'))
    assert elapsed <= 30 and max(p['sampled_peak_tree_threads'] for p in processes) <= 4
    write(out / 'COMPLETE.json', {**identity, 'status': 'passed', 'completed_at_utc': datetime.now(timezone.utc).isoformat(),
        'choices_equal': 798, 'correct': score['correct'], 'n': score['n'], 'prediction_plus_scoring_seconds': elapsed,
        'sampled_peak_task_threads': max(p['sampled_peak_tree_threads'] for p in processes), 'processes': processes,
        'output_sha256': {p.name: sha(p) for p in sorted(out.iterdir()) if p.is_file()},
        'limitation': 'Reused the prior clean checkout/environment; no second fresh installation. Official CLI replay has no audit-hook wrapper. Timing is local, not reference-Xeon certification.'})
    print(json.dumps({'status': 'passed', 'choices_equal': 798, 'correct': score['correct'], 'seconds': elapsed, 'test_opened': False}))


if __name__ == '__main__':
    main()
