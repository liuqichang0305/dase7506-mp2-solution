"""Sequential, bounded, restartable V3 experiment scheduler (train only).

The deadline is an absolute timestamp from campaign_budget.json. A restart never
renews it. Failed jobs require an explicit documented repair before retrying.
This script does not import training code or read held-out task/leaderboard data.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import psutil

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/research_v3'
QUEUE = OUT / 'campaign_queue.json'
LOCK = ROOT / 'tmp/research_v3/scheduler.lock'
PYTHON = ROOT / '.venv/Scripts/python.exe'
CORE = ['code/train_v3.py', 'code/common.py', 'code/mp2/models_v2.py',
        'code/mp2/selection_v2.py', 'code/mp2/features.py', 'code/mp2/training.py',
        'code/mp2/research_v3/training.py', 'code/mp2/research_v3/models.py',
        'code/mp2/research_v3/optimizers.py', 'code/mp2/research_v3/preprocess.py',
        'code/mp2/research_v3/selection.py']
TRAIN = ['code/data/manifest.json', 'code/data/gsm8k_train.jsonl', 'code/data/candidates/gsm8k_train.jsonl',
         'code/data/candidates/gsm8k_train.npz', 'results/research_v3/common_folds.json']


def utc():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.{os.getpid()}.pending')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    temp.replace(path)


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for part in iter(lambda: handle.read(1024 * 1024), b''):
            result.update(part)
    return result.hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def event(kind, **fields):
    record = dict(utc=utc(), event=kind, scheduler_pid=os.getpid(), **fields)
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / 'experiment_registry.jsonl').open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + '\n')
    print(json.dumps(record, ensure_ascii=False), flush=True)


def budget():
    value = read(OUT / 'campaign_budget.json')
    start = datetime.fromisoformat(value['started_at_utc'].replace('Z', '+00:00'))
    stop = datetime.fromisoformat(value['research_end_utc'].replace('Z', '+00:00'))
    deadline = datetime.fromisoformat(value['deadline_utc'].replace('Z', '+00:00'))
    actual_end = min(stop.timestamp(), start.timestamp() + value['maximum_wall_seconds'],
                     deadline.timestamp() - 3600 * value['deadline_buffer_hours'])
    return value, actual_end


def provenance():
    return {'source_files': {p: sha(ROOT / p) for p in CORE},
            'train_and_fold_files': {p: sha(ROOT / p) for p in TRAIN},
            'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()}


def load_queue():
    value = read(QUEUE) if QUEUE.exists() else {'schema': 'mp2-campaign-queue-v3', 'jobs': []}
    for job in value['jobs']:
        if canonical_sha(job['config']) != job['config_sha256']:
            raise ValueError('Queue configuration no longer matches its registration hash')
    return value


def save_queue(value):
    value['updated_at_utc'] = utc()
    atomic(QUEUE, value)


def same_process(identity):
    try:
        process = psutil.Process(identity['pid'])
        if abs(process.create_time() - identity['create_time']) > .01:
            return None
        expected = identity.get('command_fragment')
        if expected and expected.casefold() not in ' '.join(process.cmdline()).casefold():
            return None
        return process if process.is_running() else None
    except (psutil.NoSuchProcess, psutil.AccessDenied, KeyError):
        return None


def active_project_trainers():
    found = []
    for process in psutil.process_iter(['pid', 'cmdline', 'create_time']):
        try:
            command = process.info['cmdline'] or []
            if any(str(arg).replace('\\', '/').endswith('code/train_v3.py') for arg in command):
                if Path(process.cwd()).resolve() == ROOT:
                    found.append({'pid': process.pid, 'create_time': process.info['create_time'], 'command': command})
        except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
            continue
    return found


class SchedulerLock:
    def __enter__(self):
        LOCK.parent.mkdir(parents=True, exist_ok=True)
        identity = {'pid': os.getpid(), 'create_time': psutil.Process().create_time(),
                    'command_fragment': 'run_campaign_v3.py', 'utc': utc()}
        if LOCK.exists():
            previous = read(LOCK)
            if same_process(previous):
                raise RuntimeError(f"Scheduler still running: PID {previous['pid']}")
            LOCK.rename(LOCK.with_name(f'scheduler.stale.{time.time_ns()}.json'))
        with LOCK.open('x', encoding='utf-8') as stream:
            json.dump(identity, stream)
        self.identity = identity
        return self

    def __exit__(self, *_):
        if LOCK.exists() and read(LOCK) == self.identity:
            LOCK.unlink()  # Only our ephemeral lock, never experiment evidence.


def register(registration):
    value = read(registration)
    if not value.get('hypothesis') or not value.get('round'):
        raise ValueError('A round name and written hypothesis are mandatory')
    configs = value['configs']
    if not 2 <= len(configs) <= 4:
        raise ValueError('Register 2--4 hypothesis-specific configurations per round')
    plan, end = budget()
    if time.time() >= end:
        raise RuntimeError('Absolute research budget expired')
    if (OUT / 'SEARCH_CLOSED.json').exists():
        raise RuntimeError('Research is closed; no further configurations')
    queue = load_queue()
    identifiers = {job['experiment_id'] for job in queue['jobs']}
    # The timing pilot is an actual distinct configuration and consumes one slot.
    if len(identifiers | {c['experiment_id'] for c in configs}) + 1 > plan['maximum_unique_configurations']:
        raise ValueError('Unique configuration budget exceeded')
    source = provenance()
    if any(job['round'] == value['round'] for job in queue['jobs']):
        raise ValueError('Round already registered; its specification is immutable')
    prepared = []
    for config in configs:
        name = config['experiment_id']
        if not re.fullmatch(r'[a-z0-9_]+', name) or name in identifiers:
            raise ValueError(f'Invalid or previously registered experiment: {name}')
        identifiers.add(name)
        encoded = json.dumps(config, sort_keys=True).casefold()
        if any(word in encoded for word in ('validation', 'test_path', 'test.json', 'leaderboard', 'ranking_background')):
            raise ValueError('Held-out/ranking input is not permitted in a train configuration')
        config_path = ROOT / 'code/configs/research_v3' / (name + '.json')
        if config_path.exists() and read(config_path) != config:
            raise ValueError('A previous configuration must not be overwritten')
        prepared.append((config_path, config))
    for config_path, config in prepared:
        if not config_path.exists():
            atomic(config_path, config)
        job = {'experiment_id': config['experiment_id'], 'round': value['round'],
               'hypothesis': value['hypothesis'], 'config': config,
               'config_path': config_path.relative_to(ROOT).as_posix(),
               'config_sha256': canonical_sha(config), 'registered_at_utc': utc(),
               'registered_provenance': source, 'status': 'pending', 'attempts': [],
               'repair_retries': 0, 'maximum_attempt_wall_seconds': value.get('maximum_attempt_wall_seconds', 1800)}
        queue['jobs'].append(job)
        event('registered', experiment_id=job['experiment_id'], round=value['round'], status='pending',
              config_sha256=job['config_sha256'], registration=str(Path(registration).resolve().relative_to(ROOT)))
    save_queue(queue)


def summary_valid(job):
    if canonical_sha(job['config']) != job['config_sha256']:
        raise ValueError('Registered configuration identity mismatch')
    path = OUT / job['experiment_id'] / 'summary.json'
    if not path.exists():
        return None
    summary = read(path)
    if summary.get('config') != job['config']:
        raise ValueError('Summary configuration differs from the registered configuration')
    folds = summary.get('folds', [])
    if len(folds) != 3 or sorted(f['fold'] for f in folds) != [0, 1, 2]:
        return None
    if any(f.get('status') not in ('complete', 'completed') or f.get('n') != 1585 for f in folds):
        return None
    if summary.get('status') not in ('complete', 'completed') or summary.get('n') != 4755:
        return None
    if not (OUT / job['experiment_id'] / 'oof.npz').is_file():
        return None
    registered = job['registered_provenance']
    actual = summary.get('provenance', {})
    if actual.get('source_files') != registered['source_files']:
        raise ValueError('Completed summary source provenance differs from its registration')
    if 'train_and_fold_files' in actual:
        combined_data = actual['train_and_fold_files']
    else:
        combined_data = dict(actual.get('data_files', {}))
        combined_data['results/research_v3/common_folds.json'] = actual.get('folds_sha256')
    if combined_data != registered['train_and_fold_files']:
        raise ValueError('Completed summary data/fold provenance differs from its registration')
    folder = path.parent
    if summary.get('oof_sha256') != sha(folder / 'oof.npz'):
        raise ValueError('Completed OOF archive differs from its recorded hash')
    for fold in folds:
        if not fold.get('output_sha256') or not fold.get('asset_sha256'):
            raise ValueError('Completed fold is missing output/asset hashes')
        for name, expected in fold['output_sha256'].items():
            target = (folder / name).resolve()
            if not target.is_relative_to(folder.resolve()) or sha(target) != expected:
                raise ValueError('Completed fold output hash mismatch')
        asset_folder = (ROOT / fold['checkpoint']).resolve()
        if not asset_folder.is_relative_to((ROOT / 'artifacts/research_v3').resolve()):
            raise ValueError('Fold asset outside the V3 artifact directory')
        for name, expected in fold['asset_sha256'].items():
            target = (asset_folder / name).resolve()
            if not target.is_relative_to(asset_folder) or sha(target) != expected:
                raise ValueError('Completed fold asset hash mismatch')
    return {'summary_path': path.relative_to(ROOT).as_posix(), 'summary_sha256': sha(path),
            'oof_sha256': sha(path.with_name('oof.npz')), 'hard_accuracy': summary['hard_accuracy'],
            'accuracy': summary['accuracy']}


def terminate_owned(identity):
    process = same_process(identity)
    if process:
        process.terminate()
        try:
            process.wait(timeout=5)
        except psutil.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def complete(job, attempt, returncode, reason=None):
    evidence = summary_valid(job)
    if evidence and returncode in (0, None) and reason is None:
        job.update(status='completed', result=evidence)
    else:
        job['status'] = 'pruned' if reason else 'failed'
    attempt.update(ended_at_utc=utc(), returncode=returncode, stop_reason=reason,
                   observed_wall_seconds=time.time() - attempt['started_timestamp'])
    event('attempt_finished', experiment_id=job['experiment_id'], status=job['status'], attempt=attempt,
          result=job.get('result'))


def monitor(job, queue, popen=None):
    attempt = job['attempts'][-1]
    _, research_end = budget()
    next_heartbeat = 0
    while same_process(attempt):
        now = time.time()
        reason = ('absolute_campaign_budget' if now >= research_end else
                  'attempt_wall_budget' if now - attempt['started_timestamp'] >= job['maximum_attempt_wall_seconds'] else None)
        if reason:
            terminate_owned(attempt)
            complete(job, attempt, popen.poll() if popen else None, reason)
            save_queue(queue)
            return
        if now >= next_heartbeat:
            atomic(OUT / 'scheduler_heartbeat.json', {'utc': utc(), 'scheduler_pid': os.getpid(),
                'experiment_id': job['experiment_id'], 'training_pid': attempt['pid'],
                'attempt_elapsed_seconds': now - attempt['started_timestamp'],
                'campaign_remaining_seconds': max(0, research_end - now),
                'resume_command': '.venv\\Scripts\\python.exe scripts\\run_campaign_v3.py run'})
            next_heartbeat = now + 30
        time.sleep(1)
    returncode = popen.wait() if popen else None
    complete(job, attempt, returncode)
    save_queue(queue)


def run(round_name=None):
    queue = load_queue()
    if (OUT / 'SEARCH_CLOSED.json').exists():
        raise RuntimeError('Research is closed')
    # Reattach to our exact child identity; never start a duplicate trainer.
    for job in queue['jobs']:
        if job['status'] == 'running':
            attempt = job['attempts'][-1]
            if same_process(attempt):
                event('reattached', experiment_id=job['experiment_id'], training_pid=attempt['pid'])
                monitor(job, queue)
            elif summary_valid(job):
                complete(job, attempt, None)
                save_queue(queue)
            else:
                attempt.update(ended_at_utc=utc(), status='interrupted_process_missing')
                job['status'] = 'pending'  # Exact epoch/fold resume; not a new configuration or repair.
                event('interrupted', experiment_id=job['experiment_id'], status='pending')
                save_queue(queue)
    for job in queue['jobs']:
        if job['status'] != 'pending' or (round_name and job['round'] != round_name):
            continue
        _, end = budget()
        if time.time() >= end:
            job.update(status='pruned', stop_reason='absolute_campaign_budget')
            save_queue(queue)
            event('not_started', experiment_id=job['experiment_id'], status='pruned', reason=job['stop_reason'])
            continue
        if canonical_sha(job['config']) != job['config_sha256'] or read(ROOT / job['config_path']) != job['config']:
            raise RuntimeError('Registered configuration was altered')
        current = provenance()
        if current != job['registered_provenance']:
            # Git commit is informational; content hashes are binding.
            for key in ('source_files', 'train_and_fold_files'):
                if current[key] != job['registered_provenance'][key]:
                    raise RuntimeError('Source/data changed since registration; document repair before running')
        if summary_valid(job):
            # Only possible after interruption between completion and queue commit.
            job.update(status='completed', result=summary_valid(job))
            save_queue(queue)
            continue
        attempt_number = len(job['attempts']) + 1
        log = OUT / 'scheduler_logs' / f"{job['experiment_id']}.attempt{attempt_number}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        if log.exists():
            raise RuntimeError('Refuse to overwrite a previous process log')
        command = [str(PYTHON), 'code/train_v3.py', '--config', job['config_path']]
        active = active_project_trainers()
        if active:
            event('unrecorded_trainer_detected', processes=active)
            raise RuntimeError('A project trainer is alive outside this queue state; wait/reconcile it before restarting')
        env = os.environ.copy()
        env.update({key: '1' for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                    'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')})
        env['PYTHONUNBUFFERED'] = '1'
        env['PYTHONUTF8'] = '1'
        started = time.time()
        with log.open('xb') as stream:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
        identity = psutil.Process(process.pid)
        attempt = {'pid': process.pid, 'create_time': identity.create_time(),
                   'command_fragment': job['config_path'], 'command': command,
                   'started_at_utc': utc(), 'started_timestamp': started,
                   'log': log.relative_to(ROOT).as_posix(), 'provenance': current,
                   'scheduler_source_sha256': sha(__file__)}
        job['attempts'].append(attempt)
        job['status'] = 'running'
        save_queue(queue)
        event('attempt_started', experiment_id=job['experiment_id'], status='running', attempt=attempt)
        monitor(job, queue, process)
    event('queue_run_finished', statuses={s: sum(j['status'] == s for j in queue['jobs'])
          for s in ('pending', 'running', 'completed', 'failed', 'pruned')})


def retry(name, reason):
    queue = load_queue()
    plan, _ = budget()
    job = next(j for j in queue['jobs'] if j['experiment_id'] == name)
    if job['status'] not in ('failed', 'pruned') or job['repair_retries'] >= plan['maximum_repair_retries_per_experiment']:
        raise ValueError('Retry is unavailable or the two-repair cap was reached')
    if not reason.strip():
        raise ValueError('A concrete bug/repair explanation is mandatory')
    updated = provenance()
    if updated['train_and_fold_files'] != job['registered_provenance']['train_and_fold_files']:
        raise ValueError('Data/folds cannot change during repair')
    job.setdefault('repairs', []).append({'utc': utc(), 'reason': reason,
        'old_provenance': job['registered_provenance'], 'new_provenance': updated})
    job.update(status='pending', registered_provenance=updated, repair_retries=job['repair_retries'] + 1)
    save_queue(queue)
    event('repair_registered', experiment_id=name, reason=reason, repair_retries=job['repair_retries'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('register'); p.add_argument('registration', type=Path)
    p = sub.add_parser('run'); p.add_argument('--round')
    p = sub.add_parser('retry'); p.add_argument('experiment_id'); p.add_argument('--reason', required=True)
    sub.add_parser('status')
    args = parser.parse_args()
    if args.command == 'status':
        plan, end = budget()
        queue = load_queue()
        print(json.dumps({'utc': utc(), 'budget': plan, 'remaining_seconds': max(0, end - time.time()),
            'jobs': [{k: j.get(k) for k in ('experiment_id', 'round', 'status', 'result', 'repair_retries')}
                     for j in queue['jobs']]}, indent=2))
        return
    with SchedulerLock():
        if args.command == 'register': register(args.registration)
        elif args.command == 'run': run(args.round)
        elif args.command == 'retry': retry(args.experiment_id, args.reason)


if __name__ == '__main__':
    main()
