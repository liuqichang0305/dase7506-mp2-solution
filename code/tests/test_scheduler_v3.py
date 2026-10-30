"""Isolated V3 scheduler checks; synthetic files only, no training/data reads."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

REPO = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('scheduler_v3_under_test', REPO / 'scripts/run_campaign_v3.py')
scheduler = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scheduler)


class FakeProcess:
    def __init__(self, root, pid=12345, created=12.5, command=None):
        self.pid = pid
        self._created = created
        self._root = root
        self._command = command or [sys.executable, 'code/train_v3.py', '--config', 'code/configs/research_v3/mlp_a.json']
        self.info = {'pid': pid, 'create_time': created, 'cmdline': self._command, 'cwd': str(root),
                     'name': 'python.exe', 'status': scheduler.psutil.STATUS_RUNNING}

    def create_time(self): return self._created
    def cmdline(self): return self._command
    def cwd(self): return str(self._root)
    def is_running(self): return True
    def status(self): return scheduler.psutil.STATUS_RUNNING


class SchedulerV3Tests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='mp2-v3-scheduler-synthetic-')
        self.root = Path(self.temporary.name).resolve()
        self.out = self.root / 'results/research_v3'
        self.out.mkdir(parents=True)
        self.now = datetime(2026, 10, 29, 16, 22, 42, tzinfo=timezone.utc).timestamp()
        self.source = {'source_files': {'code/train_v3.py': 'source-A'},
                       'train_and_fold_files': {'code/data/manifest.json': 'manifest-A',
                                                'code/data/gsm8k_train.jsonl': 'data-A',
                                                'results/research_v3/common_folds.json': 'folds-A'},
                       'git_commit': 'synthetic-commit'}
        self.events = []
        self.patchers = [
            patch.object(scheduler, 'ROOT', self.root), patch.object(scheduler, 'OUT', self.out),
            patch.object(scheduler, 'QUEUE', self.out / 'campaign_queue.json'),
            patch.object(scheduler, 'LOCK', self.root / 'tmp/research_v3/scheduler.lock'),
            patch.object(scheduler, 'PYTHON', Path(sys.executable)),
            patch.object(scheduler, 'provenance', side_effect=lambda: deepcopy(self.source)),
            patch.object(scheduler, 'event', side_effect=lambda kind, **kw: self.events.append((kind, kw))),
            patch.object(scheduler.time, 'time', return_value=self.now + 1),
        ]
        for p in self.patchers: p.start()
        self.plan = {'schema': 'mp2-campaign-budget-v3',
                     'started_at_utc': self.iso(self.now), 'research_end_utc': self.iso(self.now + 2000),
                     'deadline_utc': self.iso(self.now + 20000), 'deadline_buffer_hours': 1,
                     'maximum_wall_seconds': 1000, 'maximum_unique_configurations': 96,
                     'maximum_repair_retries_per_experiment': 2,
                     'maximum_concurrent_training': 1, 'compute_threads_per_training': 1}
        self.write(self.out / 'campaign_budget.json', self.plan)

    def tearDown(self):
        for p in reversed(self.patchers): p.stop()
        self.temporary.cleanup()

    @staticmethod
    def iso(timestamp):
        return datetime.fromtimestamp(timestamp, timezone.utc).isoformat()

    @staticmethod
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding='utf-8')
        return path

    def register(self, round_name='round_a', names=('mlp_a', 'mlp_b'), **extra):
        value = {'round': round_name, 'hypothesis': 'Synthetic controlled optimizer comparison',
                 'configs': [{'experiment_id': name, 'family': 'mlp', 'seed': 3036900957} for name in names],
                 **extra}
        path = self.write(self.root / 'registrations' / (round_name + '.json'), value)
        scheduler.register(path)
        return scheduler.load_queue()

    def save(self, queue):
        scheduler.save_queue(queue)

    def summary(self, job, *, folds=(0, 1, 2), provenance=None):
        folder = self.out / job['experiment_id']
        folder.mkdir(parents=True, exist_ok=True)
        np.savez(folder / 'oof.npz', scores=np.zeros((4755, 4)),
                 chosen=np.zeros(4755, dtype=np.int64), fold=np.arange(4755) % 3)
        reports = []
        for f in folds:
            score_path = folder / ('fold' + str(f) + '-scores.npy')
            np.save(score_path, np.zeros((1585, 4)))
            asset_folder = self.root / 'artifacts/research_v3' / job['experiment_id'] / ('fold' + str(f))
            asset_folder.mkdir(parents=True, exist_ok=True)
            np.savez(asset_folder / 'weights.npz', weights=np.zeros(4))
            self.write(asset_folder / 'config.json', {'synthetic': True})
            reports.append({'fold': f, 'status': 'complete', 'n': 1585,
                            'checkpoint': asset_folder.relative_to(self.root).as_posix(),
                            'output_sha256': {score_path.name: scheduler.sha(score_path)},
                            'asset_sha256': {name: scheduler.sha(asset_folder / name)
                                             for name in ('weights.npz', 'config.json')}})
        registered = deepcopy(self.source if provenance is None else provenance)
        data = deepcopy(registered['train_and_fold_files'])
        folds_sha = data.pop('results/research_v3/common_folds.json', None)
        actual = {'source_files': registered['source_files'], 'data_files': data,
                  'folds_sha256': folds_sha, 'git_commit': registered.get('git_commit')}
        result = {'config': deepcopy(job['config']), 'folds': reports,
                  'status': 'completed', 'n': 4755, 'hard_accuracy': 50., 'accuracy': 70.,
                  'provenance': actual, 'oof_sha256': scheduler.sha(folder / 'oof.npz')}
        self.write(folder / 'summary.json', result)
        return result

    def assert_rejected_summary(self, job):
        try:
            value = scheduler.summary_valid(job)
        except (ValueError, RuntimeError, KeyError):
            return
        self.assertIsNone(value, 'Invalid or stale summary was accepted as completion evidence')

    def test_budget_uses_original_absolute_anchor_and_delivery_buffer(self):
        before = (self.out / 'campaign_budget.json').read_bytes()
        self.assertEqual(scheduler.budget()[1], self.now + 1000)
        with patch.object(scheduler.time, 'time', return_value=self.now + 999999):
            self.assertEqual(scheduler.budget()[1], self.now + 1000)
        self.assertEqual((self.out / 'campaign_budget.json').read_bytes(), before)
        self.plan['deadline_utc'] = self.iso(self.now + 4000)
        self.write(self.out / 'campaign_budget.json', self.plan)
        self.assertEqual(scheduler.budget()[1], self.now + 400)

    def test_registration_rejects_closed_expired_and_over_capacity(self):
        with patch.object(scheduler.time, 'time', return_value=self.now + 1001):
            with self.assertRaisesRegex(RuntimeError, 'budget'):
                self.register()
        self.assertFalse(scheduler.QUEUE.exists())
        self.write(self.out / 'SEARCH_CLOSED.json', {'closed': True})
        with self.assertRaisesRegex(RuntimeError, 'closed'):
            self.register()
        (self.out / 'SEARCH_CLOSED.json').unlink()
        self.plan['maximum_unique_configurations'] = 3  # Two jobs plus the actual timing pilot.
        self.write(self.out / 'campaign_budget.json', self.plan)
        self.register()
        before = scheduler.QUEUE.read_bytes()
        with self.assertRaisesRegex(ValueError, 'configuration budget'):
            self.register('round_b', ('mlp_c', 'mlp_d'))
        self.assertEqual(scheduler.QUEUE.read_bytes(), before)

    def test_fixed_fusions_and_pilot_share_the_unique_configuration_cap(self):
        registration = self.write(self.root / 'docs/research_v3/rounds/fusion_a.json',
            {'experiment_id': 'fusion_a', 'status': 'registered',
             'purpose': 'single_fixed_equal_probability_oof'})
        entry = {'experiment_id': 'fusion_a', 'registration': registration.relative_to(self.root).as_posix(),
                 'registration_sha256': scheduler.sha(registration)}
        registry = self.out / 'fusion_registry.jsonl'
        registry.write_text(json.dumps(entry) + '\n', encoding='utf-8')
        self.plan['maximum_unique_configurations'] = 3  # Fusion + pilot leave only one slot.
        self.write(self.out / 'campaign_budget.json', self.plan)
        with self.assertRaisesRegex(ValueError, 'configuration budget'):
            self.register()
        self.assertFalse(scheduler.QUEUE.exists())
        self.plan['maximum_unique_configurations'] = 4
        self.write(self.out / 'campaign_budget.json', self.plan)
        self.assertEqual(len(self.register()['jobs']), 2)
        self.assertEqual(scheduler.external_configuration_ids(), {'fusion_a'})
        registration.write_text('{}', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'identity/hash'):
            scheduler.external_configuration_ids()

    def test_external_fusion_cannot_reuse_a_training_identity(self):
        registration = self.write(self.root / 'docs/research_v3/rounds/mlp_a.json',
            {'experiment_id': 'mlp_a', 'status': 'registered',
             'purpose': 'single_fixed_equal_probability_oof'})
        entry = {'experiment_id': 'mlp_a', 'registration': registration.relative_to(self.root).as_posix(),
                 'registration_sha256': scheduler.sha(registration)}
        (self.out / 'fusion_registry.jsonl').write_text(json.dumps(entry) + '\n', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'distinct identities'):
            self.register()
        self.assertFalse(scheduler.QUEUE.exists())

    def test_registration_is_immutable_and_rejects_holdout_background_paths(self):
        queue = self.register()
        before = scheduler.QUEUE.read_bytes()
        with self.assertRaisesRegex(ValueError, 'already registered'):
            self.register()
        self.assertEqual(scheduler.QUEUE.read_bytes(), before)
        for key, value in [('input_path', 'validation.jsonl'), ('test_path', 'forbidden'),
                           ('input_path', 'leaderboard.json'), ('input_path', 'ranking_background.json')]:
            payload = {'round': 'forbidden', 'hypothesis': 'Must reject', 'configs': [
                {'experiment_id': 'new_a', key: value}, {'experiment_id': 'new_b'}]}
            path = self.write(self.root / 'registrations/forbidden.json', payload)
            with self.assertRaisesRegex(ValueError, 'Held-out/ranking'):
                scheduler.register(path)
        self.assertEqual(scheduler.QUEUE.read_bytes(), before)
        self.assertEqual(len(queue['jobs']), 2)

    def test_identity_requires_pid_creation_and_command(self):
        process = FakeProcess(self.root)
        identity = {'pid': process.pid, 'create_time': process.create_time(),
                    'command_fragment': 'code/configs/research_v3/mlp_a.json'}
        with patch.object(scheduler.psutil, 'Process', return_value=process):
            self.assertIs(scheduler.same_process(identity), process)
            self.assertIsNone(scheduler.same_process({**identity, 'create_time': 99}))
            self.assertIsNone(scheduler.same_process({**identity, 'command_fragment': 'different_job.json'}))
        with patch.object(scheduler.psutil, 'Process', side_effect=scheduler.psutil.NoSuchProcess(process.pid)):
            self.assertIsNone(scheduler.same_process(identity))

    def test_active_lock_refuses_and_stale_lock_is_preserved(self):
        previous = {'pid': 999999, 'create_time': 0., 'command_fragment': 'old'}
        self.write(scheduler.LOCK, previous)
        with patch.object(scheduler, 'same_process', return_value=object()):
            with self.assertRaisesRegex(RuntimeError, 'still running'):
                with scheduler.SchedulerLock(): pass
        self.assertEqual(scheduler.read(scheduler.LOCK), previous)
        with patch.object(scheduler, 'same_process', return_value=None):
            with scheduler.SchedulerLock():
                self.assertTrue(scheduler.LOCK.exists())
        self.assertFalse(scheduler.LOCK.exists())
        stale = list(scheduler.LOCK.parent.glob('scheduler.stale.*.json'))
        self.assertEqual(len(stale), 1)
        self.assertEqual(scheduler.read(stale[0]), previous)

    def test_incomplete_three_fold_summary_never_completes(self):
        job = self.register()['jobs'][0]
        for folds in ((0,), (0, 1), (0, 1, 1)):
            self.summary(job, folds=folds)
            self.assert_rejected_summary(job)
        self.summary(job)
        (self.out / job['experiment_id'] / 'oof.npz').unlink()
        self.assert_rejected_summary(job)

    def test_summary_from_changed_source_or_data_is_rejected(self):
        job = self.register()['jobs'][0]
        for key in ('source_files', 'train_and_fold_files'):
            stale = deepcopy(self.source)
            stale[key] = {'other': 'stale-content'}
            self.summary(job, provenance=stale)
            self.assert_rejected_summary(job)

    def test_complete_authentic_three_fold_summary_is_accepted_without_retraining(self):
        queue = self.register()
        job = queue['jobs'][0]
        self.summary(job)
        evidence = scheduler.summary_valid(job)
        self.assertIsInstance(evidence, dict)
        self.assertEqual(evidence['oof_sha256'], scheduler.sha(self.out / job['experiment_id'] / 'oof.npz'))
        queue['jobs'][1]['status'] = 'failed'
        self.save(queue)
        with patch.object(scheduler.subprocess, 'Popen', side_effect=AssertionError('Completed experiment retrained')):
            scheduler.run()
        self.assertEqual(scheduler.load_queue()['jobs'][0]['status'], 'completed')

    def test_completion_rejects_tampered_outputs_assets_and_oof(self):
        job = self.register()['jobs'][0]
        for relative in ('fold0-scores.npy', 'oof.npz', 'asset'):
            result = self.summary(job)
            if relative == 'asset':
                target = self.root / result['folds'][0]['checkpoint'] / 'weights.npz'
            else:
                target = self.out / job['experiment_id'] / relative
            with target.open('ab') as stream:
                stream.write(b'tampered')
            self.assert_rejected_summary(job)

    def test_completion_rejects_three_ids_with_an_incomplete_fold(self):
        job = self.register()['jobs'][0]
        for field, value in (('status', 'running'), ('n', 1584)):
            result = self.summary(job)
            result['folds'][1][field] = value
            self.write(self.out / job['experiment_id'] / 'summary.json', result)
            self.assert_rejected_summary(job)

    def test_completion_rejects_path_escape_before_reading_external_file(self):
        job = self.register()['jobs'][0]
        result = self.summary(job)
        result['folds'][0]['output_sha256'] = {'../../outside.npy': 'not-read'}
        self.write(self.out / job['experiment_id'] / 'summary.json', result)
        self.assert_rejected_summary(job)

    def test_joint_queue_and_config_mutation_cannot_bypass_registered_digest(self):
        queue = self.register()
        job = queue['jobs'][0]
        job['config']['seed'] = 999
        self.write(self.root / job['config_path'], job['config'])
        self.save(queue)
        with patch.object(scheduler.subprocess, 'Popen', side_effect=AssertionError('Mutated config reached trainer spawn')):
            with self.assertRaisesRegex((RuntimeError, ValueError), 'config|digest|hash|alter'):
                scheduler.run()

    def test_unrecorded_live_project_trainer_prevents_duplicate_spawn(self):
        self.register()
        orphan = FakeProcess(self.root)
        with patch.object(scheduler.psutil, 'process_iter', return_value=[orphan]), \
             patch.object(scheduler.subprocess, 'Popen', side_effect=AssertionError('Existing trainer was duplicated')):
            with self.assertRaisesRegex(RuntimeError, 'trainer|training|process|running'):
                scheduler.run()

    def test_changed_registered_source_or_data_rejects_launch(self):
        self.register()
        original = deepcopy(self.source)
        for key in ('source_files', 'train_and_fold_files'):
            self.source = deepcopy(original)
            self.source[key] = {'changed': 'content'}
            with patch.object(scheduler.subprocess, 'Popen', side_effect=AssertionError('Changed source reached spawn')):
                with self.assertRaisesRegex(RuntimeError, 'Source/data|source|data'):
                    scheduler.run()

    def test_absolute_expiry_prunes_pending_without_spawning(self):
        self.register()
        with patch.object(scheduler.time, 'time', return_value=self.now + 1001), \
             patch.object(scheduler.subprocess, 'Popen', side_effect=AssertionError('Expired budget launched training')):
            scheduler.run()
        jobs = scheduler.load_queue()['jobs']
        self.assertTrue(all(job['status'] == 'pruned' for job in jobs))
        self.assertTrue(all(job['stop_reason'] == 'absolute_campaign_budget' for job in jobs))

    def test_monitor_enforces_real_attempt_and_absolute_wall_budgets(self):
        for mode in ('attempt', 'campaign'):
            queue = self.register('round_' + mode, (mode + '_a', mode + '_b'), maximum_attempt_wall_seconds=30)
            job = queue['jobs'][-2]
            job['status'] = 'running'
            job['attempts'] = [{'pid': 12345, 'create_time': 12.5, 'started_timestamp': self.now,
                                'command_fragment': job['config_path']}]
            instant = self.now + (31 if mode == 'attempt' else 1001)
            with patch.object(scheduler.time, 'time', return_value=instant), \
                 patch.object(scheduler, 'same_process', return_value=object()), \
                 patch.object(scheduler, 'terminate_owned') as terminate, \
                 patch.object(scheduler, 'summary_valid', return_value=None):
                scheduler.monitor(job, queue)
                terminate.assert_called_once()
            self.assertEqual(job['status'], 'pruned')
            self.assertEqual(job['attempts'][-1]['stop_reason'],
                             'attempt_wall_budget' if mode == 'attempt' else 'absolute_campaign_budget')

    def test_retry_cap_reason_data_guard_and_failure_log_preservation(self):
        queue = self.register()
        job = queue['jobs'][0]
        job['status'] = 'failed'
        self.save(queue)
        log = self.out / 'scheduler_logs/mlp_a.attempt1.log'
        log.parent.mkdir(parents=True)
        log.write_bytes(b'original failure evidence\n')
        with self.assertRaisesRegex(ValueError, 'explanation'):
            scheduler.retry('mlp_a', '')
        for i in range(2):
            scheduler.retry('mlp_a', 'Synthetic documented engineering repair ' + str(i))
            queue = scheduler.load_queue()
            self.assertEqual(queue['jobs'][0]['repair_retries'], i + 1)
            queue['jobs'][0]['status'] = 'failed'
            self.save(queue)
        with self.assertRaisesRegex(ValueError, 'cap|unavailable'):
            scheduler.retry('mlp_a', 'Third retry forbidden')
        self.assertEqual(log.read_bytes(), b'original failure evidence\n')
        queue = scheduler.load_queue()
        queue['jobs'][1]['status'] = 'failed'
        self.save(queue)
        self.source['train_and_fold_files'] = {'different': 'forbidden'}
        with self.assertRaisesRegex(ValueError, 'Data/folds'):
            scheduler.retry('mlp_b', 'Cannot change training partition')

    def test_previous_log_is_never_overwritten(self):
        self.register()
        log = self.out / 'scheduler_logs/mlp_a.attempt1.log'
        log.parent.mkdir(parents=True)
        log.write_bytes(b'preserve this crash log\n')
        with patch.object(scheduler.subprocess, 'Popen', side_effect=AssertionError('Log collision reached spawn')):
            with self.assertRaisesRegex(RuntimeError, 'log'):
                scheduler.run()
        self.assertEqual(log.read_bytes(), b'preserve this crash log\n')


if __name__ == '__main__':
    unittest.main(verbosity=2)
