"""Temporary synthetic fixtures only; no course arrays or live registry access."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('one_fusion', REPO / 'scripts/register_reduced_tabm_attention_fusion_v3.py')
f = importlib.util.module_from_spec(spec)
spec.loader.exec_module(f)


def metric(correct, n):
    ns = [n // 3 + (i < n % 3) for i in range(3)]
    cs = [correct // 3 + (i < correct % 3) for i in range(3)]
    return {'correct': correct, 'n': n, 'pooled_accuracy': 100 * correct / n,
            'fold_mean': 100 * correct / n, 'fold_sd': 0.,
            'folds': [{'fold': i, 'correct': cs[i], 'n': ns[i], 'accuracy': 100 * cs[i] / ns[i]} for i in range(3)]}


def metrics(hard, single=388):
    return {'overall': metric(hard + 2641, 4755), 'hard': metric(hard, 2114), 'single_positive': metric(single, 711)}


class FixedFusionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='synthetic-fusion-r12-')
        self.root = Path(self.tmp.name)
        self.patchers = [patch.object(f, 'commit', return_value='a' * 40),
                         patch.object(f.psutil, 'process_iter', return_value=[])]
        for p in self.patchers: p.start()
        self.make_fixture()
        (self.root / '.venv/Lib/site-packages').mkdir(parents=True)

    def tearDown(self):
        for p in reversed(self.patchers): p.stop()
        self.tmp.cleanup()

    def write(self, name, value):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
        return f.sha(path)

    def make_fixture(self):
        for name in [f.SELF, f.DIAGNOSTIC]:
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('# synthetic source only\n', encoding='utf-8')
        sources = {f.DIAGNOSTIC: f.sha(self.root / f.DIAGNOSTIC)}
        inputs, members = {}, []
        for index, method in enumerate(f.METHODS):
            hashes = {}
            for name in ('oof.npz', 'summary.json', 'registration.json'):
                path = f.BASE + method.split(':')[1] + '/' + name
                hashes[name] = self.write(path, {'synthetic': method, 'file': name})
                inputs[path] = hashes[name]
            members.append({'method': method, 'input_sha256': hashes, 'config': {'seed': 3036900957},
                            'metrics': metrics(1668 if index == 0 else 1685, 364 if index == 0 else 388)})
        gate = {'status': 'complete', 'common_labels_groups_folds_verified': True,
                'fusion': {'status': 'not_run'}, 'methods': members, 'folds_sha256': 'b' * 64,
                'complementarity': [{'methods': f.METHODS, 'gate_passed': True,
                    'overall': {'member_unique_correct': [43, 60]},
                    'hard': {'n': 2114, 'oracle_correct': 1728, 'member_correct': [1668, 1685]}}]}
        gate_hash = self.write(f.GATE, gate)
        self.write(f.COMPLETION, {'status': 'complete', 'only_reciprocal_gate': f.METHODS,
            'fusion_run': False, 'validation_test_ranking_read': False,
            'source_sha256': sources, 'input_sha256': inputs, 'output_sha256': {f.GATE: gate_hash}})
        old_rows, old_results = [], []
        for name in ('v3_fusion_attention_tabm_s0', 'v3_fusion_attention_realmlp_s0'):
            formal = 'docs/research_v3/rounds/' + name + '.json'
            digest = self.write(formal, {'experiment_id': name, 'synthetic': True})
            old_rows.append({'experiment_id': name, 'registration': formal, 'registration_sha256': digest})
            diagnostic = f.BASE + name + '/diagnostic.json'
            old_metric = metrics(1667)
            dhash = self.write(diagnostic, {'fusion': {'status': 'complete', 'metrics': old_metric}})
            old_results.append({'experiment_id': name, 'diagnostic_path': diagnostic,
                                'diagnostic_sha256': dhash, 'metrics': old_metric,
                                'immutable_extra': [1, {'x': 'preserve'}]})
        reg = self.root / f.REGISTRY
        reg.parent.mkdir(parents=True, exist_ok=True)
        reg.write_text(''.join(json.dumps(r) + '\n' for r in old_rows), encoding='utf-8')
        self.write(f.OLD_ANALYSIS, {'schema': 'mp2-fixed-fusion-analysis-v3', 'status': 'completed',
            'configurations': 2, 'new_training_fits': 0, 'results': old_results,
            'registry_sha256': f.sha(reg), 'exceeds_registered_half_point': False,
            'terminal_observed_wall_seconds': {old_rows[0]['experiment_id']: 1.9, old_rows[1]['experiment_id']: 1.7},
            'wall_measurement_source': 'synthetic receipts'})
        now = datetime.now(timezone.utc)
        self.write(f.BASE + 'campaign_budget.json', {'started_at_utc': now.isoformat(),
            'research_end_utc': (now + timedelta(hours=1)).isoformat(),
            'deadline_utc': (now + timedelta(days=2)).isoformat(), 'maximum_wall_seconds': 3600,
            'deadline_buffer_hours': 8, 'maximum_unique_configurations': 96})
        self.write(f.BASE + 'campaign_queue.json', {'jobs': [{'experiment_id': 'synthetic_train', 'status': 'completed'}]})
        hard = metric(1700, 2114)
        self.write(f.BASE + 'synthetic_train/summary.json', {'status': 'complete', 'n': 4755,
            'config': {'seed': 3036900957}, 'hard_correct': 1700, 'hard_n': 2114,
            'hard_accuracy_fold_mean': sum(row['accuracy'] for row in hard['folds']) / 3,
            'folds': [{'fold': row['fold'], 'status': 'complete', 'n': 1585,
                       'hard_correct': row['correct'], 'hard_n': row['n']} for row in hard['folds']]})
        (self.root / f.LEDGER).write_text(json.dumps({'round': 'old', 'configurations': 2}) + '\n', encoding='utf-8')

    def fake_child(self, *args, **kwargs):
        spec = f.read(self.root / f.FORMAL)
        folder = self.root / (f.BASE + f.NAME)
        folder.mkdir()
        (folder / 'fusion-oof.npz').write_bytes(b'synthetic-not-a-real-archive')
        (folder / 'fusion-registration.json').write_bytes((self.root / f.FORMAL).read_bytes())
        actual = {'status': 'complete', 'methods': f.METHODS, 'weights': f.WEIGHTS,
            'registration_file': 'fusion-registration.json', 'registration_sha256': f.sha(self.root / f.FORMAL),
            'oof_sha256': f.sha(folder / 'fusion-oof.npz'), 'metrics': metrics(1684, 389),
            'assets_per_fold': [{'total_member_uncompressed_bytes': 1234} for _ in range(3)]}
        self.write(f.BASE + f.NAME + '/diagnostic.json', {'status': 'complete',
            'validation_test_rank_read': False, 'rejected_reads': [], 'fusion': actual,
            'source_sha256': spec['source_sha256'][f.DIAGNOSTIC], 'folds_sha256': spec['folds_sha256']})
        return SimpleNamespace(returncode=0)

    def test_prepare_has_no_registry_or_fusion_side_effect(self):
        old = (self.root / f.REGISTRY).read_bytes()
        result = f.prepare(self.root)
        self.assertEqual(result['status'], 'prepared_not_registered')
        self.assertEqual((self.root / f.REGISTRY).read_bytes(), old)
        self.assertFalse((self.root / f.FORMAL).exists())
        self.assertFalse((self.root / (f.BASE + f.NAME)).exists())
        with self.assertRaises(FileExistsError): f.prepare(self.root)

    def test_gate_fail_or_hash_tamper_rejected(self):
        gate = f.read(self.root / f.GATE)
        gate['complementarity'][0]['overall']['member_unique_correct'][0] = 9
        self.write(f.GATE, gate)
        with self.assertRaises(ValueError): f.prepare(self.root)
        complete = f.read(self.root / f.COMPLETION)
        complete['output_sha256'][f.GATE] = f.sha(self.root / f.GATE)
        self.write(f.COMPLETION, complete)
        with self.assertRaisesRegex(ValueError, 'gate failed'): f.prepare(self.root)

    def test_fixed_order_weights_policy_and_sources_enforced(self):
        result = f.prepare(self.root)
        for key, bad in [('methods', list(reversed(f.METHODS))), ('weights', [.4, .6]), ('decision_policy', {})]:
            mutated = deepcopy(result); mutated[key] = bad
            self.write(f.PREPARED, mutated)
            with self.assertRaises(ValueError): f.prepared(self.root)
        self.write(f.PREPARED, result)
        (self.root / f.SELF).write_text('# changed\n')
        with self.assertRaises(ValueError): f.prepared(self.root)

    def test_active_scheduler_lock_and_pending_queue_rejected(self):
        f.prepare(self.root)
        lock = self.root / 'tmp/research_v3/scheduler.lock'
        lock.parent.mkdir(parents=True); lock.write_text('{"owner":"other"}')
        before = lock.read_bytes()
        with self.assertRaises(FileExistsError): f.register(self.root)
        self.assertEqual(lock.read_bytes(), before)
        lock.unlink()
        self.write(f.BASE + 'campaign_queue.json', {'jobs': [{'experiment_id': 'x', 'status': 'pending'}]})
        with self.assertRaises(RuntimeError): f.register(self.root)
        self.assertFalse(lock.exists())

    def test_closed_clock_and_configuration_cap(self):
        f.prepare(self.root)
        self.write(f.BASE + 'SEARCH_CLOSED.json', {'closed': True})
        with self.assertRaises(RuntimeError): f.register(self.root)
        (self.root / (f.BASE + 'SEARCH_CLOSED.json')).unlink()
        # Budget changes are forbidden after preparation, not silently accepted.
        budget = f.read(self.root / (f.BASE + 'campaign_budget.json'))
        budget['maximum_unique_configurations'] = 2
        self.write(f.BASE + 'campaign_budget.json', budget)
        with self.assertRaises(ValueError): f.register(self.root)

    def test_register_append_preserves_bytes_and_resume_no_duplicate(self):
        f.prepare(self.root)
        before = (self.root / f.REGISTRY).read_bytes()
        receipt = f.register(self.root)
        self.assertTrue((self.root / f.REGISTRY).read_bytes().startswith(before))
        self.assertEqual(receipt['unique_configurations_including_pilot'], 5)
        self.assertEqual(f.register(self.root), receipt)
        self.assertEqual(len(f.lines(self.root / f.REGISTRY)), 3)

    def test_reject_summary_before_actual_execution(self):
        f.prepare(self.root); f.register(self.root)
        with self.assertRaises(FileNotFoundError): f.summarize(self.root)
        self.assertFalse((self.root / f.FINAL_ANALYSIS).exists())

    def test_actual_receipt_merge_preserves_old_rows_and_adds_one_round(self):
        f.prepare(self.root); f.register(self.root)
        old = f.read(self.root / f.OLD_ANALYSIS)
        with patch.object(f.subprocess, 'run', side_effect=self.fake_child) as run:
            execution = f.execute(self.root)
            self.assertEqual(run.call_count, 1)
            self.assertEqual(f.execute(self.root), execution)
            self.assertEqual(run.call_count, 1)
        receipt = f.summarize(self.root)
        aggregate = f.read(self.root / f.FINAL_ANALYSIS)
        self.assertEqual(aggregate['results'][:2], old['results'])
        self.assertEqual(f.read(self.root / f.OLD_ANALYSIS), old)
        self.assertEqual(aggregate['configurations'], 3)
        self.assertEqual(aggregate['results'][2]['hard_delta_questions_vs_attention_control'], -1)
        self.assertFalse(aggregate['results'][2]['local_retention_rule_passed'])
        self.assertEqual(receipt['new_configurations'], 1)
        self.assertGreater(receipt['child_wall_seconds'], 0)
        self.assertEqual(f.summarize(self.root), receipt)
        self.assertEqual(len(f.lines(self.root / f.LEDGER)), 2)
        self.assertEqual(aggregate['new_round_record']['global_reference'], 'research_v3:synthetic_train')
        chosen = f.read(self.root / f.FORMAL)['global_incumbent_before_evaluation']['chosen']
        expected = aggregate['results'][2]['metrics']['hard']['fold_mean'] - chosen['hard_fold_mean']
        self.assertEqual(aggregate['new_round_record']['increment_above_global_reference_pp'], expected)
        # Future stability records must not invalidate the historical receipt.
        f.atomic_append(self.root / f.LEDGER, (self.root / f.LEDGER).read_bytes(), {'round': 'r13', 'configurations': 2})
        self.assertEqual(f.summarize(self.root), receipt)
        self.assertEqual(len(f.lines(self.root / f.LEDGER)), 3)

    def test_failed_execution_kept_and_not_replayed(self):
        f.prepare(self.root); f.register(self.root)
        with patch.object(f.subprocess, 'run', return_value=SimpleNamespace(returncode=3)) as run:
            with self.assertRaises(RuntimeError): f.execute(self.root)
            with self.assertRaises(RuntimeError): f.execute(self.root)
            self.assertEqual(run.call_count, 1)
        self.assertEqual(f.read(self.root / f.EXECUTION / 'execution.json')['status'], 'failed')
        self.assertFalse((self.root / f.FINAL_ANALYSIS).exists())

    def test_changed_output_refuses_completed_execution_and_summary(self):
        f.prepare(self.root); f.register(self.root)
        with patch.object(f.subprocess, 'run', side_effect=self.fake_child): f.execute(self.root)
        (self.root / (f.BASE + f.NAME + '/fusion-oof.npz')).write_bytes(b'tampered')
        with self.assertRaises(ValueError): f.execute(self.root)
        with self.assertRaises(ValueError): f.summarize(self.root)

    def test_decision_uses_strict_hard_and_no_overall_loss(self):
        control = metrics(1685)
        self.assertFalse(f.decision(metrics(1685), control)[1])
        self.assertTrue(f.decision(metrics(1686), control)[1])
        candidate = metrics(1686); candidate['overall']['correct'] = 4325
        self.assertFalse(f.decision(candidate, control)[1])

    def test_cap_applies_before_new_registration(self):
        budget = f.read(self.root / (f.BASE + 'campaign_budget.json'))
        budget['maximum_unique_configurations'] = 4
        self.write(f.BASE + 'campaign_budget.json', budget)
        f.prepare(self.root)
        with self.assertRaises(RuntimeError): f.register(self.root)
        self.assertFalse((self.root / f.FORMAL).exists())

    def test_incumbent_queue_cannot_change_between_register_and_evaluate(self):
        f.prepare(self.root); f.register(self.root)
        self.write(f.BASE + 'campaign_queue.json', {'jobs': [{'experiment_id': 'synthetic_train', 'status': 'completed'}], 'new': True})
        with patch.object(f.subprocess, 'run') as run:
            with self.assertRaises(ValueError): f.execute(self.root)
            run.assert_not_called()


if __name__ == '__main__':
    unittest.main(verbosity=2)
