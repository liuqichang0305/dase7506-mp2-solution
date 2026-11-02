"""Synthetic metadata only: no raw inputs, training, PDF or formal evaluation."""
import hashlib
import importlib.util
import json
from pathlib import Path
import unittest

import test_report_inputs_v3 as original_tests

ROOT = Path(__file__).resolve().parents[2]
def load(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT / file)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module

c = load('closeout_inputs_under_test', 'reports/research_v3/prepare_closeout_inputs.py')
b = load('closeout_builder_under_test', 'reports/research_v3/build_closeout.py')


class CloseoutTests(unittest.TestCase):
    def setUp(self):
        self.base = original_tests.ReportInputTests(); self.base.setUp()
        self.root, self.put, self.read = self.base.root, self.base.put, self.base.read
        for path in c.FORBIDDEN:
            (self.root / path).unlink(missing_ok=True)
        val_path = c.BASE + 'validation_summary.json'
        validation = self.read(val_path)
        validation['finalists'] = [{'name': 'primary_s0', 'validation': self.base.metrics}]
        val_hash = self.put(val_path, validation)
        selected = self.read(c.SUB + 'SELECTED_MODEL.json'); selected['validation_summary_sha256'] = val_hash
        selected_hash = self.put(c.SUB + 'SELECTED_MODEL.json', selected)
        fit_path = c.BASE + 'confidence_calibration/report.json'
        fit = self.read(fit_path); fit['source_sha256']['selected_model_json'] = selected_hash
        self.put(fit_path, fit)
        prep_path = c.BASE + 'selected_confidence/preparation.json'
        prep = self.read(prep_path); prep['selected_model_sha256'] = selected_hash
        self.put(prep_path, prep)
        freeze = self.read(c.SUB + 'FREEZE.json')
        for file, key in (('REVIEW_RESULT.json', 'review_result_sha256'), ('REPRODUCTION_RESULT.json', 'reproduction_result_sha256')):
            value = self.read(c.SUB + file); value['selected_model_sha256'] = selected_hash
            freeze[key] = self.put(c.SUB + file, value)
        freeze.update(selected_model_sha256=selected_hash, validation_summary_sha256=val_hash)
        freeze_hash = self.put(c.SUB + 'FREEZE.json', freeze)
        prior_url = 'https://github.com/xudongwu-0/xudongwu-0.github.io/issues/294'
        self.put('submission/SUBMISSION_VERIFICATION.json', {'issue_number': 294, 'issue_url': prior_url,
            'student_id': '3036900957', 'actual_score': {'correct': 348, 'n': 467, 'split': 'test'}})
        blocked_hash = self.put(c.SUB + 'FORMAL_SUBMISSION_BLOCKED.json', {'created_at_utc': '2026-11-02T07:00:00+00:00',
            'missed_deadline': True, 'reason': 'Synthetic expired score phase', 'no_new_test': True, 'no_new_score_update': True})
        freeze.update(research_only=True, formal_test_allowed=False, formal_submission_blocked_sha256=blocked_hash,
                      created_at_utc='2026-11-02T07:30:00+00:00')
        freeze_hash = self.put(c.SUB + 'FREEZE.json', freeze)
        queue = self.read(c.BASE + 'campaign_queue.json')
        incomplete = []
        for i in range(4):
            name = f'partial_repeat_{i}'
            incomplete.append(name)
            log = c.BASE + f'scheduler_logs/{name}.log'
            self.put(log, 'Synthetic preserved partial attempt; no complete run')
            queue['jobs'].append({'experiment_id': name, 'round': 'r13_fixed_seed_stability', 'status': 'pruned',
                'config': {'seed': c.SEED + 1 + i % 2},
                'attempts': [{'observed_wall_seconds': 10001 + i, 'stop_reason': 'attempt_wall_budget', 'log': log}]})
        self.put(c.BASE + 'campaign_queue.json', queue)
        ledger_path = self.root / (c.BASE + 'round_decisions.jsonl')
        with ledger_path.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps({'round': 'r13_fixed_seed_stability', 'configurations': 4, 'results': []}) + '\n')
        rules = 'docs/research_v3/synthetic_rules.md'
        rules_hash = self.put(rules, 'Synthetic saved public-rule audit')
        self.put(c.SUB + 'RESEARCH_CLOSEOUT.json', {'schema': 'mp2-v3-post-deadline-closeout-v1',
            'created_at_utc': '2026-11-02T08:00:00+00:00', 'freeze_sha256': freeze_hash, 'selected_model_sha256': selected_hash,
            'formal_submission_blocked_sha256': blocked_hash, 'rules_audit_path': rules, 'rules_audit_sha256': rules_hash,
            'no_new_test': True, 'no_new_score_update': True, 'research_only': True, 'missed_deadline': True,
            'prior_issue_url': prior_url, 'prior_issue_number': 294, 'incomplete_stability_ids': incomplete,
            'monitor_gap_disclosure': 'Synthetic long monitoring gaps caused late closure. No continuous CPU-time claim is made.'})
        old = self.read(original_tests.r.DEFAULTS['fusion_analysis'])
        first = old['results'][0]
        diagnostic = self.read(first['diagnostic_path'])
        diagnostic['fusion']['metrics']['hard']['n'] = 2100
        first['metrics'] = diagnostic['fusion']['metrics']
        first['diagnostic_sha256'] = self.put(first['diagnostic_path'], diagnostic)
        first['attention_control'] = 'v3:primary_s0'
        first['hard_delta_questions_vs_attention_control'] = -75
        old['results'] = [dict(first, experiment_id=f'synthetic_fixed_fusion_{i}', decision='Reject: synthetic negative result.') for i in range(4)]
        self.put(c.DEFAULTS['fusion_analysis'], old)
        for file in ('reports/research_v3/prepare_report_inputs.py', 'reports/research_v2/build_report.py'):
            self.put(file, (ROOT / file).read_text(encoding='utf-8'))

    def tearDown(self): self.base.tearDown()

    def change_close(self, **values):
        path = c.SUB + 'RESEARCH_CLOSEOUT.json'
        record = self.read(path); record.update(values); self.put(path, record)

    def test_complete_archive_has_ten_pages_honest_status_all_rows_and_chinese_guide(self):
        record = c.prepare(self.root)
        self.assertEqual(len(record['pages']), 10)
        self.assertEqual(record['derived_aggregates']['completed_v3'], 5)
        self.assertEqual(record['derived_aggregates']['historical_v2'], 1)
        self.assertEqual(record['derived_aggregates']['fixed_fusions'], 4)
        self.assertEqual(len(record['derived_aggregates']['incomplete_stability_ids']), 4)
        text = json.dumps(record)
        self.assertIn(c.NOTICE, text)
        self.assertIn('348/467', text)
        self.assertIn('capacity and random initialization', text)
        self.assertIn('No locked V3 candidate improved', text)
        self.assertEqual(record['derived_aggregates']['confidence']['status'], 'complete')
        self.assertIn('IN-SAMPLE', text)
        self.assertNotIn('test_seconds', record['archive_facts'])
        self.assertFalse(set(c.FORBIDDEN) & set(record['evidence_sha256']))
        self.assertIn('没有进行正式 test', c.chinese_guide(record))
        self.assertFalse((self.root / 'code/data').exists())
        b.validate_inputs(record, self.root)

    def test_missing_actual_reproduction_blocks_before_aggregate_read(self):
        (self.root / (c.SUB + 'REPRODUCTION_RESULT.json')).unlink()
        self.put(c.DEFAULTS['analysis'], 'not JSON')
        with self.assertRaisesRegex(ValueError, 'not ready'):
            c.prepare(self.root)

    def test_any_v3_test_execution_record_contradicts_archive(self):
        for name in c.FORBIDDEN:
            self.put(name, {'synthetic': True})
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'conflicts'):
                c.prepare(self.root)
            (self.root / name).unlink()

    def test_deadline_no_score_and_monitoring_claims_are_required(self):
        path = c.SUB + 'RESEARCH_CLOSEOUT.json'; original = self.read(path)
        for key, value in [('created_at_utc', '2026-10-30T01:00:00+00:00'), ('no_new_score_update', False),
                           ('monitor_gap_disclosure', '')]:
            self.put(path, {**original, key: value})
            with self.subTest(key=key), self.assertRaises(ValueError): c.prepare(self.root)

    def test_rehashed_archive_freeze_cannot_allow_formal_test(self):
        path = c.SUB + 'FREEZE.json'
        freeze = self.read(path); freeze['formal_test_allowed'] = True
        self.change_close(freeze_sha256=self.put(path, freeze))
        with self.assertRaisesRegex(ValueError, 'incorrectly allows formal testing'):
            c.prepare(self.root)

    def test_optional_confidence_partial_or_wrong_choice_counts_cannot_be_reported(self):
        path = c.BASE + 'confidence_validation/report.json'
        value = self.read(path); value['raw']['correct'] -= 1
        self.put(path, value)
        with self.assertRaisesRegex(ValueError, 'event counts differ'): c.prepare(self.root)
        (self.root / path).unlink()
        with self.assertRaisesRegex(ValueError, 'confidence evidence is partial'): c.prepare(self.root)

    def test_incomplete_seeds_and_four_actual_fusions_cannot_be_omitted(self):
        self.change_close(incomplete_stability_ids=['partial_repeat_0'])
        with self.assertRaisesRegex(ValueError, 'incomplete-seed disclosure'): c.prepare(self.root)
        self.change_close(incomplete_stability_ids=[f'partial_repeat_{i}' for i in range(4)])
        fusion = self.read(c.DEFAULTS['fusion_analysis']); fusion['results'].pop()
        self.put(c.DEFAULTS['fusion_analysis'], fusion)
        with self.assertRaisesRegex(ValueError, 'four completed'): c.prepare(self.root)

    def test_builder_rejects_self_consistent_page_tamper_and_formal_schema(self):
        record = c.prepare(self.root)
        record['pages'][0]['blocks'][3]['text'] = 'V3 test completed and course score updated.'
        with self.assertRaisesRegex(ValueError, 'actual completed evidence: pages'): b.validate_inputs(record, self.root)
        record['schema'] = 'mp2-v3-final-report-inputs-v1'
        with self.assertRaisesRegex(ValueError, 'separate post-deadline'): b.validate_inputs(record, self.root)

    def test_saved_outputs_are_exclusive_and_do_not_create_pdf(self):
        record = c.prepare(self.root)
        output, chinese = self.root / 'closeout.json', self.root / 'closeout.md'
        c.write_outputs(record, output, chinese)
        before = output.read_bytes()
        with self.assertRaisesRegex(ValueError, 'Preserve'): c.write_outputs(record, output, chinese)
        self.assertEqual(before, output.read_bytes())
        self.assertEqual(list(self.root.rglob('*.pdf')), [])

    def test_old_formal_gates_still_require_completed_test(self):
        with self.assertRaisesRegex(ValueError, 'missing submission/v3/TEST_COMPLETE.json'):
            original_tests.r.prepare(self.root)

    def test_protocol_reference_and_reproduction_instructions_are_explicit(self):
        record = c.prepare(self.root)
        protocol = record['derived_aggregates']['protocol']
        self.assertEqual(protocol, {'train_n': 4755, 'folds': 3, 'outer_fit_n': 3170,
                                   'holdout_n': 1585, 'inner_fit_n': 2536, 'inner_stop_n': 634})
        text = json.dumps(record['pages'])
        self.assertIn('alternate correct answers stay positive', text)
        self.assertIn('local-path preflight is not a fresh-clone entry point', text)
        self.assertIn('separate clone at this exact source commit', text)
        self.assertIn('only the official validation files', text)
        self.assertIn('direct base-Python -S process for code/predict.py', text)
        self.assertIn('--predictions validation-predictions.json --split validation', text)
        self.assertIn('docs/research_v3/REPRODUCTION.md', text)
        self.assertNotIn('python scripts/reproduce_selected_v3.py --attempt', text)
        self.assertIn('Do not copy training state, delete guards, or open test', text)
        self.assertIn(record['archive_facts']['source_commit'], text)
        for paper in ('2410.24210', '2407.04491', '1703.06114', '1706.03762', '2604.15297v1'):
            self.assertIn('https://arxiv.org/abs/' + paper, text)
        folds = self.read(c.BASE + 'common_folds.json')
        folds['folds'][0]['inner_stop'][0] = folds['folds'][0]['inner_fit'][0]
        with self.assertRaisesRegex(ValueError, 'partitions are inconsistent'): c.fold_protocol(folds)

    def test_unstarted_job_uses_queue_reason_without_inventing_an_attempt(self):
        queue = self.read(c.BASE + 'campaign_queue.json')
        queue['jobs'][-1].update(attempts=[], stop_reason='absolute_campaign_budget')
        self.put(c.BASE + 'campaign_queue.json', queue)
        record = c.prepare(self.root)
        text = json.dumps(record['pages'][6])
        self.assertIn('Not started: absolute_campaign_budget', text)
        self.assertNotIn('No reason in the saved attempt', text)
        self.assertEqual(record['derived_aggregates']['recorded_attempt_seconds'],
                         sum(a.get('observed_wall_seconds', 0) for j in queue['jobs'] for a in j.get('attempts', [])))

    def test_fusion_delta_is_bound_to_named_member_not_weaker_attention(self):
        index = {('research_v3', 'att'): {'config': {'family': 'attention'}, 'hard_n': 100, 'hard_correct': 78},
                 ('research_v3', 'tab'): {'config': {'family': 'tabm'}, 'hard_n': 100, 'hard_correct': 80}}
        item = {'experiment_id': 'mixture', 'methods': ['v3:att', 'v3:tab'],
                'attention_control': 'v3:att', 'matched_control': 'v3:tab', 'metrics': {'hard': {'n': 100, 'correct': 80}},
                'hard_delta_questions_vs_attention_control': 2, 'hard_delta_questions_vs_matched_control': 0}
        result = c.fusion_comparison(item, index)
        self.assertEqual((result['attention_delta'], result['matched_delta']), (2, 0))
        item['hard_delta_questions_vs_matched_control'] = 2
        with self.assertRaisesRegex(ValueError, 'matched-control delta'): c.fusion_comparison(item, index)
        path = c.DEFAULTS['fusion_analysis']; fusion = self.read(path)
        fusion['results'][0]['hard_delta_questions_vs_attention_control'] += 1
        self.put(path, fusion)
        with self.assertRaisesRegex(ValueError, 'Attention delta'): c.prepare(self.root)


if __name__ == '__main__': unittest.main(verbosity=2)
