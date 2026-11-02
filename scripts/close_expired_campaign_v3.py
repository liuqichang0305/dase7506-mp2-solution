"""Seal an actually expired campaign without retrying or inventing completed fits."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code/tools'))
import confirm_validation_v3 as gate
import run_campaign_v3 as campaign


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_new(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


def main():
    out = ROOT / 'results/research_v3'
    target = out / 'SEARCH_CLOSED.json'
    if target.exists():
        raise FileExistsError('Preserve the existing search closure')
    now = datetime.now(timezone.utc)
    budget = read(out / 'campaign_budget.json')
    if now <= datetime.fromisoformat(budget['research_end_utc'].replace('Z', '+00:00')):
        raise ValueError('This entry point only closes an actually expired budget')
    with gate.exclusive_evaluation():
        queue = campaign.load_queue()
        if any(j['status'] not in ('completed', 'failed', 'pruned') for j in queue['jobs']):
            raise ValueError('First reconcile the live scheduler; never relabel pending work complete')
        complete = [j for j in queue['jobs'] if j['status'] == 'completed']
        incomplete = [j for j in queue['jobs'] if j['status'] != 'completed']
        for job in complete:
            result = campaign.summary_valid(job)
            if result != job['result']:
                raise ValueError('Completed summary differs from its scheduler receipt')
        for name in ('VALIDATION_OPENED.json', 'FINALISTS_SELECTION.json'):
            if (out / name).exists():
                raise ValueError('Unexpected earlier validation/selection; preserve and investigate')
        if (ROOT / 'submission/v3/TEST_OPENED.json').exists():
            raise ValueError('Unexpected new test opening')
        evidence = ['campaign_budget.json', 'campaign_queue.json', 'experiment_registry.jsonl',
                    'fusion_registry.jsonl', 'round_decisions.jsonl', 'training_events.jsonl',
                    'fixed_fusion_analysis_complete.json']
        record = {
            'schema': 'mp2-v3-expired-search-closure-v1', 'search_closed': True,
            'created_at_utc': now.isoformat(), 'reason': 'absolute_research_budget_and_course_score_deadline_expired',
            'research_end_utc': budget['research_end_utc'], 'course_deadline_utc': budget['deadline_utc'],
            'completed_configurations': len(complete), 'completed_outer_fits': len(complete) * 3,
            'registered_training_configurations': len(queue['jobs']),
            'fixed_fusion_configurations': len(campaign.external_configuration_ids()),
            'pilot_configurations': 1,
            'unique_configurations_including_pilot_and_fusion': len(queue['jobs']) + len(campaign.external_configuration_ids()) + 1,
            'incomplete': [{'experiment_id': j['experiment_id'], 'status': j['status'],
                            'stop_reason': j.get('stop_reason'), 'attempts': j['attempts']} for j in incomplete],
            'stability_complete': False, 'new_validation_opened': False, 'new_test_opened': False,
            'no_further_search_or_repair_retries': True,
            'allowed_remaining_scope': 'At most two existing base957 recipes may receive locked final-all-train fitting and bounded validation solely for an explicitly post-deadline research archive. No new model search, official V3 test, score update or replacement of historical submitted assets. Complete research review/report/PR and separately authorized released peer review.',
            'monitor_gap_disclosure': 'Recorded attempts span long wall-clock gaps and exceeded nominal attempt or campaign deadlines before the monitor next acted. Their exact durations are retained. The cause is not established, continuous CPU activity is not asserted, and a hard real-time cutoff is not claimed. Four stability configurations remain pruned, not completed.',
            'evidence_sha256': {f'results/research_v3/{p}': sha(out / p) for p in evidence},
            'closure_source_sha256': sha(Path(__file__)),
        }
        save_new(target, record)
        blocked = ROOT / 'submission/v3/FORMAL_SUBMISSION_BLOCKED.json'
        save_new(blocked, {
            'schema': 'mp2-v3-post-deadline-block-v1', 'created_at_utc': now.isoformat(),
            'reason': 'Official MP2 score deadline passed and published phase is review; new scores are late, score edits ignored, and new post-release artifact receipts are not accepted automatically.',
            'rules_audit_path': 'docs/research_v3/RULES_RESUMPTION_20261101.md',
            'rules_audit_sha256': sha(ROOT / 'docs/research_v3/RULES_RESUMPTION_20261101.md'),
            'no_new_test': True, 'no_new_score_update': True, 'no_automatic_artifact_replacement': True,
            'research_only': True, 'missed_deadline': True,
            'prior_issue_number': 294,
            'prior_issue_url': 'https://github.com/xudongwu-0/xudongwu-0.github.io/issues/294',
            'search_closed_sha256': sha(target),
            'existing_deadline_guards_preserved': ['submission/v3/evaluate_final.py', 'submission/v3/submit_final.py'],
        })
    print(json.dumps({'search_closed': True, 'completed': len(complete), 'incomplete': len(incomplete),
                      'formal_score_update_blocked': True, 'new_test_opened': False}))


if __name__ == '__main__':
    main()
