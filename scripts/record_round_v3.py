"""Append one evidence-bound completed-round decision, without fitting a model."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/research_v3'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--round', required=True)
    parser.add_argument('--analysis', type=Path, required=True)
    parser.add_argument('--decision', required=True)
    parser.add_argument('--next', required=True)
    parser.add_argument('--type', choices=('hypothesis', 'stability'), default='hypothesis')
    args = parser.parse_args()
    ledger = OUT / 'round_decisions.jsonl'
    existing = [json.loads(line) for line in ledger.read_text(encoding='utf-8').splitlines()] if ledger.exists() else []
    if any(row['round'] == args.round for row in existing):
        raise ValueError('A prior round decision must be preserved')
    queue = read(OUT / 'campaign_queue.json')
    selected = [j for j in queue['jobs'] if j['round'] == args.round]
    if not selected or any(j['status'] not in ('completed', 'failed', 'pruned') for j in selected):
        raise ValueError('Round still has pending/running work')
    analysis = read(args.analysis)
    ids = {j['experiment_id'] for j in selected}
    rows = [r for r in analysis['rows'] if r['version'] == 'research_v3' and r['experiment_id'] in ids]
    if {r['experiment_id'] for r in rows} != {j['experiment_id'] for j in selected if j['status'] == 'completed'}:
        raise ValueError('Analysis and queue completed results differ')
    previous_ids = set()
    for job in queue['jobs']:
        if job['round'] == args.round:
            break
        previous_ids.add(job['experiment_id'])
    eligible = [r for r in analysis['rows'] if r['seed'] == 3036900957 and
                (r['version'] == 'research_v2' or r['experiment_id'] in previous_ids)]
    reference = max(eligible, key=lambda r: r['hard_fold_mean'])
    deployment_rows = [r for r in rows if r['seed'] == 3036900957]
    best = max(deployment_rows, key=lambda r: r['hard_fold_mean']) if deployment_rows else None
    difference = best['hard_fold_mean'] - reference['hard_fold_mean'] if best else None
    record = {'utc': datetime.now(timezone.utc).isoformat(), 'round': args.round, 'round_type': args.type,
        'status': 'completed' if len(rows) == len(selected) else 'completed_with_failed_or_pruned_jobs',
        'configurations': len(selected), 'completed_outer_fits': 3 * len(rows),
        'results': [{k: r[k] for k in ('experiment_id', 'seed', 'hard_correct', 'hard_n', 'correct', 'n',
                    'hard_fold_mean', 'hard_fold_sd', 'training_seconds', 'numeric_asset_bytes')} for r in rows],
        'failed_or_pruned': [{'id': j['experiment_id'], 'status': j['status']} for j in selected if j['status'] != 'completed'],
        'observed_process_wall_seconds': sum(a.get('observed_wall_seconds', 0) for j in selected for a in j['attempts']),
        'global_reference': reference['version'] + ':' + reference['experiment_id'],
        'global_reference_hard_fold_mean': reference['hard_fold_mean'], 'increment_above_global_reference_pp': difference,
        'exceeds_registered_half_point': difference is not None and difference > .5,
        'decision': args.decision, 'next_round': args.next,
        'analysis': args.analysis.resolve().relative_to(ROOT).as_posix(),
        'analysis_sha256': hashlib.sha256(args.analysis.read_bytes()).hexdigest(),
        'resource_scope': 'CV asset audits and training costs; final fresh-process timing remains a separate gate.'}
    with ledger.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(record) + '\n')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
