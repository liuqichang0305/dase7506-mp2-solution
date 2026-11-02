"""Register two evidence-gated fixed mixtures before evaluating either one."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/research_v3'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    if (OUT / 'SEARCH_CLOSED.json').exists():
        raise RuntimeError('Research is closed')
    budget = read(OUT / 'campaign_budget.json')
    now = datetime.now(timezone.utc)
    if now >= datetime.fromisoformat(budget['research_end_utc'].replace('Z', '+00:00')):
        raise RuntimeError('Absolute research budget expired')
    ledger = OUT / 'fusion_registry.jsonl'
    if ledger.exists():
        raise FileExistsError('Preserve registered fusion choices; no automatic re-registration')
    queue = read(OUT / 'campaign_queue.json')['jobs']
    if len(queue) + 2 > budget['maximum_unique_configurations']:
        raise RuntimeError('Unique configuration cap exceeded, including fusion')
    gate_path = OUT / 'diagnostic-r03-attention/diagnostic.json'
    gate = read(gate_path)
    hashes = {row['method']: row['input_sha256']['oof.npz'] for row in gate['methods']}
    records = []
    for suffix, second in [('tabm', 'v3:v3_tabm_k8_s0'),
                           ('realmlp', 'v3:v3_realmlp_robust_all_s0')]:
        methods = ['v3:v3_attention_adamw_s0', second]
        if not any(row['methods'] == methods and row['gate_passed'] for row in gate['complementarity']):
            raise ValueError('Required prior reciprocal-error gate is missing')
        record = {'schema': 'mp2-fixed-fusion-registration-v3',
            'experiment_id': f'v3_fusion_attention_{suffix}_s0',
            'round': 'r08_fixed_heterogeneous_fusion', 'status': 'registered',
            'registered_at_utc': now.isoformat(),
            'purpose': 'single_fixed_equal_probability_oof',
            'hypothesis': 'Prior matched train-OOF errors demonstrate reciprocal corrections; evaluate exactly this two-member equal answer-group-probability mean. No learned weights, temperatures, seed selection or label-aware routing. Oracle gain is not an expected fusion gain.',
            'methods': methods, 'weights': [.5, .5],
            'representative': 'first_model_first_member',
            'oof_sha256': {key: hashes[key] for key in methods},
            'gate_evidence': {'path': gate_path.relative_to(ROOT).as_posix(),
                              'sha256': hashlib.sha256(gate_path.read_bytes()).hexdigest()},
            'source_commit': __import__('subprocess').check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
            'new_member_training_fits': 0,
            'budget_note': 'Counts as one unique configuration in the same fixed24h/96-config budget; all members reuse verified common-fold OOF.'}
        target = ROOT / 'docs/research_v3/rounds' / (record['experiment_id'] + '.json')
        with target.open('x', encoding='utf-8') as stream:
            json.dump(record, stream, indent=2)
            stream.write('\n')
        records.append({'registration': target.relative_to(ROOT).as_posix(),
                        'registration_sha256': hashlib.sha256(target.read_bytes()).hexdigest(), **record})
    with ledger.open('x', encoding='utf-8') as stream:
        for record in records:
            stream.write(json.dumps(record) + '\n')
    print(json.dumps({'registered': [r['experiment_id'] for r in records],
                      'unique_registered_including_fusion': len(queue) + len(records)}))


if __name__ == '__main__':
    main()
