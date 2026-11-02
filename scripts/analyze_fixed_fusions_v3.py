"""Summarize already completed fixed fusions without new prediction or fitting."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/research_v3'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    target = OUT / 'fixed_fusion_analysis.json'
    if target.exists():
        raise FileExistsError('Preserve completed fusion analysis')
    registered = [json.loads(line) for line in (OUT / 'fusion_registry.jsonl').read_text(encoding='utf-8').splitlines()]
    rows = []
    for registration in registered:
        path = ROOT / registration['registration']
        if sha(path) != registration['registration_sha256']:
            raise ValueError('Fusion registration changed')
        folder = OUT / registration['experiment_id']
        diagnostic = read(folder / 'diagnostic.json')
        result = diagnostic['fusion']
        if (result['status'] != 'complete' or result['registration_sha256'] != sha(path)
                or result['oof_sha256'] != sha(folder / 'fusion-oof.npz')
                or sha(folder / 'fusion-registration.json') != sha(path)):
            raise ValueError('Fusion completion/hash mismatch')
        metrics = result['metrics']
        rows.append({'experiment_id': registration['experiment_id'],
            'methods': result['methods'], 'weights': result['weights'],
            'metrics': metrics, 'hard_delta_questions_vs_attention_control': metrics['hard']['correct'] - 1682,
            'hard_delta_questions_vs_historical_best': metrics['hard']['correct'] - 1687,
            'maximum_member_asset_bytes': max(x['total_member_uncompressed_bytes'] for x in result['assets_per_fold']),
            'new_training_fits': 0, 'decision': 'Reject: fewer hard and overall correct than the matched Attention member; do not tune weights after this outcome.',
            'diagnostic_path': (folder / 'diagnostic.json').relative_to(ROOT).as_posix(),
            'diagnostic_sha256': sha(folder / 'diagnostic.json')})
    record = {'schema': 'mp2-fixed-fusion-analysis-v3',
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'round': 'r08_fixed_heterogeneous_fusion', 'round_type': 'hypothesis',
        'status': 'completed', 'configurations': len(rows), 'new_training_fits': 0,
        'results': rows, 'exceeds_registered_half_point': False,
        'global_reference': 'research_v2:attention_regularized_s0',
        'decision': 'Reciprocal-error oracle potential did not translate to either real equal-probability mixture. Both are rejected. No learned weighting or further combination sweep is justified by these outcomes.',
        'cost_note': 'Member training costs are counted once in the campaign. OOF mixing is a deterministic diagnostic, with no new optimization. Formal fresh-process latency was not measured for rejected mixtures.',
        'terminal_observed_wall_seconds': {
            'v3_fusion_attention_tabm_s0': 1.9131238000000002,
            'v3_fusion_attention_realmlp_s0': 1.7272821999999999},
        'wall_measurement_source': 'Actual exec_command completion receipts for the two fixed diagnostic invocations; includes Python startup and diagnostic output generation.',
        'registry_sha256': sha(OUT / 'fusion_registry.jsonl')}
    target.write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
    ledger = OUT / 'round_decisions.jsonl'
    prior = [json.loads(line) for line in ledger.read_text(encoding='utf-8').splitlines()]
    if any(row['round'] == record['round'] for row in prior):
        raise ValueError('Round decision already exists')
    with ledger.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(record) + '\n')
    print(json.dumps([{k: row[k] for k in ('experiment_id', 'hard_delta_questions_vs_attention_control', 'decision')} for row in rows], indent=2))


if __name__ == '__main__':
    main()
