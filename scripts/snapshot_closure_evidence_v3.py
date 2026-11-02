"""Preserve exact closure-time logs, including append-only ledger chronology."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    closed_path = ROOT / 'results/research_v3/SEARCH_CLOSED.json'
    closed = json.loads(closed_path.read_text(encoding='utf-8'))
    folder = ROOT / 'results/research_v3/closure_evidence'
    folder.mkdir(exist_ok=False)
    records = {}
    for relative, expected in closed['evidence_sha256'].items():
        source = ROOT / relative
        data = source.read_bytes()
        current = hashlib.sha256(data).hexdigest()
        note = 'Exact bytes still present at snapshot creation.'
        if current != expected:
            lines = data.splitlines(keepends=True)
            if (relative != 'results/research_v3/round_decisions.jsonl'
                    or json.loads(lines[-1])['round'] != 'r13_fixed_seed_stability'
                    or hashlib.sha256(b''.join(lines[:-1])).hexdigest() != expected):
                raise ValueError('No exact, explained closure-time evidence is available: ' + relative)
            data = b''.join(lines[:-1])
            note = 'Exact prefix before the later R13 narrative append; no training or search was resumed.'
        destination = folder / source.name
        with destination.open('xb') as stream:
            stream.write(data)
        if hashlib.sha256(destination.read_bytes()).hexdigest() != expected:
            raise ValueError('Snapshot copy differs')
        records[relative] = {'snapshot': destination.relative_to(ROOT).as_posix(),
            'sha256': expected, 'current_sha256_at_snapshot': current, 'note': note}
    output = {'schema': 'mp2-v3-closure-evidence-snapshots-v1',
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'search_closed_sha256': hashlib.sha256(closed_path.read_bytes()).hexdigest(),
        'evidence': records,
        'semantics': 'SEARCH_CLOSED records hashes at closure. Append-only reporting and permitted locked final fits may add later events; these byte-identical snapshots preserve the original captured histories.'}
    with (folder / 'SNAPSHOTS.json').open('x', encoding='utf-8') as stream:
        json.dump(output, stream, indent=2)
        stream.write('\n')
    print(json.dumps({'snapshots': len(records), 'closure_rewritten': False}))


if __name__ == '__main__':
    main()
