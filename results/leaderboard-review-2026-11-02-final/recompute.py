"""Offline independent ranking recheck using only the minimized public snapshot."""
from pathlib import Path
import hashlib
import json
import math

HERE = Path(__file__).resolve().parent
snapshot_bytes = (HERE / 'minimal_snapshot.json').read_bytes()
snapshot = json.loads(snapshot_bytes)
result = json.loads((HERE / 'ranking_result.json').read_text(encoding='utf-8'))
assert hashlib.sha256(snapshot_bytes).hexdigest() == result['minimal_snapshot_sha256']

def best(rows, normalized=False):
    selected = {}
    for row in rows:
        if row['status'] not in ('self-reported', 'verified'):
            continue
        key = (row['normalized_student_key'] if normalized else row['student_key'], row['account_key'])
        old = selected.get(key)
        available = lambda x: int(x['has_code']) + int(x['has_checkpoint'])
        if old is None or row['score'] > old['score'] or (row['score'] == old['score'] and available(row) > available(old)):
            selected[key] = row
    return sorted(selected.values(), key=lambda row: (-row['score'], row['number']))

def stats(rows):
    own = next(row for row in rows if row['is_user'])
    higher = sum(row['score'] > own['score'] for row in rows)
    tied = sum(row['score'] == own['score'] for row in rows)
    return {'count': len(rows), 'strictly_higher': higher, 'tied_including_user': tied,
            'competition_rank': higher + 1, 'tied_rank_interval': [higher+1, higher+tied],
            'floor_35_percent': math.floor(.35*len(rows)),
            'in_top_35_percent': higher+1 <= math.floor(.35*len(rows))}

official = best(snapshot['rows'])
students = [row for row in official if not row['teacher_example']]
normalized = best(students, normalized=True)
strict = [row for row in normalized if row['normalized_ten_digit_id']]
for key, rows in [('official_display', official), ('excluding_confirmed_teacher_example', students),
                  ('same_account_u_prefix_normalization', normalized), ('excluding_also_short_unverified_id', strict)]:
    assert stats(rows) == result[key], key
print(json.dumps({'passed': True, 'scopes_verified': 4, 'no_network': True,
                  'student_rank': result['same_account_u_prefix_normalization'],
                  'identity_limited_alternative': result['excluding_also_short_unverified_id']}, indent=2))
