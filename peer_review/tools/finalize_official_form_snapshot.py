"""Finalize an already retrieved exact official source graph; offline only."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'peer_review/official_form/65c16a3eef424e30b254ff58804dffe0fbda8793'
record = json.loads((BASE / 'ACQUISITION.json').read_text(encoding='utf-8'))
for item in record['files']:
    path = BASE / item['path']
    raw = path.read_bytes()
    git_hash = lambda b: hashlib.sha1(b'blob ' + str(len(b)).encode() + b'\0' + b).hexdigest()
    if git_hash(raw) != item['git_blob_sha']:
        raise ValueError('Snapshot does not match official Git blob: ' + item['path'])
    item.update(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
with (BASE / 'FORM_SOURCE_SNAPSHOT.json').open('x', encoding='utf-8') as stream:
    json.dump(record, stream, indent=2); stream.write('\n')
print(json.dumps({'status': 'exact_git_blobs_verified', 'files': len(record['files']), 'commit': record['official_site_commit']}))
