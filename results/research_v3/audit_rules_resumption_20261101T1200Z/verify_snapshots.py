"""Verify rule snapshots against Git blob IDs; no dataset or score inputs."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
BASE = Path(__file__).resolve().parent
manifest = json.loads((BASE / 'sources_and_findings.json').read_text(encoding='utf-8'))
checks = {}
def blob(raw):
    return hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()
for name, item in manifest['sources'].items():
    path = BASE / item['snapshot']
    raw = path.read_bytes()
    if blob(raw) != item['blob_sha']:
        # apply_patch may introduce a terminal blank line; only restore exact
        # official bytes when the independently returned Git blob SHA agrees.
        candidates = [raw[:-1]] if raw.endswith(b'\n') else []
        exact = [candidate for candidate in candidates if blob(candidate) == item['blob_sha']]
        if len(exact) != 1:
            raise ValueError('Official snapshot Git blob mismatch: ' + name)
        raw = exact[0]
        path.write_bytes(raw)
    checks[name] = {'git_blob_sha': blob(raw), 'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw), 'matched': True}
record = {'checked_at_utc': datetime.now(timezone.utc).isoformat(), 'official_source_bytes_verified': checks,
          'publication_entries_saved': False, 'publication_note': 'Only project review-window metadata retained; source commit/blob identify the complete public source.'}
with (BASE / 'snapshot_verification.json').open('x', encoding='utf-8') as stream:
    json.dump(record, stream, indent=2); stream.write('\n')
print(json.dumps(record, indent=2))
