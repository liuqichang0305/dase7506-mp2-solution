from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone
ROOT=Path(__file__).resolve().parents[3]
source=ROOT/'code/tools/freeze_v3.py'
raw=source.read_bytes()
old=b'The user authorized continued research archive and released peer review. The actual score deadline and publication rules prohibit a new V3 official test/score update or automatic artifact replacement.'
new=b'The user authorized continued research archive and released peer review. The course deadline/publication rules block late score updates and automatic artifact replacement. This project closes its V3 test path for the post-deadline archive; no general course ban on offline evaluation is asserted.'
assert raw.count(new)==1
reconstructed=hashlib.sha256(raw.replace(new,old)).hexdigest()
assert reconstructed=='11c9da8abdb6e651a52badb6c2b9ce90db2db5a147222f5851ee4a2ffef0b36f'
record={'checked_at_utc':datetime.now(timezone.utc).isoformat(),'current_source_sha256':hashlib.sha256(raw).hexdigest(),
        'previous_independently_tested_sha256':reconstructed,'exact_single_string_replacement_verified':True,
        'control_flow_changed':False,'tests_repeated':False,
        'finding':'Correctly distinguishes course submission cutoff from the project archive policy against a V3 test.'}
with (Path(__file__).parent/'wording-only-followup.json').open('x',encoding='utf-8') as stream:
    json.dump(record,stream,indent=2);stream.write('\n')
print(json.dumps(record,indent=2))
