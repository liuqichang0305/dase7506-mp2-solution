"""Independent narrow synthetic check of research-only archive boundary."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/research_v3/independent-research-archive-freeze-review'
OUT.mkdir(exist_ok=False)
names=['code/tools/freeze_v3.py','code/tests/test_final_freeze_v3.py','submission/v3/evaluate_final.py']
hashes=lambda:{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in names}
before=hashes();start=time.perf_counter();utc=datetime.now(timezone.utc).isoformat()
command=[sys.executable,'code/tests/test_final_freeze_v3.py',
         'FinalFreezeTests.test_post_deadline_archive_preserves_gates_but_never_authorizes_test',
         'FinalFreezeTests.test_archive_needs_real_restrictive_block_and_immutable_evidence','-v']
result=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=30)
elapsed=time.perf_counter()-start
(OUT/'stdout.log').write_text(result.stdout,encoding='utf-8');(OUT/'stderr.log').write_text(result.stderr,encoding='utf-8')
after=hashes();record={'scope':'Narrow diff source review and two generated-fixture boundary tests only. No actual freeze, archive, training, data or test worker.',
    'started_at_utc':utc,'completed_at_utc':datetime.now(timezone.utc).isoformat(),'command':command,'exit_code':result.returncode,
    'process_wall_seconds':elapsed,'source_before':before,'source_after':after,'unchanged':before==after,
    'passed':result.returncode==0 and before==after,
    'review':'Existing source/assets/review/reproduction gates retained. Restrictive actual post-deadline block binds closure and rules. Default verify_freeze/evaluator reject archive; explicit archive verification/reuse alone is permitted.',
    'wording_note':'Coordinator was advised to distinguish public score/artifact cutoff from the project restriction against a new V3 test; the public course documents did not establish a blanket ban on offline test execution.'}
(OUT/'review.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
print(json.dumps(record,indent=2));print(result.stderr)
raise SystemExit(0 if record['passed'] else 1)
