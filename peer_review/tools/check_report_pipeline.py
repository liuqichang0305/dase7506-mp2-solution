"""Capture only synthetic pipeline tests; never invoke actual peer submission."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'peer_review/audit/report-pipeline-synthetic'
OUT.mkdir(parents=True,exist_ok=False)
names=['peer_review/tools/peer_report.py','peer_review/tools/prepare_official_peer_form.mjs','code/tests/test_peer_report.py']
hashes=lambda:{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in names}
before=hashes();start=time.perf_counter();utc=datetime.now(timezone.utc).isoformat()
command=[sys.executable,'-m','unittest','discover','-s','code/tests','-p','test_peer_report.py','-v']
result=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=60)
elapsed=time.perf_counter()-start
(OUT/'stdout.log').write_text(result.stdout,encoding='utf-8');(OUT/'stderr.log').write_text(result.stderr,encoding='utf-8')
after=hashes();record={'schema':'mp2-peer-pipeline-synthetic-check-v1','started_at_utc':utc,
    'completed_at_utc':datetime.now(timezone.utc).isoformat(),'command':command,'exit_code':result.returncode,
    'process_wall_seconds':elapsed,'source_before':before,'source_after':after,'unchanged':before==after,
    'scope':'Generated temporary peer metadata and score fixtures; actual unchanged published handler offline; fake HTTP only. No actual peer scoring, preparation, credentials, GET or POST.',
    'passed':result.returncode==0 and before==after}
(OUT/'RESULT.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
print(json.dumps(record,indent=2));print(result.stderr)
raise SystemExit(0 if record['passed'] else 1)
