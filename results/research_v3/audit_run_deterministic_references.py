"""Capture the one authorized train-only deterministic-reference invocation."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
ROOT = Path(__file__).resolve().parents[2]
PREFIX = ROOT / 'results/research_v3/deterministic-train-references-run'
paths = {key: Path(str(PREFIX) + suffix) for key, suffix in
         [('stdout', '.stdout.log'), ('stderr', '.stderr.log'), ('receipt', '.json')]}
if any(path.exists() for path in paths.values()):
    raise FileExistsError('Preserve earlier execution receipts')
source = ROOT / 'code/tools/deterministic_train_references_v3.py'
before = hashlib.sha256(source.read_bytes()).hexdigest()
command = [sys.executable, 'code/tools/deterministic_train_references_v3.py']
utc = datetime.now(timezone.utc).isoformat()
started = time.perf_counter()
result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=60)
elapsed = time.perf_counter() - started
for key, value in [('stdout', result.stdout), ('stderr', result.stderr)]:
    with paths[key].open('x', encoding='utf-8') as stream: stream.write(value)
record = {'scope': 'One train-only deterministic-reference computation; no fitting or official holdout scoring.',
    'command': command, 'started_at_utc': utc, 'completed_at_utc': datetime.now(timezone.utc).isoformat(),
    'process_wall_seconds': elapsed, 'exit_code': result.returncode, 'source_sha256_before': before,
    'source_sha256_after': hashlib.sha256(source.read_bytes()).hexdigest(),
    'logs': {path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
             for key, path in paths.items() if key != 'receipt'}}
with paths['receipt'].open('x', encoding='utf-8') as stream: json.dump(record, stream, indent=2); stream.write('\n')
print(json.dumps(record, indent=2)); print(result.stdout); print(result.stderr)
raise SystemExit(result.returncode)
