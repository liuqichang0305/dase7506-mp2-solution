"""Independent generated-fixture tests only; no production finalist operation."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/research_v3/independent-finalist-preparation-review'
FILES = ['scripts/prepare_finalists_v3.py', 'code/tests/test_prepare_finalists_v3.py',
         'code/tools/confirm_validation_v3.py', 'code/tools/freeze_v3.py',
         'docs/research_v3/FINALIST_PREPARATION.md']
def hashes():
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in FILES}
OUT.mkdir(exist_ok=False)
before = hashes()
(OUT / 'prepare_finalists_v3.source.py').write_bytes((ROOT / FILES[0]).read_bytes())
command = [sys.executable, '-m', 'unittest', 'discover', '-s', 'code/tests', '-p', 'test_prepare_finalists_v3.py', '-v']
utc = datetime.now(timezone.utc).isoformat()
started = time.perf_counter()
result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=60)
elapsed = time.perf_counter() - started
(OUT / 'stdout.log').write_text(result.stdout, encoding='utf-8')
(OUT / 'stderr.log').write_text(result.stderr, encoding='utf-8')
after = hashes()
record = {'scope': 'Source review and temporary generated train/validation metadata, tiny numeric bundles and opaque state fixtures only. Actual unchanged validators exercised; no real search closure, finalist selection, lock, deployment copy, holdout bank, training, scoring or network operation.',
    'started_at_utc': utc, 'completed_at_utc': datetime.now(timezone.utc).isoformat(),
    'source_sha256_before': before, 'source_sha256_after': after, 'sources_unchanged': before == after,
    'command': command, 'process_wall_seconds': elapsed, 'exit_code': result.returncode,
    'log_sha256': {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in OUT.glob('*.log')},
    'passed': result.returncode == 0 and before == after}
(OUT / 'review.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
print(json.dumps(record, indent=2)); print(result.stderr)
raise SystemExit(0 if record['passed'] else 1)
