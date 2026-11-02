"""Independent report regression with generated fixtures, never real final data."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/research_v3/independent-report-r05-review'
FILES = ['reports/research_v3/prepare_report_inputs.py', 'code/tests/test_report_inputs_v3.py',
         'docs/research_v3/REPORT_INPUTS.md']
def hashes():
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in FILES}
OUT.mkdir(exist_ok=False)
before = hashes()
(OUT / 'prepare_report_inputs.source.py').write_bytes((ROOT / FILES[0]).read_bytes())
command = [sys.executable, '-m', 'unittest', 'discover', '-s', 'code/tests', '-p', 'test_report_inputs_v3.py', '-v']
utc = datetime.now(timezone.utc).isoformat()
start = time.perf_counter()
result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=60)
elapsed = time.perf_counter() - start
(OUT / 'stdout.log').write_text(result.stdout, encoding='utf-8')
(OUT / 'stderr.log').write_text(result.stderr, encoding='utf-8')
after = hashes()
record = {'scope': 'Source review and synthetic aggregate fixtures only. No production report preparation/rendering, raw banks, actual validation/test scores, decoded arrays, training, evaluation or network access.',
          'started_at_utc': utc, 'completed_at_utc': datetime.now(timezone.utc).isoformat(),
          'source_sha256_before': before, 'source_sha256_after': after, 'sources_unchanged': before == after,
          'command': command, 'process_wall_seconds': elapsed, 'exit_code': result.returncode,
          'log_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.glob('*.log')},
          'passed': result.returncode == 0 and before == after,
          'review': ['R05 adds saved matched TabM aggregate counts, fixed-seed 5000-draw paired intervals and per-fold gains, without recomputation.',
                     'Full historical diagnostic completion/output inventory and source snapshots are hash-bound. Raw train data hashes are reconciled to registrations without opening bank files.',
                     'Final analysis rows, exact primary-seed recipe differences, common folds, summary/OOF identities and paired contingency counts must agree.',
                     'The report explicitly distinguishes changed input/capacity/RNG consumption and conditional selected-OOF interval optimism.',
                     'Synthetic +9 gain/+2,+3,+4 folds differ from the actual saved +62 result to detect hardcoded narrative numbers. Actual compatibility report was read only; no production generator or historical OOF computation rerun.',
                     'Actual final PDF layout/page count remains a later production QA requirement.']}
(OUT / 'review.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
print(json.dumps(record, indent=2)); print(result.stderr)
raise SystemExit(0 if record['passed'] else 1)
