"""Independent synthetic-only report utility verification; no production helper call."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
CASES = {
    'figures': ('reports/research_v3/prepare_figures.py', 'test_prepare_figures_v3.py'),
    'report': ('reports/research_v3/prepare_report_inputs.py', 'test_report_inputs_v3.py'),
}

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    name = sys.argv[1]
    source, tests = CASES[name]
    output = ROOT / ('results/research_v3/independent-report-' + name + '-review')
    output.mkdir(exist_ok=False)
    before = sha(ROOT / source)
    (output / Path(source).name).write_bytes((ROOT / source).read_bytes())
    command = [sys.executable, '-m', 'unittest', 'discover', '-s', 'code/tests', '-p', tests, '-v']
    started = datetime.now(timezone.utc).isoformat()
    begin = time.perf_counter()
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                            encoding='utf-8', errors='replace', timeout=60)
    elapsed = time.perf_counter() - begin
    (output / 'stdout.log').write_text(result.stdout, encoding='utf-8')
    (output / 'stderr.log').write_text(result.stderr, encoding='utf-8')
    after = sha(ROOT / source)
    record = {
        'scope': 'Independent source review and generated-fixture tests only; no real holdout, scoring, training, network or production report generation.',
        'started_at_utc': started,
        'completed_at_utc': datetime.now(timezone.utc).isoformat(),
        'source_path': source,
        'source_sha256_before': before,
        'source_sha256_after': after,
        'source_unchanged': before == after,
        'command': command,
        'exit_code': result.returncode,
        'process_wall_seconds': elapsed,
        'test_source_path': 'code/tests/' + tests,
        'test_source_sha256': sha(ROOT / 'code/tests' / tests),
        'log_sha256': {p.name: sha(p) for p in output.glob('*.log')},
        'passed': result.returncode == 0 and before == after,
    }
    (output / 'review.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(record, indent=2))
    print(result.stderr)
    raise SystemExit(0 if record['passed'] else 1)

if __name__ == '__main__':
    main()
