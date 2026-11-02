"""Independent synthetic regression; never runs an actual fusion stage."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/research_v3/independent-reduced-fusion-helper-review'
FILES = ['scripts/register_reduced_tabm_attention_fusion_v3.py',
         'code/tests/test_reduced_tabm_fusion_registration_v3.py',
         'code/tools/diagnose_oof_v3.py',
         'docs/research_v3/REDUCED_TABM_FIXED_FUSION.md']
def hashes():
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in FILES}
OUT.mkdir(exist_ok=False)
before = hashes()
(OUT / 'register_reduced_tabm_attention_fusion_v3.source.py').write_bytes((ROOT / FILES[0]).read_bytes())
command = [sys.executable, '-m', 'unittest', 'discover', '-s', 'code/tests', '-p', 'test_reduced_tabm_fusion_registration_v3.py', '-v']
utc = datetime.now(timezone.utc).isoformat()
start = time.perf_counter()
result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=60)
elapsed = time.perf_counter() - start
(OUT / 'stdout.log').write_text(result.stdout, encoding='utf-8')
(OUT / 'stderr.log').write_text(result.stderr, encoding='utf-8')
after = hashes()
record = {'scope': 'Independent source review plus synthetic temporary train-OOF metadata and mocked diagnostic process. No production prepare/register/evaluate/summarize, training, holdout, leaderboard or network access.',
          'started_at_utc': utc, 'completed_at_utc': datetime.now(timezone.utc).isoformat(),
          'source_sha256_before': before, 'source_sha256_after': after, 'sources_unchanged': before == after,
          'command': command, 'process_wall_seconds': elapsed, 'exit_code': result.returncode,
          'log_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.glob('*.log')},
          'passed': result.returncode == 0 and before == after,
          'review': ['Exact fixed two-member order/equal weights and prior reciprocal-error gate retained.',
                     'Pre-evaluation registration binds current completed BASE957 common-fold summaries, preserved V2 and earlier registered fusions; strict half-point continuation uses actual hard fold mean against the frozen maximum.',
                     'Direct base Python -S and fixed venv site-packages avoid the Windows launcher timeout boundary; existing diagnostic guard forbids worker spawning. Synthetic execution is mocked, not a fresh measured resource qualification.',
                     'Shared scheduler lock, terminal queue, absolute deadline and configuration cap are enforced; original registry/analysis are preserved and failed execution cannot silently rerun.',
                     'Final receipt binds immutable ledger snapshot plus exact R12 record, allowing later stability records on the live ledger.']}
(OUT / 'review.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
print(json.dumps(record, indent=2)); print(result.stderr)
raise SystemExit(0 if record['passed'] else 1)
