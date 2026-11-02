"""Independent R15 synthetic regression; no production fusion stage."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/research_v3/independent-reduced-ema-fusion-helper-review'
FILES = ['scripts/register_reduced_tabm_ema_attention_fusion_v3.py',
         'scripts/register_reduced_tabm_attention_fusion_v3.py',
         'code/tests/test_reduced_tabm_ema_fusion_registration_v3.py',
         'code/tools/diagnose_oof_v3.py',
         'docs/research_v3/REDUCED_TABM_EMA_FIXED_FUSION.md']
def hashes():
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in FILES}
OUT.mkdir(exist_ok=False)
before = hashes()
(OUT / 'register_reduced_tabm_ema_attention_fusion_v3.source.py').write_bytes((ROOT / FILES[0]).read_bytes())
command = [sys.executable, '-m', 'unittest', 'discover', '-s', 'code/tests', '-p', 'test_reduced_tabm_ema_fusion_registration_v3.py', '-v']
utc = datetime.now(timezone.utc).isoformat()
start = time.perf_counter()
result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=60)
elapsed = time.perf_counter() - start
(OUT / 'stdout.log').write_text(result.stdout, encoding='utf-8')
(OUT / 'stderr.log').write_text(result.stderr, encoding='utf-8')
after = hashes()
record = {'scope': 'Independent source-diff review plus synthetic temporary train-OOF metadata and mocked diagnostic process. No production prepare/register/evaluate/summarize, raw arrays, training, holdout, leaderboard or network access.',
          'started_at_utc': utc, 'completed_at_utc': datetime.now(timezone.utc).isoformat(),
          'source_sha256_before': before, 'source_sha256_after': after, 'sources_unchanged': before == after,
          'command': command, 'process_wall_seconds': elapsed, 'exit_code': result.returncode,
          'log_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.glob('*.log')},
          'passed': result.returncode == 0 and before == after,
          'review': ['R12 typed final and successful execution receipts, exact three-entry prior inventory and frozen R12 helper source are required before preparing R15.',
                     'Exact fixed improved reduced TabM EMA then Attention EMA order and equal model-level probabilities; TabM four-member averaging precedes the two-model mean.',
                     'Local retention compares to stronger improved TabM EMA; Attention-specific fields retain their own baseline. Current global BASE957 hard fold mean inventory includes actual R12.',
                     'New aggregate preserves all three prior objects and appends one measured fourth outcome; no prior source/registration/analysis rewrite.',
                     'Inherited idle ownership, absolute budget/configuration cap, direct-base guarded worker and partial-attempt preservation remain unchanged.',
                     'Immutable R15 ledger and record snapshots allow later stability append. Child execution was mocked, so this is software validation rather than measured fusion/resource evidence.']}
(OUT / 'review.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
print(json.dumps(record, indent=2)); print(result.stderr)
raise SystemExit(0 if record['passed'] else 1)
