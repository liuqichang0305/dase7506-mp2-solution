"""Prepare fresh inference-only environment and verified official test copies."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
RUNTIME = ROOT / 'tmp/peer_review/submission_240_7e3a2792'
ENV = RUNTIME / 'venv'
EVALUATOR = RUNTIME / 'official'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2); stream.write('\n')


def main():
    if ENV.exists() or EVALUATOR.exists():
        raise FileExistsError('Preserve prior setup; do not silently recreate')
    EVALUATOR.mkdir()
    (RUNTIME / 'temp').mkdir()
    package = ROOT / 'code/PACKAGE_MANIFEST.json'
    if sha(package) != '2e9b714598e1addba61545f0d7ce4891d9d11aaf567fb341420d36694fcfd656':
        raise ValueError('Official manifest changed')
    manifest = json.loads(package.read_text(encoding='utf-8'))
    files = ['code/common.py', 'code/predict.py', 'code/score_predictions.py',
             'code/data/manifest.json', 'code/data/candidates/gsm8k_test.jsonl',
             'code/data/candidates/gsm8k_test.npz', 'code/data/gsm8k_test.jsonl',
             'code/data/LICENSE.txt', 'code/data/NOTICE.txt']
    inventory = []
    for name in files:
        source = ROOT / name
        content = source.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        if digest != manifest['files'][name]:
            raise ValueError('Official file differs: ' + name)
        destination = EVALUATOR / Path(name).relative_to('code')
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open('xb') as stream:
            stream.write(content)
        inventory.append({'path': str(destination), 'official_path': name, 'sha256': digest, 'bytes': len(content)})
    write(OUT / 'official_files.json', {'schema': 'mp2-peer-official-files-v1',
        'official_commit': 'bb801f4f2e7629d270df765ab12c333b2dc616bb',
        'package_manifest_sha256': sha(package), 'files': inventory,
        'references_copied_for_independent_scorer_only': True, 'question_text_displayed': False})
    environment = {key: os.environ[key] for key in ('SYSTEMROOT', 'WINDIR', 'COMSPEC', 'PATH', 'PATHEXT') if key in os.environ}
    environment.update({'TEMP': str(RUNTIME / 'temp'), 'TMP': str(RUNTIME / 'temp'), 'PYTHONUTF8': '1'})
    commands = [
        [str(Path(sys._base_executable).resolve()), '-m', 'venv', str(ENV)],
        [str(ENV / 'Scripts/python.exe'), '-m', 'pip', '--isolated', 'install', '--only-binary=:all:',
         '--index-url', 'https://pypi.org/simple', 'numpy==2.5.3', 'threadpoolctl==3.7.0', 'psutil==7.2.2'],
    ]
    for index, command in enumerate(commands):
        tick = time.perf_counter()
        with (OUT / f'setup-{index}.stdout.log').open('xb') as out, (OUT / f'setup-{index}.stderr.log').open('xb') as err:
            result = subprocess.run(command, cwd=RUNTIME, env=environment, stdout=out, stderr=err, timeout=120)
        write(OUT / f'setup-{index}.json', {'command': command, 'returncode': result.returncode,
              'wall_seconds': time.perf_counter()-tick, 'completed_at_utc': datetime.now(timezone.utc).isoformat()})
        if result.returncode:
            raise RuntimeError('Environment setup failed; preserve logs and no score')
    print(json.dumps({'status': 'setup_complete', 'venv': str(ENV), 'official': str(EVALUATOR)}))


if __name__ == '__main__':
    main()
