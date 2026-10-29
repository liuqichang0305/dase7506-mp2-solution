"""Time fresh official prediction + scoring processes; test requires a verified freeze."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time

from audit_assets import audit, sha256

THREAD_ENV = {key: '1' for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                                   'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS',
                                   'BLIS_NUM_THREADS', 'OMP_THREAD_LIMIT')}


def verify_freeze(root, freeze_path, method, checkpoint):
    freeze = json.loads(Path(freeze_path).read_text(encoding='utf-8'))
    if freeze.get('schema') != 'dase7506-mp2-freeze-v1' or freeze.get('frozen') is not True:
        raise ValueError('Explicit frozen=true manifest required')
    if not re.fullmatch('[0-9a-f]{40}', freeze.get('source_commit', '')):
        raise ValueError('Freeze requires full source commit SHA')
    if freeze.get('upstream_commit') != 'bb801f4f2e7629d270df765ab12c333b2dc616bb':
        raise ValueError('Frozen upstream version differs')
    if not freeze.get('validation_selection_reason'):
        raise ValueError('Validation-based selection reason missing')
    for section in ('source_files', 'assets'):
        entries = freeze.get(section)
        if not isinstance(entries, dict) or not entries:
            raise ValueError(f'Nonempty {section} hashes required')
        for relative, digest in entries.items():
            path = (root / relative).resolve()
            if not path.is_relative_to(root) or not path.is_file() or path.is_symlink():
                raise ValueError(f'Invalid frozen file: {relative}')
            if sha256(path) != digest:
                raise ValueError(f'Frozen bytes changed: {relative}')
            if section == 'source_files':
                committed = subprocess.run(['git', 'show', f'{freeze["source_commit"]}:{relative}'], cwd=root, capture_output=True)
                if committed.returncode or hashlib.sha256(committed.stdout).hexdigest() != digest:
                    raise ValueError(f'Frozen source does not match recorded commit: {relative}')
    required = {method.relative_to(root).as_posix(), 'code/common.py', 'code/predict.py', 'code/score_predictions.py'}
    # Freeze the full custom module directory so newly added inference modules cannot go unnoticed.
    if (root / 'code/mp2').exists():
        required.update(p.relative_to(root).as_posix() for p in (root / 'code/mp2').rglob('*.py'))
    if not required.issubset(freeze['source_files']):
        raise ValueError(f'Freeze omits source files: {sorted(required - freeze["source_files"].keys())}')
    actual_assets = {p.relative_to(root).as_posix() for p in checkpoint.rglob('*') if p.is_file()}
    if actual_assets != set(freeze['assets']):
        raise ValueError('Bundle file list differs from freeze')
    resources = audit(checkpoint)
    if not resources['within_limit']:
        raise ValueError('Inference assets exceed 16 MiB')
    return {'path': str(freeze_path), 'sha256': sha256(freeze_path), 'source_commit': freeze['source_commit'],
            'verified_source_files': len(freeze['source_files']), 'verified_assets': len(freeze['assets'])}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    p.add_argument('--python', default=sys.executable)
    p.add_argument('--method', type=Path, default=Path('code/solution.py'))
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--split', choices=('train', 'validation', 'test'), default='validation')
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--freeze', type=Path)
    a = p.parse_args()
    root = a.root.resolve()
    method = (root / a.method).resolve()
    checkpoint = (root / a.checkpoint).resolve(strict=True)
    destination = (root / a.output_dir).resolve()
    if not method.is_relative_to(root) or not checkpoint.is_relative_to(root):
        raise ValueError('Method and assets must be inside the project root')
    freeze = None
    if a.split == 'test':
        if a.freeze is None:
            raise ValueError('Test is locked: supply a pre-test freeze manifest after model selection is complete')
        freeze = verify_freeze(root, (root / a.freeze).resolve(), method, checkpoint)
    destination.mkdir(parents=True, exist_ok=True)
    prefix = a.split
    outputs = {key: destination / f'{prefix}-{key}.json' for key in ('predictions', 'score', 'timing')}
    if any(path.exists() for path in outputs.values()):
        raise FileExistsError('Preserve previous results; use a new output directory for another run')
    env = os.environ.copy()
    env.update(THREAD_ENV)
    env['PYTHONUTF8'] = '1'
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    commands = [
        [a.python, 'predict.py', '--method', str(method), '--checkpoint', str(checkpoint),
         '--split', a.split, '--output', str(outputs['predictions'])],
        [a.python, 'score_predictions.py', '--split', a.split, '--predictions', str(outputs['predictions']),
         '--output', str(outputs['score'])],
    ]
    report = {'started_at_utc': datetime.now(timezone.utc).isoformat(), 'split': a.split,
              'platform': platform.platform(), 'python': a.python, 'parent_python': sys.version,
              'processor': platform.processor(), 'logical_cpus': os.cpu_count(), 'thread_environment': THREAD_ENV,
              'processes': [], 'freeze': freeze,
              'measurement': 'Local fresh prediction process + independent scoring process; sequential; includes startup, imports, inputs, assets and I/O. Not a reference-machine eligibility certification.'}
    start = time.perf_counter()
    for index, command in enumerate(commands):
        tick = time.perf_counter()
        completed = subprocess.run(command, cwd=root / 'code', env=env, capture_output=True, text=True, encoding='utf-8', errors='replace')
        elapsed = time.perf_counter() - tick
        name = ('prediction', 'scoring')[index]
        (destination / f'{prefix}-{name}.stdout.log').write_text(completed.stdout, encoding='utf-8')
        (destination / f'{prefix}-{name}.stderr.log').write_text(completed.stderr, encoding='utf-8')
        report['processes'].append({'command': command, 'cwd': str(root / 'code'), 'returncode': completed.returncode,
                                    'wall_seconds': elapsed, 'stdout': f'{prefix}-{name}.stdout.log',
                                    'stderr': f'{prefix}-{name}.stderr.log'})
        if completed.returncode != 0:
            break
    report['total_wall_seconds'] = time.perf_counter() - start
    report['status'] = 'passed' if len(report['processes']) == 2 and all(r['returncode'] == 0 for r in report['processes']) else 'failed'
    report['local_under_30_seconds'] = report['status'] == 'passed' and report['total_wall_seconds'] <= 30
    if report['status'] == 'passed':
        report['official_score'] = json.loads(outputs['score'].read_text(encoding='utf-8'))
    outputs['timing'].write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report['status'] == 'passed' else 1)


if __name__ == '__main__':
    main()
