"""Inventory every file in an inference bundle, including uncompressed containers."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile

LIMIT = 16 * 1024 * 1024


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def audit(bundle):
    bundle = Path(bundle).resolve(strict=True)
    if not bundle.is_dir():
        raise ValueError('Checkpoint must be an explicit bundle directory')
    files = []
    for path in sorted(bundle.rglob('*')):
        if path.is_symlink():
            raise ValueError(f'External/symlink asset prohibited: {path}')
        if not path.is_file():
            continue
        row = {'path': path.relative_to(bundle).as_posix(), 'sha256': sha256(path),
               'disk_bytes': path.stat().st_size}
        row['uncompressed_bytes'] = row['disk_bytes']
        if zipfile.is_zipfile(path):
            with zipfile.ZipFile(path) as archive:
                row['members'] = [{'name': x.filename, 'uncompressed_bytes': x.file_size}
                                  for x in archive.infolist() if not x.is_dir()]
                # Count at least the physical archive; never benefit from compression.
                row['uncompressed_bytes'] = max(row['disk_bytes'], sum(x['uncompressed_bytes'] for x in row['members']))
        if path.suffix == '.npz':
            import numpy as np
            with np.load(path, allow_pickle=False) as values:
                row['arrays'] = [{'name': name, 'shape': list(values[name].shape),
                                  'dtype': str(values[name].dtype), 'array_bytes': values[name].nbytes}
                                 for name in values.files]
                if any(values[name].dtype.hasobject for name in values.files):
                    raise ValueError('Object/pickle arrays are not permitted')
        files.append(row)
    if not files:
        raise ValueError('Empty bundle')
    total = sum(x['uncompressed_bytes'] for x in files)
    return {'bundle': str(bundle), 'files': files, 'uncompressed_bytes': total,
            'limit_bytes': LIMIT, 'within_limit': total <= LIMIT,
            'counting_rule': 'All files; for ZIP/NPZ use max(physical bytes, sum uncompressed member bytes). Array payload bytes also disclosed.',
            'scope': 'Bundle inventory. Inference source must separately be reviewed for hidden parameters and external dependencies.'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('bundle', type=Path)
    p.add_argument('--output', type=Path)
    a = p.parse_args()
    result = audit(a.bundle)
    payload = json.dumps(result, indent=2) + '\n'
    if a.output:
        if a.output.resolve().is_relative_to(a.bundle.resolve()):
            raise ValueError('Write audit report outside the audited bundle')
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(payload, encoding='utf-8')
    print(payload)
    raise SystemExit(0 if result['within_limit'] else 1)


if __name__ == '__main__':
    main()
