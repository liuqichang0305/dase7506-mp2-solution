"""Fetch only official package data at the recorded immutable upstream commit."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

from verify_official import MANIFEST_SHA256

COMMIT = 'bb801f4f2e7629d270df765ab12c333b2dc616bb'
BASE = f'https://raw.githubusercontent.com/xudongwu-0/DASE7506-MP2/{COMMIT}/'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    a = p.parse_args()
    root = a.root.resolve()
    manifest_bytes = (root / 'code/PACKAGE_MANIFEST.json').read_bytes()
    if hashlib.sha256(manifest_bytes).hexdigest() != MANIFEST_SHA256:
        raise ValueError('Official package manifest bytes changed')
    manifest = json.loads(manifest_bytes)
    for relative, expected in manifest['files'].items():
        if not relative.startswith('code/data/'):
            continue
        destination = (root / relative).resolve()
        if not destination.is_relative_to(root / 'code/data'):
            raise ValueError('Unsafe destination path')
        if destination.exists():
            actual = hashlib.sha256(destination.read_bytes()).hexdigest()
            if actual == expected:
                print(json.dumps({'path': relative, 'status': 'already_verified'}))
                continue
            raise ValueError(f'Existing file differs; preserve and inspect before retrying: {relative}')
        with urllib.request.urlopen(BASE + relative, timeout=120) as response:
            content = response.read()
        actual = hashlib.sha256(content).hexdigest()
        if actual != expected:
            raise ValueError(f'Downloaded official file has wrong hash: {relative}')
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
        print(json.dumps({'path': relative, 'status': 'downloaded_verified', 'sha256': actual, 'bytes': len(content)}))


if __name__ == '__main__':
    main()
