"""Hash-verified official train/validation download; no test option."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request
from verify_official import MANIFEST_SHA256

COMMIT = 'bb801f4f2e7629d270df765ab12c333b2dc616bb'

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    p.add_argument('--split', choices=('train', 'validation'), nargs='+', default=['train', 'validation'])
    a = p.parse_args()
    root = a.root.resolve()
    manifest = (root / 'code/PACKAGE_MANIFEST.json').read_bytes()
    if hashlib.sha256(manifest).hexdigest() != MANIFEST_SHA256:
        raise ValueError('Official manifest changed')
    for relative, expected in json.loads(manifest)['files'].items():
        if not relative.startswith('code/data/') or 'test' in relative:
            continue
        if not any(split in Path(relative).name for split in a.split) and Path(relative).name not in ('manifest.json', 'LICENSE.txt', 'NOTICE.txt'):
            continue
        target = (root / relative).resolve()
        if not target.is_relative_to(root / 'code/data'):
            raise ValueError('Unsafe official package destination')
        existed = target.exists()
        if existed:
            content = target.read_bytes()
        else:
            url = f'https://raw.githubusercontent.com/xudongwu-0/DASE7506-MP2/{COMMIT}/{relative}'
            with urllib.request.urlopen(url, timeout=180) as response:
                content = response.read()
        if hashlib.sha256(content).hexdigest() != expected:
            raise ValueError(f'Official bytes mismatch: {relative}')
        if not existed:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
        print(json.dumps({'file': relative, 'sha256': expected, 'bytes': len(content),
                          'status': 'already_verified' if existed else 'downloaded_verified'}), flush=True)
