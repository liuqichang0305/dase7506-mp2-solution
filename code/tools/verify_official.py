"""Verify package bytes against the unmodified official package manifest."""
import argparse
import hashlib
import json
from pathlib import Path

MANIFEST_SHA256 = '2e9b714598e1addba61545f0d7ce4891d9d11aaf567fb341420d36694fcfd656'


def verify(root):
    root = Path(root).resolve()
    manifest_path = root / 'code' / 'PACKAGE_MANIFEST.json'
    if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != MANIFEST_SHA256:
        raise ValueError('Official package manifest bytes changed')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    mismatches = []
    checked = []
    for relative, expected in manifest['files'].items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root):
            raise ValueError('Unsafe package manifest path')
        actual = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
        checked.append({'path': relative, 'expected_sha256': expected, 'actual_sha256': actual})
        if actual != expected:
            mismatches.append(relative)
    return {'protocol': manifest['protocol'], 'package_manifest_sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            'status': 'passed' if not mismatches else 'failed', 'checked': checked, 'mismatches': mismatches}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    p.add_argument('--output', type=Path)
    a = p.parse_args()
    result = verify(a.root)
    text = json.dumps(result, indent=2) + '\n'
    if a.output:
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(text, encoding='utf-8')
    print(text)
    raise SystemExit(0 if result['status'] == 'passed' else 1)


if __name__ == '__main__':
    main()
