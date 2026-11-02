"""Create V3-local copies of the reviewed original form adapters, once.

Only local path/freeze references change. The official website handler remains
unmodified and will be retrieved/verified after the unique final freeze.
This operation neither accesses the network nor creates a course issue.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def create(source, destination, replacements, guard=False):
    source, destination = ROOT / source, ROOT / destination
    if destination.exists():
        raise FileExistsError('Preserve the existing V3 adapter')
    text = source.read_text(encoding='utf-8')
    for old, new in replacements:
        if old not in text:
            raise ValueError(f'Original adapter changed: {old}')
        text = text.replace(old, new)
    if guard:
        text = text.replace('args=parser.parse_args()',
            "args=parser.parse_args()\n"
            "if not (root/'submission/v3/FREEZE.json').exists() or not (root/'results/research_v3/SEARCH_CLOSED.json').exists():\n"
            "    raise SystemExit('V3 form snapshot is fetched only after research closure and final freeze')")
    destination.write_text(text, encoding='utf-8')
    print(destination.relative_to(ROOT))


if __name__ == '__main__':
    create('submission/prepare_official_form.mjs', 'submission/v3/prepare_official_form.mjs', [
        ("path.dirname(fileURLToPath(import.meta.url)), '..')", "path.dirname(fileURLToPath(import.meta.url)), '..', '..')"),
        ('artifacts/final_bundle', 'artifacts/research_v3/final_selected'),
        ('results/test-score.json', 'submission/v3/test-score.json'),
        ("'tmp/form-source'", "'tmp/form-source-v3'"),
        ("||'submission'", "||'submission/v3'"),
        ('submission/PRE_TEST_FREEZE.json', 'submission/v3/FREEZE.json'),
        ('submission/FORM_SOURCE_SNAPSHOT.json', 'submission/v3/FORM_SOURCE_SNAPSHOT.json'),
        ('node submission/prepare_official_form.mjs', 'node submission/v3/prepare_official_form.mjs'),
    ])
    create('submission/fetch_form_sources.py', 'submission/v3/fetch_form_sources.py', [
        ('parents[1]', 'parents[2]'), ('tmp/form-source', 'tmp/form-source-v3'),
        ('submission/FORM_SOURCE_SNAPSHOT.json', 'submission/v3/FORM_SOURCE_SNAPSHOT.json'),
    ], guard=True)
