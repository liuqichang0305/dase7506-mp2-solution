"""Independent saved-aggregate/PDF check; no raw banks, inference or scoring."""
from pathlib import Path
import hashlib
import json
import math
import re
import statistics
from datetime import datetime, timezone
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
def read(name):
    return json.loads((ROOT / name).read_text(encoding='utf-8-sig'))
def sha(name):
    path = ROOT / name
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()
def close(a, b):
    assert math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-10), (a, b)

pdf = 'reports/research_v3/research_closeout/research_closeout_v3.pdf'
inputs = 'reports/research_v3/closeout_inputs.json'
markdown = 'reports/research_v3/research_closeout/report_content.md'
build = 'reports/research_v3/research_closeout/build_record.json'
initial = {p: sha(p) for p in (pdf, inputs, markdown, build)}
assert initial[pdf] == '53945861fe0adeb71b29cda388c338720344feb59acf3248bfc75eeaa2beae1f'
r, receipt = read(inputs), read(build)
assert receipt['output_sha256'] == initial[pdf] and receipt['inputs_sha256'] == initial[inputs]
pages = PdfReader(ROOT / pdf).pages
texts = [page.extract_text() for page in pages]
assert len(pages) == receipt['pages'] == len(r['pages']) == 10
evidence = r['evidence_sha256']
for path, expected in evidence.items():
    canonical = (ROOT / path).resolve().relative_to(ROOT.resolve()).as_posix()
    assert canonical == path and not canonical.startswith('code/data/')
    assert sha(path) == expected, path
analysis = read(r['paths']['analysis'])
counts, totals = {}, {}
for version in ('research_v3', 'research_v2'):
    rows = [row for row in analysis['rows'] if row['version'] == version]
    counts[version] = len(rows); totals[version] = sum(row['training_seconds'] for row in rows)
    document = re.sub(r'\s+', ' ', '\n'.join(texts[2:4] if version == 'research_v3' else texts[4:5]))
    for row in rows:
        summary = read(f"results/{version}/{row['experiment_id']}/summary.json")
        assert row['status'] == 'completed' and len(summary['folds']) == 3
        assert row['config'] == summary['config']
        for key in ('n', 'correct', 'hard_n', 'hard_correct'):
            assert row[key] == summary[key] == sum(fold[key] for fold in summary['folds'])
        accuracies = [100 * fold['hard_correct'] / fold['hard_n'] for fold in summary['folds']]
        close(row['hard_fold_mean'], statistics.mean(accuracies))
        close(row['hard_fold_sd'], statistics.stdev(accuracies))
        close(row['training_seconds'], summary['training_seconds'])
        expected = (f"{row['experiment_id']} {row['correct']}/{row['n']} "
                    f"{row['hard_correct']}/{row['hard_n']} "
                    f"{row['hard_fold_mean']:.2f} +/- {row['hard_fold_sd']:.2f} {row['training_seconds']:.1f}")
        assert expected in document, row['experiment_id']
assert counts == {'research_v3': 35, 'research_v2': 19}
derived = r['derived_aggregates']
close(totals['research_v3'], derived['v3_fit_seconds'])
close(totals['research_v2'], derived['historical_fit_seconds'])
queue = read('results/research_v3/campaign_queue.json')
attempts = [a for job in queue['jobs'] for a in job.get('attempts', [])]
close(sum(a.get('observed_wall_seconds', 0) for a in attempts), derived['recorded_attempt_seconds'])
assert sum('observed_wall_seconds' not in a for a in attempts) == derived['attempts_without_wall_seconds'] == 0
pruned = [job for job in queue['jobs'] if job['status'] == 'pruned']
assert len(pruned) == 4 and sorted(j['experiment_id'] for j in pruned) == sorted(derived['incomplete_stability_ids'])
freeze, selected = read('submission/v3/FREEZE.json'), read('submission/v3/SELECTED_MODEL.json')
assert freeze['research_only'] is True and freeze['formal_test_allowed'] is False
assert selected['name'] == r['archive_facts']['selected_name'] == 'deepsets_regularized_s0'
assert freeze['source_commit'] == r['archive_facts']['source_commit'] == 'eee478b4959e555b6ec8c902b6db8b69f31df508'
validation = read('results/research_v3/validation_summary.json')
assert (validation['baseline_validation']['correct'], validation['baseline_validation']['hard_correct']) == (740, 297)
assert sorted((v['validation']['correct'], v['validation']['hard_correct']) for v in validation['finalists']) == [(735, 292), (736, 293)]
assert all(not (ROOT / 'submission/v3' / name).exists() for name in
           ('TEST_OPENED.json', 'TEST_COMPLETE.json', 'test-score.json', 'SUBMISSION_SUCCESS.json'))
all_text = '\n'.join(texts)
for phrase in ('AI assistance was substantive', 'no accepted replacement is claimed',
               'IN-SAMPLE', 'monitoring gaps', 'No V3 official test was run',
               'no claim of uninterrupted 24-hour CPU work', 'known historical test aggregate'):
    assert phrase in all_text, phrase
assert initial == {path: sha(path) for path in initial}
visual = {f'tmp/pdfs/research-closeout-v3/page-{i:02d}.png': sha(f'tmp/pdfs/research-closeout-v3/page-{i:02d}.png')
          for i in (3, 4, 9, 10)}
review = {
    'schema': 'mp2-closeout-first-pdf-independent-qa-v1',
    'created_at_utc': datetime.now(timezone.utc).isoformat(),
    'scope': 'First actual PDF; saved aggregate reconciliation and visual pages 3,4,9,10 only. No raw bank or per-question arrays parsed.',
    'status': 'facts_and_layout_verified_with_five_documented_clarifications',
    'source_files_unchanged': initial, 'evidence_files_verified': len(evidence),
    'all_54_experiment_table_rows_match_saved_summaries_and_pdf': True,
    'independent_fit_seconds': totals, 'attempt_wall_seconds': derived['recorded_attempt_seconds'],
    'confidence_limitations_disclosed': True, 'no_new_test_or_score_update_verified': True,
    'visual_page_hashes': visual,
    'visual_observation': 'Legible tables/paragraphs, consistent margins and footers; no visible clipping, overlap or missing glyphs on the four inspected pages. Root separately reviews all ten.',
    'clarifications_for_second_attempt': [
        'Page8 fourth mixture: +2 versus Attention EMA1685 is 0 versus stronger matched TabM EMA1687; spell out named controls.',
        'Page7 Attention seed2 has zero attempts and an existing queue absolute_campaign_budget stop reason; identify it as not started.',
        'Page10 reproduce from separate exact source-commit clone before later FREEZE metadata, with fixed official data prepared; current archive intentionally refuses reproduction preflight.',
        'Page10 add primary paper title/year/URL citations instead of only pointers to local provenance docs.',
        'Page2 spell out split/fold sizes, all-four-candidate labels and selection optimism.'
    ],
    'primary_metadata_verified_urls': ['https://arxiv.org/abs/' + paper for paper in
        ('2410.24210', '2407.04491', '1703.06114', '1706.03762', '2604.15297')],
    'revised_producer_sha256': sha('reports/research_v3/prepare_closeout_inputs.py'),
    'revised_tests': {'path': 'results/research_v3/closeout-clarity-revision-tests.log',
                      'sha256': sha('results/research_v3/closeout-clarity-revision-tests.log'), 'passed': 13,
                      'note': 'Final subsequent wording-only data-fetch clarification awaits root combined suite.'},
    'final_revised_pdf_qa': 'Pending actual second render; not certified by this record.'
}
with (OUT / 'review.json').open('x', encoding='utf-8') as stream:
    json.dump(review, stream, indent=2); stream.write('\n')
print(json.dumps({'status': review['status'], 'table_rows': sum(counts.values()), 'evidence_files': len(evidence)}))
