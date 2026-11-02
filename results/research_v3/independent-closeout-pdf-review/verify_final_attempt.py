"""Final PDF evidence check; only saved summaries and rendered report artifacts."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import re
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
def sha(name):
    with (ROOT / name).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()
def read(name):
    return json.loads((ROOT / name).read_text(encoding='utf-8-sig'))
def normalized(text):
    return re.sub(r'\s+', ' ', text).strip()

folder = 'reports/research_v3/research_closeout_final/'
pdf = folder + 'research_closeout_v3.pdf'
inputs = 'reports/research_v3/closeout_inputs_final.json'
names = [pdf, inputs, folder + 'report_content.md', folder + 'build_record.json',
         'reports/research_v3/closeout_zh_final.md', 'reports/research_v3/prepare_closeout_inputs.py',
         'reports/research_v3/build_closeout.py']
before = {name: sha(name) for name in names}
assert before[pdf] == 'bb8bebb96c7007403fbc782ea8cce26c2b421ced6551e3fbe99b990f3c1035b3'
r, build = read(inputs), read(folder + 'build_record.json')
assert r['generator_sha256'] == build['producer_sha256'] == before[names[-2]]
assert build['builder_sha256'] == before[names[-1]]
assert build['inputs_sha256'] == before[inputs] and build['output_sha256'] == before[pdf]
assert r['research_only'] and r['no_new_test'] and r['no_new_score_update']
for path, digest in r['evidence_sha256'].items():
    canonical = (ROOT / path).resolve().relative_to(ROOT.resolve()).as_posix()
    assert canonical == path and not canonical.startswith('code/data/')
    assert sha(path) == digest, path
new = [normalized(p.extract_text()) for p in PdfReader(ROOT / pdf).pages]
old = [normalized(p.extract_text()) for p in PdfReader(ROOT / 'reports/research_v3/research_closeout/research_closeout_v3.pdf').pages]
assert len(new) == build['pages'] == build['intended_pages'] == 10
for page in (1, 3, 4, 5, 6, 9):
    assert new[page - 1] == old[page - 1], page
# Therefore all 54 table rows, unchanged metrics/costs and validation/confidence
# already independently reconciled in review.json remain exact in this PDF.
prior = read('results/research_v3/independent-closeout-pdf-review/review.json')
assert prior['all_54_experiment_table_rows_match_saved_summaries_and_pdf'] is True
for path, digest in prior['source_files_unchanged'].items():
    assert sha(path) == digest, path
protocol = r['derived_aggregates']['protocol']
assert protocol == {'train_n': 4755, 'folds': 3, 'outer_fit_n': 3170, 'holdout_n': 1585,
                    'inner_fit_n': 2536, 'inner_stop_n': 634}
for text in ('4755 training questions', '798 validation questions', '467 official test questions (V3 test unopened)',
             '2536 inner-fit and 634 stopping', 'alternate correct answers stay positive', 'selection-optimistic'):
    assert text in new[1], text
assert 'Not started: absolute_campaign_budget' in new[6]
assert 'No reason in the saved attempt' not in new[6]
last = r['derived_aggregates']['fusion_comparisons'][-1]
assert (last['attention_correct'], last['matched_correct'], last['hard_correct']) == (1685, 1687, 1687)
assert (last['attention_delta'], last['matched_delta']) == (2, 0)
assert '+2 hard questions versus v3_attention_ema_s0 (1685 correct)' in new[7]
assert '+0 versus the matched v3_tabm_embedding_stats_ema_s0 (1687 correct); rejected' in new[7]
for text in ('separate clone checked out at this exact source commit', 'code/tools/fetch_official.py',
             'isolated evaluation checkout receives validation files only', 'preflight intentionally refuses the current frozen',
             'do not delete FREEZE', 'AI assistance was substantive', 'not instructor approval'):
    assert text in new[9], text
assert r['archive_facts']['source_commit'] in new[9]
for paper in ('2410.24210', '2407.04491', '1703.06114', '1706.03762', '2604.15297v1'):
    assert 'https://arxiv.org/abs/' + paper in new[9]
assert before == {name: sha(name) for name in names}
images = {f'tmp/pdfs/research-closeout-v3-final/page-{n:02d}.png': sha(f'tmp/pdfs/research-closeout-v3-final/page-{n:02d}.png')
          for n in (2, 7, 8, 10)}
result = {'schema': 'mp2-closeout-final-pdf-independent-qa-v1',
    'created_at_utc': datetime.now(timezone.utc).isoformat(), 'status': 'passed', 'blocking_findings': [],
    'scope': 'Saved evidence and PDF extraction; visual pages2,7,8,10. No raw bank, training, inference, scoring or network submission.',
    'final_artifact_sha256': before, 'evidence_files_reverified': len(r['evidence_sha256']),
    'first_review': {'path': 'results/research_v3/independent-closeout-pdf-review/review.json',
                     'sha256': sha('results/research_v3/independent-closeout-pdf-review/review.json')},
    'unchanged_pages_extracted_text_equal_to_verified_first_attempt': [1,3,4,5,6,9],
    'clarifications_verified': ['saved nested fold sizes and all-candidate labels',
        'zero-attempt queue reason', 'named fusion controls and zero improvement versus stronger member',
        'exact source commit separate-clone preflight plus fixed-data preparation', 'five primary title/year/URL references'],
    'visual_page_sha256': images,
    'visual_observation': 'All four revised pages fit their intended page with clear, legible type, intact table rows, aligned columns and adequate margins. No clipping, overlaps, overflow or missing-glyph blocks observed. Citations and explicit archive limitation remain readable.',
    'all_ten_pages_visual_scope': 'Root separately performs the complete ten-page visual QA; this independent review does not claim it.'}
with (OUT / 'final_review.json').open('x', encoding='utf-8') as stream:
    json.dump(result, stream, indent=2); stream.write('\n')
print(json.dumps({'status': result['status'], 'pages': len(new), 'evidence_files': len(r['evidence_sha256'])}))
