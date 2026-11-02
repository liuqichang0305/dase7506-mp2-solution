"""Render the separate post-deadline research archive after evidence completion.

Reuses the preserved V2 typography. Never changes or bypasses the formal
TEST_COMPLETE-gated report builder. Root performs final rendered-page visual QA.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('post_deadline_closeout_inputs', HERE / 'prepare_closeout_inputs.py')
producer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(producer)


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_inputs(record, root=ROOT):
    producer.require(record.get('schema') == 'mp2-v3-post-deadline-report-inputs-v1'
        and all(record.get(key) is True for key in ('research_only', 'no_new_test', 'no_new_score_update')),
        'Only the separate post-deadline research report schema is accepted')
    producer.require(record.get('generator_sha256') == digest(HERE / 'prepare_closeout_inputs.py'),
                     'Closeout input producer changed; regenerate into a new path')
    # Reconcile the exact report text/derived numbers with the completed records,
    # not merely a self-consistent arbitrary evidence map or altered page text.
    actual = producer.prepare(root, **record['paths'])
    for key in ('title', 'student_id', 'pages', 'evidence_sha256', 'derived_aggregates', 'archive_facts'):
        producer.require(record[key] == actual[key], 'Closeout inputs differ from actual completed evidence: ' + key)
    producer.require(1 <= len(record['pages']) <= 10, 'At most ten intended pages are allowed')
    return actual


def render(inputs_path, output_directory, root=ROOT):
    inputs_path, output_directory = Path(inputs_path), Path(output_directory)
    inputs = json.loads(inputs_path.read_text(encoding='utf-8-sig'))
    validate_inputs(inputs, root)
    output = output_directory / 'research_closeout_v3.pdf'
    receipt = output_directory / 'build_record.json'
    content = output_directory / 'report_content.md'
    producer.require(not any(path.exists() for path in (output, receipt, content)),
                     'Preserve previous build evidence; use a new output directory')
    source = ROOT / 'reports/research_v2/build_report.py'
    producer.require(digest(source) == inputs['evidence_sha256']['reports/research_v2/build_report.py'],
                     'Preserved layout source differs from the bound source')
    spec = importlib.util.spec_from_file_location('preserved_v2_closeout_layout', source)
    layout = importlib.util.module_from_spec(spec); spec.loader.exec_module(layout)
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import BaseDocTemplate, Frame, PageTemplate
    from pypdf import PdfReader
    def footer(canvas, doc):
        canvas.saveState(); canvas.setStrokeColor(colors.HexColor('#C2D0DA'))
        canvas.line(42, 42, A4[0] - 42, 42); canvas.setFont('Helvetica', 7)
        canvas.setFillColor(colors.HexColor('#4D6374'))
        canvas.drawString(42, 29, 'DASE7506 MP2 | Research archive only - no V3 test or score update | 3036900957')
        canvas.drawRightString(A4[0] - 42, 29, str(doc.page)); canvas.restoreState()
    report = layout.Report(); width = A4[0] - 84
    for index, page in enumerate(inputs['pages']):
        if index: report.page()
        for item in page['blocks']:
            kind = item['kind']
            if kind in ('title', 'subtitle', 'paragraph', 'small'):
                style = {'title': 'ReportTitle', 'subtitle': 'ReportSubtitle',
                         'paragraph': 'ReportBody', 'small': 'ReportSmall'}[kind]
                report.text(escape(item['text']).replace('\n', '<br/>'), style)
            elif kind == 'heading': report.heading(escape(item['text']), int(item.get('level', 1)))
            elif kind == 'table':
                proportions = item['widths']
                report.table(item['headers'], item['rows'], [width * value / sum(proportions) for value in proportions])
            else: raise ValueError('Unexpected closeout block: ' + kind)
    output_directory.mkdir(parents=True, exist_ok=True)
    document = BaseDocTemplate(str(output), pagesize=A4, title=inputs['title'],
        author='Student 3036900957; substantive AI assistance disclosed',
        leftMargin=42, rightMargin=42, topMargin=42, bottomMargin=53)
    document.addPageTemplates(PageTemplate(id='body', frames=Frame(42, 53, width, A4[1] - 95,
        leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0), onPage=footer))
    document.build(report.story)
    pages = len(PdfReader(output).pages)
    with content.open('x', encoding='utf-8', newline='\n') as stream: stream.write('\n\n'.join(report.markdown) + '\n')
    record = {'schema': 'mp2-v3-post-deadline-report-build-v1', 'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'research_only': True, 'no_new_test': True, 'no_new_score_update': True, 'pages': pages,
        'within_ten_pages': pages <= 10, 'intended_pages': len(inputs['pages']),
        'output_sha256': digest(output), 'inputs_sha256': digest(inputs_path), 'builder_sha256': digest(__file__),
        'producer_sha256': digest(HERE / 'prepare_closeout_inputs.py'), 'layout_source_sha256': digest(source),
        'evidence_sha256': inputs['evidence_sha256'], 'visual_qa_status': 'pending_render_and_individual_page_inspection'}
    with receipt.open('x', encoding='utf-8', newline='\n') as stream: json.dump(record, stream, indent=2); stream.write('\n')
    producer.require(pages <= 10, 'Rendered archive exceeds ten pages; preserve this attempt and revise layout')
    return {'output': str(output), 'pages': pages, 'sha256': record['output_sha256']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, default=HERE / 'closeout_inputs.json')
    parser.add_argument('--output-directory', type=Path, default=HERE / 'research_closeout')
    args = parser.parse_args()
    print(json.dumps(render(args.inputs, args.output_directory)))


if __name__ == '__main__': main()
