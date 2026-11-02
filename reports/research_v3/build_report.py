"""Render the evidence-bound final V3 report; never train or evaluate data.

The established V2 report typography is reused without editing its source or
outputs. Content is supplied only after search closure and final evaluation.
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
spec = importlib.util.spec_from_file_location('preserved_v2_report_layout', ROOT / 'reports/research_v2/build_report.py')
layout = importlib.util.module_from_spec(spec)
spec.loader.exec_module(layout)
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.platypus import BaseDocTemplate, Frame, PageTemplate, Image
from pypdf import PdfReader


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def footer(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor('#C2D0DA'))
    canvas.line(42, 42, A4[0] - 42, 42)
    canvas.setFont('Helvetica', 7)
    canvas.setFillColor(colors.HexColor('#4D6374'))
    canvas.drawString(42, 29, 'DASE7506 MP2 | Autonomous research v3 | Student 3036900957')
    canvas.drawRightString(A4[0] - 42, 29, str(doc.page))
    canvas.restoreState()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, default=HERE / 'report_inputs.json')
    args = parser.parse_args()
    inputs = read(args.inputs)
    if inputs.get('schema') != 'mp2-v3-final-report-inputs-v1':
        raise ValueError('Final report inputs require the versioned evidence schema')
    sources = inputs['evidence_sha256']
    for relative, expected in sources.items():
        path = (ROOT / relative).resolve()
        if not path.is_relative_to(ROOT) or digest(path) != expected:
            raise ValueError(f'Report evidence mismatch: {relative}')
    required = ('results/research_v3/SEARCH_CLOSED.json',
                'results/research_v3/validation_summary.json',
                'submission/v3/SELECTED_MODEL.json', 'submission/v3/FREEZE.json',
                'submission/v3/REVIEW_RESULT.json', 'submission/v3/REPRODUCTION_RESULT.json',
                'submission/v3/TEST_COMPLETE.json', 'submission/v3/test-score.json',
                'submission/v3/test-timing.json')
    if not set(required).issubset(sources):
        raise ValueError('Final report must bind every actual selection/freeze/test evidence file')
    if read(ROOT / required[0]).get('search_closed') is not True:
        raise ValueError('Research must be closed')
    if read(ROOT / 'submission/v3/FREEZE.json').get('frozen') is not True:
        raise ValueError('A genuine completed freeze is required')
    complete = read(ROOT / 'submission/v3/TEST_COMPLETE.json')
    if complete.get('status') != 'passed':
        raise ValueError('A final result cannot be manufactured from unfinished evaluation')
    sections = inputs['pages']
    if not 1 <= len(sections) <= 10:
        raise ValueError('At most ten intended pages')
    report = layout.Report()
    width = A4[0] - 84
    for index, page in enumerate(sections):
        if index:
            report.page()
        for block in page['blocks']:
            kind = block['kind']
            if kind == 'title':
                report.text(escape(block['text']).replace('\n', '<br/>'), 'ReportTitle')
            elif kind == 'subtitle':
                report.text(escape(block['text']).replace('\n', '<br/>'), 'ReportSubtitle')
            elif kind == 'heading':
                report.heading(escape(block['text']), int(block.get('level', 1)))
            elif kind in ('paragraph', 'small'):
                report.text(escape(block['text']).replace('\n', '<br/>'),
                            'ReportSmall' if kind == 'small' else 'ReportBody')
            elif kind == 'table':
                proportions = block['widths']
                report.table(block['headers'], block['rows'],
                             [width * part / sum(proportions) for part in proportions])
            elif kind == 'image':
                path = (ROOT / block['path']).resolve()
                if block['path'] not in sources:
                    raise ValueError('Figure must be bound to evidence map')
                report.story.append(Image(str(path), width=width, height=block['height']))
                report.text(escape(block['caption']), 'ReportSmall')
            else:
                raise ValueError(f'Unknown report block: {kind}')
    output = HERE / 'report_v3.pdf'
    document = BaseDocTemplate(str(output), pagesize=A4, title=inputs['title'],
        author='Student 3036900957; substantive AI assistance disclosed in report',
        leftMargin=42, rightMargin=42, topMargin=42, bottomMargin=53)
    document.addPageTemplates(PageTemplate(id='body',
        frames=Frame(42, 53, width, A4[1] - 95, leftPadding=0,
                     rightPadding=0, topPadding=0, bottomPadding=0), onPage=footer))
    document.build(report.story)
    pages = len(PdfReader(output).pages)
    (HERE / 'report_content.md').write_text('\n\n'.join(report.markdown) + '\n', encoding='utf-8')
    record = {'schema': 'mp2-v3-report-build-v1',
              'created_at_utc': datetime.now(timezone.utc).isoformat(),
              'pages': pages, 'within_ten_pages': pages <= 10,
              'intended_pages': len(sections),
              'output_sha256': digest(output),
              'inputs_sha256': digest(args.inputs),
              'builder_sha256': digest(__file__),
              'layout_source_sha256': digest(ROOT / 'reports/research_v2/build_report.py'),
              'evidence_sha256': sources,
              'visual_qa_status': 'pending_render_and_individual_page_inspection'}
    (HERE / 'build_record.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'output': str(output), 'pages': pages, 'sha256': record['output_sha256']}))
    if pages > 10:
        raise ValueError('Report exceeds ten-page limit; revise layout before publication')


if __name__ == '__main__':
    main()
