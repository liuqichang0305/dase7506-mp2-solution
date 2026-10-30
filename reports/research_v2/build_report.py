"""Build the bounded v2 research report from recorded experiment artifacts.

No training or scoring is performed. The final summary gate must exist before
authoring the final PDF. Existing original reports are never changed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
from xml.sax.saxutils import escape

for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                 'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS', 'OMP_THREAD_LIMIT'):
    os.environ[variable] = '1'

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, Frame, PageTemplate, Paragraph,
                               Spacer, Table, TableStyle, PageBreak, KeepTogether, Image)
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / 'results' / 'research_v2'
HERE = Path(__file__).resolve().parent
SOURCE_RECORDS = {}


def read(path):
    path = Path(path)
    payload = path.read_bytes()
    SOURCE_RECORDS[path.relative_to(ROOT).as_posix()] = hashlib.sha256(payload).hexdigest()
    return json.loads(payload.decode('utf-8-sig'))


def load_experiments():
    rows = []
    for path in sorted(RESULTS.glob('*/summary.json')):
        item = read(path)
        if item.get('status') == 'complete' and 'accuracy_fold_mean' in item:
            rows.append(item)
    return rows


def fmt(value, digits=2):
    return 'Not recorded' if value is None else f'{float(value):.{digits}f}'


def pct_count(row, correct='correct', total='n'):
    if correct not in row or total not in row:
        return 'Not recorded'
    return f'{row[correct]}/{row[total]} ({100 * row[correct] / row[total]:.2f}%)'


def label(experiment_id):
    names = {
        'linear_original': 'Linear / original listwise',
        'linear_group_a0': 'Linear / group CE alpha=0',
        'linear_group_a0p5': 'Linear / group CE alpha=0.5',
        'linear_group_a1': 'Linear / group CE alpha=1',
        'linear_group_hard': 'Linear / hard-normalized group CE',
        'mlp_group_s0': 'MLP / group CE',
        'resmlp_group_s0': 'ResMLP / group CE',
        'deepsets_group_s0': 'Deep Sets / group CE',
        'attention_group_s0': 'Attention / group CE',
        'mlp_listwise_s0': 'MLP / original listwise',
        'mlp_capacity_s0': 'MLP / matched capacity',
        'attention_regularized_s0': 'Attention / stronger regularization',
        'deepsets_regularized_s0': 'Deep Sets / stronger regularization',
        'attention_hard_ablation': 'Attention / hard-normalized CE',
        'deepsets_no_context': 'Deep Sets / learned pooling removed',
    }
    return names.get(experiment_id, experiment_id.replace('_', ' '))


def styles():
    font_dir = Path('C:/Windows/Fonts')
    if (font_dir / 'arial.ttf').exists():
        for name, filename in [('Body', 'arial.ttf'), ('BodyBold', 'arialbd.ttf'), ('BodyItalic', 'ariali.ttf')]:
            pdfmetrics.registerFont(TTFont(name, str(font_dir / filename)))
        pdfmetrics.registerFontFamily('Body', normal='Body', bold='BodyBold', italic='BodyItalic', boldItalic='BodyBold')
        regular, bold = 'Body', 'BodyBold'
    else:
        regular, bold = 'Helvetica', 'Helvetica-Bold'
    base = getSampleStyleSheet()
    base.add(ParagraphStyle(name='ReportTitle', fontName=bold, fontSize=23, leading=27, spaceAfter=12, textColor=colors.HexColor('#102D47')))
    base.add(ParagraphStyle(name='ReportSubtitle', fontName=regular, fontSize=10, leading=15, spaceAfter=15, textColor=colors.HexColor('#43576A')))
    base.add(ParagraphStyle(name='ReportH1', fontName=bold, fontSize=16, leading=20, spaceAfter=10, textColor=colors.HexColor('#102D47')))
    base.add(ParagraphStyle(name='ReportH2', fontName=bold, fontSize=11.2, leading=15, spaceBefore=10, spaceAfter=6, textColor=colors.HexColor('#164B68')))
    base.add(ParagraphStyle(name='ReportBody', fontName=regular, fontSize=9.4, leading=14.2, spaceAfter=8))
    base.add(ParagraphStyle(name='ReportSmall', fontName=regular, fontSize=8.0, leading=11.5, spaceAfter=6, textColor=colors.HexColor('#405061')))
    base.add(ParagraphStyle(name='ReportCell', fontName=regular, fontSize=7.3, leading=10.0, spaceAfter=0))
    base.add(ParagraphStyle(name='ReportHeadCell', fontName=bold, fontSize=7.3, leading=10.0, textColor=colors.white))
    return base


class Report:
    def __init__(self):
        self.styles = styles()
        self.story = []
        self.markdown = []

    def text(self, text, style='ReportBody'):
        for separator in ('\u2010', '\u2011', '\u2012', '\u2013', '\u2014'):
            text = text.replace(separator, '-')
        self.story.append(Paragraph(text, self.styles[style]))
        self.markdown.append(text.replace('<b>', '**').replace('</b>', '**').replace('<br/>', '\n'))

    def heading(self, text, level=1):
        self.text(text, 'ReportH1' if level == 1 else 'ReportH2')

    def page(self):
        self.story.append(PageBreak())
        self.markdown.append('\n---\n')

    def table(self, headers, rows, widths, font='ReportCell'):
        content = [[Paragraph(escape(str(c)), self.styles['ReportHeadCell']) for c in headers]]
        content += [[Paragraph(escape(str(c)), self.styles[font]) for c in row] for row in rows]
        table = Table(content, colWidths=widths, repeatRows=1, hAlign='LEFT')
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#164B68')),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('LEFTPADDING', (0, 0), (-1, -1), 6), ('RIGHTPADDING', (0, 0), (-1, -1), 6),
            ('TOPPADDING', (0, 0), (-1, -1), 5), ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#F1F5F8'), colors.white]),
            ('LINEBELOW', (0, -1), (-1, -1), 0.4, colors.HexColor('#B2C2CC')),
        ]))
        self.story.extend([table, Spacer(1, 10)])
        self.markdown.append('| ' + ' | '.join(headers) + ' |')
        self.markdown.append('| ' + ' | '.join(['---'] * len(headers)) + ' |')
        self.markdown.extend('| ' + ' | '.join(map(str, row)) + ' |' for row in rows)


def footer(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor('#C2D0DA'))
    canvas.line(42, 42, A4[0] - 42, 42)
    canvas.setFont('Helvetica', 7)
    canvas.setFillColor(colors.HexColor('#4D6374'))
    canvas.drawString(42, 29, 'DASE7506 MP2 | Research v2 | Student 3036900957')
    canvas.drawRightString(A4[0] - 42, 29, str(doc.page))
    canvas.restoreState()


def confidence_figure(finalists):
    """Plot only saved reliability bins; never open task data or recompute labels."""
    os.environ.setdefault('MPLCONFIGDIR', str(HERE / '.matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.5, 2.5), layout='constrained')
    ax.plot([0, 1], [0, 1], color='#9AA7AF', linestyle='--', linewidth=1, label='Perfect calibration')
    palette = ['#176B8E', '#C26137']
    for row, color in zip(finalists, palette):
        bins = [item for item in row['reliability']['bins'] if item['n'] > 0]
        ax.plot([item['confidence'] for item in bins], [item['accuracy'] for item in bins],
                marker='o', markersize=4, color=color, label=label(row['experiment_id']))
    ax.set(xlabel='Mean chosen-group probability within bin', ylabel='Observed correctness',
           xlim=(-0.02, 1.02), ylim=(-0.02, 1.02))
    ax.grid(alpha=0.2)
    ax.legend(loc='upper left', fontsize=7.2, frameon=False)
    ax.tick_params(labelsize=8)
    ax.xaxis.label.set_size(9)
    ax.yaxis.label.set_size(9)
    path = HERE / 'validation_reliability.png'
    fig.savefig(path, dpi=180, facecolor='white')
    plt.close(fig)
    return path


def build(summary_path, output):
    if not output.resolve().is_relative_to(HERE.resolve()):
        raise ValueError('V2 report outputs must remain under reports/research_v2; old reports are protected')
    final = read(summary_path)
    if final.get('status') in (None, 'running', 'pending'):
        raise ValueError('A completed development/finalist summary is required before PDF authoring')
    experiments = load_experiments()
    script_path = Path(__file__).resolve()
    SOURCE_RECORDS[script_path.relative_to(ROOT).as_posix()] = hashlib.sha256(script_path.read_bytes()).hexdigest()
    freeze = read(ROOT / final['freeze_path']) if final.get('freeze_path') else None
    environment = read(RESULTS / 'environment.json')
    by_id = {row['experiment_id']: row for row in experiments}
    analysis = read(ROOT / final['analysis_path']) if final.get('analysis_path') else None
    event_path = RESULTS / 'events.jsonl'
    event_bytes = event_path.read_bytes()
    SOURCE_RECORDS[event_path.relative_to(ROOT).as_posix()] = hashlib.sha256(event_bytes).hexdigest()
    final_fits = {event['experiment_id']: event for event in
                  (json.loads(line) for line in event_bytes.decode('utf-8').splitlines() if line.strip())
                  if event.get('status') == 'final_train_complete'}
    report = Report()
    W = A4[0] - 84
    report.text('Reliable answer selection\nPerformance research v2'.replace('\n', '<br/>'), 'ReportTitle')
    report.text('DASE7506 - Advanced Machine Learning Methods<br/>Student 3036900957 | 29 October 2026 | Independent research/performance-v2 branch', 'ReportSubtitle')
    report.heading('Research question', 2)
    report.text('Can answer-group alignment and small learned nonlinear candidate interactions improve the difficult disagreements in the supplied four-candidate bank, within a small CPU inference budget? The research separates decoding, objective and architecture changes, then uses common train-only out-of-fold (OOF) comparisons before a one-way validation confirmation.')
    report.heading('Scope and status', 2)
    report.text('The original submitted release and frozen checkpoint remain intact. Its previously observed test total, 348/467 (74.5182%), was already known when this work began. The present study is subsequent development, not an untouched independent experiment. No new test score or rank is reported. New formal test scoring and score updates remain closed unless an explicit applicable course process permits them.')
    for paragraph in final.get('narrative', []):
        wording = str(paragraph)
        for compact, spaced in (('from724', 'from 724'), ('to740', 'to 740'), ('hard282', 'hard 282'),
                                ('to297', 'to 297'), ('single-positive59', 'single-positive 59'),
                                ('to68', 'to 68'), ('Issue294', 'Issue #294')):
            wording = wording.replace(compact, spaced)
        report.text(escape(wording))
    report.heading('Original model and motivation', 2)
    report.text('The original 2064-dimensional representation combines each 1024-dimensional embedding, its absolute deviation from the same-question mean, and 16 cosine/parse/vote/text statistics. A standardized linear scorer uses a multi-positive listwise objective but selects the highest-scoring individual candidate. Thus it already has engineered relations, while lacking learned multilayer interactions. Shared scalar features cancel in linear within-question ranking; the absolute deviation features do not.')
    report.table(['Known original validation', 'Correct / questions', 'Interpretation'], [
        ['Overall', '724 / 798 (90.73%)', '379 questions have four correct candidates'],
        ['Hard: >=2 distinct valid answers', '282 / 355 (79.44%)', 'Fixed input-derived difficult subset'],
        ['Exactly one correct candidate', '59 / 117 (50.43%)', 'Label-derived diagnostic, never an inference feature'],
    ], [165, 130, W - 295])
    report.text('The official scorer checks the final numeric answer, not the logical validity of every reasoning step. All new numeric results below come from saved execution artifacts.', 'ReportSmall')

    report.page()
    report.heading('1. Task-aligned selection and losses')
    report.text('The unchanged official parser groups candidates by normalized numeric answer. Invalid parses have no answer group. With at least one valid candidate, they are excluded; when only one valid group remains, the course guarantee of at least one correct candidate determines the answer group. A fully unparseable synthetic bank falls back deterministically to original index 0, with diagnostics.')
    report.table(['Decoder', 'Definition / role'], [
        ['Raw candidate max', 'Original argmax over all four scores; reproduces the old baseline'],
        ['Masked candidate max / group max', 'Exclude invalid parses. Both rules are mathematically equivalent, including the implemented tie convention'],
        ['Group LSE', 'S(a) = log sum exp(s(j)/tau), j in answer group a'],
        ['Normalized group LSE', 'S(a) = log sum exp(s(j)/tau) - alpha log |G(a)|; alpha in {0, 0.5, 1}, tau=1'],
    ], [150, W - 150])
    report.text('After choosing a group, return its highest-scoring member as an original Python integer index. Equal scores use the smallest original index; exact indistinguishable ties are the documented exception to permutation equivariance. Repeated same-answer trajectories are correlated: LSE is an aggregation rule, not a proof of independent evidence.')
    report.heading('Objective controls', 2)
    report.text('The legacy loss is LSE(all candidate scores) - LSE(correct candidate scores). The group objective is LSE(valid group logits) - logit(correct answer group). With alpha=0, tau=1 and all candidates valid, these are identical. Parse masks, alpha or the normalization denominator must be identified separately when attributing gains. BCE preserves every positive candidate; multiple correct candidates are never converted into mutually exclusive negative labels.')
    report.text('Questions whose candidates are all correct provide zero listwise ranking gradient. After masking, a single valid-answer group likewise supplies no group discrimination gradient. A hard-only mean rescales the data loss. Linear L2 is adjusted by the fitting subset hard fraction to preserve the equivalent objective; AdamW decay scaling is only a controlled comparison, not an identical optimization trajectory.')
    report.heading('Small architectures, randomly initialized', 2)
    report.table(['Architecture', 'Task-specific structure', 'Parameters'], [
        ['MLP', '2064 -> 128 -> 64 -> 1, shared candidate scorer', '272,641'],
        ['ResMLP', 'Width 128, pre-LayerNorm residual block, expansion 2, shared head', '330,625'],
        ['Deep Sets', '1024 -> 128 -> 128 phi; h, mean, max, |h-mean|, h*mean and 16 statistics -> rho', '231,937'],
        ['Attention', 'Four tokens, width 128, four heads, FFN 256, no positional encoding', '273,025'],
    ], [80, W - 150, 70])
    report.text('These are original compact adaptations inspired by the cited papers [2-4], not claims to reproduce their complete benchmark systems. No pretrained weights, extra task data, teacher LLM outputs or external answer retrieval are used.', 'ReportSmall')

    report.page()
    report.heading('2. Common-fold research protocol')
    report.text('Only the official 4,755 training questions fit parameters. Three outer question-level folds are shared by every main family, stratified by positive count and valid-answer count with rare strata merged. All four candidates stay with their question. A fixed 20% inner subset of each outer training partition selects the stopping epoch. A fresh model and scaler are then fitted on the complete outer training partition for that fixed epoch count. Outer labels never choose epochs.')
    report.text('Scalers use the corresponding fitting partition only. Final all-train neural epochs are the rounded-up median of the three inner-selected epochs. The initial seed is 3036900957; promising structures add the two fixed seeds 3036900958 and 3036900959. Deployment does not choose the luckiest seed. Selection prioritizes mean hard OOF accuracy, then full accuracy, stability and resource cost.')
    report.text('All accuracy values are question-level. Hard always means at least two distinct valid parsed answers. The complete official denominator is retained; the single-positive subset is diagnostic only. OOF is still optimistic after architecture/hyperparameter selection, and seed repeats do not create independent observations. Question-paired bootstrap intervals quantify a selected comparison, not the full uncertainty of adaptive research.')
    report.heading('Main common-three-fold results', 2)
    primary = [row for row in experiments if row['experiment_id'] in ('linear_original', 'linear_group_a0', 'linear_group_a0p5', 'linear_group_a1', 'linear_group_hard', 'attention_hard_ablation', 'deepsets_no_context') or row['experiment_id'].endswith('_s0')]
    primary = sorted(primary, key=lambda x: (x['config'].get('family') != 'linear', x['experiment_id']))
    analysis_rows = {row['row_id']: row for row in analysis['rows']} if analysis else {}
    rows = []
    for r in primary:
        audit = analysis_rows.get(r['experiment_id'], {})
        mib = audit.get('audited_uncompressed_mib_max')
        parameters = max(fold['parameters'] for fold in r['folds'])
        rows.append([label(r['experiment_id']), f"{r['accuracy_fold_mean']:.2f} +/- {r['accuracy_fold_sd']:.2f}",
                     f"{r['hard_accuracy_fold_mean']:.2f} +/- {r['hard_accuracy_fold_sd']:.2f}",
                     f'{parameters:,}', fmt(mib, 3), fmt(r.get('training_seconds'), 1)])
    report.table(['Completed model / objective', 'Full mean +/- SD (%)', 'Hard mean +/- SD (%)', 'Parameters', 'Actual MiB', 'Fit seconds'],
                 rows, [155, 84, 84, 65, 58, W - 446])
    report.text('All rows completed the same three outer folds; SD is the sample fold SD. Actual MiB is the maximum audited complete fold-bundle size, counting uncompressed containers, not just numeric parameter bytes. Fit seconds include recorded inner-stopping and refit work; invocation wall time is retained separately. All nonfinalist validation entries are not evaluated by protocol. Counts and every additional seed/configuration are retained in the complete linked analysis.', 'ReportSmall')
    if 'deepsets_no_context' in by_id:
        report.text('The no-pooling Deep Sets ablation retains the 16 input statistics (including existing vote/cosine information) and reduces parameter count. It removes learned pooled interactions, not every source of context, and alone does not separate interaction from capacity. The matched-capacity MLP is an additional control.', 'ReportSmall')
    if not any(row['config'].get('family') == 'tabm' for row in experiments):
        report.text('TabM was not run. The optional additional family actually evaluated was four-candidate attention; no TabM performance is estimated or implied.', 'ReportSmall')

    if analysis:
        report.page()
        report.heading('3. Controlled decoding and seed stability')
        decoder_rows = [row for row in analysis['rows'] if row['experiment_id'] == 'linear_original']
        table_rows = []
        for row in decoder_rows:
            overall, hard = row['metrics']['overall'], row['metrics']['hard']
            table_rows.append([row['variant'], f"{overall['correct']}/{overall['n']}",
                               f"{hard['correct']}/{hard['n']}",
                               f"{hard['fold_mean']:.2f} +/- {hard['fold_sd']:.2f}"])
        report.table(['Same original linear scores', 'Full correct/n', 'Hard correct/n', 'Hard mean +/- SD (%)'],
                     table_rows, [205, 95, 95, W - 395])
        report.text('These rows isolate inference changes on identical OOF scores. Group-max is an implementation-equivalence check against masked candidate-max. None values are never counted as an answer group. The already-known v1 validation baseline is replayed separately, without opening new test evaluation.', 'ReportSmall')
        stability = analysis.get('seed_stability', [])
        if stability:
            report.heading('Fixed-seed stability', 2)
            table_rows = []
            for row in stability:
                cfg = row['configuration_without_seed']
                title = cfg.get('family', str(cfg)) if isinstance(cfg, dict) else str(cfg)
                if isinstance(cfg, dict):
                    title += f" / {cfg.get('objective', '')} / wd={cfg.get('weight_decay', 'n/a')}"
                table_rows.append([title, ', '.join(str(seed) for seed in row['seeds']),
                                   f"{row['hard_fold_mean_across_seeds']:.2f}",
                                   f"{row['hard_fold_mean_sd_across_seeds']:.2f}"])
            report.table(['Configuration', 'Fixed seeds', 'Mean hard (%)', 'Seed SD (pp)'],
                         table_rows, [190, 150, 84, W - 424])
            report.text('For each seed, the same three outer folds are reused; the reported seed SD is across mean hard fold accuracies, not an uncertainty estimate from independent datasets. Deployment uses the preregistered base seed rather than choosing the highest seed score.', 'ReportSmall')
        comparisons = analysis.get('paired_comparisons', [])
        requested = final.get('report_comparisons')
        if requested:
            pairs = {tuple(pair) for pair in requested}
            comparisons = [row for row in comparisons if (row['first'], row['second']) in pairs]
        else:
            important = {row['experiment_id'] for row in final.get('finalists', [])}
            selected = final.get('selected_experiment_id')
            comparisons = sorted(comparisons, key=lambda row: (
                not (row['first'] == selected and row['second'] == 'linear_original'),
                not (row['first'] in important or row['second'] in important),
                row['first'], row['second']))
        if comparisons:
            report.heading('Selected question-paired OOF comparisons', 2)
            table_rows = []
            for row in comparisons[:4]:
                full, hard = row['overall'], row['hard']
                lo, hi = hard['paired_bootstrap_95_interval']
                table_rows.append([row['first'] + ' vs ' + row['second'],
                                   str(full['correct_difference']), str(hard['correct_difference']),
                                   f"{hard['difference_percentage_points']:+.2f} [{lo:+.2f}, {hi:+.2f}]"])
            report.table(['First minus second', 'Full extra correct', 'Hard extra correct', 'Hard delta pp [95% CI]'],
                         table_rows, [230, 72, 72, W - 374])
            report.text('5,000 question-paired bootstrap draws; intervals are conditional on selected OOF predictions and do not remove model-selection optimism. Further comparisons, all individual folds and selected-candidate confidence diagnostics are preserved in analysis.json.', 'ReportSmall')
        if all(name in by_id for name in ('mlp_group_s0', 'mlp_listwise_s0', 'deepsets_regularized_s0', 'deepsets_no_context')):
            report.heading('What the ablations support', 2)
            mlp_group, mlp_old = by_id['mlp_group_s0'], by_id['mlp_listwise_s0']
            contextual, independent = by_id['deepsets_regularized_s0'], by_id['deepsets_no_context']
            report.text(f"Group-objective alignment was not uniformly better: the MLP group-loss control selected {mlp_group['hard_correct']}/{mlp_group['hard_n']} hard questions correctly, versus {mlp_old['hard_correct']}/{mlp_old['hard_n']} with the original listwise loss. Removing learned Deep Sets pooling gave {independent['hard_correct']}/{independent['hard_n']}, compared with {contextual['hard_correct']}/{contextual['hard_n']} for pooled Deep Sets. This does not establish a benefit from that specific pooling design; retained statistics and changed capacity limit causal attribution.", 'ReportSmall')
            report.text('The regularized neural configuration changed dropout, learning rate and AdamW decay together. Its comparison evaluates that registered configuration, not the separate causal effect of each knob. The hard-normalization control is reported independently; a smaller loss denominator is not itself evidence for a stronger architecture.', 'ReportSmall')

    ensemble_path = ROOT / final['ensemble_path'] if final.get('ensemble_path') else None
    if ensemble_path is not None and ensemble_path.is_dir():
        ensemble_path = ensemble_path / 'summary.json'
    ensemble = read(ensemble_path) if ensemble_path is not None else None
    if ensemble and ensemble.get('ensemble_evaluated'):
        report.page()
        report.heading('4. Complementary errors did not guarantee a gain')
        gate = ensemble['gate']
        report.text(f"The predeclared complementarity gate was met: the first model alone corrected {gate['first_only_correct']} errors of the second, and the second alone corrected {gate['second_only_correct']} of the first. Their hard-set oracle gain over the stronger member was {gate['hard_oracle_gain_over_stronger_percentage_points']:.2f} percentage points. An oracle uses unavailable correctness information and is only a diagnostic upper bound.")
        report.text('Exactly one equal-weight average of aligned answer-group probabilities was evaluated on the shared OOF predictions. Raw logits were not averaged, no mixture weight grid was tried, and no validation references were used to tune or fit the combination.')
        table_rows = []
        for name in (ensemble['first'], ensemble['second']):
            metric = ensemble['member_metrics'][name]
            table_rows.append([label(name), f"{metric['overall']['fold_mean']:.2f} +/- {metric['overall']['fold_sd']:.2f}",
                               f"{metric['hard']['fold_mean']:.2f} +/- {metric['hard']['fold_sd']:.2f}",
                               f"{metric['hard']['correct']}/{metric['hard']['n']}"])
        metric = ensemble['metrics']
        table_rows.append(['Equal probability mixture', f"{metric['overall']['fold_mean']:.2f} +/- {metric['overall']['fold_sd']:.2f}",
                           f"{metric['hard']['fold_mean']:.2f} +/- {metric['hard']['fold_sd']:.2f}",
                           f"{metric['hard']['correct']}/{metric['hard']['n']}"])
        report.table(['OOF method', 'Full mean +/- SD (%)', 'Hard mean +/- SD (%)', 'Hard correct/n'],
                     table_rows, [185, 115, 115, W - 415])
        strongest = max(ensemble['member_metrics'], key=lambda name: ensemble['member_metrics'][name]['hard']['correct'])
        comparison = next(row for row in ensemble['paired_comparisons'] if row['second'] == strongest)
        effect = comparison['hard']
        lo, hi = effect['paired_bootstrap_95_interval']
        report.text(f"The mixture changed the hard result by {effect['correct_difference']:+d} correct questions relative to {escape(label(strongest))}: {effect['difference_percentage_points']:+.2f} percentage points, with a paired 95% interval [{lo:+.2f}, {hi:+.2f}]. It was not retained for deployment or new validation scoring. The interval does not establish a systematic loss, but the predeclared comparison provided no reason to add the extra model.")
        member_bytes = max(row['member_audited_uncompressed_bytes_sum'] for row in ensemble['assets_per_fold'])
        parameters = max(row['parameters'] for row in ensemble['assets_per_fold'])
        report.table(['Measured OOF combination cost', 'Value'], [
            ['Combined model parameters', f'{parameters:,}'],
            ['Sum of audited member bundle assets', f'{member_bytes:,} bytes ({member_bytes / 2**20:.3f} MiB)'],
            ['Additional fitting', f"{ensemble['additional_training_seconds']:.3f} seconds"],
            ['Previously recorded member fitting reused', f"{ensemble['existing_training_seconds_reused']:.3f} seconds"],
            ['Probability combination (in-process)', f"{ensemble['group_probability_combine_seconds']:.4f} seconds"],
            ['New official validation / test', 'Not evaluated / disabled'],
        ], [290, W - 290])
        report.text('The member-asset sum is an actual inventory, not a deployed wrapper size. A deployed mixture would also have to count its configuration and measure a fresh prediction-plus-scoring process; no such claim is made for this rejected OOF combination.', 'ReportSmall')
        report.heading('Recorded timing deviation', 2)
        deviation = ROOT / 'docs/research_v2/BUDGET_DEVIATION.md'
        if deviation.exists():
            SOURCE_RECORDS[deviation.relative_to(ROOT).as_posix()] = hashlib.sha256(deviation.read_bytes()).hexdigest()
            report.text('The first seed batch had a 900-second wall budget. Its first experiment completed after 1124.3837 seconds (1098.3894 seconds of summed fitting wall intervals); the parent stopped before starting the next experiment. This cap was exceeded in the observed run. The completed seed, fold predictions and logs were retained, then remaining registered seeds resumed. The cause of the long wall interval is unestablished; no guessed CPU-only time replaces it. See BUDGET_DEVIATION.md.', 'ReportSmall')

    finalists = final.get('finalists', [])
    if finalists:
        report.page()
        report.heading(('5' if ensemble else '4') + '. Locked validation confirmation')
        report.text('The search was closed and at most two finalists, configurations, decoders and assets were locked before new validation access. All 798 validation questions were predicted before their references were loaded for diagnostics. No result on this page reopens architecture or hyperparameter search.')
        validation_path = RESULTS / 'validation_summary.json'
        validation_record = read(validation_path) if validation_path.exists() else {}
        baseline = validation_record.get('baseline', {'correct': 724, 'n': 798, 'hard_correct': 282, 'hard_n': 355,
                                                     'single_positive_correct': 59, 'single_positive_n': 117})
        table_rows = [['Frozen v1 (replayed)', pct_count(baseline), pct_count(baseline, 'hard_correct', 'hard_n'),
                       pct_count(baseline, 'single_positive_correct', 'single_positive_n')]]
        for row in finalists:
            metric = row['validation']
            title = label(row['experiment_id'])
            if row['experiment_id'] == final.get('selected_experiment_id'):
                title += ' [selected]'
            table_rows.append([title, pct_count(metric), pct_count(metric, 'hard_correct', 'hard_n'),
                               pct_count(metric, 'single_positive_correct', 'single_positive_n')])
        report.table(['Finalist', 'Full correct/n', 'Hard correct/n', 'Single-positive correct/n'],
                     table_rows, [175, 112, 112, W - 399])
        report.heading('Paired differences from the preserved v1 selector', 2)
        table_rows = []
        for row in finalists:
            effects = row['bootstrap_vs_frozen_v1']
            cells = [label(row['experiment_id'])]
            for subset in ('overall', 'hard'):
                effect = effects[subset]
                lo, hi = effect['paired_bootstrap_95_interval']
                cells.append(f"{effect['correct_difference']:+d} correct; {effect['difference_percentage_points']:+.2f} pp [{lo:+.2f}, {hi:+.2f}]")
            table_rows.append(cells)
        report.table(['Finalist minus v1', 'Full delta [95% paired CI]', 'Hard delta [95% paired CI]'],
                     table_rows, [170, 170, W - 340])
        report.text('The intervals use 5,000 paired question resamples and remain conditional on the small preselected finalist set. Counts on the single-positive subset disclose a possible tradeoff rather than serving as a new selection objective.', 'ReportSmall')
        if len(finalists) == 2:
            chosen = next(row for row in finalists if row['experiment_id'] == final['selected_experiment_id'])
            other = next(row for row in finalists if row['experiment_id'] != final['selected_experiment_id'])
            full_difference = chosen['validation']['correct'] - other['validation']['correct']
            single_difference = chosen['validation']['single_positive_correct'] - other['validation']['single_positive_correct']
            report.text(f"The preregistered validation rule selected {escape(label(chosen['experiment_id']))}. Relative to the other locked finalist, it changed full validation by {full_difference:+d} questions and the single-positive subset by {single_difference:+d}. This small between-finalist difference does not establish stable architectural superiority; train-OOF and fixed-seed evidence must be read alongside it. No hyperparameters or seeds were changed after confirmation.")
        report.page()
        report.heading(('6' if ensemble else '5') + '. Resource checks and reliability')
        report.text('Reported deployment costs include every selected model and preprocessing artifact. Training cost appears separately in the common-fold table. The official constraints are at most 16 MiB of uncompressed extra assets and at most 30 seconds for complete CPU prediction plus independent scoring, with no more than four total compute threads.')
        table_rows = []
        for row in finalists:
            table_rows.append([label(row['experiment_id']), final_fits.get(row['experiment_id'], {}).get('epochs_run', 'Not recorded'),
                               f"{row['resource']['uncompressed_bytes'] / 2**20:.3f}",
                               fmt(row['prediction_plus_scoring_seconds'], 3), fmt(row['reliability']['brier'], 4),
                               fmt(row['reliability']['ece'], 4)])
        report.table(['Finalist', 'Final epochs', 'All assets MiB', 'Fresh predict + score seconds', 'Brier', 'ECE'],
                     table_rows, [170, 50, 65, 105, 60, W - 450])
        report.text('Assets count every deployed file using uncompressed container sizes. Time encloses fresh CPU prediction and a separate official scoring process, including imports, loading and preprocessing. Brier/ECE concern the actual selected group; probabilities are uncalibrated and the old calibrator is not reused. This is local measurement, not reference-Xeon certification.', 'ReportSmall')
        figure = confidence_figure(finalists)
        report.story.append(Image(str(figure), width=430, height=165.4, hAlign='LEFT'))
        report.markdown.append('![Validation chosen-group reliability](validation_reliability.png)')
        report.text('Reliability diagram: ten equal-width probability bins; empty bins are omitted. Each plotted value is an aggregate of already saved validation diagnostics. A point near the diagonal does not establish calibration under hidden-bank distribution shift.', 'ReportSmall')
        reproduction = final.get('reproduction', {})
        if reproduction:
            report.heading('Independent environment reproduction', 2)
            if isinstance(reproduction, dict) and 'matching_choices' in reproduction:
                report.text(f"A fresh public clone at frozen commit {escape(reproduction['source_commit'])} and a new NumPy-only environment independently downloaded official validation inputs/references. The preserved checkpoint reproduced {reproduction['correct']}/{reproduction['n']}; all {reproduction['matching_choices']} selected indices matched the original confirmation. All {reproduction['source_and_asset_hashes_verified']} frozen source/asset hashes matched. Complete fresh prediction plus separate scoring took {reproduction['prediction_plus_scoring_seconds']:.4f} seconds. The checkpoint was not retrained; no train or test data were present.", 'ReportSmall')
                report.text('Reproduction dependencies: ' + escape(', '.join(reproduction['packages'])) + '. The different local timing is retained alongside the original confirmation timing, rather than replacing it. Evidence: ' + escape(reproduction['evidence']) + '.', 'ReportSmall')
            elif isinstance(reproduction, dict):
                for key in ('status', 'environment', 'python', 'checkpoint', 'prediction_plus_scoring_seconds', 'choices_equal', 'note'):
                    if key in reproduction:
                        report.text(escape(f'{key}: {reproduction[key]}'), 'ReportSmall')
            else:
                report.text(escape(str(reproduction)), 'ReportSmall')
        report.text('The artifact audit inventories numeric arrays and complete uncompressed containers. Synthetic contract checks exercise arbitrary N, changed IDs, question/candidate permutations, valid/invalid parses, reference-free offline inference and NumPy-only imports. Full finalist export comparison separately checks all validation input scores and selected indices against Torch without opening references.', 'ReportSmall')

    # Optional factual notes supplied by the finalizer, never inferred from
    # incomplete experiments or from unstored performance estimates.
    sections = final.get('report_sections', [])
    for section in sections:
        report.page()
        report.heading(escape(section['title']))
        for paragraph in section.get('paragraphs', []):
            report.text(escape(str(paragraph)))
        for table in section.get('tables', []):
            headers = table['headers']
            widths = table.get('widths') or [W / len(headers)] * len(headers)
            report.table(headers, table['rows'], widths)
            if table.get('note'):
                report.text(escape(table['note']), 'ReportSmall')

    report.page()
    report.heading('Reproducibility, limits and disclosure')
    if freeze:
        commit = freeze['source_commit']
        report.text(f'Frozen v2 inference source: <link href="https://github.com/liuqichang0305/dase7506-mp2-solution/tree/{commit}">{commit}</link><br/>Checkpoint: {escape(freeze["checkpoint"])}<br/>Evaluation state: {escape(freeze["allowed_evaluation_state"])}', 'ReportSmall')
    report.text('Every completed experiment retains its ID, configuration, seed, source/data hashes, fold membership, OOF scores, fitting logs, failure information and numeric model assets. Existing search guards and v1 assets are preserved. Research entry points reject test evaluation. The independent v2 interface loads only supplied inputs and numeric checkpoint arrays; no reference file, ID lookup, online model or new generation is needed.')
    report.text('Training uses PyTorch on the local CPU. Inference is NumPy-only. Export verification compares complete finalist validation input scores and actual choices against the reconstructed Torch model, without reading reference answers. Fresh-process inference plus a separate official scoring process includes startup, imports, asset loading and features. All deployed members and preprocessing statistics count toward the 16 MiB uncompressed asset limit.')
    report.text(f"Recorded host: {escape(environment['processor'])}; {escape(environment['platform'])}; {environment['ram_bytes'] / 2**30:.2f} GiB RAM. Training/inference use CPU with one configured compute thread. Python 3.12; PyTorch {escape(environment['packages']['torch'])}; NumPy {escape(environment['packages']['numpy'])}. Local timing does not certify performance on the course reference Xeon Platinum 8457C.")
    report.heading('Interpretation limits', 2)
    report.text('The research bank is small, includes many unambiguous or all-positive questions, and differs from difficult hidden evaluation inputs. Full accuracy can mask weakness on disagreements or single-positive cases. Validation was already used in v1 development and is therefore a development confirmation, not a newly untouched test set. Bootstrap intervals do not eliminate OOF selection bias, repeated-development bias or distribution shift. A validation-hard percentage cannot be read as a promised test percentage or rank.')
    report.text('Confidence is evaluated for the actually selected answer group. Raw group probabilities are not automatically calibrated correctness probabilities, and the old candidate-max calibrator is not reused. Brier and ECE are descriptive diagnostics; formal selection never abstains. Ensembles are considered only after paired OOF error complementarity and count all members toward compute and storage.')
    report.heading('AI and implementation disclosure', 2)
    report.text('OpenAI Codex agents assisted with design, source inspection, implementation, experiments, testing, documentation and this report. Numerical claims are sourced from saved real runs rather than generated estimates. Neural modules are original task-specific implementations using PyTorch and NumPy; cited architectures are conceptual sources, not imported pretrained artifacts. Independent AI code review is documented as such and is not a substitute for instructor verification.')
    report.text('Reference-repository license checks are saved in docs/research_v2/rules_and_sources.md and results/research_v2/rules_evidence.json: rtdl-revisiting-models and DeepSets use Apache-2.0, while set_transformer uses MIT. These codebases were not copied into the project; the licenses and paper sources are disclosed to distinguish architectural inspiration from code reuse.', 'ReportSmall')
    report.text('Original release: 419e1158ba2f83b8c08796429e635f1bf8ab2977. The v2 freeze records the final source commit, configuration, asset hashes and permitted evaluation state. See docs/research_v2, results/research_v2 and the v2 README commands for the complete table, exact checkpoints, review and fresh-environment reproduction evidence.', 'ReportSmall')
    report.heading('References', 2)
    references = [
        '[1] Course guide and official interface. https://github.com/xudongwu-0/DASE7506-MP2/tree/bb801f4f2e7629d270df765ab12c333b2dc616bb',
        '[2] Gorishniy et al. Revisiting Deep Learning Models for Tabular Data. NeurIPS 2021. https://arxiv.org/abs/2106.11959',
        '[3] Zaheer et al. Deep Sets. NeurIPS 2017. https://arxiv.org/abs/1703.06114',
        '[4] Lee et al. Set Transformer. ICML 2019. https://proceedings.mlr.press/v97/lee19d.html',
        '[5] Gorishniy et al. TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling. ICLR 2025. https://arxiv.org/abs/2410.24210. Optional route; no run is claimed unless present in the experiment table.',
        '[6] PyTorch project, https://pytorch.org/ ; NumPy project, https://numpy.org/ . Software versions and notices are recorded with the environment.',
    ]
    for item in references:
        report.text(escape(item), 'ReportSmall')
    output.parent.mkdir(parents=True, exist_ok=True)
    doc = BaseDocTemplate(str(output), pagesize=A4, leftMargin=42, rightMargin=42,
                          topMargin=43, bottomMargin=53, title='DASE7506 MP2: Performance research v2',
                          author='Student 3036900957')
    doc.addPageTemplates(PageTemplate(id='report', frames=Frame(42, 53, W, A4[1] - 96, leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0), onPage=footer))
    doc.build(report.story)
    pages = len(PdfReader(str(output)).pages)
    if pages > 10:
        raise ValueError(f'Report exceeds ten-page course limit: {pages}')
    (HERE / 'report_content.md').write_text('\n\n'.join(report.markdown) + '\n', encoding='utf-8')
    record = {'generated_utc': datetime.now(timezone.utc).isoformat(), 'pages': pages,
              'report_sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
              'source_artifacts_sha256': SOURCE_RECORDS, 'output': output.relative_to(ROOT).as_posix(),
              'visual_qa': 'Pending render and visual inspection; see VISUAL_QA.md after inspection'}
    (HERE / 'build_record.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'pages': pages, 'pdf': str(output), 'source_artifacts': len(SOURCE_RECORDS)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', type=Path, default=RESULTS / 'final_summary.json')
    parser.add_argument('--output', type=Path, default=HERE / 'report_v2.pdf')
    args = parser.parse_args()
    build(args.summary, args.output)


if __name__ == '__main__':
    main()
