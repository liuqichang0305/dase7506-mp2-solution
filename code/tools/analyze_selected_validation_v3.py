"""Post-selection question-paired validation comparisons; no fitting or test."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'code/tools'), str(ROOT / 'code')]
import confirm_validation_v3 as confirmation
import freeze_v3 as freezing
from analyze_oof_v2 import paired_comparison


def main():
    access = confirmation.install_data_guard()
    destination = ROOT / 'results/research_v3/selected_validation_comparison.json'
    if destination.exists():
        raise FileExistsError('Preserve completed post-selection comparison')
    with confirmation.exclusive_evaluation():
        if any((ROOT / 'submission/v3' / name).exists() for name in ('FREEZE.json', 'TEST_OPENED.json')):
            raise ValueError('Complete development reporting before final freeze/test')
        selected, summary, verified, assets, inventory = freezing.validate_selected(ROOT)
        selected_hash = confirmation.sha(ROOT / 'submission/v3/SELECTED_MODEL.json')
        initial_sources = freezing.source_hashes(ROOT)
        access['validation_open'] = True
        import numpy as np
        from common import load_inputs, read_jsonl, parse_answer, validate_choices
        from mp2.selection_v2 import answer_groups
        manifest = confirmation.read(ROOT / 'code/data/manifest.json')
        reference_path = ROOT / 'code/data/gsm8k_validation.jsonl'
        if confirmation.sha(reference_path) != manifest['sha256']['gsm8k_validation.jsonl']:
            raise ValueError('Official validation reference hash changed')
        bank, _ = load_inputs('validation')
        references = read_jsonl(ROOT / 'code/data/gsm8k_validation.jsonl')
        if len(bank) != 798 or len(references) != 798 or any(
                b['id'] != r['id'] or b['question'] != r['question'] for b, r in zip(bank, references)):
            raise ValueError('Official validation alignment failed')
        labels = np.asarray([[parse_answer(c['text']) is not None and
                              parse_answer(c['text']) == parse_answer(r['answer'])
                              for c in b['candidates']] for b, r in zip(bank, references)], dtype=bool)
        preparation = confirmation.read(ROOT / 'results/research_v3/selected_confidence/preparation.json')
        if (preparation.get('schema') != 'mp2-v3-selected-confidence-preparation-v1'
                or preparation['selected_model_sha256'] != selected_hash):
            raise ValueError('Selected confidence events belong to another model')
        event_path = ROOT / 'results/research_v3/selected_confidence/validation' / selected['name'] / 'validation-events.json'
        if (ROOT / preparation['events']).resolve() != event_path.resolve():
            raise ValueError('Selected events must use the canonical selected-model directory')
        events = confirmation.read(event_path)
        expected_sources = dict(initial_sources)
        expected_sources['code/tools/prepare_selected_confidence_v3.py'] = confirmation.sha(ROOT / 'code/tools/prepare_selected_confidence_v3.py')
        expected_sources['submission/v3/SELECTED_MODEL.json'] = selected_hash
        if (events.get('schema') != 'mp2-v3-confidence-events-v1' or events.get('split') != 'validation'
                or events.get('selector') != selected['selector'] or events.get('source_sha256') != expected_sources
                or events.get('selection_lock') != {'path': 'results/research_v3/FINALISTS_LOCK.json', 'sha256': verified['lock_sha256']}):
            raise ValueError('Selected events do not bind the actual selector, sources and lock')
        archive = event_path.parent / 'validation-events.npz'
        if events['predictions_npz']['path'] != 'validation-events.npz':
            raise ValueError('Selected events must use the canonical NPZ')
        if confirmation.sha(archive) != events['predictions_npz']['sha256']:
            raise ValueError('Selected event array changed')
        with np.load(archive, allow_pickle=False) as arrays:
            ids, choices, correct, saved_hard, groups = [arrays[key].copy() for key in ('ids', 'choices', 'correct', 'hard', 'groups')]
        if (ids.tolist() != [row['id'] for row in bank] or choices.shape != (798,)
                or not np.issubdtype(choices.dtype, np.integer) or np.any((choices < 0) | (choices > 3))
                or correct.shape != (798,) or not np.array_equal(groups, answer_groups(bank))):
            raise ValueError('Selected events are not aligned to the complete official bank')
        hard = np.asarray([len(set(row[row >= 0])) >= 2 for row in groups], dtype=bool)
        if not np.array_equal(hard, saved_hard):
            raise ValueError('Saved hard subset differs from the fixed input-derived definition')
        generation = 'research_v2' if selected['historical_fallback'] else 'research_v3'
        origin = ROOT / 'results' / generation / 'validation' / selected['name']
        origin_predictions, origin_score = origin / 'validation-predictions.json', origin / 'validation-score.json'
        original_hashes = {str(p.relative_to(ROOT)).replace('\\', '/'): confirmation.sha(p)
                           for p in (origin_predictions, origin_score)}
        if (preparation['original_official_output_sha256'] != original_hashes
                or events['prediction_sha256'] != confirmation.sha(origin_predictions)):
            raise ValueError('Selected event evidence is not bound to the original official output')
        original_prediction, original_score = confirmation.read(origin_predictions), confirmation.read(origin_score)
        validate_choices(bank, original_prediction['ids'], original_prediction['selected'])
        if (choices.tolist() != original_prediction['selected'] or original_score['n'] != 798
                or int(correct.sum()) != original_score['correct']):
            raise ValueError('Selected events differ from the unique saved official choices or score')
        if not np.array_equal(correct, labels[np.arange(798), choices]):
            raise ValueError('Saved selected-answer correctness differs from official parsing')
        correct = correct.astype(bool)
        old = confirmation.read(ROOT / 'results/research_v2/V2_FREEZE.json')
        baselines = [('frozen_v1', ROOT / 'results'),
                     ('historical_v2_selected', ROOT / 'results/research_v2/validation' / old['selected_experiment_id'])]
        comparisons, evidence = {}, {str(event_path.relative_to(ROOT)).replace('\\', '/'): confirmation.sha(event_path),
                                     str(archive.relative_to(ROOT)).replace('\\', '/'): confirmation.sha(archive)}
        evidence.update(original_hashes)
        evidence['results/research_v3/selected_confidence/preparation.json'] = confirmation.sha(ROOT / 'results/research_v3/selected_confidence/preparation.json')
        for name, folder in baselines:
            prediction_path, score_path = folder / 'validation-predictions.json', folder / 'validation-score.json'
            prediction, score = confirmation.read(prediction_path), confirmation.read(score_path)
            if any(item.get('split') != 'validation' or item.get('protocol') != confirmation.PROTOCOL
                   for item in (prediction, score)):
                raise ValueError('Only preserved official validation outputs may serve as baselines')
            validate_choices(bank, prediction['ids'], prediction['selected'])
            baseline_correct = labels[np.arange(798), np.asarray(prediction['selected'], dtype=int)]
            if score['n'] != 798 or int(baseline_correct.sum()) != score['correct']:
                raise ValueError('Preserved baseline predictions and official score disagree')
            result = paired_comparison(correct, baseline_correct, hard, 5000, 3036901059)
            result['interval_caveat'] = 'Question-paired descriptive interval on a previously used, post-selection validation set; does not remove adaptive selection bias or establish an untouched confirmatory test.'
            comparisons[name] = result
            for path in (prediction_path, score_path):
                evidence[str(path.relative_to(ROOT)).replace('\\', '/')] = confirmation.sha(path)
        final_selected, final_summary, final_verified, final_assets, _ = freezing.validate_selected(ROOT)
        if (final_selected != selected or final_summary != summary or final_assets != assets
                or final_verified['lock_sha256'] != verified['lock_sha256']
                or freezing.source_hashes(ROOT) != initial_sources
                or confirmation.sha(ROOT / 'submission/v3/SELECTED_MODEL.json') != selected_hash):
            raise ValueError('Unique selection, source, assets or validation identity changed')
        confirmation.atomic(destination, {'schema': 'mp2-v3-selected-validation-comparison-v1',
            'completed_at_utc': confirmation.utc(), 'selected_model_sha256': selected_hash,
            'selected_name': selected['name'], 'split': 'validation',
            'comparisons': comparisons, 'evidence_sha256': evidence,
            'comparison_source_sha256': confirmation.sha(__file__),
            'bootstrap_source_sha256': confirmation.sha(ROOT / 'code/tools/analyze_oof_v2.py'),
            'data_manifest_sha256': confirmation.sha(ROOT / 'code/data/manifest.json'),
            'validation_reference_sha256': confirmation.sha(reference_path),
            'new_official_score_runs': 0, 'model_selection_changed': False,
            'test_opened': False, 'rejected_data_reads': access['rejected']}, exclusive=True)
        print(json.dumps({'status': 'complete', 'output': str(destination), 'new_official_score_runs': 0}))


if __name__ == '__main__':
    main()
