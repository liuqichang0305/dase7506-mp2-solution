"""Prepare validation confidence events for the unique selected model only.

This post-selection diagnostic copies the already saved official predictions and
score, verifies unchanged choices, and derives chosen-group probabilities. It
does not evaluate another candidate, invoke official scoring, fit calibration,
modify the model, or open test. It also supports the preserved V2 fallback.
"""
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code/tools'))
import confirm_validation_v3 as confirmation
import freeze_v3 as freezing


def main():
    access = confirmation.install_data_guard()
    with confirmation.exclusive_evaluation():
        if any((ROOT / 'submission/v3' / name).exists() for name in ('FREEZE.json', 'TEST_OPENED.json')):
            raise ValueError('Create chosen-event validation analysis before final freeze/test')
        selected, summary, verified, assets, inventory = freezing.validate_selected(ROOT)
        selected_hash = confirmation.sha(ROOT / 'submission/v3/SELECTED_MODEL.json')
        initial_sources = freezing.source_hashes(ROOT)
        access['validation_open'] = True
        output = ROOT / 'results/research_v3/selected_confidence'
        if output.exists():
            raise FileExistsError('Preserve existing selected confidence evidence')
        generation = 'research_v2' if selected['historical_fallback'] else 'research_v3'
        original = ROOT / 'results' / generation / 'validation' / selected['name']
        folder = output / 'validation' / selected['name']
        folder.mkdir(parents=True, exist_ok=False)
        copied = {}
        for stem in ('predictions', 'score'):
            name = 'validation-' + stem + '.json'
            with (original / name).open('rb') as src, (folder / name).open('xb') as dst:
                shutil.copyfileobj(src, dst)
            if confirmation.sha(original / name) != confirmation.sha(folder / name):
                raise ValueError('Copied official evidence differs')
            copied[str((original / name).relative_to(ROOT)).replace('\\', '/')] = confirmation.sha(original / name)
        sources = dict(initial_sources)
        sources['code/tools/prepare_selected_confidence_v3.py'] = confirmation.sha(__file__)
        sources['submission/v3/SELECTED_MODEL.json'] = confirmation.sha(ROOT / 'submission/v3/SELECTED_MODEL.json')
        diagnostic_verified = {'lock_sha256': verified['lock_sha256'],
                               'lock': {'inference_source_sha256': sources}}
        finalist = {'name': selected['name'], 'method': freezing.METHOD,
                    'checkpoint': freezing.CHECKPOINT, 'selector': selected['selector']}
        original_output = confirmation.OUT
        try:
            confirmation.OUT = output
            metrics = confirmation.diagnostic_events(finalist, diagnostic_verified, {})
        finally:
            confirmation.OUT = original_output
        final_selected, final_summary, final_verified, final_assets, final_inventory = freezing.validate_selected(ROOT)
        if (final_selected != selected or final_summary != summary or final_assets != assets
                or final_verified['lock_sha256'] != verified['lock_sha256']
                or freezing.source_hashes(ROOT) != initial_sources
                or confirmation.sha(ROOT / 'submission/v3/SELECTED_MODEL.json') != selected_hash):
            raise ValueError('Selected model, source, assets or validation identity changed during analysis')
        confirmation.atomic(output / 'preparation.json', {
            'schema': 'mp2-v3-selected-confidence-preparation-v1',
            'completed_at_utc': confirmation.utc(), 'selected_model_sha256': sources['submission/v3/SELECTED_MODEL.json'],
            'historical_fallback': selected['historical_fallback'],
            'original_official_output_sha256': copied, 'metrics': metrics,
            'new_official_score_runs': 0, 'candidate_selection_changed': False,
            'rejected_data_reads': access['rejected'],
            'events': str((folder / 'validation-events.json').relative_to(ROOT)).replace('\\', '/'),
            'purpose': 'Post-selection confidence analysis of the exact saved official choices; no model/seed/calibration fitting or new candidate evaluation.'}, exclusive=True)
        print(json.dumps({'status': 'complete', 'events': str(folder / 'validation-events.json'),
                          'new_official_score_runs': 0, 'test_opened': False}))


if __name__ == '__main__':
    main()
