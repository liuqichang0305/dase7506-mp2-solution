"""Close research, then seal exact assets for at most two confirmations."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/research_v2'
sys.path.insert(0, str(ROOT / 'code/tools'))
from audit_assets import audit


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seal', action='store_true')
    a = p.parse_args()
    selection_path = OUT / 'FINALISTS_SELECTION.json'
    if not a.seal:
        if selection_path.exists() or (OUT / 'FINALISTS_LOCK.json').exists():
            raise FileExistsError('Research already closed')
        for name in ('regularization', 'stability', 'ablations'):
            for cfg in json.loads((OUT / (name + '-configs.json')).read_text()):
                if not (OUT / cfg['experiment_id'] / 'summary.json').exists():
                    raise ValueError('Complete the registered bounded research before locking')
        ensemble = json.loads((OUT / 'ensemble-selected/summary.json').read_text())
        if ensemble['status'] != 'complete':
            raise ValueError('Prespecified single ensemble analysis incomplete')
        ids = ['attention_regularized_s0', 'deepsets_regularized_s0']
        record = {'selected_at_utc': datetime.now(timezone.utc).isoformat(), 'search_closed': True,
                  'allowed_final_fit_experiments': ids, 'new_test_enabled': False,
                  'finalists': [{'experiment_id': name, 'checkpoint': 'artifacts/research_v2/final_' + family,
                                 'method': 'code/solution_v2.py'} for name, family in zip(ids, ['attention', 'deepsets'])],
                  'selection_reason': 'Best two stable nonlinear families from common train OOF. All three fixed seeds retained; deployment seed was fixed at 3036900957 before trials. No new finalist validation opened.',
                  'ensemble_rejected': 'Only equal mixture lost 13 hard OOF questions versus fixed-seed Attention; no weight search or deployment.',
                  'ablation_note': 'Hard-normalization and context-removal runs were preregistered diagnostic controls, not new finalist candidates.',
                  'analysis_sha256': digest(OUT / 'analysis-final/analysis.json'),
                  'configuration_sha256': {name: digest(ROOT / 'code/configs/research_v2' / (name + '.json')) for name in ids}}
        selection_path.write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
    else:
        target = OUT / 'FINALISTS_LOCK.json'
        if target.exists():
            raise FileExistsError('Asset lock already sealed')
        record = json.loads(selection_path.read_text())
        record['sealed_at_utc'] = datetime.now(timezone.utc).isoformat()
        record['selection_sha256'] = digest(selection_path)
        record['asset_sha256'] = {}
        for item in record['finalists']:
            folder = ROOT / item['checkpoint']
            if not audit(folder)['within_limit']:
                raise ValueError('Finalist exceeds uncompressed asset limit')
            fitting = json.loads((OUT / item['experiment_id'] / 'final_fit.json').read_text())
            if fitting['config'] != json.loads((ROOT / 'code/configs/research_v2' / (item['experiment_id'] + '.json')).read_text()):
                raise ValueError('Final fitting configuration changed')
            for path in folder.rglob('*'):
                if path.is_file():
                    record['asset_sha256'][path.relative_to(ROOT).as_posix()] = digest(path)
        files = ['code/common.py', 'code/predict.py', 'code/score_predictions.py', 'code/solution_v2.py',
                 'code/mp2/__init__.py', 'code/mp2/features.py', 'code/mp2/models_v2.py', 'code/mp2/selection_v2.py']
        record['inference_source_sha256'] = {name: digest(ROOT / name) for name in files}
        target.write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(record))
