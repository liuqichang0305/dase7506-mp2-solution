"""One-way confirmation of at most two locked finalists on full validation."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS', 'OMP_THREAD_LIMIT'):
    os.environ[key] = '1'
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code'))
sys.path.insert(0, str(ROOT / 'code/tools'))
from mp2.training import load_development
from mp2.selection_v2 import answer_groups
from audit_assets import audit


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def counts(labels, choices, groups):
    correct = labels[np.arange(len(labels)), choices].astype(bool)
    result = {}
    for prefix, mask in [('', np.ones(len(labels), bool)), ('hard_', groups.max(1) >= 1),
                         ('single_positive_', labels.sum(1) == 1)]:
        result.update({prefix + 'correct': int(correct[mask].sum()), prefix + 'n': int(mask.sum()),
                       prefix + 'accuracy': float(100 * correct[mask].mean())})
    return result, correct


def bootstrap(event, baseline, subset):
    delta = event[subset].astype(float) - baseline[subset]
    rng = np.random.default_rng(3036901059)
    draws = np.array([delta[rng.integers(len(delta), size=len(delta))].mean() for _ in range(5000)])
    return {'correct_difference': int(delta.sum()), 'difference_percentage_points': float(100 * delta.mean()),
            'paired_bootstrap_95_interval': (100 * np.quantile(draws, [.025, .975])).tolist(),
            'resampling_unit': 'question', 'repetitions': 5000, 'seed': 3036901059}


def reliability(confidence, event):
    bins = []
    for index in range(10):
        selected = (confidence >= index / 10) & ((confidence < (index + 1) / 10) if index < 9 else (confidence <= 1))
        bins.append({'lower': index / 10, 'upper': (index + 1) / 10, 'n': int(selected.sum()),
                     'confidence': float(confidence[selected].mean()) if selected.any() else None,
                     'accuracy': float(event[selected].mean()) if selected.any() else None})
    return {'brier': float(np.mean((confidence - event) ** 2)),
            'ece': sum(r['n'] / len(event) * abs(r['confidence'] - r['accuracy']) for r in bins if r['n']),
            'n': len(event), 'bins': bins, 'calibration': 'none; raw chosen-answer-group probability',
            'event': 'actual deployed selector chose a correct numerical-answer group'}


def main():
    destination = ROOT / 'results/research_v2'
    lock_path = destination / 'FINALISTS_LOCK.json'
    lock = json.loads(lock_path.read_text())
    if lock.get('search_closed') is not True or not lock.get('asset_sha256') or not 1 <= len(lock['finalists']) <= 2:
        raise ValueError('One or two fully locked finalists required before validation')
    output = destination / 'validation_summary.json'
    if output.exists():
        raise FileExistsError('Confirmation already complete; never reopen search')
    for relative, digest in {**lock['asset_sha256'], **lock['inference_source_sha256']}.items():
        if sha(ROOT / relative) != digest:
            raise ValueError(f'Locked finalist asset changed: {relative}')
    # Record the first new-model validation access before the first process.
    opened = destination / 'VALIDATION_OPENED.json'
    if opened.exists() and json.loads(opened.read_text())['lock_sha256'] != sha(lock_path):
        raise ValueError('Validation was opened under a different lock; refuse to resume')
    if not opened.exists():
        opened.write_text(json.dumps({'utc': datetime.now(timezone.utc).isoformat(), 'lock_sha256': sha(lock_path),
                                      'new_model_count': len(lock['finalists']), 'new_test_enabled': False}, indent=2) + '\n')
    for finalist in lock['finalists']:
        folder = destination / 'validation' / finalist['experiment_id']
        if (folder / 'validation-timing.json').exists():
            prior = json.loads((folder / 'validation-timing.json').read_text())
            if prior['status'] != 'passed':
                raise RuntimeError('Prior evaluation failed; inspect preserved evidence before resuming')
            continue
        folder.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, 'code/tools/evaluate.py', '--method', finalist.get('method', 'code/solution_v2.py'),
                   '--checkpoint', finalist['checkpoint'], '--split', 'validation', '--output-dir', str(folder)]
        proc = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, encoding='utf-8')
        (folder / 'evaluate.stdout.log').write_text(proc.stdout, encoding='utf-8')
        (folder / 'evaluate.stderr.log').write_text(proc.stderr, encoding='utf-8')
        if proc.returncode:
            raise RuntimeError(proc.stderr)
    # References are first loaded here, after all locked official predictions.
    bank, embeddings, labels = load_development('validation')
    groups = answer_groups(bank)
    baseline = json.loads((destination / 'baseline_reproduction/validation/validation-predictions.json').read_text())
    baseline_metrics, baseline_event = counts(labels, baseline['selected'], groups)
    results = []
    for finalist in lock['finalists']:
        folder = destination / 'validation' / finalist['experiment_id']
        prediction = json.loads((folder / 'validation-predictions.json').read_text())
        timing = json.loads((folder / 'validation-timing.json').read_text())
        score = json.loads((folder / 'validation-score.json').read_text())
        metric, correct = counts(labels, prediction['selected'], groups)
        if metric['correct'] != score['correct']:
            raise ValueError('Official scorer and diagnostic event disagree')
        import importlib.util
        spec = importlib.util.spec_from_file_location('confirmed_v2_method', ROOT / finalist.get('method', 'code/solution_v2.py'))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        confidence = np.asarray(module.predict_confidence(bank, {'embeddings': embeddings}, ROOT / finalist['checkpoint']))
        summary = {**finalist, 'validation': metric, 'official_score': score,
                   'prediction_plus_scoring_seconds': timing['total_wall_seconds'], 'resource': audit(ROOT / finalist['checkpoint']),
                   'reliability': reliability(confidence, correct),
                   'bootstrap_vs_frozen_v1': {name: bootstrap(correct, baseline_event, mask) for name, mask in
                       [('overall', np.ones(len(labels), bool)), ('hard', groups.max(1) >= 1)]}}
        summary['local_resource_eligible'] = bool(summary['resource']['within_limit'] and timing['local_under_30_seconds'])
        np.savez(folder / 'diagnostics.npz', selected=np.asarray(prediction['selected']), correct=correct, confidence=confidence)
        results.append(summary)
    eligible = [r for r in results if r['local_resource_eligible']]
    if not eligible:
        raise RuntimeError('No finalist passed local time/asset limits; official outputs preserved')
    winner = max(eligible, key=lambda r: (r['validation']['hard_correct'], r['validation']['correct'],
                                        -r['resource']['uncompressed_bytes']))
    record = {'completed_at_utc': datetime.now(timezone.utc).isoformat(), 'baseline': baseline_metrics,
              'finalists': results, 'selected_experiment_id': winner['experiment_id'],
              'selection_rule': 'Among the prelocked at-most-two candidates: hard correct, full correct, smaller assets',
              'new_test_evaluated': False, 'search_remains_closed': True,
              'limitation': 'Train-OOF is selected/optimistic; confirmation has at most two candidates; intervals do not remove selection bias.'}
    output.write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(record))


if __name__ == '__main__':
    main()
