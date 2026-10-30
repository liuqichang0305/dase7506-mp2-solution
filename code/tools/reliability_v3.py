"""Optional chosen-answer confidence diagnostics; never used by official predict.

Fit one fixed C=1 logistic map only after selecting the final method. Its train
OOF calibration metrics are in-sample and optimistic. Evaluate it on separately
provided, selection-locked validation events to assess generalization. This
tool does not open official banks/references, select methods, or change choices.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
import warnings

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code'))
from mp2.research_v3.selection import member_probabilities, select_raw

EVENT_SCHEMA = 'mp2-v3-confidence-events-v1'
CALIBRATION_SCHEMA = 'mp2-v3-confidence-calibration-v1'
SEED = 3036900957
CLIP = 1e-6
C = 1.0
NOTE = ('Optional report-only analysis. Official predict does not load this map; '
        'choices are unchanged. These files are outside the inference bundle. '
        'If a future prediction path loads them, all such assets must be counted.')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    path = Path(path).resolve()
    if path.is_relative_to((ROOT / 'code/data').resolve()) or any(
            token in path.name.lower() for token in ('leaderboard', 'ranking_background')):
        raise ValueError('Confidence analysis cannot open official data or ranking files')
    return json.loads(path.read_text(encoding='utf-8'))


def write_new(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def chosen_confidence(groups, probability, choices):
    groups, probability, choices = np.asarray(groups), np.asarray(probability, dtype=np.float64), np.asarray(choices)
    if groups.ndim != 2 or groups.shape[1] != 4 or probability.shape != groups.shape or choices.shape != (len(groups),):
        raise ValueError('Expected groups/probabilities [N,4] and choices [N]')
    if not len(groups) or not np.issubdtype(groups.dtype, np.integer) or np.any((groups < -1) | (groups > 3)):
        raise ValueError('Nonempty integer group IDs -1..3 are required')
    if not np.issubdtype(choices.dtype, np.integer) or np.any((choices < 0) | (choices > 3)):
        raise ValueError('Choices must be original candidate indices 0..3')
    active = (groups[:, :, None] == np.arange(4)[None, None, :]).any(axis=1)
    if not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
        raise ValueError('Group probabilities must be finite in [0,1]')
    if np.any(probability[~active] != 0) or not np.allclose(probability.sum(axis=1), active.any(axis=1), atol=1e-10, rtol=1e-10):
        raise ValueError('Probability mass must be normalized over active answer groups')
    selected_group = groups[np.arange(len(groups)), choices]
    confidence = np.zeros(len(groups), dtype=np.float64)
    valid = selected_group >= 0
    confidence[valid] = probability[np.arange(len(groups))[valid], selected_group[valid]]
    return confidence


def metrics(correct, confidence):
    correct, confidence = np.asarray(correct), np.asarray(confidence, dtype=np.float64)
    if correct.ndim != 1 or correct.shape != confidence.shape or not len(correct) or not np.isin(correct, (0, 1)).all():
        raise ValueError('Nonempty matching binary outcomes and confidence vectors required')
    if not np.isfinite(confidence).all() or np.any((confidence < 0) | (confidence > 1)):
        raise ValueError('Invalid confidence')
    correct = correct.astype(bool)
    # [0,.1), ..., [.9,1]; values exactly one belong to the last bin.
    bins = np.minimum((confidence * 10).astype(np.int64), 9)
    records, ece = [], 0.
    for number in range(10):
        selected = bins == number
        count = int(selected.sum())
        accuracy = float(correct[selected].mean()) if count else None
        mean = float(confidence[selected].mean()) if count else None
        gap = abs(mean - accuracy) if count else None
        ece += count / len(correct) * (gap or 0.)
        records.append({'bin': number, 'lower': number / 10, 'upper': (number + 1) / 10,
                        'upper_inclusive': number == 9, 'n': count, 'accuracy': accuracy,
                        'mean_confidence': mean, 'absolute_gap': gap})
    order = np.argsort(-confidence, kind='stable')
    retained = np.arange(1, len(correct) + 1)
    errors = np.cumsum(~correct[order])
    risk = errors / retained
    return {'n': len(correct), 'correct': int(correct.sum()), 'accuracy': float(correct.mean()),
            'brier': float(np.mean((confidence - correct) ** 2)), 'ece_10_equal_width': float(ece),
            'bins': records, 'aurc_discrete_mean': float(risk.mean()),
            'risk_coverage': {'retained': retained.tolist(), 'coverage': (retained / len(correct)).tolist(),
                              'risk': risk.tolist(), 'errors': errors.tolist(),
                              'ordered_original_rows': order.tolist(),
                              'tie_rule': 'Stable original row order; partial coverage within confidence ties is arbitrary.',
                              'interpretation': 'Descriptive confidence ranking, not a deployment abstention policy.'}}


def _validate_folds(fold, manifest):
    fold = np.asarray(fold)
    if fold.shape != (manifest['questions'],) or not np.issubdtype(fold.dtype, np.integer) or set(fold.tolist()) != {0, 1, 2}:
        raise ValueError('Complete three-fold train OOF is required')
    if sorted(part['fold'] for part in manifest['folds']) != [0, 1, 2]:
        raise ValueError('Invalid common fold manifest')
    for part in manifest['folds']:
        if not np.array_equal(np.flatnonzero(fold == part['fold']), np.sort(np.asarray(part['holdout'], dtype=int))):
            raise ValueError('OOF rows differ from common held-out folds')


def load_train_oof(folder):
    """Read immutable complete single-model or fixed equal-probability OOF only."""
    folder = Path(folder).resolve()
    if (folder / 'diagnostic.json').is_file():
        from reliability_fusion_v3 import load_fusion_oof
        return load_fusion_oof(folder)
    if folder.is_relative_to((ROOT / 'code/data').resolve()):
        raise ValueError('OOF folder cannot be official data')
    summary, registration = (read_json(folder / name) for name in ('summary.json', 'registration.json'))
    if summary.get('status') != 'complete':
        raise ValueError('OOF summary is incomplete')
    mixture = bool(summary.get('ensemble_evaluated'))
    name = 'oof-mixture.npz' if mixture else 'oof.npz'
    archive_path = folder / name
    expected_hash = summary.get('oof_mixture_sha256' if mixture else 'oof_sha256')
    if (mixture and expected_hash is None) or (expected_hash is not None and digest(archive_path) != expected_hash):
        raise ValueError('Missing or inconsistent immutable OOF archive hash')
    spec = registration.get('specification', {}) if mixture else registration
    fold_path = folder.parent / 'common_folds.json'
    if digest(fold_path) != spec.get('folds_sha256'):
        raise ValueError('OOF common fold manifest hash mismatch')
    manifest = read_json(fold_path)
    with np.load(archive_path, allow_pickle=False) as data:
        arrays = {key: data[key].copy() for key in data.files}
    if expected_hash is None and 'raw_scores' in arrays:
        raise ValueError('V3 OOF archive must have its immutable summary hash')
    groups, labels, fold = (arrays[key] for key in ('groups', 'labels', 'fold'))
    _validate_folds(fold, manifest)
    if len(labels) != len(fold) or labels.shape != groups.shape or not np.isin(labels, (0, 1)).all() or np.any(labels.sum(axis=1) < 1) or np.any((labels > 0) & (groups < 0)):
        raise ValueError('Malformed train OOF labels')
    for group in range(4):
        member = groups == group
        positive = (labels.astype(bool) & member).any(axis=1)
        if np.any(member & (labels.astype(bool) != positive[:, None])):
            raise ValueError('Candidates sharing an answer must share correctness')
    if mixture:
        weights = np.asarray(summary.get('weights'), dtype=float)
        if weights.shape not in ((2,), (3,)) or not np.array_equal(weights, np.full(len(weights), 1 / len(weights))):
            raise ValueError('Only a fixed equal-probability mixture is supported')
        if spec.get('weights') != summary['weights'] or registration.get('split') != 'train OOF only':
            raise ValueError('Mixture registration differs or is not train OOF')
        probability, choices = arrays['probability'], arrays['choices']
        selector = {'kind': 'fixed_equal_probability', 'specification': spec}
    else:
        config = summary['config']
        if registration['config'] != config or sorted(r['fold'] for r in summary['folds']) != [0, 1, 2]:
            raise ValueError('Model registration/fold reports are inconsistent')
        data_files = registration.get('data_files', {})
        if not data_files or any(Path(path).name != 'manifest.json' and ('train' not in Path(path).name or any(
                token in Path(path).name for token in ('validation', 'test', 'audit'))) for path in data_files):
            raise ValueError('Only registered train inputs are permitted')
        raw = arrays.get('raw_scores', arrays['scores'])
        if raw.shape[0] != len(fold) or not np.isfinite(raw).all():
            raise ValueError('Incomplete/nonfinite raw OOF scores')
        for part in manifest['folds']:
            number, held = part['fold'], np.asarray(part['holdout'], dtype=int)
            report = next(r for r in summary['folds'] if r['fold'] == number)
            names = [f'fold{number}-scores.npy'] + ([f'fold{number}-raw-scores.npy'] if 'raw_scores' in arrays else [])
            for filename in names:
                source = folder / filename
                expected = report.get('output_sha256', {}).get(filename, report.get('scores_sha256') if '-raw-' not in filename else None)
                key = 'raw_scores' if '-raw-' in filename else 'scores'
                if expected is None or digest(source) != expected or not np.array_equal(np.load(source, allow_pickle=False), arrays[key][held]):
                    raise ValueError('OOF archive differs from immutable held-out fold scores')
        probability = member_probabilities(raw, groups, config['decoder'])
        choices = np.asarray(select_raw(raw, groups, config['decoder']), dtype=np.int64)
        selector = {'kind': 'single_model', 'experiment_id': config['experiment_id'], 'family': config['family'],
                    'decoder': config['decoder'], 'member_probability_average': raw.ndim == 3}
    confidence = chosen_confidence(groups, probability, choices)
    correct = labels[np.arange(len(labels)), choices].astype(bool)
    if 'correct' in arrays and not np.array_equal(arrays['correct'], correct):
        raise ValueError('Saved correctness differs from actual chosen candidate')
    if 'selected_probability' in arrays and not np.allclose(arrays['selected_probability'], confidence, atol=1e-12, rtol=1e-12):
        raise ValueError('Saved confidence is not the actual selected group probability')
    return {'groups': groups, 'group_probabilities': probability, 'choices': choices, 'correct': correct,
            'confidence': confidence, 'fold': fold, 'selector': selector, 'split': 'train_oof',
            'source_sha256': {name: digest(folder / name) for name in ('summary.json', 'registration.json', name)},
            'folds_sha256': digest(fold_path), 'oof_directory': str(folder)}


def _logit(confidence):
    value = np.clip(np.asarray(confidence, dtype=np.float64), CLIP, 1 - CLIP)
    return (np.log(value) - np.log1p(-value)).reshape(-1, 1)


def fit_mapping(correct, confidence):
    """Fixed L2-regularized slope, unpenalized intercept, no hyperparameter search."""
    import sklearn
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    metrics(correct, confidence)
    if set(np.asarray(correct).astype(int).tolist()) != {0, 1}:
        raise ValueError('Logistic calibration requires both correct and incorrect outcomes')
    model = LogisticRegression(C=C, l1_ratio=0., solver='lbfgs', max_iter=1000, tol=1e-10, random_state=SEED)
    with warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        model.fit(_logit(confidence), np.asarray(correct, dtype=int))
    return {'schema': CALIBRATION_SCHEMA, 'coefficient': float(model.coef_[0, 0]), 'intercept': float(model.intercept_[0]),
            'C': C, 'logit_clip_epsilon': CLIP, 'seed': SEED, 'sklearn_version': sklearn.__version__,
            'numpy_version': np.__version__, 'iterations': int(model.n_iter_[0]),
            'objective': 'Binary logistic loss with C=1 L2-regularized one-dimensional slope and unpenalized intercept.',
            'fit_n': len(correct), 'fit_correct': int(np.asarray(correct).sum()), 'purpose': NOTE}


def apply_mapping(confidence, calibration):
    if calibration['schema'] != CALIBRATION_SCHEMA or calibration['C'] != C or calibration['logit_clip_epsilon'] != CLIP:
        raise ValueError('Unsupported calibration specification')
    coefficient, intercept = calibration['coefficient'], calibration['intercept']
    if not np.isfinite([coefficient, intercept]).all():
        raise ValueError('Nonfinite calibration coefficients')
    confidence = np.asarray(confidence, dtype=np.float64)
    if not np.isfinite(confidence).all() or np.any((confidence < 0) | (confidence > 1)):
        raise ValueError('Invalid confidence')
    z = coefficient * _logit(confidence).ravel() + intercept
    return np.exp(-np.logaddexp(0., -z))


def _write_analysis(output, events, calibration, interpretation):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    calibrated = apply_mapping(events['confidence'], calibration)
    report = {'schema': 'mp2-v3-confidence-report-v1', 'split': events['split'], 'interpretation': interpretation,
              'selector': events['selector'], 'source_sha256': events['source_sha256'],
              'analysis_source_sha256': digest(__file__), 'created_at_utc': datetime.now(timezone.utc).isoformat(),
              'choices_changed': 0, 'purpose': NOTE,
              'raw': metrics(events['correct'], events['confidence']),
              'calibrated': metrics(events['correct'], calibrated)}
    np.savez(output / 'confidence-events.npz', groups=events['groups'], group_probabilities=events['group_probabilities'],
             choices=events['choices'], correct=events['correct'], raw_confidence=events['confidence'],
             calibrated_confidence=calibrated)
    report['events_sha256'] = digest(output / 'confidence-events.npz')
    write_new(output / 'report.json', report)
    return report


def selected_fit_guard(events):
    """Real repository OOF can fit a map only for its unique locked selection."""
    directory = Path(events['oof_directory']).resolve()
    if not any(directory.is_relative_to((ROOT / part).resolve()) for part in ('results/research_v2', 'results/research_v3')):
        return  # Temporary generated test fixtures are outside production results.
    selected_path = ROOT / 'submission/v3/SELECTED_MODEL.json'
    summary_path = ROOT / 'results/research_v3/validation_summary.json'
    selected, summary = read_json(selected_path), read_json(summary_path)
    if selected.get('schema') != 'mp2-selected-model-v3' or selected.get('validation_summary_sha256') != digest(summary_path):
        raise ValueError('Confidence fit requires the immutable unique selected-model record')
    if summary.get('schema') != 'mp2-validation-confirmation-v3' or selected.get('name') != summary.get('selected', {}).get('name'):
        raise ValueError('Confidence fit model differs from the unique validation choice')
    if (ROOT / selected['oof_directory']).resolve() != directory or selected.get('selector') != events['selector']:
        raise ValueError('Confidence fit OOF/selector differs from the unique selected method')
    events['source_sha256'].update({'selected_model_json': digest(selected_path), 'validation_summary_json': digest(summary_path)})


def fit(oof_dir, output):
    if Path(output).exists():
        raise FileExistsError('Refusing to overwrite calibration evidence')
    events = load_train_oof(oof_dir)
    selected_fit_guard(events)
    calibration = fit_mapping(events['correct'], events['confidence'])
    calibration.update({'source_split': 'train_oof', 'selector': events['selector'], 'source_sha256': events['source_sha256'],
                        'folds_sha256': events['folds_sha256'], 'oof_directory': events['oof_directory'],
                        'analysis_source_sha256': digest(__file__), 'fit_metrics_interpretation':
                        'IN-SAMPLE: the mapping was fitted to these OOF outcomes. Calibration metrics are optimistic; '
                        'OOF model selection adds further optimism. No unbiased/generalization claim.'})
    report = _write_analysis(output, events, calibration, calibration['fit_metrics_interpretation'])
    write_new(Path(output) / 'calibration.json', calibration)
    return report


def load_prediction_events(path):
    record = read_json(path)
    if record.get('schema') != EVENT_SCHEMA or record.get('split') not in ('synthetic', 'validation'):
        raise ValueError('Only synthetic or selection-locked validation prediction events are supported; test is disabled')
    if not isinstance(record.get('selector'), dict) or not record['selector']:
        raise ValueError('Prediction selector metadata is required')
    source = record.get('source_sha256')
    if not isinstance(source, dict) or not source or any(not isinstance(value, str) or not re.fullmatch('[0-9a-f]{64}', value) for value in source.values()):
        raise ValueError('Declared prediction-source SHA256 hashes are required')
    if record['split'] == 'validation':
        lock = record.get('selection_lock', {})
        lock_path = Path(lock.get('path', '')).resolve()
        allowed = {(ROOT / name).resolve() for name in ('results/research_v3/FINALISTS_LOCK.json',
                  'results/research_v3/FINALISTS_SELECTION.json', 'submission/v3/FREEZE.json')}
        if lock_path not in allowed or not lock_path.is_file() or digest(lock_path) != lock.get('sha256'):
            raise ValueError('Validation requires a matching pre-existing V3 selection lock')
        record['source_sha256']['selection_lock'] = digest(lock_path)
    if 'predictions_npz' in record:
        reference = record['predictions_npz']
        archive_path = Path(reference['path'])
        if not archive_path.is_absolute():
            archive_path = Path(path).resolve().parent / archive_path
        archive_path = archive_path.resolve()
        if archive_path.is_relative_to((ROOT / 'code/data').resolve()) or archive_path.suffix != '.npz':
            raise ValueError('Prediction events must be an analysis NPZ, not official data')
        if digest(archive_path) != reference['sha256']:
            raise ValueError('Prediction event NPZ hash mismatch')
        with np.load(archive_path, allow_pickle=False) as archive:
            arrays = {key: archive[key].copy() for key in archive.files}
        record['source_sha256']['prediction_events_npz'] = digest(archive_path)
    else:
        arrays = record
    groups, probability, choices = (np.asarray(arrays[name]) for name in ('groups', 'group_probabilities', 'choices'))
    if 'selected' in arrays and not np.array_equal(arrays['selected'], choices):
        raise ValueError('Selected alias differs from actual choices')
    confidence = chosen_confidence(groups, probability, choices)
    if 'confidence' in arrays and not np.allclose(arrays['confidence'], confidence, atol=1e-12, rtol=1e-12):
        raise ValueError('Event confidence differs from actual chosen group probability')
    correct = np.asarray(arrays['correct'])
    metrics(correct, confidence)
    if np.any(correct.astype(bool) & (groups[np.arange(len(groups)), choices] < 0)):
        raise ValueError('An unparseable selected candidate cannot be correct')
    return {'groups': groups, 'group_probabilities': probability, 'choices': choices, 'correct': correct.astype(bool),
            'confidence': confidence, 'selector': record['selector'], 'split': record['split'],
            'source_sha256': {**record['source_sha256'], 'prediction_events_json': digest(path)}}


def evaluate(predictions, calibration_dir, output):
    if Path(output).exists():
        raise FileExistsError('Refusing to overwrite confidence evaluation evidence')
    calibration_path = Path(calibration_dir) / 'calibration.json'
    calibration = read_json(calibration_path)
    events = load_prediction_events(predictions)
    if events['selector'] != calibration['selector']:
        raise ValueError('Prediction selector differs from the fitted confidence map')
    events['source_sha256']['calibration_json'] = digest(calibration_path)
    return _write_analysis(output, events, calibration,
        'Held-out validation assessment of a previously fitted train-OOF map; no choices or calibration parameters were updated.'
        if events['split'] == 'validation' else 'SYNTHETIC verification only; no claim about course data.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    fitting = sub.add_parser('fit', help='Run only after the final selected method is unique')
    fitting.add_argument('--oof-dir', type=Path, required=True)
    fitting.add_argument('--output', type=Path, required=True)
    evaluating = sub.add_parser('evaluate')
    evaluating.add_argument('--predictions', type=Path, required=True)
    evaluating.add_argument('--calibration', type=Path, required=True)
    evaluating.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = fit(args.oof_dir, args.output) if args.command == 'fit' else evaluate(args.predictions, args.calibration, args.output)
    print(json.dumps({key: result[key] for key in ('split', 'interpretation', 'choices_changed')}, indent=2))


if __name__ == '__main__':
    main()
