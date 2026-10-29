"""Development and fitting. This module is not imported by inference."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from scipy.optimize import minimize

from common import DATA, PROTOCOL, check_data, load_inputs, parse_answer, read_jsonl
from .features import build_features, hard_mask, majority_choices, parsed_answers
from .model import score_with_bundle, confidence_features

SEED = 3036900957
ROOT = Path(__file__).resolve().parents[2]
DEV = ROOT / 'results' / 'development'


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def source_hash():
    paths = [ROOT/'code'/'solution.py', ROOT/'code'/'train.py', *sorted((ROOT/'code'/'mp2').glob('*.py'))]
    h = hashlib.sha256()
    for path in paths:
        h.update(path.relative_to(ROOT).as_posix().encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def log_event(event):
    event = {'timestamp_utc': datetime.now(timezone.utc).isoformat(), 'source_sha256': source_hash(),
             'upstream_commit': 'bb801f4f2e7629d270df765ab12c333b2dc616bb', **event}
    DEV.mkdir(parents=True, exist_ok=True)
    with (DEV/'experiments.jsonl').open('a', encoding='utf-8') as f:
        f.write(json.dumps(event, ensure_ascii=False) + '\n')
    print(json.dumps(event, ensure_ascii=False), flush=True)


def load_development(split):
    if split not in ('train', 'validation'):
        raise ValueError('Development loader permits only train or validation')
    check_data(DATA, (split,))
    bank, arrays = load_inputs(split)
    refs = read_jsonl(DATA/f'gsm8k_{split}.jsonl')
    if len(refs) != len(bank) or any(r['id'] != b['id'] or r['question'] != b['question'] for r, b in zip(refs, bank)):
        raise ValueError('Reference alignment mismatch')
    labels = np.array([[a is not None and a == parse_answer(ref['answer']) for a in row]
                       for row, ref in zip(parsed_answers(bank), refs)], dtype=np.float64)
    if not (labels.sum(axis=1) >= 1).all():
        raise ValueError('Every training/development question must have a positive')
    return bank, arrays['embeddings'], labels


def loss_gradient(theta, x, y, loss, l2):
    n, k, d = x.shape
    flat = x.reshape(-1, d)
    s = (flat @ theta[:-1] + theta[-1]).reshape(n, k)
    if loss == 'bce':
        objective = np.mean(np.logaddexp(0, s) - y*s)
        derivative = (1/(1+np.exp(-np.clip(s, -700, 700))) - y)/(n*k)
    elif loss == 'listwise':
        shift = s.max(axis=1, keepdims=True)
        a = np.exp(s-shift)
        positive_scores = np.where(y > 0, s, -np.inf)
        positive_shift = positive_scores.max(axis=1, keepdims=True)
        positives = np.exp(positive_scores-positive_shift)
        full_sum = a.sum(axis=1, keepdims=True)
        pos_sum = positives.sum(axis=1, keepdims=True)
        objective = np.mean(shift + np.log(full_sum) - positive_shift - np.log(pos_sum))
        derivative = (a/full_sum - positives/pos_sum)/n
    else:
        raise ValueError(f'Unsupported loss: {loss}')
    objective += 0.5*l2*np.dot(theta[:-1], theta[:-1])
    gradient = np.r_[flat.T @ derivative.ravel() + l2*theta[:-1], derivative.sum()]
    return float(objective), gradient


def fit_features(features, labels, config):
    start = time.perf_counter()
    mean = features.mean(axis=(0, 1))
    scale = features.std(axis=(0, 1))
    scale[scale < 1e-8] = 1
    x = np.ascontiguousarray((features-mean)/scale)
    result = minimize(loss_gradient, np.zeros(x.shape[-1]+1), args=(x, labels, config['loss'], config['l2']),
                      method='L-BFGS-B', jac=True, options={'maxiter': config.get('maxiter', 200), 'ftol': 1e-9, 'gtol': 1e-6, 'maxls': 30})
    if not np.isfinite(result.fun) or not np.isfinite(result.x).all():
        raise ValueError('Optimizer produced nonfinite parameters')
    arrays = {'mean': mean, 'scale': scale, 'weight': result.x[:-1], 'bias': np.asarray(result.x[-1])}
    info = {'training_seconds': time.perf_counter()-start, 'optimizer_success': bool(result.success),
            'optimizer_message': str(result.message), 'iterations': int(result.nit),
            'objective': float(result.fun), 'max_gradient': float(np.abs(result.jac).max()),
            'train_questions': len(labels), 'parameters': len(result.x)}
    return arrays, info


def save_bundle(path, config, arrays):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    write_json(path/'config.json', {'schema': 'mp2-linear-v1', 'features': config['features'], 'loss': config['loss'],
               'l2': config['l2'], 'maxiter': config.get('maxiter', 200), 'seed': SEED,
               'fit_split': 'train', 'tie_rule': 'first maximum', 'calibration': 'train-three-fold-OOF-logistic' if 'calibration_weight' in arrays else None})
    np.savez(path/'weights.npz', **arrays)
    return sum(p.stat().st_size for p in path.iterdir() if p.is_file())


def metrics(labels, choices, hard):
    correct = labels[np.arange(len(labels)), np.asarray(choices, dtype=int)]
    return {'correct': int(correct.sum()), 'n': len(labels), 'accuracy': float(100*correct.mean()),
            'hard_correct': int(correct[hard].sum()), 'hard_n': int(hard.sum()),
            'hard_accuracy': float(100*correct[hard].mean()) if hard.any() else None}


def describe(bank, labels):
    answers = parsed_answers(bank)
    lengths = np.array([[len(c['text']) for c in r['candidates']] for r in bank])
    return {'questions': len(bank), 'correct_candidates_histogram': {str(i): int((labels.sum(axis=1)==i).sum()) for i in range(1, 5)},
            'parseable_candidate_fraction': sum(a is not None for row in answers for a in row)/(4*len(bank)),
            'distinct_valid_answer_histogram': {str(i): sum(len({a for a in row if a is not None})==i for row in answers) for i in range(5)},
            'candidate_characters_quantiles': np.quantile(lengths, [0, .25, .5, .75, 1]).tolist(),
            'random_selection_expected_accuracy': float(100*labels.mean())}


def official_validation(name, checkpoint):
    output = DEV/name
    output.mkdir(parents=True, exist_ok=True)
    commands = [[sys.executable, 'predict.py', '--method', 'solution.py', '--checkpoint', str(checkpoint), '--split', 'validation', '--output', str(output/'predictions.json')],
                [sys.executable, 'score_predictions.py', '--split', 'validation', '--predictions', str(output/'predictions.json'), '--output', str(output/'score.json')]]
    start = time.perf_counter()
    for i, command in enumerate(commands):
        result = subprocess.run(command, cwd=ROOT/'code', capture_output=True, text=True, encoding='utf-8')
        (output/f'{i}.stdout.txt').write_text(result.stdout, encoding='utf-8')
        (output/f'{i}.stderr.txt').write_text(result.stderr, encoding='utf-8')
        write_json(output/f'{i}.process.json', {'command': command, 'returncode': result.returncode})
        if result.returncode:
            raise RuntimeError(f'Official validation process failed: {command}; {result.stderr}')
    elapsed = time.perf_counter()-start
    write_json(output/'timing.json', {'fresh_prediction_plus_scoring_wall_seconds': elapsed, 'local_measurement': True})
    return json.loads((output/'score.json').read_text()), elapsed


def calibrate_oof(bank, embeddings, labels, config, features):
    order = np.random.default_rng(SEED+1).permutation(len(bank))
    folds = np.array_split(order, 3)
    oof = np.empty((len(bank), 4))
    fold_info = []
    for i, holdout in enumerate(folds):
        fitting = np.setdiff1d(np.arange(len(bank)), holdout)
        arrays, info = fit_features(features[fitting], labels[fitting], config)
        scores = ((features[holdout]-arrays['mean'])/arrays['scale']) @ arrays['weight'] + arrays['bias']
        oof[holdout] = scores
        info.update({'fold': i, 'holdout_questions': len(holdout),
                     **metrics(labels[holdout], scores.argmax(axis=1), hard_mask([bank[j] for j in holdout]))})
        fold_info.append(info)
        log_event({'experiment_id': f"oof-{config['name']}-{i}", 'phase': 'train_oof', 'config': config, **info})
    signals = confidence_features(bank, oof)
    chosen = oof.argmax(axis=1)
    event = labels[np.arange(len(bank)), chosen]
    mean, scale = signals.mean(axis=0), signals.std(axis=0)
    scale[scale<1e-8] = 1
    x = ((signals-mean)/scale)[:, None, :]
    opt = minimize(loss_gradient, np.zeros(x.shape[-1]+1), args=(x, event[:, None], 'bce', .01), method='L-BFGS-B', jac=True,
                   options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-7})
    if not opt.success:
        raise RuntimeError(f'Calibration did not converge: {opt.message}')
    np.savez(DEV/'train_oof.npz', scores=oof, selected_correct=event, confidence_signals=signals)
    write_json(DEV/'oof_folds.json', {'seed': SEED+1, 'folds': [f.tolist() for f in folds], 'results': fold_info,
                                    'calibration_fit_split': 'train OOF only', 'calibration_l2': .01})
    return {'calibration_mean': mean, 'calibration_scale': scale,
            'calibration_weight': opt.x[:-1], 'calibration_bias': np.asarray(opt.x[-1])}, fold_info


def train_one(config, output, calibrate=False):
    bank, e, y = load_development('train')
    x = build_features(bank, e, config['features'])
    arrays, info = fit_features(x, y, config)
    if calibrate:
        calibration, _ = calibrate_oof(bank, e, y, config, x)
        arrays.update(calibration)
    size = save_bundle(output, config, arrays)
    log_event({'experiment_id': config.get('name', 'explicit-training'), 'phase': 'explicit_train', 'config': config,
               'asset_file_bytes': size, 'checkpoint': str(output), **info})


def run_search():
    if (DEV/'search_summary.json').exists():
        raise RuntimeError('Search already completed; preserve outcomes instead of rerunning selection')
    start = time.perf_counter()
    bank, e, y = load_development('train')
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(bank))
    cut = int(.8*len(bank))
    fitting, holding = order[:cut], order[cut:]
    write_json(DEV/'train_inner_split.json', {'seed': SEED, 'fitting_rows': fitting.tolist(), 'holdout_rows': holding.tolist()})
    write_json(DEV/'train_statistics.json', describe(bank, y))
    features = {mode: build_features(bank, e, mode) for mode in ('embedding', 'relation')}
    families = [('embedding', 'bce'), ('relation', 'bce'), ('relation', 'listwise')]
    configurations = [{'name': f'{mode}-{loss}-l2-{l2}', 'features': mode, 'loss': loss, 'l2': l2, 'maxiter': 200, 'seed': SEED}
                      for mode, loss in families for l2 in (.01, .1)]
    write_json(ROOT/'code'/'configs'/'search.json', configurations)
    screens = []
    h = hard_mask([bank[i] for i in holding])
    for config in configurations:
        if time.perf_counter()-start > 7200:
            raise TimeoutError('Prespecified two-hour search budget reached')
        arrays, info = fit_features(features[config['features']][fitting], y[fitting], config)
        holdfeatures = features[config['features']][holding]
        scores = ((holdfeatures-arrays['mean'])/arrays['scale']) @ arrays['weight'] + arrays['bias']
        result = {'experiment_id': config['name'], 'parent_experiment': None, 'phase': 'train_inner_screen', 'config': config,
                  'hypothesis': {'embedding': 'H1 shared verifier', 'relation': 'H2/H3 relative evidence and objective'}[config['features']],
                  'fitting_split': 'train 80 percent grouped', 'selection_split': 'train 20 percent grouped',
                  **info, **metrics(y[holding], scores.argmax(axis=1), h)}
        screens.append(result)
        log_event(result)
    winners = [max((r for r in screens if (r['config']['features'], r['config']['loss']) == family),
                   key=lambda r: (r['accuracy'], r['config']['l2']))['config'] for family in families]
    # Only now load official validation: never used to fit or screen the six configurations.
    vb, ve, vy = load_development('validation')
    vh = hard_mask(vb)
    write_json(DEV/'validation_statistics.json', describe(vb, vy))
    vx = {mode: build_features(vb, ve, mode) for mode in ('embedding', 'relation')}
    evaluations, predictions, trained = [], {}, {}
    for name, choices in [('fixed0', [0]*len(vb)), ('majority', majority_choices(vb))]:
        data = {'protocol': PROTOCOL, 'split': 'validation', 'ids': [r['id'] for r in vb], 'selected': choices}
        write_json(DEV/name/'predictions.json', data)
        command = [sys.executable, 'score_predictions.py', '--split', 'validation', '--predictions', str(DEV/name/'predictions.json'), '--output', str(DEV/name/'score.json')]
        proc = subprocess.run(command, cwd=ROOT/'code', capture_output=True, text=True, encoding='utf-8')
        (DEV/name/'scorer.stdout.txt').write_text(proc.stdout, encoding='utf-8')
        (DEV/name/'scorer.stderr.txt').write_text(proc.stderr, encoding='utf-8')
        write_json(DEV/name/'process.json', {'command': command, 'returncode': proc.returncode})
        if proc.returncode:
            raise RuntimeError(proc.stderr)
        result = {'experiment_id': name, 'phase': 'validation_baseline', **metrics(vy, choices, vh)}
        predictions[name] = np.asarray(choices)
        evaluations.append(result)
        log_event(result)
    for config in winners:
        arrays, info = fit_features(features[config['features']], y, config)
        folder = ROOT/'artifacts'/'experiments'/config['name']
        size = save_bundle(folder, config, arrays)
        official, wall = official_validation(config['name'], folder)
        scores = ((vx[config['features']]-arrays['mean'])/arrays['scale']) @ arrays['weight'] + arrays['bias']
        choices = scores.argmax(axis=1)
        result = {'experiment_id': config['name'], 'phase': 'full_train_validation', 'config': config, **info,
                  **metrics(vy, choices, vh), 'official_score': official, 'full_local_wall_seconds': wall, 'asset_file_bytes': size}
        if result['correct'] != official['correct']:
            raise RuntimeError('Official scorer mismatch')
        predictions[config['name']] = choices
        np.save(DEV/config['name']/'scores.npy', scores)
        evaluations.append(result)
        trained[config['name']] = (config, arrays)
        log_event(result)
    learned = [r for r in evaluations if r['phase'] == 'full_train_validation']
    selected = max(learned, key=lambda r: (r['accuracy'], r['config']['features']=='embedding', r['config']['loss']=='bce', r['config']['l2']))
    config, arrays = trained[selected['experiment_id']]
    calibration, folds = calibrate_oof(bank, e, y, config, features[config['features']])
    arrays.update(calibration)
    calibrated = ROOT/'artifacts'/'experiments'/'recommended_bundle'
    size = save_bundle(calibrated, config, arrays)
    write_json(ROOT/'code'/'configs'/'recommended.json', config)
    # Bootstrap resamples questions and preserves pairing, after all selection decisions.
    correct = {name: vy[np.arange(len(vy)), choices] for name, choices in predictions.items()}
    bootrng = np.random.default_rng(SEED+2)
    sample = bootrng.integers(0, len(vy), size=(2000, len(vy)))
    intervals = {}
    for name, values in correct.items():
        delta = correct[selected['experiment_id']] - values
        interval = 100*np.quantile(delta[sample].mean(axis=1), [.025, .975])
        intervals[name] = {'difference_percentage_points': float(100*delta.mean()), 'paired_bootstrap_95_interval': interval.tolist()}
    summary = {'screens': screens, 'validation': evaluations, 'recommended_learned': selected['experiment_id'],
               'recommended_checkpoint': 'artifacts/experiments/recommended_bundle', 'recommended_config': config,
               'recommended_asset_file_bytes': size, 'selected_train_only_oof': folds, 'paired_bootstrap': intervals,
               'bootstrap_repetitions': 2000, 'bootstrap_seed': SEED+2, 'wall_seconds': time.perf_counter()-start,
               'note': 'Recommended learned method must still be compared with majority baseline before final freeze. No test access.'}
    write_json(DEV/'search_summary.json', summary)
    log_event({'experiment_id': 'search-complete', 'phase': 'complete', 'recommended_learned': selected['experiment_id'],
               'wall_seconds': summary['wall_seconds'], 'test_access': False})
