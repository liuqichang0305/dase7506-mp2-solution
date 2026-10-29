"""Versioned, resumable train-only research; never loads official validation/test."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time

import numpy as np
from scipy.optimize import minimize
from sklearn.model_selection import StratifiedKFold, train_test_split

from .features import build_features
from .training import load_development
from .selection_v2 import answer_groups, select_candidates, score_loss_gradient, torch_loss
from .models_v2 import build_model, export_state, numpy_forward, save_bundle, load_bundle

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/research_v2'
ASSETS = ROOT / 'artifacts/research_v2'
SEED = 3036900957


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.pending')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def config_hash(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def provenance():
    paths = ['code/mp2/training_v2.py', 'code/mp2/selection_v2.py',
             'code/mp2/models_v2.py', 'code/mp2/features.py', 'code/mp2/training.py']
    data = [p for p in (ROOT / 'code/data').rglob('*') if p.is_file() and 'train' in p.name]
    return {'source_files': {p: digest(ROOT / p) for p in paths},
            'data_files': {p.relative_to(ROOT).as_posix(): digest(p) for p in data},
            'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()}


def event(record):
    record = {'utc': datetime.now(timezone.utc).isoformat(), **record}
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / 'events.jsonl').open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(record) + '\n')
    print(json.dumps(record), flush=True)


def data_and_folds():
    bank, embeddings, labels = load_development('train')
    groups = answer_groups(bank)
    features = build_features(bank, embeddings, 'relation')
    strata = np.array([f'{int(p)}-{int(g)}' for p, g in zip(labels.sum(1), groups.max(1) + 1)])
    unique, counts = np.unique(strata, return_counts=True)
    rare = set(unique[counts < 15])
    strata = np.array(['rare' if s in rare else s for s in strata])
    # If the merged rare stratum is still too small, use positive count alone.
    if min(np.unique(strata, return_counts=True)[1]) < 3:
        strata = labels.sum(1).astype(int).astype(str)
    splits = []
    for fold, (outer_fit, holdout) in enumerate(StratifiedKFold(3, shuffle=True, random_state=SEED).split(np.zeros(len(bank)), strata)):
        inner_strata = strata[outer_fit]
        stratify = inner_strata if min(np.unique(inner_strata, return_counts=True)[1]) >= 2 else None
        fitting, stopping = train_test_split(outer_fit, test_size=.2, random_state=SEED + 100 + fold, stratify=stratify)
        splits.append({'fold': fold, 'outer_fit': outer_fit.tolist(), 'holdout': holdout.tolist(),
                       'inner_fit': sorted(fitting.tolist()), 'inner_stop': sorted(stopping.tolist())})
    value = {'seed': SEED, 'questions': len(bank), 'folds': splits,
             'train_ids_sha256': hashlib.sha256(json.dumps([r['id'] for r in bank]).encode()).hexdigest()}
    path = OUT / 'common_folds.json'
    if path.exists() and json.loads(path.read_text()) != value:
        raise ValueError('Existing common folds differ; preserve rather than replace')
    if not path.exists():
        write_json(path, value)
    return features, labels, groups, splits


def metrics(labels, scores, groups, decoder):
    choices = np.asarray(select_candidates(scores, groups, **decoder))
    correct = labels[np.arange(len(labels)), choices].astype(bool)
    hard = groups.max(1) >= 1
    one = labels.sum(1) == 1
    result = {}
    for prefix, subset in [('', np.ones(len(labels), bool)), ('hard_', hard), ('single_positive_', one)]:
        result.update({prefix + 'correct': int(correct[subset].sum()), prefix + 'n': int(subset.sum()),
                       prefix + 'accuracy': float(100 * correct[subset].mean()) if subset.any() else None})
    return result


def scaler(features):
    mean, scale = features.mean((0, 1)), features.std((0, 1))
    scale[scale < 1e-8] = 1
    return mean, scale


def loss_kwargs(config):
    return {'objective': config.get('objective', 'group'), 'alpha': config.get('alpha', 0.),
            'tau': config.get('tau', 1.), 'reduction': config.get('reduction', 'full')}


def fit_linear(features, labels, groups, config):
    start = time.perf_counter()
    mean, scale = scaler(features)
    x = np.ascontiguousarray((features - mean) / scale)
    hard_fraction = float((groups.max(1) >= 1).mean())
    l2 = config.get('l2', .1) / (hard_fraction if config.get('scale_regularization') else 1.)

    def objective(theta):
        scores = x @ theta[:-1] + theta[-1]
        loss, gradient = score_loss_gradient(scores, labels, groups, **loss_kwargs(config))
        value = loss + .5 * l2 * np.dot(theta[:-1], theta[:-1])
        grad = np.r_[np.einsum('nkd,nk->d', x, gradient) + l2 * theta[:-1], gradient.sum()]
        return value, grad

    result = minimize(objective, np.zeros(x.shape[-1] + 1), method='L-BFGS-B', jac=True,
                      options={'maxiter': 200, 'ftol': 1e-9, 'gtol': 1e-6, 'maxls': 30})
    if not np.isfinite(result.fun) or not np.isfinite(result.x).all():
        raise ValueError('Nonfinite linear fit')
    state = {'weight': result.x[:-1], 'bias': np.asarray(result.x[-1])}
    return state, mean, scale, {'seconds': time.perf_counter() - start, 'iterations': int(result.nit),
        'optimizer_success': bool(result.success), 'optimizer_message': str(result.message),
        'objective': float(result.fun), 'effective_l2': l2, 'hard_fraction': hard_fraction}


def fit_neural(features, labels, groups, fit_rows, stop_rows, config, seed, log_path, epochs=None):
    import torch
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    mean, scale = scaler(features[fit_rows])
    x = torch.from_numpy(((features[fit_rows] - mean) / scale).astype(np.float32))
    y = torch.from_numpy(labels[fit_rows].astype(np.float32))
    g = torch.from_numpy(groups[fit_rows])
    if stop_rows is not None:
        sx = torch.from_numpy(((features[stop_rows] - mean) / scale).astype(np.float32))
        sy = torch.from_numpy(labels[stop_rows].astype(np.float32))
        sg = torch.from_numpy(groups[stop_rows])
    model = build_model(config)
    hard_fraction = float((groups[fit_rows].max(1) >= 1).mean())
    decay = config.get('weight_decay', .001) / (hard_fraction if config.get('scale_regularization') else 1.)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.get('lr', .001), weight_decay=decay)
    maximum = int(epochs if epochs is not None else config.get('max_epochs', 60))
    best_loss, best_epoch, history = float('inf'), 1, []
    start = time.perf_counter()
    for epoch in range(1, maximum + 1):
        model.train()
        order = rng.permutation(len(x))
        total = 0.
        for offset in range(0, len(x), config.get('batch_size', 128)):
            batch = order[offset:offset + config.get('batch_size', 128)]
            optimizer.zero_grad(set_to_none=True)
            loss = torch_loss(model(x[batch]), y[batch], g[batch], **loss_kwargs(config))
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite neural loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            total += float(loss.detach()) * len(batch)
        row = {'epoch': epoch, 'train_loss': total / len(x), 'seconds': time.perf_counter() - start}
        if stop_rows is not None:
            model.eval()
            with torch.no_grad():
                stop_loss = float(torch_loss(model(sx), sy, sg, **loss_kwargs(config)))
            row['inner_stop_loss'] = stop_loss
            if stop_loss < best_loss - 1e-6:
                best_loss, best_epoch = stop_loss, epoch
        history.append(row)
        write_json(log_path, {'seed': seed, 'effective_weight_decay': decay, 'hard_fraction': hard_fraction,
                              'history': history, 'best_epoch': best_epoch if stop_rows is not None else maximum})
        if epoch == 1 or epoch % 5 == 0:
            print(json.dumps({'phase': str(log_path.relative_to(ROOT)), **row}), flush=True)
        if stop_rows is not None and epoch - best_epoch >= config.get('patience', 8):
            break
    model.eval()
    state = export_state(model)
    # Independent numeric export check on actual fit rows, before OOF scoring.
    check_x = x[:min(64, len(x))]
    with torch.no_grad():
        torch_scores = model(check_x).numpy()
    np_scores = numpy_forward(check_x.numpy(), config, state)
    max_error = float(np.max(np.abs(torch_scores - np_scores)))
    if not np.allclose(torch_scores, np_scores, atol=2e-5, rtol=2e-5):
        raise ValueError(f'NumPy export score mismatch {max_error}')
    if select_candidates(torch_scores, groups[fit_rows][:len(check_x)], **config['decoder']) != select_candidates(np_scores, groups[fit_rows][:len(check_x)], **config['decoder']):
        raise ValueError('NumPy export changes selected answers')
    info = {'seconds': time.perf_counter() - start, 'epochs_run': len(history),
            'best_epoch': best_epoch if stop_rows is not None else maximum,
            'inner_best_loss': best_loss if stop_rows is not None else None,
            'numpy_max_abs_error': max_error, 'effective_weight_decay': decay, 'hard_fraction': hard_fraction,
            'torch_version': torch.__version__, 'device': 'cpu', 'torch_threads': torch.get_num_threads()}
    return state, mean, scale, info


def predict_features(features, config, state, mean, scale):
    dtype = np.float64 if config['family'] == 'linear' else np.float32
    return np.concatenate([numpy_forward(((features[i:i+256] - mean) / scale).astype(dtype), config, state)
                           for i in range(0, len(features), 256)])


def run_cv(config, only_fold=None):
    start = time.perf_counter()
    experiment = config['experiment_id']
    destination = OUT / experiment
    destination.mkdir(parents=True, exist_ok=True)
    features, labels, groups, folds = data_and_folds()
    prov = provenance()
    registration = {'config': config, 'config_sha256': config_hash(config), **prov,
                    'folds_sha256': digest(OUT / 'common_folds.json')}
    existing = destination / 'registration.json'
    if existing.exists():
        old = json.loads(existing.read_text())
        for key in ('config_sha256', 'source_files', 'data_files', 'folds_sha256'):
            if old[key] != registration[key]:
                raise ValueError(f'Resume provenance mismatch: {experiment}: {key}')
    else:
        write_json(existing, registration)
        write_json(ROOT / 'code/configs/research_v2' / (experiment + '.json'), config)
    all_scores = np.full((len(labels), 4), np.nan)
    reports = []
    for split in folds:
        index = split['fold']
        if only_fold is not None and index != only_fold:
            continue
        result_path = destination / f'fold{index}.json'
        score_path = destination / f'fold{index}-scores.npy'
        holdout = np.array(split['holdout'])
        if result_path.exists():
            report = json.loads(result_path.read_text())
            if report['scores_sha256'] != digest(score_path):
                raise ValueError('Changed completed OOF scores')
            all_scores[holdout] = np.load(score_path, allow_pickle=False)
            reports.append(report)
            continue
        attempt = 1
        while (destination / f'fold{index}-attempt{attempt}').exists():
            attempt += 1
        attempt_dir = destination / f'fold{index}-attempt{attempt}'
        attempt_dir.mkdir()
        checkpoint = ASSETS / experiment / f'fold{index}-attempt{attempt}'
        event({'experiment_id': experiment, 'fold': index, 'attempt': attempt, 'status': 'started',
               'config_sha256': registration['config_sha256'], 'seed': config['seed'],
               'registration': str(existing.relative_to(ROOT))})
        try:
            fitting = np.array(split['outer_fit'])
            if config['family'] == 'linear':
                state, mean, scale, fit_info = fit_linear(features[fitting], labels[fitting], groups[fitting], config)
                stop_info = None
            else:
                _, _, _, stop_info = fit_neural(features, labels, groups, np.array(split['inner_fit']),
                    np.array(split['inner_stop']), config, config['seed'] + index, attempt_dir / 'inner_epochs.json')
                state, mean, scale, fit_info = fit_neural(features, labels, groups, fitting, None, config,
                    config['seed'] + index, attempt_dir / 'refit_epochs.json', epochs=stop_info['best_epoch'])
            scores = predict_features(features[holdout], config, state, mean, scale)
            if not np.isfinite(scores).all():
                raise ValueError('Nonfinite held-out scores')
            save_bundle(checkpoint, config, state, mean, scale, config['decoder'])
            np.save(score_path, scores)
            byte_count = sum(v.nbytes for v in state.values()) + mean.nbytes + scale.nbytes
            report = {'experiment_id': experiment, 'fold': index, 'status': 'complete', 'seed': config['seed'],
                'fit': fit_info, 'inner_stopping': stop_info, 'parameters': sum(v.size for v in state.values()),
                'numeric_asset_bytes': byte_count, 'checkpoint': checkpoint.relative_to(ROOT).as_posix(),
                'scores_sha256': digest(score_path), **metrics(labels[holdout], scores, groups[holdout], config['decoder'])}
            write_json(result_path, report)
            event(report)
            all_scores[holdout] = scores
            reports.append(report)
        except Exception as exc:
            event({'experiment_id': experiment, 'fold': index, 'attempt': attempt, 'status': 'failed',
                   'exception': repr(exc)})
            raise
    if len(reports) == 3:
        np.savez(destination / 'oof.npz', scores=all_scores, groups=groups, labels=labels,
                 fold=np.array([next(s['fold'] for s in folds if i in s['holdout']) for i in range(len(labels))]))
        summary = {'experiment_id': experiment, 'status': 'complete', 'config': config, 'folds': reports,
                   'wall_seconds_this_invocation': time.perf_counter() - start,
                   'training_seconds': sum(r['fit']['seconds'] + (r['inner_stopping']['seconds'] if r['inner_stopping'] else 0) for r in reports),
                   **metrics(labels, all_scores, groups, config['decoder'])}
        for key in ('accuracy', 'hard_accuracy', 'single_positive_accuracy'):
            values = [r[key] for r in reports]
            summary[key + '_fold_mean'] = float(np.mean(values))
            summary[key + '_fold_sd'] = float(np.std(values, ddof=1))
        write_json(destination / 'summary.json', summary)
        event({'experiment_id': experiment, 'status': 'cv_complete', 'hard_accuracy': summary['hard_accuracy'],
               'accuracy': summary['accuracy'], 'training_seconds': summary['training_seconds']})
        return summary
    return reports


def fit_final(config, output):
    """All-train fit using epochs determined by that configuration's inner CV."""
    destination = OUT / config['experiment_id']
    summary = json.loads((destination / 'summary.json').read_text())
    if summary['config'] != config:
        raise ValueError('Final config differs from registered CV')
    features, labels, groups, _ = data_and_folds()
    if config['family'] == 'linear':
        state, mean, scale, info = fit_linear(features, labels, groups, config)
    else:
        epochs = int(math.ceil(np.median([r['inner_stopping']['best_epoch'] for r in summary['folds']])))
        state, mean, scale, info = fit_neural(features, labels, groups, np.arange(len(labels)), None,
            config, config['seed'], destination / 'final_epochs.json', epochs)
    save_bundle(output, config, state, mean, scale, config['decoder'])
    write_json(destination / 'final_fit.json', {'config': config, 'checkpoint': str(output), 'info': info,
               'parameters': sum(v.size for v in state.values()), 'provenance': provenance()})
    event({'experiment_id': config['experiment_id'], 'status': 'final_train_complete', **info})

