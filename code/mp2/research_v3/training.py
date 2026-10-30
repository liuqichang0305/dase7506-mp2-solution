"""Train-only nested CV with atomic epoch-level recovery and immutable evidence."""
from __future__ import annotations

from copy import deepcopy
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import uuid

import numpy as np
import torch

from common import DATA
from ..features import build_features
from ..training import load_development
from ..selection_v2 import answer_groups, select_candidates, score_loss_gradient
from .models import build_model, export_state, numpy_forward, save_bundle, load_bundle, model_config
from .optimizers import build_optimizers, EMA
from .preprocess import fit_preprocessor, transform_features
from .selection import member_loss, collapse_scores, select_raw, oof_decoder
from tools.audit_assets import audit

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'results/research_v3'
ASSETS = ROOT / 'artifacts/research_v3'
STATE_ROOT = ROOT / 'tmp/research_v3/training_states'
SOURCE_FILES = ('code/train_v3.py', 'code/common.py', 'code/mp2/features.py', 'code/mp2/training.py',
                'code/mp2/selection_v2.py', 'code/mp2/models_v2.py', 'code/mp2/research_v3/training.py',
                'code/mp2/research_v3/models.py', 'code/mp2/research_v3/optimizers.py',
                'code/mp2/research_v3/preprocess.py', 'code/mp2/research_v3/selection.py')
_GUARD_INSTALLED = False


class TrainingPaused(RuntimeError):
    """Explicit synthetic interruption, after a durable epoch checkpoint."""


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def array_hash(value):
    value = np.ascontiguousarray(value)
    h = hashlib.sha256(str((value.shape, value.dtype.str)).encode())
    h.update(value.tobytes())
    return h.hexdigest()


def write_json(path, value, immutable=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if immutable and path.exists():
        if json.loads(path.read_text(encoding='utf-8')) != value:
            raise FileExistsError(f'Refusing to replace immutable evidence: {path}')
        return
    temporary = path.with_suffix(path.suffix + '.pending')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def _check_access(path, mode=None, flags=0):
    if not isinstance(path, (str, bytes, os.PathLike)):
        return
    target = Path(os.fsdecode(path)).resolve()
    name = target.name.lower()
    writing = (isinstance(mode, str) and any(c in mode for c in 'wa+')) or bool((flags or 0) & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC))
    if 'leaderboard' in name or 'ranking_background' in name:
        raise PermissionError('Research trainer cannot read ranking feedback')
    data = DATA.resolve()
    if target.is_relative_to(data):
        if writing:
            raise PermissionError('Official data is immutable')
        relative = target.relative_to(data).as_posix()
        if relative not in {'manifest.json', 'gsm8k_train.jsonl', 'candidates/gsm8k_train.jsonl', 'candidates/gsm8k_train.npz'}:
            raise PermissionError('Research data allowlist permits only official train and public manifest')
    protected = ('results/development', 'results/reliability', 'results/research_v2', 'artifacts/final_bundle', 'artifacts/experiments', 'artifacts/research_v2')
    if writing and any(target.is_relative_to((ROOT / prefix).resolve()) for prefix in protected):
        raise PermissionError('V3 cannot modify preserved prior experiment evidence/assets')


def install_training_guard():
    global _GUARD_INSTALLED
    if not _GUARD_INSTALLED:
        def hook(event, args):
            if event == 'open':
                _check_access(args[0], args[1], args[2])
        sys.addaudithook(hook)
        _GUARD_INSTALLED = True


@contextmanager
def training_lock():
    """Serialize project training, with PID creation-time/command identity."""
    import psutil
    lock = ROOT / 'tmp/research_v3/trainer.lock'
    lock.parent.mkdir(parents=True, exist_ok=True)
    process = psutil.Process(os.getpid())
    owner = {'pid': process.pid, 'create_time': process.create_time(), 'command_line': process.cmdline(),
             'nonce': uuid.uuid4().hex, 'created_at_utc': datetime.now(timezone.utc).isoformat()}
    while True:
        try:
            with lock.open('x', encoding='utf-8') as stream:
                json.dump(owner, stream)
            break
        except FileExistsError:
            try:
                old = json.loads(lock.read_text(encoding='utf-8'))
                other = psutil.Process(old['pid'])
                live = abs(other.create_time() - old['create_time']) < 1e-6 and other.cmdline() == old['command_line']
            except (psutil.NoSuchProcess, KeyError):
                live = False
            except psutil.AccessDenied as exc:
                raise RuntimeError('Cannot verify existing trainer lock owner; refusing a second trainer') from exc
            except (ValueError, OSError):
                if lock.exists() and time.time() - lock.stat().st_mtime < 30:
                    raise RuntimeError('Trainer lock is being initialized; refusing a second trainer')
                live = False
            if live:
                raise RuntimeError(f'Another project trainer is active: PID {old["pid"]}')
            stale = lock.with_name('trainer.stale-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '.json')
            try:
                lock.rename(stale)
            except FileNotFoundError:
                continue
    try:
        yield owner
    finally:
        if lock.exists():
            try:
                current = json.loads(lock.read_text(encoding='utf-8'))
                if current == owner:
                    lock.unlink()
            except (OSError, ValueError):
                pass


def normalize_config(config):
    result = deepcopy(config)
    experiment = result.get('experiment_id', result.get('id'))
    if not isinstance(experiment, str) or re.fullmatch('[a-z0-9_]+', experiment) is None:
        raise ValueError('A single safe experiment_id/id directory name is required')
    if 'id' in result and result['id'] != experiment:
        raise ValueError('id and experiment_id differ')
    result['experiment_id'] = experiment
    if result.get('lr_schedule', 'constant') != 'constant':
        raise ValueError('Only explicitly constant learning rates are implemented in this recipe')
    return result


def check_stage_permission(final_experiment=None):
    if (ROOT / 'submission/v3/FREEZE.json').exists():
        raise RuntimeError('V3 is frozen: use prediction-only reproduction, not fitting or search')
    locks = [OUT / 'FINALISTS_LOCK.json', OUT / 'FINALISTS_SELECTION.json']
    lock = next((path for path in locks if path.exists()), None)
    if final_experiment is None:
        if lock is not None or (OUT / 'SEARCH_CLOSED.json').exists():
            raise RuntimeError('V3 search is closed; no further CV is permitted')
    else:
        if lock is None:
            raise RuntimeError('Final fitting requires an explicit finalist-component lock')
        allowed = json.loads(lock.read_text(encoding='utf-8'))['allowed_final_fit_experiments']
        if final_experiment not in allowed:
            raise RuntimeError('Final fit is not a locked finalist component')


def provenance():
    train = ['code/data/manifest.json', 'code/data/gsm8k_train.jsonl', 'code/data/candidates/gsm8k_train.jsonl', 'code/data/candidates/gsm8k_train.npz']
    return {'source_files': {name: digest(ROOT / name) for name in SOURCE_FILES},
            'data_files': {name: digest(ROOT / name) for name in train},
            'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
            'folds_sha256': digest(OUT / 'common_folds.json'),
            'torch_version': str(torch.__version__), 'numpy_version': np.__version__, 'python': sys.version}


def event(record):
    OUT.mkdir(parents=True, exist_ok=True)
    value = {'utc': datetime.now(timezone.utc).isoformat(), 'pid': os.getpid(), **record}
    with (OUT / 'training_events.jsonl').open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(value) + '\n')
    print(json.dumps(value), flush=True)


def data_and_folds():
    install_training_guard()
    bank, embeddings, labels = load_development('train')
    groups = answer_groups(bank)
    features = build_features(bank, embeddings, 'relation')
    path = OUT / 'common_folds.json'
    if not path.exists():
        old = ROOT / 'results/research_v2/common_folds.json'
        value = json.loads(old.read_text(encoding='utf-8'))
        write_json(path, value, immutable=True)
    manifest = json.loads(path.read_text(encoding='utf-8'))
    ids_hash = hashlib.sha256(json.dumps([r['id'] for r in bank]).encode()).hexdigest()
    if manifest['questions'] != len(bank) or manifest['train_ids_sha256'] != ids_hash:
        raise ValueError('Common folds do not match train bank order')
    universe, held = set(range(len(bank))), []
    for split in manifest['folds']:
        a, b, c, d = (set(split[key]) for key in ('outer_fit', 'holdout', 'inner_fit', 'inner_stop'))
        if a & b or a | b != universe or c & d or c | d != a:
            raise ValueError('Invalid nested fold partition')
        held.extend(split['holdout'])
    if len(manifest['folds']) != 3 or sorted(held) != list(range(len(bank))):
        raise ValueError('Every train question must be held out exactly once')
    duplicate_groups = {}
    for index, row in enumerate(bank):
        duplicate_groups.setdefault(' '.join(row['question'].split()), []).append(index)
    membership = {i: split['fold'] for split in manifest['folds'] for i in split['holdout']}
    if any(len({membership[i] for i in rows}) > 1 for rows in duplicate_groups.values() if len(rows) > 1):
        raise ValueError('Duplicate question crosses outer folds; grouped split revision required')
    return features, labels, groups, manifest['folds']


def metrics(labels, scores, groups, decoder):
    choices = np.asarray(select_candidates(scores, groups, **decoder), dtype=np.int64)
    correct = labels[np.arange(len(labels)), choices].astype(bool)
    result = {}
    hard = np.array([len(set(row[row >= 0])) >= 2 for row in groups])
    for prefix, subset in (('', np.ones(len(labels), bool)), ('hard_', hard), ('single_positive_', labels.sum(1) == 1)):
        result.update({prefix + 'correct': int(correct[subset].sum()), prefix + 'n': int(subset.sum()),
                       prefix + 'accuracy': float(100 * correct[subset].mean()) if subset.any() else None})
    return result


def loss_kwargs(config):
    return {'objective': config.get('objective', 'group'), 'alpha': config.get('alpha', 0.),
            'tau': config.get('tau', 1.), 'reduction': config.get('reduction', 'full')}


def _torch_preprocess(preprocess):
    return {key: (torch.from_numpy(value.copy()) if isinstance(value, np.ndarray) else value) for key, value in preprocess.items()}


def _numpy_preprocess(preprocess):
    return {key: (value.detach().cpu().numpy().copy() if isinstance(value, torch.Tensor) else value) for key, value in preprocess.items()}


def _atomic_checkpoint(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.pending.pt')
    torch.save(value, temporary)
    temporary.replace(path)


def fit_neural(features, labels, groups, fit_rows, stop_rows, config, seed, state_dir, log_path,
               context, epochs=None, max_new_epochs=None):
    """Resume at an epoch boundary; test-only max_new_epochs simulates interruption."""
    started = time.perf_counter()
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    fit_rows = np.asarray(fit_rows, dtype=np.int64)
    stop_rows = None if stop_rows is None else np.asarray(stop_rows, dtype=np.int64)
    if not len(fit_rows) or stop_rows is not None and (not len(stop_rows) or np.intersect1d(fit_rows, stop_rows).size):
        raise ValueError('Fitting and stopping partitions must be nonempty and disjoint')
    if config.get('lr_schedule', 'constant') != 'constant':
        raise ValueError('Unsupported unregistered learning-rate schedule')
    preprocess = fit_preprocessor(features[fit_rows], config.get('preprocess', 'standard'))
    x = torch.from_numpy(transform_features(features[fit_rows], preprocess))
    y = torch.from_numpy(labels[fit_rows].astype(np.float32))
    g = torch.from_numpy(groups[fit_rows].astype(np.int64))
    if stop_rows is not None:
        sx = torch.from_numpy(transform_features(features[stop_rows], preprocess))
        sy = torch.from_numpy(labels[stop_rows].astype(np.float32))
        sg = torch.from_numpy(groups[stop_rows].astype(np.int64))
    model = build_model(config)
    optimizer_config = deepcopy(config)
    hard_fraction = float((groups[fit_rows].max(axis=1) >= 1).mean())
    if config.get('scale_regularization'):
        if hard_fraction == 0:
            raise ValueError('Cannot rescale regularization without hard questions')
        optimizer_config['weight_decay'] = config.get('weight_decay', .001) / hard_fraction
        if 'muon_weight_decay' in config:
            optimizer_config['muon_weight_decay'] = config['muon_weight_decay'] / hard_fraction
    optimizer = build_optimizers(model, optimizer_config)
    ema = EMA(model, config['ema_decay']) if config.get('ema_decay') is not None else None
    maximum = int(epochs if epochs is not None else config.get('max_epochs', 60))
    if maximum < 1:
        raise ValueError('At least one training epoch is required')
    identity = {'config_sha256': object_hash(config), 'context': context, 'seed': int(seed),
                'fitting_rows_sha256': array_hash(fit_rows), 'stopping_rows_sha256': None if stop_rows is None else array_hash(stop_rows),
                'maximum_epochs': maximum, 'stopping': stop_rows is not None,
                'preprocess_hashes': {key: array_hash(value) if isinstance(value, np.ndarray) else value for key, value in preprocess.items()}}
    checkpoint_path = Path(state_dir) / 'current.pt'
    history, best_loss, best_epoch, best_state = [], float('inf'), 0, None
    previous_seconds, completed, resumed_epoch = 0., False, 0
    if checkpoint_path.exists():
        saved = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        if saved.get('schema') != 'mp2-v3-epoch-state-v1' or saved['identity'] != identity:
            raise ValueError('Checkpoint source/config/data/fold/preprocessing identity differs')
        saved_preprocess = _numpy_preprocess(saved['preprocess'])
        if any(not np.array_equal(preprocess[key], saved_preprocess[key]) for key in preprocess):
            raise ValueError('Recovered model and preprocessing coordinates differ')
        model.load_state_dict(saved['model'], strict=True)
        optimizer.load_state_dict(saved['optimizer'])
        if (saved['ema'] is None) != (ema is None):
            raise ValueError('EMA policy differs on resume')
        if ema is not None:
            ema.load_state_dict(saved['ema'])
        torch.set_rng_state(saved['torch_rng'])
        rng.bit_generator.state = saved['numpy_rng']
        history, best_loss, best_epoch, best_state = saved['history'], saved['best_loss'], saved['best_epoch'], saved['best_eval_state']
        previous_seconds, completed, resumed_epoch = saved['elapsed_seconds'], saved['complete'], saved['epoch']
        if saved['epoch'] != len(history):
            raise ValueError('Checkpoint epoch/history mismatch')
    new_epochs = 0
    if not completed:
        for epoch in range(resumed_epoch + 1, maximum + 1):
            model.train()
            order = rng.permutation(len(x))
            total, normalizer = 0., 0
            batch_size = int(config.get('batch_size', 128))
            if batch_size < 1:
                raise ValueError('Invalid batch size')
            for start in range(0, len(x), batch_size):
                batch = order[start:start + batch_size]
                bx, by, bg = x[batch], y[batch], g[batch]
                if config.get('permute_candidates', False):
                    permutations = np.stack([rng.permutation(4) for _ in batch])
                    rows = torch.arange(len(batch))[:, None]
                    permutation = torch.from_numpy(permutations)
                    bx, by, bg = bx[rows, permutation], by[rows, permutation], bg[rows, permutation]
                optimizer.zero_grad(set_to_none=True)
                loss = member_loss(model(bx), by, bg, **loss_kwargs(config))
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite training objective')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.get('clip_grad_norm', 5.)))
                optimizer.step()
                if ema is not None:
                    ema.update(model)
                weight = int((bg.max(dim=1).values >= 1).sum()) if config.get('reduction', 'full') == 'hard' else len(batch)
                total += float(loss.detach()) * weight
                normalizer += weight
            model.eval()
            evaluation = ema.module if ema is not None else model
            evaluation.eval()
            row = {'epoch': epoch, 'train_loss': total / max(normalizer, 1),
                   'elapsed_seconds': previous_seconds + time.perf_counter() - started,
                   'optimizer_steps': optimizer.step_count, 'ema_updates': ema.num_updates if ema is not None else None}
            if stop_rows is not None:
                with torch.no_grad():
                    stop_loss = float(member_loss(evaluation(sx), sy, sg, **loss_kwargs(config)))
                if not math.isfinite(stop_loss):
                    raise FloatingPointError('Nonfinite inner stopping objective')
                row['inner_stop_loss'] = stop_loss
                if stop_loss < best_loss - float(config.get('min_delta', 1e-6)):
                    best_loss, best_epoch = stop_loss, epoch
                    best_state = {name: value.detach().clone() for name, value in evaluation.state_dict().items()}
            else:
                best_epoch = epoch
            history.append(row)
            completed = epoch == maximum or stop_rows is not None and epoch - best_epoch >= int(config.get('patience', 8))
            saved = {'schema': 'mp2-v3-epoch-state-v1', 'identity': identity, 'epoch': epoch, 'complete': bool(completed),
                     'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'ema': ema.state_dict() if ema is not None else None,
                     'preprocess': _torch_preprocess(preprocess), 'torch_rng': torch.get_rng_state(), 'numpy_rng': rng.bit_generator.state,
                     'scheduler': {'name': 'constant', 'epoch': epoch}, 'history': history, 'best_loss': best_loss,
                     'best_epoch': best_epoch, 'best_eval_state': best_state,
                     'elapsed_seconds': previous_seconds + time.perf_counter() - started}
            _atomic_checkpoint(checkpoint_path, saved)
            write_json(log_path, {'identity': identity, 'optimizer_groups': optimizer.group_report,
                'evaluation_weights': 'ema' if ema is not None else 'live', 'history': history,
                'best_epoch': best_epoch, 'complete': bool(completed), 'checkpoint': str(checkpoint_path)})
            print(json.dumps({'phase': str(log_path), **row, 'complete': bool(completed)}), flush=True)
            new_epochs += 1
            if completed:
                break
            if max_new_epochs is not None and new_epochs >= max_new_epochs:
                raise TrainingPaused('Synthetic stop after saved epoch; resume current.pt')
    evaluation = deepcopy(ema.module if ema is not None else model).eval()
    if stop_rows is not None:
        if best_state is None:
            raise ValueError('No valid inner-selected state')
        evaluation.load_state_dict(best_state)
    state = export_state(evaluation)
    check = x[:min(64, len(x))]
    with torch.no_grad():
        reference = evaluation(check).numpy()
    exported = numpy_forward(check.numpy(), config, state)
    if not np.allclose(reference, exported, atol=2e-5, rtol=2e-5):
        raise ValueError('Exported NumPy scores disagree with actual selected training weights')
    check_groups = groups[fit_rows][:len(check)]
    if select_raw(reference, check_groups, config['decoder']) != select_raw(exported, check_groups, config['decoder']):
        raise ValueError('NumPy export changes selected original indices')
    info = {'seconds': previous_seconds + time.perf_counter() - started, 'epochs_run': len(history),
            'best_epoch': best_epoch, 'inner_best_loss': best_loss if stop_rows is not None else None,
            'evaluation_weights': 'ema' if ema is not None else 'live', 'ema_updates': ema.num_updates if ema is not None else None,
            'optimizer_steps': optimizer.step_count, 'optimizer_groups': optimizer.group_report,
            'hard_fraction': hard_fraction, 'resumed_from_epoch': resumed_epoch,
            'numpy_max_abs_error': float(np.max(np.abs(reference - exported))), 'device': 'cpu',
            'torch_threads': torch.get_num_threads(), 'torch_version': torch.__version__,
            'checkpoint': str(checkpoint_path), 'checkpoint_sha256': digest(checkpoint_path), 'context': context}
    return state, preprocess, info


def fit_linear(features, labels, groups, config):
    from scipy.optimize import minimize
    start = time.perf_counter()
    preprocess = fit_preprocessor(features, config.get('preprocess', 'standard'))
    x = transform_features(features, preprocess, np.float64)
    hard_fraction = float((groups.max(axis=1) >= 1).mean())
    l2 = config.get('l2', .1) / (hard_fraction if config.get('scale_regularization') else 1.)
    def objective(theta):
        scores = x @ theta[:-1] + theta[-1]
        loss, gradient = score_loss_gradient(scores, labels, groups, **loss_kwargs(config))
        return loss + .5 * l2 * np.dot(theta[:-1], theta[:-1]), np.r_[np.einsum('nkd,nk->d', x, gradient) + l2 * theta[:-1], gradient.sum()]
    result = minimize(objective, np.zeros(x.shape[-1] + 1), jac=True, method='L-BFGS-B',
        options={'maxiter': int(config.get('maxiter', 200)), 'ftol': 1e-9, 'gtol': 1e-6, 'maxls': 30})
    if not np.isfinite(result.fun) or not np.isfinite(result.x).all():
        raise FloatingPointError('Invalid linear optimizer result')
    return {'weight': result.x[:-1], 'bias': np.asarray(result.x[-1])}, preprocess, {
        'seconds': time.perf_counter() - start, 'iterations': int(result.nit), 'optimizer_success': bool(result.success),
        'optimizer_message': str(result.message), 'effective_l2': l2, 'hard_fraction': hard_fraction}


def predict_features(features, config, state, preprocess):
    dtype = np.float64 if config['family'] == 'linear' else np.float32
    if not len(features):
        return np.empty((0, config.get('k', 4), 4) if config['family'] == 'tabm' else (0, 4), dtype=dtype)
    return np.concatenate([numpy_forward(transform_features(features[i:i+128], preprocess, dtype), config, state)
                           for i in range(0, len(features), 128)])


def _save_or_verify_bundle(path, config, state, preprocess):
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        complete = (path / 'config.json').is_file() and (path / 'weights.npz').is_file()
        if complete:
            record, arrays = load_bundle(path)
            expected = {**state, 'mean': preprocess['mean'], 'scale': preprocess['scale'],
                        'prep_clip_mask': np.asarray(preprocess['clip_mask'], np.uint8), 'prep_constant_mask': np.asarray(preprocess['constant_mask'], np.uint8)}
            if set(arrays) != set(expected) or any(not np.array_equal(arrays[key], value) for key, value in expected.items()) or record['decoder'] != config['decoder'] or record['architecture'] != model_config(config) or record['preprocess']['mode'] != preprocess['mode']:
                raise ValueError('Existing complete bundle differs; preserve it rather than overwrite')
            return
        resolved = path.resolve()
        if not resolved.is_relative_to(ASSETS.resolve()):
            raise ValueError('Can quarantine only an incomplete V3 artifact directory')
        quarantine = path.with_name(path.name + '.incomplete-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
        path.rename(quarantine)
        event({'status': 'incomplete_bundle_quarantined', 'original': str(path), 'preserved_at': str(quarantine)})
    save_bundle(path, config, state, preprocess['mean'], preprocess['scale'], config['decoder'], preprocess=preprocess)


def _save_array_once(path, value):
    if path.exists():
        if not np.array_equal(np.load(path, allow_pickle=False), value):
            raise ValueError('Existing output array differs; refusing to overwrite')
        return
    pending = path.with_suffix('.pending.npy')
    np.save(pending, value)
    pending.replace(path)


def _verify_completed_fold(folder, report):
    for name, sha in report['output_sha256'].items():
        if digest(folder / name) != sha:
            raise ValueError('Completed fold output changed')
    for name, sha in report['asset_sha256'].items():
        if digest(ROOT / report['checkpoint'] / name) != sha:
            raise ValueError('Completed fold artifact changed')


def run_cv(config, only_fold=None):
    install_training_guard()
    check_stage_permission()
    initial_config_hash = object_hash(config)
    config = normalize_config(config)
    features, labels, groups, folds = data_and_folds()
    experiment, started = config['experiment_id'], time.perf_counter()
    folder = OUT / experiment
    folder.mkdir(parents=True, exist_ok=True)
    prov = provenance()
    registration = {'config': config, 'config_sha256': object_hash(config), 'input_config_sha256': initial_config_hash,
                    'folds_sha256': digest(OUT / 'common_folds.json'), **prov}
    registration_path = folder / 'registration.json'
    if registration_path.exists():
        saved = json.loads(registration_path.read_text(encoding='utf-8'))
        for key in ('config_sha256', 'source_files', 'data_files', 'folds_sha256'):
            if saved[key] != registration[key]:
                raise ValueError(f'Resume provenance differs: {experiment}: {key}')
    else:
        write_json(registration_path, registration, immutable=True)
    effective = oof_decoder(config)
    all_scores = np.full((len(labels), 4), np.nan)
    raw_shape = (len(labels), config.get('k', 4), 4) if config['family'] == 'tabm' else (len(labels), 4)
    all_raw = np.full(raw_shape, np.nan, dtype=np.float64 if config['family'] == 'linear' else np.float32)
    reports = []
    for split in folds:
        number = split['fold']
        if only_fold is not None and number != only_fold:
            continue
        report_path = folder / f'fold{number}.json'
        held = np.asarray(split['holdout'], dtype=np.int64)
        if report_path.exists():
            report = json.loads(report_path.read_text(encoding='utf-8'))
            if any(report['provenance'][key] != prov[key] for key in ('source_files', 'data_files', 'folds_sha256')):
                raise ValueError('Completed fold provenance differs')
            _verify_completed_fold(folder, report)
            all_scores[held] = np.load(folder / f'fold{number}-scores.npy', allow_pickle=False)
            all_raw[held] = np.load(folder / f'fold{number}-raw-scores.npy', allow_pickle=False)
            reports.append(report)
            continue
        context = {'source_files': prov['source_files'], 'data_files': prov['data_files'],
                   'folds_sha256': registration['folds_sha256'], 'fold': number}
        event({'experiment_id': experiment, 'fold': number, 'status': 'running', 'seed': config['seed'] + number})
        try:
            fitting = np.asarray(split['outer_fit'], dtype=np.int64)
            if config['family'] == 'linear':
                state, preprocess, info = fit_linear(features[fitting], labels[fitting], groups[fitting], config)
                inner = None
            else:
                _, _, inner = fit_neural(features, labels, groups, split['inner_fit'], split['inner_stop'], config,
                    config['seed'] + number, STATE_ROOT / experiment / f'fold{number}' / 'inner',
                    folder / f'fold{number}-inner-epochs.json', {**context, 'phase': 'inner'})
                state, preprocess, info = fit_neural(features, labels, groups, fitting, None, config,
                    config['seed'] + number, STATE_ROOT / experiment / f'fold{number}' / 'outer',
                    folder / f'fold{number}-outer-epochs.json', {**context, 'phase': 'outer'}, epochs=inner['best_epoch'])
            raw = predict_features(features[held], config, state, preprocess)
            scores = collapse_scores(raw, groups[held], config['decoder'])
            if select_raw(raw, groups[held], config['decoder']) != select_candidates(scores, groups[held], **effective):
                raise ValueError('Collapsed OOF representation changes authoritative prediction')
            if provenance()['source_files'] != prov['source_files']:
                raise ValueError('Training source changed while a fold ran')
            checkpoint = ASSETS / experiment / f'fold{number}'
            _save_or_verify_bundle(checkpoint, config, state, preprocess)
            score_path, raw_path = folder / f'fold{number}-scores.npy', folder / f'fold{number}-raw-scores.npy'
            _save_array_once(score_path, scores)
            _save_array_once(raw_path, raw)
            resource = audit(checkpoint)
            report = {'experiment_id': experiment, 'fold': number, 'status': 'complete', 'seed': config['seed'] + number,
                'provenance': prov,
                'fit': info, 'inner_stopping': inner, 'parameters': sum(value.size for value in state.values()),
                'numeric_asset_bytes': sum(value.nbytes for value in state.values()) + sum(value.nbytes for value in preprocess.values() if isinstance(value, np.ndarray)),
                'audited_uncompressed_bytes': resource['uncompressed_bytes'], 'checkpoint': checkpoint.relative_to(ROOT).as_posix(),
                'scores_sha256': digest(score_path), 'output_sha256': {score_path.name: digest(score_path), raw_path.name: digest(raw_path)},
                'asset_sha256': {item['path']: item['sha256'] for item in resource['files']}, 'oof_decoder': effective,
                **metrics(labels[held], scores, groups[held], effective)}
            write_json(report_path, report, immutable=True)
            reports.append(report)
            all_scores[held], all_raw[held] = scores, raw
            event({'experiment_id': experiment, 'fold': number, 'status': 'completed', 'hard_accuracy': report['hard_accuracy'],
                   'accuracy': report['accuracy'], 'seconds': info['seconds'] + (inner['seconds'] if inner else 0)})
        except Exception as exc:
            event({'experiment_id': experiment, 'fold': number, 'status': 'failed', 'reason': repr(exc)})
            raise
    if len(reports) != 3:
        return reports
    summary_path = folder / 'summary.json'
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding='utf-8'))
        if summary['config'] != config or any(summary['provenance'][key] != prov[key] for key in ('source_files', 'data_files', 'folds_sha256')):
            raise ValueError('Completed summary provenance differs')
        if digest(folder / 'oof.npz') != summary['oof_sha256']:
            raise ValueError('Completed OOF archive changed')
        return summary
    fold_numbers = np.empty(len(labels), dtype=np.int64)
    for split in folds:
        fold_numbers[split['holdout']] = split['fold']
    contents = {'scores': all_scores, 'raw_scores': all_raw, 'groups': groups, 'labels': labels, 'fold': fold_numbers}
    oof_path = folder / 'oof.npz'
    if oof_path.exists():
        with np.load(oof_path, allow_pickle=False) as archive:
            if set(archive.files) != set(contents) or any(not np.array_equal(archive[key], value) for key, value in contents.items()):
                raise ValueError('Incomplete prior OOF archive differs')
    else:
        temporary = folder / 'oof.pending.npz'
        np.savez(temporary, **contents)
        temporary.replace(oof_path)
    summary = {'experiment_id': experiment, 'status': 'complete', 'config': config, 'oof_decoder': effective, 'folds': reports,
               'provenance': prov,
               'oof_sha256': digest(oof_path), 'wall_seconds_this_invocation': time.perf_counter() - started,
               'training_seconds': sum(r['fit']['seconds'] + (r['inner_stopping']['seconds'] if r['inner_stopping'] else 0) for r in reports),
               **metrics(labels, all_scores, groups, effective)}
    for key in ('accuracy', 'hard_accuracy', 'single_positive_accuracy'):
        values = [report[key] for report in reports]
        summary[key + '_fold_mean'] = float(np.mean(values))
        summary[key + '_fold_sd'] = float(np.std(values, ddof=1))
    write_json(summary_path, summary, immutable=True)
    event({'experiment_id': experiment, 'status': 'cv_completed', 'accuracy': summary['accuracy'],
           'hard_accuracy': summary['hard_accuracy'], 'training_seconds': summary['training_seconds']})
    return summary


def fit_final(config, output):
    install_training_guard()
    config = normalize_config(config)
    check_stage_permission(config['experiment_id'])
    folder = OUT / config['experiment_id']
    summary = json.loads((folder / 'summary.json').read_text(encoding='utf-8'))
    registration = json.loads((folder / 'registration.json').read_text(encoding='utf-8'))
    prov = provenance()
    if summary['config'] != config or any(registration[key] != prov[key] for key in ('source_files', 'data_files')):
        raise ValueError('Final fit differs from completed CV source/data/config')
    features, labels, groups, _ = data_and_folds()
    report_path = folder / 'final_fit.json'
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding='utf-8'))
        for name, sha in report['asset_sha256'].items():
            if digest(Path(output) / name) != sha:
                raise ValueError('Completed final bundle changed')
        return report
    if config['family'] == 'linear':
        state, preprocess, info = fit_linear(features, labels, groups, config)
    else:
        epochs = int(math.ceil(np.median([fold['inner_stopping']['best_epoch'] for fold in summary['folds']])))
        context = {'source_files': prov['source_files'], 'data_files': prov['data_files'],
                   'folds_sha256': registration['folds_sha256'], 'phase': 'final-all-train'}
        state, preprocess, info = fit_neural(features, labels, groups, np.arange(len(labels)), None, config, config['seed'],
            STATE_ROOT / config['experiment_id'] / 'final', folder / 'final-epochs.json', context, epochs=epochs)
    output = Path(output).resolve()
    if not output.is_relative_to(ASSETS.resolve()):
        raise ValueError('Final V3 bundle must stay in artifacts/research_v3')
    _save_or_verify_bundle(output, config, state, preprocess)
    resource = audit(output)
    report = {'config': config, 'checkpoint': output.relative_to(ROOT).as_posix(), 'info': info,
              'parameters': sum(value.size for value in state.values()), 'provenance': prov,
              'asset_sha256': {item['path']: item['sha256'] for item in resource['files']}, 'resource': resource}
    write_json(report_path, report, immutable=True)
    event({'experiment_id': config['experiment_id'], 'status': 'final_train_complete', 'seconds': info['seconds']})
    return report
