"""Check a frozen v2 bundle's full-validation Torch/NumPy numerical parity.

This tool opens only validation candidate inputs, never gold references. It
reports numerical errors and choice equality, not accuracy or model selection
statistics. It must be run only after the finalist and its decoder are fixed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

THREAD_VARIABLES = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                    'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS',
                    'VECLIB_MAXIMUM_THREADS', 'OMP_THREAD_LIMIT')
for variable in THREAD_VARIABLES:
    os.environ[variable] = '1'
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from common import DATA, load_inputs
from mp2.features import build_features
from mp2.models_v2 import build_model, load_bundle, score_with_bundle
from mp2.selection_v2 import answer_groups, select_candidates, selection_details


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _install_data_guard():
    data = DATA.resolve()
    allowed = {data / 'manifest.json',
               data / 'candidates' / 'gsm8k_validation.jsonl',
               data / 'candidates' / 'gsm8k_validation.npz'}
    rejected = []

    def guard(event, arguments):
        if event != 'open' or isinstance(arguments[0], int):
            return
        path = Path(os.fsdecode(arguments[0])).resolve()
        if path.is_relative_to(data) and path not in allowed:
            rejected.append(str(path))
            raise PermissionError('Export parity permits validation candidate inputs only; gold and other splits are prohibited')

    sys.addaudithook(guard)
    return rejected


def check(checkpoint):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    rejected_reads = _install_data_guard()
    started = time.perf_counter()
    bundle = load_bundle(checkpoint)
    config, arrays = bundle
    bank, inputs = load_inputs('validation')
    groups = answer_groups(bank)
    features = build_features(bank, inputs['embeddings'], 'relation')
    linear = config['architecture']['family'] == 'linear'
    dtype = np.float64 if linear else np.float32
    model = build_model(config['architecture'])
    if linear:
        model = model.double()
    state = {name: torch.from_numpy(value.copy())
             for name, value in arrays.items() if name not in ('mean', 'scale')}
    model.load_state_dict(state, strict=True)
    model.eval()
    torch_scores = np.empty((len(bank), 4), dtype=dtype)
    tick = time.perf_counter()
    with torch.no_grad():
        for start in range(0, len(bank), 256):
            stop = min(start + 256, len(bank))
            x = ((features[start:stop] - arrays['mean']) / arrays['scale']).astype(dtype)
            torch_scores[start:stop] = model(torch.from_numpy(x)).cpu().numpy()
    torch_forward_seconds = time.perf_counter() - tick
    tick = time.perf_counter()
    numpy_scores = score_with_bundle(bank, inputs['embeddings'], bundle)
    numpy_features_and_forward_seconds = time.perf_counter() - tick
    error = np.abs(torch_scores.astype(np.float64) - numpy_scores.astype(np.float64))
    absolute_tolerance = 3e-6 if not linear else 1e-10
    relative_tolerance = 3e-5 if not linear else 1e-10
    allowed_error = absolute_tolerance + relative_tolerance * np.abs(torch_scores)
    torch_choices = np.asarray(select_candidates(torch_scores, groups, **config['decoder']))
    numpy_choices = np.asarray(select_candidates(numpy_scores, groups, **config['decoder']))
    differences = np.flatnonzero(torch_choices != numpy_choices)
    details = selection_details(numpy_scores, groups, **config['decoder'])
    margin = details['group_probability_margin']
    result = {
        'status': 'passed' if np.all(error <= allowed_error) and len(differences) == 0 else 'failed',
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'purpose': 'Frozen full-validation numerical export check; no accuracy, references, or model selection',
        'split': 'validation', 'questions': len(bank), 'candidate_scores': int(error.size),
        'reference_data_guard': 'Only data/manifest.json and the two validation candidate files can be opened',
        'rejected_data_reads': rejected_reads,
        'checkpoint': str(Path(checkpoint).resolve()),
        'assets': {name: sha256(Path(checkpoint) / name) for name in ('config.json', 'weights.npz')},
        'architecture': config['architecture'], 'decoder': config['decoder'],
        'torch_version': torch.__version__, 'numpy_version': np.__version__,
        'dtype': str(np.dtype(dtype)), 'device': 'cpu',
        'threads': {'torch_intraop': torch.get_num_threads(), 'torch_interop': torch.get_num_interop_threads(),
                    'environment': {variable: os.environ[variable] for variable in THREAD_VARIABLES}},
        'absolute_tolerance': absolute_tolerance, 'relative_tolerance': relative_tolerance,
        'maximum_absolute_error': float(error.max(initial=0)),
        'mean_absolute_error': float(error.mean()) if error.size else 0.0,
        'scores_outside_tolerance': int((error > allowed_error).sum()),
        'choice_agreement': {'equal': int((torch_choices == numpy_choices).sum()), 'n': len(bank)},
        'differing_row_indices': differences.tolist(),
        'differing_torch_choices': torch_choices[differences].tolist(),
        'differing_numpy_choices': numpy_choices[differences].tolist(),
        'raw_argmax_difference_count': int((torch_scores.argmax(axis=1) != numpy_scores.argmax(axis=1)).sum()),
        'selected_group_probability_margin_minimum': float(margin.min(initial=1)),
        'selected_group_probability_margin_below_1e_5_count': int((np.abs(margin) < 1e-5).sum()),
        'torch_forward_seconds': torch_forward_seconds,
        'numpy_features_and_forward_seconds': numpy_features_and_forward_seconds,
        'audit_total_seconds': time.perf_counter() - started,
        'timing_note': 'Diagnostic in-process timings only; not the official fresh-process inference-plus-scoring budget',
    }
    try:
        from threadpoolctl import threadpool_info
        result['threadpool_info'] = threadpool_info()
    except ImportError:
        result['threadpool_info'] = 'threadpoolctl is not installed; environment and Torch settings recorded above'
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Refusing to overwrite an export audit')
    if args.output.resolve().is_relative_to(args.checkpoint.resolve()):
        raise ValueError('Write parity evidence outside the immutable checkpoint directory')
    result = check(args.checkpoint)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({key: result[key] for key in ('status', 'questions', 'maximum_absolute_error', 'choice_agreement')}))
    raise SystemExit(0 if result['status'] == 'passed' else 1)


if __name__ == '__main__':
    main()
