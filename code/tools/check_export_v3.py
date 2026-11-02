"""Input-only Torch/NumPy audit for frozen V3/V2 models and fixed manifests.

Development uses synthetic or train candidate inputs. Validation is allowed only
after finalist freezing, with --allow-frozen-validation. No labels, accuracy,
test inputs or model-selection statistics are read or produced by this tool.
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
                    'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'OMP_THREAD_LIMIT')
for variable in THREAD_VARIABLES:
    os.environ[variable] = '1'
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from common import load_inputs
from mp2.features import build_features
from mp2 import models_v2
from mp2.research_v3 import models
from mp2.research_v3.preprocess import preprocess_from_bundle, transform_features
from mp2.selection_v2 import answer_groups
from solution_v3 import checkpoint_asset_paths, load_inference_checkpoint, score_members
from check_contract_v3 import install_input_guard, reference_decision, synthetic_inputs


def check(checkpoint, input_source='synthetic'):
    if input_source not in ('synthetic', 'train', 'validation'):
        raise ValueError('This audit never permits test inputs')
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    rejected = install_input_guard(input_source)
    started = time.perf_counter()
    loaded = load_inference_checkpoint(checkpoint)
    bank, inputs = synthetic_inputs() if input_source == 'synthetic' else load_inputs(input_source)
    groups = answer_groups(bank)
    features = build_features(bank, inputs['embeddings'], 'relation')
    torch_scores, reports = [], []
    for version, bundle in loaded['members']:
        config, arrays = bundle
        linear = config['architecture']['family'] == 'linear'
        dtype = np.float64 if linear else np.float32
        module = models if version == 'v3' else models_v2
        model = module.build_model(config['architecture'])
        if linear:
            model = model.double()
        reserved = models.RESERVED if version == 'v3' else ('mean', 'scale')
        state = {name: torch.from_numpy(value.copy()) for name, value in arrays.items() if name not in reserved}
        model.load_state_dict(state, strict=True)
        model.eval()
        shape = (len(bank), config['architecture']['k'], 4) if config['architecture']['family'] == 'tabm' else (len(bank), 4)
        scores = np.empty(shape, dtype=dtype)
        tick = time.perf_counter()
        with torch.no_grad():
            for start in range(0, len(bank), 128):
                stop = min(start + 128, len(bank))
                if version == 'v3':
                    x = transform_features(features[start:stop], preprocess_from_bundle(config, arrays), dtype=dtype)
                else:
                    x = ((features[start:stop] - arrays['mean']) / arrays['scale']).astype(dtype)
                scores[start:stop] = model(torch.from_numpy(x)).cpu().numpy()
        torch_scores.append(scores)
        reports.append({'version': version, 'architecture': config['architecture'], 'decoder': config['decoder'],
                        'dtype': str(np.dtype(dtype)), 'score_shape': list(scores.shape),
                        'torch_forward_seconds': time.perf_counter() - tick})
    tick = time.perf_counter()
    numpy_scores = score_members(bank, inputs, checkpoint)
    numpy_seconds = time.perf_counter() - tick
    all_scores_passed = True
    for expected, actual, report in zip(torch_scores, numpy_scores, reports):
        linear = report['architecture']['family'] == 'linear'
        atol, rtol = (1e-10, 1e-10) if linear else (3e-6, 3e-5)
        error = np.abs(expected.astype(np.float64) - actual.astype(np.float64))
        allowed = atol + rtol * np.abs(expected)
        passed = bool(np.all(error <= allowed))
        all_scores_passed &= passed
        report.update({'status': 'passed' if passed else 'failed', 'absolute_tolerance': atol,
                       'relative_tolerance': rtol, 'maximum_absolute_error': float(error.max(initial=0)),
                       'mean_absolute_error': float(error.mean()) if error.size else 0.,
                       'scores_outside_tolerance': int((error > allowed).sum()),
                       'raw_argmax_difference_count': int((expected.argmax(axis=-1) != actual.argmax(axis=-1)).sum())})
    configs = [item[1][0] for item in loaded['members']]
    expected_choice, expected_prob, _ = reference_decision(torch_scores, groups, configs, loaded['kind'])
    actual_choice, actual_prob, _ = reference_decision(numpy_scores, groups, configs, loaded['kind'])
    expected_choice, actual_choice = np.asarray(expected_choice), np.asarray(actual_choice)
    mismatch = np.flatnonzero(expected_choice != actual_choice)
    probability_error = np.abs(expected_prob - actual_prob)
    probability_passed = bool(np.all(probability_error <= 3e-6 + 3e-5 * np.abs(expected_prob)))
    probability_order = np.sort(actual_prob, axis=1)
    margin = probability_order[:, -1] - probability_order[:, -2]
    paths = checkpoint_asset_paths(checkpoint)
    result = {'status': 'passed' if all_scores_passed and probability_passed and len(mismatch) == 0 else 'failed',
        'timestamp_utc': datetime.now(timezone.utc).isoformat(), 'input_source': input_source,
        'purpose': 'Full selected-input numerical export check only; no references, accuracy or model selection',
        'questions': len(bank), 'kind': loaded['kind'], 'members': reports,
        'candidate_scores': int(sum(score.size for score in torch_scores)),
        'reference_files_read': False, 'rejected_input_reads': rejected,
        'checkpoint': str(Path(checkpoint).resolve()),
        'assets_sha256': {str(path.relative_to(Path(checkpoint).resolve())): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
        'torch_version': torch.__version__, 'numpy_version': np.__version__, 'device': 'cpu',
        'threads': {'torch_intraop': torch.get_num_threads(), 'torch_interop': torch.get_num_interop_threads(),
                    'environment': {variable: os.environ[variable] for variable in THREAD_VARIABLES}},
        'maximum_absolute_error': max(report['maximum_absolute_error'] for report in reports),
        'maximum_group_probability_error': float(probability_error.max(initial=0)),
        'group_probabilities_within_tolerance': probability_passed,
        'choice_agreement': {'equal': int((expected_choice == actual_choice).sum()), 'n': len(bank)},
        'differing_row_indices': mismatch.tolist(),
        'differing_torch_choices': expected_choice[mismatch].tolist(),
        'differing_numpy_choices': actual_choice[mismatch].tolist(),
        'top_group_probability_margin_minimum': float(margin.min(initial=1)),
        'top_group_probability_margin_below_1e_5_count': int((margin < 1e-5).sum()),
        'numpy_features_and_forward_seconds': numpy_seconds,
        'audit_total_seconds': time.perf_counter() - started,
        'timing_note': 'In-process diagnostics only; not official fresh-process inference-plus-scoring timing'}
    try:
        from threadpoolctl import threadpool_info
        result['threadpool_info'] = threadpool_info()
    except ImportError:
        result['threadpool_info'] = 'Unavailable; Torch and environment thread limits recorded.'
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--input-source', choices=('synthetic', 'train', 'validation'), default='synthetic')
    parser.add_argument('--allow-frozen-validation', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.input_source == 'validation' and not args.allow_frozen_validation:
        parser.error('Validation requires a frozen finalist and --allow-frozen-validation')
    if args.output.exists():
        raise FileExistsError('Refusing to overwrite export evidence')
    if args.output.resolve().is_relative_to(args.checkpoint.resolve()):
        raise ValueError('Write export evidence outside the immutable checkpoint')
    result = check(args.checkpoint, args.input_source)
    result['command'] = [sys.executable, *sys.argv]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({key: result[key] for key in ('status', 'questions', 'maximum_absolute_error', 'maximum_group_probability_error', 'choice_agreement')}))
    raise SystemExit(0 if result['status'] == 'passed' else 1)


if __name__ == '__main__':
    main()
