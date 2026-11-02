"""Synthetic V3 entry/export/isolated-contract audit. Never opens task data."""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS',
             'BLIS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'OMP_THREAD_LIMIT'):
    os.environ[name] = '1'
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code'))
sys.path.insert(0, str(ROOT / 'code' / 'tools'))

import numpy as np
import torch
from mp2 import models_v2
from mp2.features import build_features
from mp2.research_v3.models import build_model, export_state, save_bundle
from mp2.research_v3.preprocess import fit_preprocessor
from mp2.research_v3.selection import select_raw
from mp2.selection_v2 import answer_groups
from solution_v3 import (ENSEMBLE_SCHEMA, load_inference_checkpoint, predict,
                         prediction_details, score_members)
from check_contract_v3 import reference_decision, synthetic_inputs


def write_manifest(folder, members):
    folder.mkdir()
    for i, member in enumerate(members):
        shutil.copytree(member, folder / f'member{i}')
    config = {'schema': ENSEMBLE_SCHEMA, 'aggregation': 'equal_group_probability',
              'representative': 'first_model_first_member', 'members': [f'member{i}' for i in range(len(members))]}
    (folder / 'config.json').write_text(json.dumps(config, indent=2) + '\n', encoding='utf-8')


def run(output):
    if output.exists():
        raise FileExistsError('Refusing to overwrite synthetic inference evidence')
    output.mkdir(parents=True)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    started = time.perf_counter()
    bank, arrays = synthetic_inputs()
    features = build_features(bank, arrays['embeddings'], 'relation')
    decoder = {'mode': 'group_lse', 'alpha': 0., 'tau': 1., 'mask': True}
    reports = []
    with tempfile.TemporaryDirectory(prefix='mp2-v3-synthetic-assets-') as temporary:
        base = Path(temporary)
        paths = {}
        for family in ('mlp', 'tabm', 'realmlp'):
            torch.manual_seed(3036900957)
            config = {'family': family}
            model = build_model(config).eval()
            if family == 'realmlp':
                with torch.no_grad():
                    model.head.weight.normal_(0, .4)
                    model.head.bias.fill_(.2)
            prep = fit_preprocessor(features[:53], 'robust_stats' if family == 'realmlp' else 'standard')
            paths[family] = base / family
            save_bundle(paths[family], config, export_state(model), prep['mean'], prep['scale'], decoder, preprocess=prep)
        paths['v2_linear'] = base / 'v2_linear'
        rng = np.random.default_rng(219)
        models_v2.save_bundle(paths['v2_linear'], {'family': 'linear'},
            {'weight': rng.normal(size=2064), 'bias': np.array(.0123456789012345)},
            np.zeros(2064), np.ones(2064), decoder)
        paths['ensemble2'] = base / 'ensemble2'
        paths['ensemble3'] = base / 'ensemble3'
        write_manifest(paths['ensemble2'], [paths['tabm'], paths['realmlp']])
        write_manifest(paths['ensemble3'], [paths['mlp'], paths['tabm'], paths['v2_linear']])

        groups = answer_groups(bank)
        for name, path in paths.items():
            loaded = load_inference_checkpoint(path)
            raw = score_members(bank, arrays, path)
            configs = [member[1][0] for member in loaded['members']]
            expected, probability, _ = reference_decision(raw, groups, configs, loaded['kind'])
            details = prediction_details(bank, arrays, path)
            assert details['chosen'] == expected
            np.testing.assert_array_equal(details['group_probabilities'], probability)
            assert details['chosen'][0] == 0 and details['chosen_group_probability'][0] == 0
            assert details['chosen'][3] == 0, 'Identical candidate tie must choose smallest index'
            assert predict([], {'embeddings': arrays['embeddings'][:0]}, path) == []
            for tool in ('check_export_v3.py', 'check_contract_v3.py'):
                report_path = output / f'{name}-{tool.removesuffix(".py")}.json'
                command = [sys.executable, str(ROOT / 'code' / 'tools' / tool), '--checkpoint', str(path),
                           '--input-source', 'synthetic', '--output', str(report_path)]
                completed = subprocess.run(command, cwd=ROOT, env=os.environ.copy(),
                    capture_output=True, text=True, encoding='utf-8', errors='replace')
                (output / f'{name}-{tool.removesuffix(".py")}.log').write_text(
                    completed.stdout + completed.stderr, encoding='utf-8')
                assert completed.returncode == 0, f'{name}/{tool}: {completed.stdout} {completed.stderr}'
                record = json.loads(report_path.read_text(encoding='utf-8'))
                assert record['status'] == 'passed'
                reports.append({'fixture': name, 'tool': tool, 'status': record['status'],
                                'evidence': report_path.name})
                print(json.dumps(reports[-1]), flush=True)

        # Direct exact group/within-group tie: member zero prefers index 1 in
        # group zero, but averaged groups are exactly equal and group zero wins.
        tie_groups = np.array([[0, 0, 1, 1], [-1, -1, -1, -1]], dtype=np.int64)
        scores = np.array([[[0., 2., 0., 2.], [2., 0., 2., 0.]], [[0., 1., 2., 3.], [3., 2., 1., 0.]]])
        assert select_raw(scores, tie_groups, decoder) == [1, 0]
        invalid = base / 'invalid_manifest'
        invalid.mkdir()
        original = json.loads((paths['ensemble2'] / 'config.json').read_text())
        for change in ({'members': ['../tabm', '../realmlp']}, {'members': ['x', 'x']}, {'weights': [.8, .2]}):
            value = copy.deepcopy(original)
            value.update(change)
            (invalid / 'config.json').write_text(json.dumps(value))
            try:
                load_inference_checkpoint(invalid)
            except (ValueError, FileNotFoundError):
                pass
            else:
                raise AssertionError('Unsafe or non-equal manifest accepted')
    summary = {'status': 'passed', 'source': 'synthetic only; no official data files',
               'isolated_checks': reports, 'exact_ties': 'passed', 'manifest_rejections': 'passed',
               'seconds': time.perf_counter() - started,
               'temporary_assets': 'Deterministically generated synthetic weights removed with TemporaryDirectory; no deployable ensemble was selected.'}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'status': summary['status'], 'checks': len(reports), 'seconds': summary['seconds']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args().output)
