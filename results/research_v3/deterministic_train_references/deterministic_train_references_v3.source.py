"""Once-only fixed-candidate/valid-answer-majority references on saved train labels.

No fitting, model hyperparameter configuration, raw reference/embedding read,
holdout access or new official scoring. Shared folds are descriptive partitions
for these label-independent deterministic rules, not separately trained models.
"""
from __future__ import annotations
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code'))
sys.path.insert(0, str(ROOT / 'code/tools'))
import diagnose_oof_v3 as diagnostic
from mp2.features import majority_choices
from mp2.selection_v2 import answer_groups
from analyze_oof_v2 import measure

SOURCE_METHOD = 'v3:v3_mlp_adamw_s0'
OUTPUT = ROOT / 'results/research_v3/deterministic_train_references'


def majority_from_groups(groups):
    groups = np.asarray(groups)
    if groups.ndim != 2 or groups.shape[1] != 4 or not np.issubdtype(groups.dtype, np.integer):
        raise ValueError('Majority requires integer answer groups [N,4]')
    if np.any((groups < -1) | (groups > 3)):
        raise ValueError('Invalid parsed-answer group index')
    votes = ((groups[:, :, None] == groups[:, None, :]) & (groups[:, :, None] >= 0)).sum(2)
    return votes.argmax(1).astype(np.int64)


def main():
    if len(sys.argv) != 1:
        raise ValueError('Fixed train references accept no data/model/seed overrides')
    started = time.perf_counter()
    denied = diagnostic.install_guard()
    if OUTPUT.exists():
        raise FileExistsError('Preserve the completed or partial deterministic references')
    OUTPUT.mkdir(parents=True, exist_ok=False)
    try:
        manifest = json.loads((diagnostic.DATA / 'manifest.json').read_text(encoding='utf-8'))
        base = diagnostic.load_method(SOURCE_METHOD, manifest)
        bank = diagnostic.load_bank('train')
        labels, groups, fold = (base[key] for key in ('labels', 'groups', 'fold'))
        if labels.shape != (4755, 4) or len(bank) != 4755:
            raise ValueError('Complete official train coverage is required')
        fold_path = ROOT / 'results/research_v3/common_folds.json'
        fold_record = json.loads(fold_path.read_text(encoding='utf-8'))
        ids = [row['id'] for row in bank]
        ids_hash = hashlib.sha256(json.dumps(ids).encode()).hexdigest()
        if fold_record['train_ids_sha256'] != ids_hash or not np.array_equal(answer_groups(bank), groups):
            raise ValueError('Train IDs/order/official parsed groups differ from saved labels')
        fixed = np.zeros(len(bank), dtype=np.int64)
        majority = np.asarray(majority_choices(bank), dtype=np.int64)
        group_reference = majority_from_groups(groups)
        if not np.array_equal(majority, group_reference):
            raise ValueError('Existing majority and independent group counts disagree')
        choices = {'fixed0': fixed, 'valid_answer_majority': majority}
        metrics = {name: measure(labels[np.arange(len(bank)), selected].astype(bool), labels, groups, fold)
                   for name, selected in choices.items()}
        sources = set(base['registration']['source_files']) | {
            'code/tools/deterministic_train_references_v3.py', 'code/tools/diagnose_oof_v3.py',
            'code/tools/analyze_oof_v2.py', 'code/tools/audit_assets.py'}
        inputs = {'code/data/manifest.json', 'code/data/candidates/gsm8k_train.jsonl',
                  'results/research_v3/common_folds.json'}
        inputs.update((base['folder'] / name).relative_to(ROOT).as_posix()
                      for name in ('registration.json', 'summary.json', 'oof.npz'))
        for report in base['summary']['folds']:
            inputs.update((base['folder'] / name).relative_to(ROOT).as_posix() for name in report['output_sha256'])
        source_hashes = {name: diagnostic.digest(ROOT / name) for name in sorted(sources)}
        input_hashes = {name: diagnostic.digest(ROOT / name) for name in sorted(inputs)}
        selection_file = OUTPUT / 'selections.npz'
        with selection_file.open('xb') as stream:
            np.savez(stream, ids=np.asarray(ids), fold=fold, groups=groups,
                     fixed0=fixed, valid_answer_majority=majority,
                     fixed0_correct=labels[np.arange(len(bank)), fixed].astype(bool),
                     valid_answer_majority_correct=labels[np.arange(len(bank)), majority].astype(bool))
        with (OUTPUT / 'deterministic_train_references_v3.source.py').open('xb') as stream:
            stream.write(Path(__file__).read_bytes())
        if any(diagnostic.digest(ROOT / name) != expected for name, expected in {**source_hashes, **input_hashes}.items()):
            raise ValueError('Source or train-only input changed during computation')
        report = {'schema': 'mp2-v3-deterministic-train-references-v1', 'status': 'complete',
            'created_at_utc': datetime.now(timezone.utc).isoformat(), 'split': 'train', 'n': 4755,
            'source_label_archive': SOURCE_METHOD, 'train_ids_sha256': ids_hash,
            'folds_sha256': diagnostic.digest(fold_path), 'source_sha256': source_hashes,
            'input_sha256': input_hashes, 'official_training_data_sha256': base['registration']['data_files'],
            'selections': {'path': selection_file.relative_to(ROOT).as_posix(), 'sha256': diagnostic.digest(selection_file)},
            'rules': {'fixed0': 'Always original candidate index 0, including when unparseable; an unconditional reference.',
                'valid_answer_majority': 'Use existing mp2.features.majority_choices: None has zero votes; count equal official parsed answers; tie chooses the earliest original candidate index; all-None returns index 0.'},
            'majority_independent_group_count_choices_equal': True,
            'all_unparseable_questions': int((groups < 0).all(1).sum()),
            'results': metrics, 'new_fitting_runs': 0, 'model_hyperparameter_configurations_created': 0,
            'raw_reference_or_embeddings_read': False, 'validation_test_ranking_read': False,
            'guard_denied_accesses': denied, 'numpy_version': np.__version__,
            'computation_wall_seconds': time.perf_counter() - started,
            'cost_scope': 'This process internal elapsed includes guard, hash/source/OOF verification, train text parsing, rules and output; not formal fresh inference/scorer latency. Training continued independently; no resource certification is claimed.',
            'report_integration_note': 'Add these two label-independent reference rows to the train common-fold mechanism table. Report pooled full/hard/single counts and fold mean/sample SD from results. They have zero fitted parameters and no seed tuning; partitioning by common held-out folds is descriptive, not three trained fits. Do not mix their computation cost with fitted model cost or send them to test.',
            'command': '.venv/Scripts/python.exe code/tools/deterministic_train_references_v3.py'}
        diagnostic.write_new(OUTPUT / 'summary.json', report)
        output_hashes = {path.relative_to(ROOT).as_posix(): diagnostic.digest(path) for path in OUTPUT.iterdir() if path.is_file()}
        diagnostic.write_new(OUTPUT / 'COMPLETION.json', {'schema': 'mp2-v3-deterministic-train-references-complete-v1',
            'status': 'complete', 'created_at_utc': datetime.now(timezone.utc).isoformat(),
            'source_sha256': source_hashes, 'input_sha256': input_hashes, 'output_sha256': output_hashes,
            'new_fitting_runs': 0, 'validation_test_ranking_read': False})
        print(json.dumps({'status': 'complete', 'output': OUTPUT.relative_to(ROOT).as_posix(),
                          'wall_seconds': report['computation_wall_seconds'], 'results': metrics}, indent=2))
    except BaseException as error:
        diagnostic.write_new(OUTPUT / 'INCOMPLETE.json', {'status': 'incomplete',
            'created_at_utc': datetime.now(timezone.utc).isoformat(), 'error_type': type(error).__name__,
            'error': str(error), 'guard_denied_accesses': denied})
        raise


if __name__ == '__main__':
    main()
