"""First train-only v3 timing pilot, using preserved v2 numerical primitives."""
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'OMP_THREAD_LIMIT'):
    os.environ[name] = '1'
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code'))
from mp2 import training_v2 as old
from mp2.models_v2 import save_bundle
OUT = ROOT / 'results/research_v3'
old.OUT = OUT


def record(value):
    value = {'utc': datetime.now(timezone.utc).isoformat(), 'pid': os.getpid(), **value}
    with (OUT / 'experiment_registry.jsonl').open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(value) + '\n')
    print(json.dumps(value), flush=True)


if __name__ == '__main__':
    run = OUT / 'pilot_mlp_adamw_v3'
    if (run / 'result.json').exists():
        raise FileExistsError('Completed pilot is preserved; do not repeat')
    run.mkdir(parents=True, exist_ok=True)
    cfg = {'experiment_id': 'pilot_mlp_adamw_v3', 'family': 'mlp', 'width': 128, 'hidden': 64,
           'dropout': .3, 'objective': 'group', 'alpha': 0, 'reduction': 'full', 'lr': .0003,
           'weight_decay': .001, 'batch_size': 128, 'max_epochs': 40, 'patience': 8,
           'seed': 3036900957, 'decoder': {'mode': 'group_lse', 'alpha': 0, 'tau': 1, 'mask': True}}
    old.write_json(ROOT / 'code/configs/research_v3/pilot_mlp_adamw_v3.json', cfg)
    start = time.perf_counter()
    record({'experiment_id': cfg['experiment_id'], 'status': 'running', 'config': cfg,
            'config_sha256': old.config_hash(cfg), 'source': old.provenance(),
            'pilot_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    try:
        bank, _, _ = old.load_development('train')
        counts = {}
        for i, row in enumerate(bank):
            key = ' '.join(row['question'].split())
            counts.setdefault(key, []).append(i)
        duplicates = [rows for rows in counts.values() if len(rows) > 1]
        old.write_json(OUT / 'duplicate_question_audit.json', {'questions': len(bank), 'duplicate_groups': duplicates,
                        'normalization': 'whitespace only; exact question strings otherwise', 'test_access': False})
        if duplicates:
            raise ValueError('Duplicate questions require grouped fold reconstruction before pilot')
        x, y, groups, folds = old.data_and_folds()
        split = folds[0]
        _, _, _, inner = old.fit_neural(x, y, groups, np.array(split['inner_fit']), np.array(split['inner_stop']),
                cfg, cfg['seed'], run / 'inner_epochs.json')
        state, mean, scale, refit = old.fit_neural(x, y, groups, np.array(split['outer_fit']), None,
                cfg, cfg['seed'], run / 'refit_epochs.json', inner['best_epoch'])
        held = np.array(split['holdout'])
        scores = old.predict_features(x[held], cfg, state, mean, scale)
        checkpoint = ROOT / 'artifacts/research_v3/pilot_mlp_adamw_v3/fold0'
        save_bundle(checkpoint, cfg, state, mean, scale, cfg['decoder'])
        np.save(run / 'fold0_scores.npy', scores)
        result = {'status': 'completed', 'experiment_id': cfg['experiment_id'], 'fold': 0,
                  'inner': inner, 'fit': refit, 'wall_seconds': time.perf_counter()-start,
                  'schema_note': 'Timing pilot deliberately reuses immutable v2 training/export primitives; new v3 main-table runs use their registered trainer.',
                  **old.metrics(y[held], scores, groups[held], cfg['decoder'])}
        old.write_json(run / 'result.json', result)
        record(result)
    except Exception as exc:
        record({'experiment_id': cfg['experiment_id'], 'status': 'failed', 'exception': repr(exc), 'wall_seconds': time.perf_counter()-start})
        raise
