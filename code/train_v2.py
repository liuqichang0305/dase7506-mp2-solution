"""Explicit train-only v2 entry point; official validation/test are unavailable."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS', 'OMP_THREAD_LIMIT'):
    os.environ[key] = '1'

import argparse
import json
from pathlib import Path
from mp2.training_v2 import run_cv, fit_final

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--fold', type=int, choices=(0, 1, 2))
    parser.add_argument('--final-output', type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding='utf-8'))
    lock = Path(__file__).resolve().parents[1] / 'results/research_v2/FINALISTS_LOCK.json'
    if not lock.exists():
        lock = lock.with_name('FINALISTS_SELECTION.json')
    if lock.exists():
        locked = json.loads(lock.read_text(encoding='utf-8'))
        if not args.final_output:
            parser.error('Research selection is locked: no further CV/search after finalist selection')
        if config['experiment_id'] not in locked['allowed_final_fit_experiments']:
            parser.error('This configuration is not a locked finalist component')
    if args.final_output:
        if args.fold is not None:
            parser.error('--fold and --final-output are mutually exclusive')
        fit_final(config, args.final_output)
    else:
        run_cv(config, args.fold)
