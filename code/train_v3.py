"""Explicit train-only V3 entry; epoch checkpoints resume automatically."""
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS', 'OMP_THREAD_LIMIT'):
    os.environ[name] = '1'

import argparse
import json
from pathlib import Path
from mp2.research_v3.training import run_cv, fit_final, install_training_guard, training_lock


if __name__ == '__main__':
    install_training_guard()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--fold', type=int, choices=(0, 1, 2))
    parser.add_argument('--final-output', type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding='utf-8'))
    root = Path(__file__).resolve().parents[1]
    locks = [root / 'results/research_v3/FINALISTS_LOCK.json', root / 'results/research_v3/FINALISTS_SELECTION.json']
    lock = next((path for path in locks if path.exists()), None)
    if (root / 'submission/v3/FREEZE.json').exists():
        parser.error('V3 is frozen: use prediction-only reproduction; new fitting/search is disabled')
    if (root / 'results/research_v3/SEARCH_CLOSED.json').exists() and not args.final_output:
        parser.error('V3 search is closed; further CV is disabled')
    if args.final_output and lock is None:
        parser.error('Final fitting requires an explicit finalist-component lock')
    if lock is not None:
        locked = json.loads(lock.read_text(encoding='utf-8'))
        if not args.final_output:
            parser.error('Finalists locked: no further model search')
        if config.get('experiment_id', config.get('id')) not in locked['allowed_final_fit_experiments']:
            parser.error('This final fit is not a locked finalist component')
    with training_lock():
        if args.final_output:
            if args.fold is not None:
                parser.error('--fold and --final-output cannot be combined')
            fit_final(config, args.final_output)
        else:
            run_cv(config, args.fold)
