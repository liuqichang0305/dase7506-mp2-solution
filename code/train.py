"""CPU training/search entry point. Never fits on validation or reads test."""
import os
for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[variable] = '1'

import argparse
import json
from pathlib import Path
import traceback

from mp2.training import log_event, run_search, train_one


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--search', action='store_true', help='Run the prespecified train-only screen and validation comparison')
    mode.add_argument('--config', type=Path, help='Configuration to fit on official train only')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--calibrate', action='store_true', help='Fit selected-event calibration on train-only three-fold OOF')
    args = parser.parse_args()
    if args.search:
        run_search()
    else:
        if args.output is None:
            parser.error('--config requires --output')
        config = json.loads(args.config.read_text(encoding='utf-8'))
        train_one(config, args.output, args.calibrate)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        log_event({'phase': 'failed', 'error': str(exc), 'traceback': traceback.format_exc()})
        raise
