"""Finite sequential experiment batches, recording commands and failures."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results/research_v2'
CONFIGS = ROOT / 'code/configs/research_v2'


def configurations(round_name):
    linear = json.loads((CONFIGS / 'linear_original.json').read_text())
    neural = json.loads((CONFIGS / 'mlp_group_s0.json').read_text())
    if round_name == 'linear':
        configs = []
        for alpha in (0, .5, 1):
            configs.append({**linear, 'experiment_id': f'linear_group_a{str(alpha).replace(".", "p")}',
                'objective': 'group', 'alpha': alpha,
                'decoder': {'mode': 'group_lse', 'alpha': alpha, 'tau': 1, 'mask': True}})
        configs.append({**configs[0], 'experiment_id': 'linear_group_hard', 'reduction': 'hard', 'scale_regularization': True})
        return configs
    if round_name == 'neural':
        return [neural,
            {**neural, 'experiment_id': 'resmlp_group_s0', 'family': 'resmlp', 'blocks': 1},
            {**neural, 'experiment_id': 'deepsets_group_s0', 'family': 'deepsets'},
            {**neural, 'experiment_id': 'attention_group_s0', 'family': 'attention', 'heads': 4},
            {**neural, 'experiment_id': 'mlp_listwise_s0', 'objective': 'listwise'},
            {**neural, 'experiment_id': 'mlp_capacity_s0', 'width': 109}]
    raise ValueError(round_name)


def run(configs, batch_id, budget):
    env = os.environ.copy()
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS',
                'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS', 'OMP_THREAD_LIMIT'):
        env[key] = '1'
    env['PYTHONUTF8'] = '1'
    start = time.perf_counter()
    batch = RESULTS / (batch_id + '-batch.jsonl')
    for config in configs:
        remaining = budget - (time.perf_counter() - start)
        if remaining <= 0:
            raise TimeoutError('Registered batch wall budget exhausted; completed fits preserved')
        path = CONFIGS / (config['experiment_id'] + '.json')
        if path.exists() and json.loads(path.read_text()) != config:
            raise ValueError('Existing experiment configuration differs')
        path.write_text(json.dumps(config, indent=2) + '\n', encoding='utf-8')
        command = [sys.executable, 'code/train_v2.py', '--config', str(path)]
        attempt = 1
        log = RESULTS / f'{config["experiment_id"]}-process-{attempt}.log'
        while log.exists():
            attempt += 1
            log = RESULTS / f'{config["experiment_id"]}-process-{attempt}.log'
        print('RUN ' + config['experiment_id'], flush=True)
        tick = time.perf_counter()
        with log.open('w', encoding='utf-8') as stream:
            try:
                result = subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                        timeout=remaining)
                returncode = result.returncode
            except subprocess.TimeoutExpired:
                returncode = 'timeout'
        record = {'experiment_id': config['experiment_id'], 'command': command, 'returncode': returncode,
                  'wall_seconds': time.perf_counter() - tick, 'log': log.relative_to(ROOT).as_posix()}
        with batch.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record) + '\n')
        print(json.dumps(record), flush=True)
        if returncode != 0:
            raise RuntimeError(f'Experiment stopped; see {log}')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--round', choices=('linear', 'neural'))
    p.add_argument('--config-list', type=Path)
    p.add_argument('--batch-id')
    p.add_argument('--budget-seconds', type=float, required=True)
    args = p.parse_args()
    if bool(args.round) == bool(args.config_list):
        p.error('Choose --round or --config-list')
    configs = configurations(args.round) if args.round else json.loads(args.config_list.read_text())
    run(configs, args.batch_id or args.round, args.budget_seconds)
