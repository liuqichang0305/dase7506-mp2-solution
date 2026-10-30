"""Prepare fixed extra seeds for one or two explicitly chosen completed recipes."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = 3036900957


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(ids, rationale):
    if len(ids) not in (1, 2) or len(set(ids)) != len(ids):
        raise ValueError('Choose one or two distinct completed base-seed recipes')
    if not rationale.strip():
        raise ValueError('An actual train-only selection rationale is required')
    if (ROOT / 'results/research_v3/SEARCH_CLOSED.json').exists():
        raise ValueError('Research already closed')
    queue = read(ROOT / 'results/research_v3/campaign_queue.json')
    jobs = {j['experiment_id']: j for j in queue['jobs']}
    evidence, configs = {}, []
    for name in ids:
        job = jobs[name]
        if job['status'] != 'completed' or job['config']['seed'] != BASE or not name.endswith('_s0'):
            raise ValueError('Only complete base-seed recipes can receive stability repeats')
        path = ROOT / job['result']['summary_path']
        summary = read(path)
        if sha(path) != job['result']['summary_sha256'] or summary['config'] != job['config']:
            raise ValueError('Completed recipe identity changed')
        if (summary.get('n') != 4755 or len(summary['folds']) != 3
                or sorted(f['fold'] for f in summary['folds']) != [0, 1, 2]
                or any(f.get('n') != 1585 or f.get('status') != 'complete' for f in summary['folds'])):
            raise ValueError('A complete common three-fold result is required')
        oof = path.with_name('oof.npz')
        if sha(oof) != summary['oof_sha256']:
            raise ValueError('OOF archive identity changed')
        evidence[path.relative_to(ROOT).as_posix()] = sha(path)
        evidence[oof.relative_to(ROOT).as_posix()] = sha(oof)
        evidence[job['config_path']] = sha(ROOT / job['config_path'])
        for offset in (1, 2):
            config = copy.deepcopy(job['config'])
            config.update(experiment_id=name[:-3] + f'_s{offset}', seed=BASE + offset)
            if config['experiment_id'] in jobs:
                raise ValueError('Do not repeat an already registered seed')
            configs.append(config)
    return {
        'round': 'r13_fixed_seed_stability', 'round_type': 'stability',
        'hypothesis': 'Assess seed variation for the explicitly chosen completed recipes without changing any hyperparameter. Base seeds3036900958 and3036900959 are fixed in the original search plan; seed3036900957 remains the only deployment seed. This is stability confirmation, not a new hypothesis or a way to count saturation rounds.',
        'selection_rationale': rationale,
        'base_recipes': ids, 'evidence_sha256': evidence,
        'producer_sha256': sha(Path(__file__)),
        'configs': configs,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', action='append', required=True)
    parser.add_argument('--rationale', required=True)
    args = parser.parse_args()
    spec = prepare(args.experiment, args.rationale)
    destination = ROOT / 'docs/research_v3/rounds/r13_fixed_seed_stability.json'
    if destination.exists():
        if read(destination) != spec:
            raise ValueError('Preserve the prior stability selection')
    else:
        with destination.open('x', encoding='utf-8', newline='\n') as stream:
            json.dump(spec, stream, indent=2)
            stream.write('\n')
    print(json.dumps({'prepared': destination.relative_to(ROOT).as_posix(),
                      'configurations': len(spec['configs']), 'registered': False}))


if __name__ == '__main__':
    main()
