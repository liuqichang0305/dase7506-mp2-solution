"""Prepare exactly two train-only variance controls after the actual R10 gain."""
import copy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    control = 'v3_realmlp_robust_embedding_stats_s0'
    path = ROOT / 'code/configs/research_v3' / (control + '.json')
    config = json.loads(path.read_text(encoding='utf-8'))
    reduced_path = ROOT / 'results/research_v3' / control / 'summary.json'
    full_path = ROOT / 'results/research_v3/v3_realmlp_robust_all_s0/summary.json'
    reduced, full = [json.loads(p.read_text(encoding='utf-8')) for p in (reduced_path, full_path)]
    if (reduced['status'] != 'complete' or reduced['config'] != config
            or reduced['n'] != 4755 or reduced['hard_correct'] != 1665
            or full['hard_correct'] != 1611):
        raise ValueError('The actual completed R10 evidence differs from the proposed follow-up')
    configs = []
    for suffix, updates in [('drop01', {'dropout': .1}), ('ema', {'ema_decay': .99})]:
        item = copy.deepcopy(config)
        item.update(experiment_id=f'v3_realmlp_embedding_stats_{suffix}_s0', **updates)
        configs.append(item)
    evidence = [path, reduced_path, full_path]
    spec = {
        'round': 'r14_realmlp_reduced_variance',
        'hypothesis': 'The completed R10 reduced-input RealMLP reached1665/2114 hard versus1611 for its full-input robust-all control (+54 questions). The family remains below the global Attention incumbent and this representation change also changes first-layer capacity/initialization. Before rejecting it, independently test dropout0.1 and same-trajectory EMA0.99 against the exact reduced-input control (dropout0, no EMA). Each changes one setting; learning rate/width/robust preprocessing/objective remain fixed. Lower learning rate and greater width on full input did not help, so there is no further LR/width grid.',
        'evidence_sha256': {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in evidence},
        'configs': configs,
    }
    destination = ROOT / 'docs/research_v3/rounds/r14_realmlp_reduced_variance.json'
    if destination.exists():
        if json.loads(destination.read_text(encoding='utf-8')) != spec:
            raise ValueError('Preserve the previously prepared follow-up')
    else:
        with destination.open('x', encoding='utf-8', newline='\n') as stream:
            json.dump(spec, stream, indent=2)
            stream.write('\n')
    print(destination.relative_to(ROOT))


if __name__ == '__main__':
    main()
