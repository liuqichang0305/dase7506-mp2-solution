"""Fixed descriptive slices and complementarity of complete aligned train OOF.

No training, answer generation, validation/test/leaderboard access or automatic
fusion search. A fusion is evaluated only from a separate exact registration
whose prior, hash-locked complementarity gate passes.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

for variable in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[variable] = '1'
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code'))
from common import DATA, load_bank
from mp2.selection_v2 import answer_groups, group_probabilities, select_candidates
from mp2.research_v3.selection import member_probabilities, select_raw
from analyze_oof_v2 import measure
from audit_assets import audit

DEFAULT_METHODS = ['v3:v3_mlp_adamw_s0', 'v3:v3_mlp_ema_s0', 'v2:attention_regularized_s0']
DEFAULT_PAIRS = [(DEFAULT_METHODS[0], DEFAULT_METHODS[1]),
                 (DEFAULT_METHODS[0], DEFAULT_METHODS[2]),
                 (DEFAULT_METHODS[1], DEFAULT_METHODS[2])]
MIN_RECIPROCAL = 10
MIN_ORACLE_PP = 1.


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, data):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(data, stream, indent=2, ensure_ascii=False)
        stream.write('\n')


def install_guard():
    data = DATA.resolve()
    allowed = {data / 'manifest.json', data / 'candidates' / 'gsm8k_train.jsonl'}
    denied = []
    def guard(event, arguments):
        if event.startswith('socket.') or event in ('subprocess.Popen', 'os.system', 'os.spawn'):
            denied.append(event)
            raise PermissionError('OOF diagnostics are offline and do not launch processes')
        if event != 'open' or isinstance(arguments[0], int):
            return
        path = Path(os.fsdecode(arguments[0])).resolve()
        if path.is_relative_to(data) and path not in allowed:
            denied.append(str(path))
            raise PermissionError('Only the train candidate text and manifest may be read')
        name = path.name.lower()
        if path.is_relative_to(ROOT) and ('leaderboard' in name or 'ranking_background' in name
            or name in ('validation_summary.json', 'final_summary.json', 'final_test.json')):
            denied.append(str(path))
            raise PermissionError('Holdout and ranking background are forbidden in this diagnostic')
    sys.addaudithook(guard)
    return denied


def method_path(key):
    version, separator, name = key.partition(':')
    if not separator or version not in ('v2', 'v3') or not name or any(x in name for x in ('/', '\\', ':')) or name in ('.', '..'):
        raise ValueError('Method must be v2:<registered_name> or v3:<registered_name>')
    return ROOT / 'results' / f'research_{version}' / name


def load_method(key, manifest):
    folder = method_path(key)
    registration = json.loads((folder / 'registration.json').read_text(encoding='utf-8'))
    summary = json.loads((folder / 'summary.json').read_text(encoding='utf-8'))
    config = registration['config']
    if config['experiment_id'] != folder.name or summary['status'] != 'complete' or summary['config'] != config:
        raise ValueError(f'{key}: incomplete or inconsistent registration/summary')
    if sorted(row['fold'] for row in summary['folds']) != [0, 1, 2]:
        raise ValueError(f'{key}: complete three-fold OOF is required')
    folds_path = folder.parent / 'common_folds.json'
    if digest(folds_path) != registration['folds_sha256']:
        raise ValueError(f'{key}: fold-manifest hash mismatch')
    folds = json.loads(folds_path.read_text(encoding='utf-8'))
    for filename, expected in registration['source_files'].items():
        source = (ROOT / filename).resolve()
        if not source.is_relative_to(ROOT / 'code') or source.suffix != '.py' or digest(source) != expected:
            raise ValueError(f'{key}: source provenance mismatch: {filename}')
    expected_data = {f'code/data/{name}': value for name, value in manifest['sha256'].items()
                     if name in ('gsm8k_train.jsonl', 'candidates/gsm8k_train.jsonl', 'candidates/gsm8k_train.npz')}
    expected_data['code/data/manifest.json'] = digest(DATA / 'manifest.json')
    if any(filename not in expected_data or expected_data[filename] != expected
           for filename, expected in registration['data_files'].items()):
        raise ValueError(f'{key}: registered data is not the same official train bank')
    required = set(expected_data) - {'code/data/manifest.json'}
    if not required <= set(registration['data_files']):
        raise ValueError(f'{key}: incomplete train provenance')
    with np.load(folder / 'oof.npz', allow_pickle=False) as archive:
        contents = {name: archive[name].copy() for name in archive.files}
    scores, labels, groups, fold = (contents[name] for name in ('scores', 'labels', 'groups', 'fold'))
    if scores.shape != labels.shape or groups.shape != labels.shape or labels.shape != (folds['questions'], 4) or fold.shape != (len(labels),):
        raise ValueError(f'{key}: malformed OOF shapes')
    if not np.isfinite(scores).all() or not np.isin(labels, (0, 1)).all() or np.any(labels.sum(1) < 1):
        raise ValueError(f'{key}: nonfinite scores or invalid labels')
    if not np.issubdtype(groups.dtype, np.integer) or np.any((groups < -1) | (groups > 3)) or np.any((labels > 0) & (groups < 0)):
        raise ValueError(f'{key}: invalid answer groups')
    membership = groups[:, :, None] == np.arange(4)[None, None, :]
    counts = membership.sum(1)
    positive_counts = (membership * labels[:, :, None]).sum(1)
    if np.any((positive_counts > 0) & (positive_counts != counts)) or np.any((positive_counts > 0).sum(1) != 1):
        raise ValueError(f'{key}: labels do not identify one wholly correct numeric group')
    if set(fold.tolist()) != {0, 1, 2}:
        raise ValueError(f'{key}: incomplete folds')
    for split in folds['folds']:
        number, held = split['fold'], np.asarray(split['holdout'], dtype=int)
        if not np.array_equal(np.flatnonzero(fold == number), np.sort(held)):
            raise ValueError(f'{key}: OOF fold membership mismatch')
        report = next(row for row in summary['folds'] if row['fold'] == number)
        if report.get('status') != 'complete':
            raise ValueError(f'{key}: incomplete per-fold report')
        path = folder / f'fold{number}-scores.npy'
        if digest(path) != report['scores_sha256'] or not np.array_equal(np.load(path, allow_pickle=False), scores[held]):
            raise ValueError(f'{key}: per-fold score hash/order mismatch')
        for filename, expected in report.get('output_sha256', {}).items():
            if Path(filename).name != filename or digest(folder / filename) != expected:
                raise ValueError(f'{key}: saved output hash mismatch')
        if 'raw_scores' in contents and not np.array_equal(np.load(folder / f'fold{number}-raw-scores.npy', allow_pickle=False), contents['raw_scores'][held]):
            raise ValueError(f'{key}: raw member scores mismatch')
    oof_hash = digest(folder / 'oof.npz')
    if 'oof_sha256' in summary and summary['oof_sha256'] != oof_hash:
        raise ValueError(f'{key}: OOF archive hash mismatch')
    raw = contents.get('raw_scores', scores)
    choices = np.asarray(select_raw(raw, groups, config['decoder']), dtype=np.int64)
    effective = summary.get('oof_decoder', config['decoder'])
    if choices.tolist() != select_candidates(scores, groups, **effective):
        raise ValueError(f'{key}: saved OOF representation differs from member decisions')
    correct = labels[np.arange(len(labels)), choices].astype(bool)
    metrics = measure(correct, labels, groups, fold)
    for field, metric in [('correct', 'overall'), ('hard_correct', 'hard'), ('single_positive_correct', 'single_positive')]:
        if int(summary[field]) != metrics[metric]['correct']:
            raise ValueError(f'{key}: summary correctness is inconsistent')
    return {'key': key, 'folder': folder, 'registration': registration, 'summary': summary,
            'config': config, 'scores': scores, 'raw_scores': raw, 'labels': labels, 'groups': groups,
            'fold': fold, 'choices': choices, 'correct': correct, 'metrics': metrics,
            'hashes': {name: digest(folder / name) for name in ('registration.json', 'summary.json', 'oof.npz')},
            'folds_sha256': registration['folds_sha256']}


def complementarity(records, hard):
    """Two or three specified methods; no mixture scores are computed here."""
    if len(records) not in (2, 3):
        raise ValueError('Gate requires exactly two or three explicit methods')
    events = np.stack([record['correct'] for record in records])
    subsets = {}
    for label, mask in [('overall', np.ones(events.shape[1], bool)), ('hard', hard)]:
        chosen = events[:, mask]
        n = chosen.shape[1]
        oracle = chosen.any(0)
        unique = [int((chosen[i] & ~np.delete(chosen, i, axis=0).any(0)).sum()) for i in range(len(records))]
        correct = chosen.sum(1)
        gain = int(oracle.sum()) - int(correct.max(initial=0))
        subsets[label] = {'n': n, 'member_correct': correct.tolist(), 'member_unique_correct': unique,
            'oracle_correct': int(oracle.sum()), 'all_correct': int(chosen.all(0).sum()),
            'all_wrong': int((~oracle).sum()), 'oracle_gain_over_strongest_percentage_points': 100 * gain / n if n else None}
    full, difficult = subsets['overall'], subsets['hard']
    passed = min(full['member_unique_correct']) >= MIN_RECIPROCAL and difficult['n'] > 0 and (
        100 * (difficult['oracle_correct'] - max(difficult['member_correct'])) >= difficult['n'])
    return {'methods': [row['key'] for row in records], **subsets, 'gate_passed': bool(passed),
            'requirements': {'minimum_each_unique_correct': MIN_RECIPROCAL, 'minimum_hard_oracle_gain_pp': MIN_ORACLE_PP},
            'interpretation': 'Oracle is an unattainable diagnostic upper bound; no fusion prediction has been evaluated. For three methods, each must uniquely correct errors shared by both others.'}


def fixed_slices(bank, labels, groups):
    group_count = np.array([len(set(row[row >= 0])) for row in groups])
    positive_count = labels.sum(1).astype(int)
    none_count = (groups < 0).sum(1)
    question_words = np.array([len(row['question'].split()) for row in bank])
    candidate_words = np.array([[len(candidate['text'].split()) for candidate in row['candidates']] for row in bank])
    question_chars = np.array([len(row['question']) for row in bank])
    candidate_chars = np.array([[len(candidate['text']) for candidate in row['candidates']] for row in bank])
    slices = []
    for dimension, values, bins in [('answer_groups', group_count, range(1, 5)),
                                    ('positive_candidates', positive_count, range(1, 5)),
                                    ('unparseable_candidates', none_count, range(5))]:
        for value in bins:
            slices.append((dimension, str(value), values == value))
    # Fixed cutpoints are declared in source, never fitted to errors or accuracy.
    lengths = [('question_words', question_words, [0, 40, 80, 120, np.inf]),
               ('question_characters', question_chars, [0, 200, 400, 600, np.inf]),
               ('candidate_mean_words', candidate_words.mean(1), [0, 100, 200, 400, np.inf]),
               ('candidate_max_words', candidate_words.max(1), [0, 100, 200, 400, np.inf]),
               ('candidate_word_spread', np.ptp(candidate_words, axis=1), [0, 50, 100, 200, np.inf]),
               ('candidate_mean_characters', candidate_chars.mean(1), [0, 500, 1000, 2000, np.inf])]
    definitions = {}
    for dimension, values, cutpoints in lengths:
        definitions[dimension] = [None if np.isinf(x) else x for x in cutpoints]
        for low, high in zip(cutpoints[:-1], cutpoints[1:]):
            slices.append((dimension, f'[{low:g},{high:g})', (values >= low) & (values < high)))
    return slices, group_count >= 2, definitions


def evaluate_registered_fusion(path, records, hard, output):
    registration = json.loads(Path(path).read_text(encoding='utf-8'))
    keys = registration.get('methods', [])
    if registration.get('status') != 'registered' or registration.get('purpose') != 'single_fixed_equal_probability_oof':
        raise ValueError('A separate explicit OOF fusion registration is required')
    if len(keys) not in (2, 3) or len(set(keys)) != len(keys) or any(key not in records for key in keys):
        raise ValueError('Registered fusion requires two or three loaded distinct methods')
    if registration.get('weights') != [1. / len(keys)] * len(keys):
        raise ValueError('Only fixed equal probability weights are permitted')
    members = [records[key] for key in keys]
    if any(member['config']['seed'] != 3036900957 for member in members):
        raise ValueError('Fusion cannot select a lucky seed')
    if any(member['config']['decoder'].get('mode') != 'group_lse' or not member['config']['decoder'].get('mask', True) for member in members):
        raise ValueError('Fusion members require the deployable masked group-LSE decoder')
    expected_hashes = {member['key']: member['hashes']['oof.npz'] for member in members}
    if registration.get('oof_sha256') != expected_hashes:
        raise ValueError('Registration does not lock the exact OOF arrays')
    evidence = registration['gate_evidence']
    gate_path = (ROOT / evidence['path']).resolve()
    if not gate_path.is_relative_to(ROOT / 'results' / 'research_v3') or digest(gate_path) != evidence['sha256']:
        raise ValueError('Prior gate evidence hash/path mismatch')
    previous = json.loads(gate_path.read_text(encoding='utf-8'))
    previous_hashes = {item['method']: item['input_sha256']['oof.npz'] for item in previous['methods']}
    if any(previous_hashes.get(key) != value for key, value in expected_hashes.items()):
        raise ValueError('Prior gate evidence used different OOF arrays')
    registered_at = datetime.fromisoformat(registration['registered_at_utc'].replace('Z', '+00:00'))
    evidence_at = datetime.fromisoformat(previous['created_at_utc'].replace('Z', '+00:00'))
    if not evidence_at <= registered_at <= datetime.now(timezone.utc):
        raise ValueError('Gate evidence must precede this independent registration')
    gate = complementarity(members, hard)
    if not gate['gate_passed'] or not any(item['methods'] == keys and item['gate_passed'] for item in previous['complementarity']):
        raise ValueError('Previously documented reciprocal/oracle gate is not satisfied')
    # Write the verified gate before forming any equal-probability mixture.
    write_new(output / 'fusion-gate.json', {'gate': gate, 'registration_sha256': digest(path)})
    probabilities = [member_probabilities(member['raw_scores'], member['groups'], member['config']['decoder']) for member in members]
    probability = np.mean(probabilities, axis=0)
    groups, labels, fold = (members[0][name] for name in ('groups', 'labels', 'fold'))
    valid = groups >= 0
    first = np.where(valid, np.take_along_axis(probability, np.maximum(groups, 0), axis=1), -np.inf).argmax(1)
    winner = groups[np.arange(len(groups)), first]
    raw = members[0]['raw_scores']
    anchor = raw[:, 0] if raw.ndim == 3 else raw
    choices = np.where(valid & (groups == winner[:, None]), anchor, -np.inf).argmax(1)
    choices = np.where(valid.any(1), choices, 0)
    correct = labels[np.arange(len(labels)), choices].astype(bool)
    assets = []
    for number in (0, 1, 2):
        children = [audit(ROOT / next(row for row in member['summary']['folds'] if row['fold'] == number)['checkpoint']) for member in members]
        assets.append({'fold': number, 'member_uncompressed_bytes': [row['uncompressed_bytes'] for row in children],
                       'total_member_uncompressed_bytes': sum(row['uncompressed_bytes'] for row in children)})
    np.savez(output / 'fusion-oof.npz', probabilities=probability, choices=choices, correct=correct, fold=fold)
    return {'status': 'complete', 'methods': keys, 'weights': registration['weights'], 'gate': gate,
            'metrics': measure(correct, labels, groups, fold), 'assets_per_fold': assets,
            'note': 'Only the separately registered mixture; no pair/weight sweep. Wrapper bytes still count at final deployment. OOF selection optimism remains.'}


def diagnose(methods, pairs, output, fusion_registration=None):
    output = Path(output)
    if output.exists():
        raise FileExistsError('Choose a new immutable diagnostic directory')
    denied = install_guard()
    manifest = json.loads((DATA / 'manifest.json').read_text(encoding='utf-8'))
    if digest(DATA / 'candidates/gsm8k_train.jsonl') != manifest['sha256']['candidates/gsm8k_train.jsonl']:
        raise ValueError('Official train candidate hash mismatch')
    records = {key: load_method(key, manifest) for key in methods}
    if len(records) != len(methods) or not records:
        raise ValueError('Specify unique completed methods')
    first = next(iter(records.values()))
    for record in records.values():
        if record['folds_sha256'] != first['folds_sha256'] or any(not np.array_equal(record[key], first[key]) for key in ('labels', 'groups', 'fold')):
            raise ValueError('Methods do not have exactly aligned question-level folds, labels and groups')
    bank = load_bank('train')
    labels, groups, fold = (first[key] for key in ('labels', 'groups', 'fold'))
    if len(bank) != len(labels) or not np.array_equal(answer_groups(bank), groups):
        raise ValueError('OOF groups/order do not match official train candidate inputs')
    slices, hard, definitions = fixed_slices(bank, labels, groups)
    comparisons = []
    for keys in pairs:
        if len(set(keys)) != len(keys) or any(key not in records for key in keys):
            raise ValueError('Every requested gate must name distinct loaded methods')
        comparisons.append(complementarity([records[key] for key in keys], hard))
    rows = []
    for dimension, bucket, mask in slices:
        row = {'dimension': dimension, 'bucket': bucket, 'n': int(mask.sum()), 'hard_n': int((mask & hard).sum()), 'methods': {}}
        for key, record in records.items():
            event = record['correct']
            row['methods'][key] = {'correct': int(event[mask].sum()),
                'accuracy_percent': float(100 * event[mask].mean()) if mask.any() else None,
                'hard_correct': int(event[mask & hard].sum()),
                'hard_accuracy_percent': float(100 * event[mask & hard].mean()) if (mask & hard).any() else None}
        rows.append(row)
    result = {'status': 'complete', 'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'Complete aligned train OOF only; official train text used only for predefined lengths and parser alignment.',
        'question_texts_or_answers_output': False, 'validation_test_rank_read': False,
        'folds_sha256': first['folds_sha256'], 'common_labels_groups_folds_verified': True,
        'train_candidate_sha256': digest(DATA / 'candidates/gsm8k_train.jsonl'),
        'source_sha256': digest(__file__), 'rejected_reads': denied,
        'methods': [{'method': key, 'config': record['config'], 'metrics': record['metrics'],
                     'input_sha256': record['hashes'], 'registered_source_files': record['registration']['source_files'],
                     'historical_comparator': key.startswith('v2:')} for key, record in records.items()],
        'slice_definitions': {'length_cutpoints': definitions, 'intervals': 'lower inclusive, upper exclusive; null denotes infinity',
            'unit': 'Question, never individual candidate; candidate mean/max/spread are question-level summaries',
            'selection': 'All fixed bins reported, including empty bins; no error-adaptive thresholds'},
        'slices': rows, 'complementarity': comparisons,
        'fusion': {'status': 'not_run', 'reason': 'No independent fusion registration supplied'},
        'limitations': ['Descriptive OOF slices overlap and are not confirmatory significance tests.',
                       'Historical architecture comparison includes optimizer/regularization recipe differences; it is not a clean causal architecture ablation.',
                       'Oracle gains are not expected ensemble gains. OOF-guided future hypotheses require registration and fresh fitting within the same bounded campaign.']}
    output.mkdir(parents=True)
    if fusion_registration:
        result['fusion'] = evaluate_registered_fusion(fusion_registration, records, hard, output)
    write_new(output / 'diagnostic.json', result)
    # Preserve the exact audit implementation beside its immutable hash.
    (output / 'diagnose_oof_v3.source.py').write_bytes(Path(__file__).read_bytes())
    lines = ['# Fixed train-OOF diagnostic', '', result['scope'], '',
             '| Method | Overall correct/N | Hard correct/N | Single-positive correct/N |', '|---|---:|---:|---:|']
    for item in result['methods']:
        metric = item['metrics']
        lines.append('| ' + item['method'] + ' | ' + ' | '.join(f"{metric[name]['correct']}/{metric[name]['n']}" for name in ('overall', 'hard', 'single_positive')) + ' |')
    lines += ['', 'Oracle/reciprocal results are diagnostic; no fusion is implied.', '',
              '| Explicit methods | Unique corrections per member | Hard oracle gain over strongest (pp) | Gate |', '|---|---:|---:|---|']
    for item in comparisons:
        gain = item['hard']['oracle_gain_over_strongest_percentage_points']
        lines.append(f"| {' + '.join(item['methods'])} | {item['overall']['member_unique_correct']} | {gain:.4f} | {item['gate_passed']} |")
    lines += ['', 'All fixed slices, provenance and checks are in `diagnostic.json`. No question text or answer is emitted.', '',
              'Fusion status: ' + result['fusion']['status'] + '.', '', *result['limitations']]
    (output / 'RESULTS.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', action='append', help='Explicit v2:name or v3:name; repeat for each completed method')
    parser.add_argument('--pair', nargs=2, action='append', help='Explicit diagnostic pair; does not form a mixture')
    parser.add_argument('--triple', nargs=3, action='append', help='Explicit diagnostic three-member gate')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fusion-registration', type=Path)
    args = parser.parse_args()
    methods = args.method or DEFAULT_METHODS
    pairs = (args.pair or (DEFAULT_PAIRS if args.method is None else [])) + (args.triple or [])
    result = diagnose(methods, pairs, args.output, args.fusion_registration)
    print(json.dumps({'status': result['status'], 'methods': methods, 'slices': len(result['slices']), 'fusion': result['fusion']['status']}))


if __name__ == '__main__':
    main()
