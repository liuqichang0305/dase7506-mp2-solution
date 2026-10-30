"""Prepare one/two explicit single-model V3 finalists after search closure.

Select registers permitted full-train fits; lock binds their completed bundles.
Neither operation trains, chooses a winner, reads holdout data, or opens validation.
"""
from __future__ import annotations

from datetime import datetime, timezone
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
BASE = 'results/research_v3/'
SELECTION = BASE + 'FINALISTS_SELECTION.json'
LOCK = BASE + 'FINALISTS_LOCK.json'
PRODUCER = 'scripts/prepare_finalists_v3.py'
SEED = 3036900957
PROVENANCE = ('source_files', 'data_files', 'folds_sha256')


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(part)
    return result.hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def payload(value):
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode('utf-8')


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def path(root, relative, prefix=None):
    require(isinstance(relative, str) and not Path(relative).is_absolute(), 'Expected a repository-relative path')
    root = Path(root).resolve()
    target = (root / relative).resolve()
    require(target.is_relative_to(root) and target.relative_to(root).as_posix() == relative,
            'Noncanonical or escaping evidence path')
    require(prefix is None or target.is_relative_to(root / prefix), 'Evidence escapes its permitted directory')
    require(not any(word in relative.casefold() for word in ('ranking_background', 'leaderboard')),
            'Ranking/background evidence is prohibited')
    if target.is_relative_to(root / 'code/data'):
        require(relative in {'code/data/manifest.json', 'code/data/gsm8k_train.jsonl',
                             'code/data/candidates/gsm8k_train.jsonl', 'code/data/candidates/gsm8k_train.npz'},
                'Only official train and its public manifest may be opened')
    return target


def load_gate(root):
    sys.path.insert(0, str(Path(root) / 'code/tools'))
    location = Path(root) / 'code/tools/confirm_validation_v3.py'
    spec = importlib.util.spec_from_file_location('finalist_preopen_confirmation', location)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_freezer(root):
    location = Path(root) / 'code/tools/freeze_v3.py'
    spec = importlib.util.spec_from_file_location('finalist_copy_freezer', location)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_map(root, values, evidence):
    require(isinstance(values, dict) and values, 'A nonempty hash map is required')
    for relative, expected in values.items():
        require(re.fullmatch('[0-9a-f]{64}', str(expected)) is not None, 'Invalid evidence SHA256')
        require(sha(path(root, relative)) == expected, 'Evidence changed: ' + relative)
        evidence[relative] = expected


def bind(root, relative, evidence):
    evidence[relative] = sha(path(root, relative))
    return read(path(root, relative))


def before_holdout(root):
    for relative in (BASE + 'VALIDATION_OPENED.json', BASE + 'validation_summary.json',
                     'submission/v3/FREEZE.json', 'submission/v3/TEST_OPENED.json'):
        require(not (Path(root) / relative).exists(), 'Finalist preparation is closed after holdout opening/freeze')
    closed = read(path(root, BASE + 'SEARCH_CLOSED.json'))
    require(closed.get('search_closed') is True, 'Explicit SEARCH_CLOSED is required; this helper never closes research')
    instant = datetime.fromisoformat(closed['created_at_utc'].replace('Z', '+00:00'))
    require(instant.tzinfo is not None and instant <= datetime.now(timezone.utc), 'Invalid search closure timestamp')
    return closed


def common_folds(root, evidence):
    folds = bind(root, BASE + 'common_folds.json', evidence)
    require(folds.get('questions') == 4755 and sorted(r['fold'] for r in folds['folds']) == [0, 1, 2],
            'All three official-train question folds are required')
    universe, all_held = set(range(4755)), []
    for row in folds['folds']:
        parts = [row[key] for key in ('outer_fit', 'holdout', 'inner_fit', 'inner_stop')]
        require(all(len(v) == len(set(v)) and all(type(i) is int for i in v) for v in parts), 'Duplicate/noninteger fold rows')
        outer, held, inner, stop = map(set, parts)
        require(len(held) == 1585 and not outer & held and outer | held == universe
                and inner and stop and not inner & stop and inner | stop == outer,
                'Question folds or nested stopping partition are incomplete/overlapping')
        all_held.extend(held)
    require(sorted(all_held) == list(range(4755)), 'Every train question must be held out exactly once')
    old = 'results/research_v2/common_folds.json'
    require(sha(path(root, old)) == evidence[BASE + 'common_folds.json'], 'Preserved V2 and V3 common folds differ')
    evidence[old] = evidence[BASE + 'common_folds.json']
    return evidence[BASE + 'common_folds.json']


def candidate(root, identifier, queue, analysis, folds_hash, gate, evidence):
    jobs = [j for j in queue['jobs'] if j['experiment_id'] == identifier]
    require(len(jobs) == 1 and jobs[0]['status'] == 'completed', 'Explicit candidate must be a completed registered job')
    job = jobs[0]
    config_path = job['config_path']
    path(root, config_path, 'code/configs/research_v3')
    config = bind(root, config_path, evidence)
    require(config == job['config'] and canonical_sha(config) == job['config_sha256'], 'Registered configuration changed')
    require(config['experiment_id'] == identifier and config['seed'] == SEED, 'Only the fixed base deployment seed 3036900957 is eligible')
    directory = BASE + identifier + '/'
    registration = bind(root, directory + 'registration.json', evidence)
    summary = bind(root, directory + 'summary.json', evidence)
    receipt = job['result']
    require(receipt['summary_path'] == directory + 'summary.json'
            and receipt['summary_sha256'] == evidence[directory + 'summary.json']
            and receipt['oof_sha256'] == summary['oof_sha256']
            and receipt['accuracy'] == summary['accuracy'] and receipt['hard_accuracy'] == summary['hard_accuracy'],
            'Completed scheduler receipt differs from the actual CV summary/OOF')
    require(registration['config'] == summary['config'] == config and registration['config_sha256'] == canonical_sha(config),
            'CV registration/summary/config disagree')
    require(summary.get('status') == 'complete' and summary.get('n') == 4755,
            'A complete full-train three-fold summary is required')
    prov = summary['provenance']
    require(set(prov['source_files']) == gate.TRAIN_SOURCE and set(prov['data_files']) == gate.TRAIN_PATHS
            and prov['folds_sha256'] == folds_hash, 'CV training source/data/fold provenance differs from the reviewed protocol')
    require(all(registration[key] == prov[key] for key in PROVENANCE), 'Registration provenance differs from completed CV')
    registered = job['registered_provenance']
    require(registered['source_files'] == prov['source_files']
            and registered['train_and_fold_files'] == {**prov['data_files'], BASE + 'common_folds.json': folds_hash},
            'Queue provenance differs from the actual completed training inputs')
    check_map(root, prov['source_files'], evidence)
    check_map(root, prov['data_files'], evidence)
    reports = sorted(summary['folds'], key=lambda item: item['fold'])
    require([item['fold'] for item in reports] == [0, 1, 2]
            and all(item['status'] == 'complete' and item['n'] == 1585 for item in reports),
            'Missing or incomplete outer fold')
    from audit_assets import audit
    for item in reports:
        require(item['experiment_id'] == identifier, 'Fold report belongs to another experiment')
        require(all(item['provenance'][key] == prov[key] for key in PROVENANCE), 'Fold provenance differs')
        require(item == bind(root, directory + f"fold{item['fold']}.json", evidence), 'Stored fold report differs from summary')
        require(item['seed'] == SEED + item['fold'], 'Fold seed differs from the fixed deterministic offset')
        check_map(root, {directory + name: digest for name, digest in item['output_sha256'].items()}, evidence)
        checkpoint = path(root, item['checkpoint'], 'artifacts/research_v3')
        inventory = audit(checkpoint)
        require(inventory['within_limit'], 'A CV bundle exceeds the inference asset budget')
        require({x['path']: x['sha256'] for x in inventory['files']} == item['asset_sha256'], 'CV bundle inventory changed')
        check_map(root, {item['checkpoint'] + '/' + key: digest for key, digest in item['asset_sha256'].items()}, evidence)
    check_map(root, {directory + 'oof.npz': summary['oof_sha256']}, evidence)
    matches = [row for row in analysis['rows'] if row['version'] == 'research_v3' and row['experiment_id'] == identifier]
    require(len(matches) == 1, 'Final train-only analysis must contain this explicit candidate once')
    row = matches[0]
    require(row['status'] == 'completed' and row['seed'] == SEED and row['config'] == config
            and row['summary_sha256'] == evidence[directory + 'summary.json'] and row['oof_sha256'] == summary['oof_sha256'],
            'Selection rationale analysis is stale or belongs to another configuration')
    for key in ('n', 'correct', 'hard_n', 'hard_correct', 'training_seconds'):
        require(row[key] == summary[key], 'Analysis and original summary disagree: ' + key)
    means = [100 * item['hard_correct'] / item['hard_n'] for item in reports]
    require(math.isclose(row['hard_fold_mean'], statistics.mean(means), abs_tol=1e-10)
            and math.isclose(row['hard_fold_sd'], statistics.stdev(means), abs_tol=1e-10), 'Analysis fold metrics differ')
    require(sum(item['correct'] for item in reports) == summary['correct']
            and sum(item['hard_correct'] for item in reports) == summary['hard_correct']
            and sum(item['hard_n'] for item in reports) == summary['hard_n'], 'Fold counts do not pool to the summary')
    require(row['parameters'] == max(item['parameters'] for item in reports)
            and row['numeric_asset_bytes'] == max(item['numeric_asset_bytes'] for item in reports),
            'Analysis parameter/byte counts differ from the actual completed fold reports')
    selector = {'kind': 'single_model', 'experiment_id': identifier, 'family': config['family'],
                'decoder': config['decoder'], 'member_probability_average': config['family'] == 'tabm'}
    return {'name': identifier, 'component_ids': [identifier], 'selector': selector,
            'checkpoint': 'artifacts/research_v3/final_' + identifier,
            'config_path': config_path, 'config_file_sha256': evidence[config_path],
            'train_only_evidence': {key: row[key] for key in ('seed', 'correct', 'n', 'hard_correct', 'hard_n',
                'hard_fold_mean', 'hard_fold_sd', 'training_seconds', 'parameters', 'numeric_asset_bytes')},
            'cv_summary_sha256': evidence[directory + 'summary.json']}


def selection_value(root, identifiers, analysis_path, rationale, gate, created_at):
    require(1 <= len(identifiers) <= 2 and len(set(identifiers)) == len(identifiers)
            and all(isinstance(name, str) and re.fullmatch('[a-z0-9_]+', name) for name in identifiers),
            'Specify one or two distinct safe experiment IDs; no automatic candidate selection')
    require(isinstance(rationale, str) and rationale.strip(), 'An explicit train-only selection rationale is required')
    closed = before_holdout(root)
    require(gate.instant(closed['created_at_utc']) <= gate.instant(created_at) <= datetime.now(timezone.utc).timestamp(),
            'Selection timestamp must follow closure and cannot be in the future')
    path(root, analysis_path, BASE + 'analysis')
    evidence = {}
    bind(root, BASE + 'SEARCH_CLOSED.json', evidence)
    queue = bind(root, BASE + 'campaign_queue.json', evidence)
    require(queue.get('schema') == 'mp2-campaign-queue-v3'
            and all(job['status'] in ('completed', 'failed', 'pruned') for job in queue['jobs']), 'Campaign queue is not terminal')
    analysis = bind(root, analysis_path, evidence)
    folds_hash = common_folds(root, evidence)
    require(analysis['folds_sha256'] == folds_hash, 'Final selection analysis used different folds')
    check_map(root, {'scripts/analyze_campaign_v3.py': analysis['source_sha256']}, evidence)
    finalists = [candidate(root, name, queue, analysis, folds_hash, gate, evidence) for name in identifiers]
    producer_hash = sha(path(root, PRODUCER))
    require(producer_hash == sha(__file__), 'Producer in the checkout differs from the running helper')
    return {'schema': 'mp2-finalists-selection-v3', 'created_at_utc': created_at, 'deployment_seed': SEED,
            'allowed_final_fit_experiments': identifiers, 'finalists': finalists,
            'selection_basis': {'split': 'train_oof_only', 'analysis_path': analysis_path,
                'analysis_sha256': evidence[analysis_path], 'rationale': rationale.strip(),
                'policy': 'Explicit fixed-base-seed candidates; no new validation/test or ranking inputs. Historical fallback remains governed by the existing confirmation rule.'},
            'evidence_sha256': dict(sorted(evidence.items())), 'producer': {'path': PRODUCER, 'sha256': producer_hash},
            'final_fit_commands': [['.venv/Scripts/python.exe', 'code/train_v3.py', '--config', row['config_path'],
                                    '--final-output', row['checkpoint']] for row in finalists]}


def write_once(target, record):
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('xb') as stream:
        stream.write(payload(record))
        stream.flush()
        os.fsync(stream.fileno())


def select(root, identifiers, analysis_path, rationale, gate=None):
    root = Path(root).resolve()
    gate = gate or load_gate(root)
    require(not (root / LOCK).exists(), 'Finalists are already locked; use lock verification, not selection')
    target = root / SELECTION
    existing = read(target) if target.exists() else None
    result = selection_value(root, identifiers, analysis_path, rationale, gate,
                             existing['created_at_utc'] if existing else now())
    if existing is not None:
        require(existing == result, 'Immutable finalist selection differs; changing or appending candidates is prohibited')
        return existing
    for row in result['finalists']:
        require(not (root / row['checkpoint']).exists()
                and not (root / BASE / row['name'] / 'final_fit.json').exists()
                and not (root / 'tmp/research_v3/training_states' / row['name'] / 'final').exists(),
                'New selection must precede all selected full-train fits/assets')
    write_once(target, result)
    return result


def prevalidate_lock(gate, value):
    """Run the unchanged strict validator against one in-memory proposed file.

    Only read/hash of the not-yet-published lock path are substituted. Every
    source, training provenance, full-train state, bundle and fallback is real.
    No validator bypass or holdout access is introduced; real lock is written
    only after this full check succeeds.
    """
    target = (gate.OUT / 'FINALISTS_LOCK.json').resolve()
    original_read, original_sha = gate.read, gate.sha
    digest = hashlib.sha256(payload(value)).hexdigest()
    gate.read = lambda p: value if Path(p).resolve() == target else original_read(p)
    gate.sha = lambda p: digest if Path(p).resolve() == target else original_sha(p)
    try:
        return gate.validate_lock()
    finally:
        gate.read, gate.sha = original_read, original_sha


def lock(root, gate=None):
    root = Path(root).resolve()
    gate = gate or load_gate(root)
    before_holdout(root)
    selected = read(root / SELECTION)
    basis = selected['selection_basis']
    rebuilt = selection_value(root, selected['allowed_final_fit_experiments'], basis['analysis_path'], basis['rationale'], gate, selected['created_at_utc'])
    require(rebuilt == selected, 'Selected CV/config/source/data or producer evidence changed before full-train locking')
    from audit_assets import audit
    sys.path.insert(0, str(root / 'code'))
    from mp2.research_v3.models import model_config
    finalists, assets, fit_evidence = [], {}, {}
    for row in selected['finalists']:
        identifier, checkpoint = row['name'], row['checkpoint']
        report_path = BASE + identifier + '/final_fit.json'
        report = bind(root, report_path, fit_evidence)
        summary = read(root / BASE / identifier / 'summary.json')
        config = summary['config']
        require(report['config'] == config and report['checkpoint'] == checkpoint
                and all(report['provenance'][key] == summary['provenance'][key] for key in PROVENANCE),
                'Full-train report differs from selected config/source/data/folds or output directory')
        inventory = audit(path(root, checkpoint, 'artifacts/research_v3'))
        require(inventory['within_limit'] and report['asset_sha256'] == {x['path']: x['sha256'] for x in inventory['files']},
                'Full-train asset inventory differs or exceeds 16 MiB')
        bundle = read(root / checkpoint / 'config.json')
        require(bundle['schema'] == 'mp2-research-v3' and bundle['architecture'] == model_config(config)
                and bundle['decoder'] == config['decoder'] and bundle['preprocess']['mode'] == config.get('preprocess', 'standard'),
                'Actual exported architecture/preprocessing/decoder differs from selected full-train recipe')
        if config['family'] != 'linear':
            info = report['info']
            epochs = math.ceil(statistics.median(item['inner_stopping']['best_epoch'] for item in summary['folds']))
            require(info['context'] == {**{key: summary['provenance'][key] for key in PROVENANCE}, 'phase': 'final-all-train'}
                    and info['epochs_run'] == epochs and info['evaluation_weights'] == ('ema' if config.get('ema_decay') is not None else 'live'),
                    'Full-train epoch/context/evaluation-weight recipe differs')
            state = Path(info['checkpoint']).resolve()
            expected_state = root / 'tmp/research_v3/training_states' / identifier / 'final/current.pt'
            require(state == expected_state and sha(state) == info['checkpoint_sha256'], 'Actual trusted full-train state is missing or changed')
            fit_evidence[state.relative_to(root).as_posix()] = info['checkpoint_sha256']
        assets.update({checkpoint + '/' + key: value for key, value in report['asset_sha256'].items()})
        finalists.append({'name': identifier, 'method': 'code/solution_v3.py', 'checkpoint': checkpoint,
            'selector': row['selector'], 'components': [{'kind': 'new', 'experiment_id': identifier, 'seed': SEED,
                'bundle_subdir': '.', 'final_fit_path': report_path, 'final_fit_sha256': fit_evidence[report_path],
                'cv_summary_sha256': row['cv_summary_sha256']}]})
    target = root / LOCK
    existing = read(target) if target.exists() else None
    result = {'schema': 'mp2-finalists-lock-v3', 'search_closed': True,
        'created_at_utc': existing['created_at_utc'] if existing else now(),
        'selection_sha256': sha(root / SELECTION), 'search_closed_sha256': sha(root / BASE / 'SEARCH_CLOSED.json'),
        'data_manifest_sha256': sha(root / 'code/data/manifest.json'),
        'baseline_freeze_sha256': sha(root / 'results/research_v2/V2_FREEZE.json'),
        'inference_source_sha256': {name: sha(path(root, name)) for name in sorted(gate.INFERENCE_SOURCE)},
        'asset_sha256': dict(sorted(assets.items())), 'finalists': finalists,
        'allowed_final_fit_experiments': selected['allowed_final_fit_experiments'],
        'full_train_evidence_sha256': dict(sorted(fit_evidence.items())), 'producer': selected['producer']}
    if existing is not None:
        require(existing == result, 'Immutable finalist lock changed; replacement or adding candidates is prohibited')
        gate.validate_lock()
        return existing
    verified = prevalidate_lock(gate, result)
    require(all(item['within_limit'] for item in verified['resources'].values()), 'Strict pre-open resource audit failed')
    # Detect mutation during the strict model/asset/fallback check before sealing.
    check_map(root, {**selected['evidence_sha256'], **fit_evidence, **assets, **result['inference_source_sha256']}, {})
    require(sha(root / SELECTION) == result['selection_sha256'], 'Selection changed during lock verification')
    write_once(target, result)
    gate.validate_lock()
    return result


def copy_selected(root, gate=None, freezer=None):
    """Copy the already chosen method; never run a model/scorer or choose anew."""
    root = Path(root).resolve()
    gate, freezer = gate or load_gate(root), freezer or load_freezer(root)
    require(not any((root / 'submission/v3' / name).exists() for name in ('FREEZE.json', 'TEST_OPENED.json')),
            'Selected-copy preparation is closed after freeze/test opening')
    verified = gate.validate_lock()
    summary_path = BASE + 'validation_summary.json'
    summary = read(root / summary_path)
    require(summary.get('schema') == 'mp2-validation-confirmation-v3'
            and summary.get('lock_sha256') == verified['lock_sha256']
            and summary.get('search_remains_closed') is True and summary.get('new_test_evaluated') is False,
            'Completed locked validation summary is required')
    expected = gate.choose_finalist(summary['finalists'], verified['baseline'])
    require(summary['selected'] == expected, 'Saved unique selection differs from the fixed confirmation rule')
    check_map(root, summary['evidence_sha256'], {})
    if expected['historical_fallback']:
        identifier = verified['baseline']['selected_experiment_id']
        config = read(root / f'results/research_v2/{identifier}/final_fit.json')['config']
        selector = {'kind': 'single_model', 'experiment_id': identifier, 'family': config['family'],
                    'decoder': config['decoder'], 'member_probability_average': False}
        components = [{'kind': 'historical_v2', 'experiment_id': identifier, 'bundle_subdir': '.',
            'freeze_path': 'results/research_v2/V2_FREEZE.json',
            'freeze_sha256': sha(root / 'results/research_v2/V2_FREEZE.json')}]
        oof = f'results/research_v2/{identifier}'
    else:
        candidate_ = next(row for row in verified['lock']['finalists'] if row['name'] == expected['name'])
        require(len(candidate_['components']) == 1 and candidate_['components'][0]['kind'] == 'new',
                'This helper copies single-model new finalists or the documented historical fallback only')
        identifier = candidate_['components'][0]['experiment_id']
        selector, components = candidate_['selector'], candidate_['components']
        oof = BASE + identifier
    from audit_assets import audit
    original = path(root, expected['checkpoint'], 'artifacts')
    inventory = audit(original)
    require(inventory['within_limit'], 'Selected original inference bundle exceeds the resource limit')
    original_assets = {item['path']: item['sha256'] for item in inventory['files']}
    selected_path = root / 'submission/v3/SELECTED_MODEL.json'
    target = root / 'artifacts/research_v3/final_selected'
    existing = read(selected_path) if selected_path.exists() else None
    producer_hash = sha(path(root, PRODUCER))
    require(producer_hash == sha(__file__), 'Producer in checkout differs from the running helper')
    selected = {'schema': 'mp2-selected-model-v3', 'created_at_utc': existing['created_at_utc'] if existing else now(),
        'name': expected['name'], 'method': 'code/solution_v3.py', 'checkpoint': 'artifacts/research_v3/final_selected',
        'origin_checkpoint': expected['checkpoint'], 'historical_fallback': expected['historical_fallback'],
        'deployment_seed': SEED, 'selector': selector, 'components': components, 'oof_directory': oof,
        'validation_summary_sha256': sha(root / summary_path),
        'origin_asset_sha256': original_assets, 'producer': {'path': PRODUCER, 'sha256': producer_hash}}
    if existing is not None:
        require(target.is_dir() and existing == selected, 'Existing selected record/bundle differs; never overwrite it')
        freezer.validate_selected(root, verified)
        return existing
    marker = root / 'submission/v3/SELECTED_COPY_OPENED.json'
    require(not target.exists() and not marker.exists(), 'Partial/existing deployment copy is preserved; automatic overwrite/resume is prohibited')
    require(gate.instant(summary['completed_at_utc']) <= gate.instant(selected['created_at_utc']), 'Selected copy predates completed validation')
    write_once(marker, {'schema': 'mp2-v3-selected-copy-opened-v1', 'created_at_utc': now(),
        'validation_summary_sha256': selected['validation_summary_sha256'],
        'origin_checkpoint': expected['checkpoint'], 'destination': selected['checkpoint'],
        'origin_asset_sha256': original_assets, 'producer': selected['producer']})
    target.mkdir(parents=True, exist_ok=False)
    for relative, digest in original_assets.items():
        source = (original / relative).resolve()
        destination = (target / relative).resolve()
        require(source.is_relative_to(original) and destination.is_relative_to(target), 'Copy entry escapes its bundle')
        destination.parent.mkdir(parents=True, exist_ok=True)
        with source.open('rb') as incoming, destination.open('xb') as outgoing:
            shutil.copyfileobj(incoming, outgoing)
            outgoing.flush()
            os.fsync(outgoing.fileno())
        require(sha(destination) == digest, 'Copied selected asset bytes differ')
    require({item['path']: item['sha256'] for item in audit(original)['files']} == original_assets,
            'Selected original assets changed during copying')
    require(sha(root / summary_path) == selected['validation_summary_sha256'], 'Validation selection changed during copying')
    # All actual copied assets exist, but publish the selection record only when
    # the unchanged freeze validator accepts its proposed exact metadata.
    original_read = freezer.read
    freezer.read = lambda p: selected if Path(p).resolve() == selected_path.resolve() else original_read(p)
    try:
        freezer.validate_selected(root, verified)
    finally:
        freezer.read = original_read
    write_once(selected_path, selected)
    freezer.validate_selected(root, verified)
    return selected


def install_no_holdout_guard(root):
    root = Path(root).resolve()
    def hook(event, arguments):
        if event != 'open' or isinstance(arguments[0], int):
            return
        target = Path(os.fsdecode(arguments[0])).resolve()
        if target.is_relative_to(root / 'code/data'):
            path(root, target.relative_to(root).as_posix())
            mode, flags = arguments[1], arguments[2]
            if isinstance(mode, str) and any(c in mode for c in 'wax+') or flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC):
                raise PermissionError('Official training inputs are read-only')
        if any(word in str(target).casefold() for word in ('ranking_background', 'leaderboard')):
            raise PermissionError('Finalist preparation cannot read ranking/background inputs')
    sys.addaudithook(hook)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest='mode', required=True)
    choosing = modes.add_parser('select', help='Register explicit candidates; never choose or train automatically')
    choosing.add_argument('--experiment', action='append', required=True)
    choosing.add_argument('--analysis', default=BASE + 'analysis/final_search/analysis.json')
    choosing.add_argument('--rationale', required=True)
    modes.add_parser('lock', help='Verify actual full-train bundles and seal the strict pre-validation lock')
    modes.add_parser('copy-selected', help='Copy the unique saved post-validation selection, including historical fallback')
    args = parser.parse_args()
    install_no_holdout_guard(ROOT)
    if args.mode != 'copy-selected':
        before_holdout(ROOT)
    sys.path.insert(0, str(ROOT / 'code/tools'))
    gate = load_gate(ROOT)
    with gate.exclusive_evaluation():
        if args.mode == 'select':
            result = select(ROOT, args.experiment, args.analysis, args.rationale, gate)
        elif args.mode == 'lock':
            result = lock(ROOT, gate)
        else:
            result = copy_selected(ROOT, gate)
    print(json.dumps({'status': 'verified_existing_or_created_once', 'stage': args.mode,
        'finalists': [item['name'] for item in result['finalists']] if 'finalists' in result else [result['name']],
        'final_fit_commands': result.get('final_fit_commands', []),
        'validation_opened_by_this_command': False, 'test_opened': False}, indent=2))


if __name__ == '__main__':
    main()
