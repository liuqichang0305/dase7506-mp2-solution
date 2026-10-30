"""Freeze one reviewed, reproduced V3 selection before its single new test run.

This command reads only train/manifest and completed development evidence. It
never opens official test inputs or references and never evaluates a model.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code/tools'))
import confirm_validation_v3 as timing
from audit_assets import audit

METHOD = 'code/solution_v3.py'
CHECKPOINT = 'artifacts/research_v3/final_selected'
SEED = 3036900957
SCHEMA = 'dase7506-mp2-final-freeze-v3'
CORE = {'code/train_v3.py', 'code/common.py', 'code/mp2/features.py', 'code/mp2/training.py',
        'code/mp2/selection_v2.py', 'code/mp2/models_v2.py', 'code/mp2/research_v3/training.py',
        'code/mp2/research_v3/models.py', 'code/mp2/research_v3/optimizers.py',
        'code/mp2/research_v3/preprocess.py', 'code/mp2/research_v3/selection.py'}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_new(path, value):
    import os
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def safe(root, relative):
    if not isinstance(relative, str) or '\\' in relative or Path(relative).is_absolute():
        raise ValueError('Expected a POSIX repository-relative path')
    value = (Path(root) / relative).resolve()
    if value == Path(root).resolve() or not value.is_relative_to(Path(root).resolve()):
        raise ValueError('Evidence path escapes repository')
    if value.is_relative_to((Path(root) / 'code/data').resolve()) and relative != 'code/data/manifest.json':
        raise ValueError('Freeze/evidence checks cannot read raw official data')
    return value


def source_hashes(root=ROOT):
    root = Path(root)
    paths = set(timing.INFERENCE_SOURCE) | CORE | {
        'code/tools/freeze_v3.py', 'submission/v3/evaluate_final.py',
        'code/tools/reliability_v3.py', 'code/tools/reliability_fusion_v3.py',
        'code/tools/prepare_selected_confidence_v3.py',
        'scripts/reproduce_selected_v3.py',
        'code/tools/check_contract_v3.py', 'code/tools/check_export_v3.py'}
    paths.update(path.relative_to(root).as_posix() for path in (root / 'code/mp2').rglob('*.py'))
    paths.update(path.relative_to(root).as_posix() for path in root.glob('requirements*.txt'))
    paths.update(path.relative_to(root).as_posix() for path in (root / 'code').glob('requirements*.txt'))
    return {name: sha(safe(root, name)) for name in sorted(paths)}


def asset_hashes(root, checkpoint=CHECKPOINT):
    folder = safe(root, checkpoint)
    inventory = audit(folder)
    if not inventory['within_limit']:
        raise ValueError('Selected inference assets exceed the 16 MiB limit')
    return {f'{checkpoint}/{item["path"]}': item['sha256'] for item in inventory['files']}, inventory


def check_hashes(root, mapping):
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError('Nonempty evidence hash mapping required')
    for name, digest in mapping.items():
        target = safe(root, name)
        if target.is_symlink() or re.fullmatch('[0-9a-f]{64}', str(digest)) is None or sha(target) != digest:
            raise ValueError('Frozen/evidence bytes changed: ' + name)


def commit_bytes(root, commit, mapping):
    if re.fullmatch('[0-9a-f]{40}', commit) is None:
        raise ValueError('Full source commit required')
    for relative, expected in mapping.items():
        result = subprocess.run(['git', 'show', f'{commit}:{relative}'], cwd=root, capture_output=True)
        if result.returncode or hashlib.sha256(result.stdout).hexdigest() != expected:
            raise ValueError('Frozen source/asset differs from committed bytes: ' + relative)


def _instant(value):
    result = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if result.tzinfo is None:
        raise ValueError('Timezone-aware evidence timestamp required')
    return result


def validate_selected(root=ROOT, verified=None):
    """Validate unique post-lock selection and a byte-identical deployment copy."""
    root = Path(root)
    folder, out = root / 'submission/v3', root / 'results/research_v3'
    selected = read(folder / 'SELECTED_MODEL.json')
    summary = read(out / 'validation_summary.json')
    verified = timing.validate_lock() if verified is None else verified
    if selected.get('schema') != 'mp2-selected-model-v3' or selected.get('method') != METHOD or selected.get('checkpoint') != CHECKPOINT:
        raise ValueError('The unique selected V3 method/checkpoint must use the fixed deployment paths')
    if selected.get('deployment_seed') != SEED or not isinstance(selected.get('selector'), dict) or not selected['selector']:
        raise ValueError('Selected deployment seed/selector is missing or differs')
    if selected['validation_summary_sha256'] != sha(out / 'validation_summary.json') or summary.get('schema') != 'mp2-validation-confirmation-v3':
        raise ValueError('Selection is not bound to completed locked validation')
    if summary['lock_sha256'] != verified['lock_sha256'] or summary.get('new_test_evaluated') is not False or summary.get('search_remains_closed') is not True:
        raise ValueError('Validation summary differs from the closed final lock')
    expected = timing.choose_finalist(summary['finalists'], verified['baseline'])
    if summary['selected'] != expected:
        raise ValueError('Validation selection differs from the fixed selection rule')
    for key in ('name', 'historical_fallback'):
        if selected.get(key) != expected[key]:
            raise ValueError('Selected model differs from the unique validation choice')
    if selected['origin_checkpoint'] != expected['checkpoint']:
        raise ValueError('Selected bundle origin differs from validation choice')
    if _instant(selected['created_at_utc']) < _instant(summary['completed_at_utc']):
        raise ValueError('Final selection predates validation completion')
    check_hashes(root, summary['evidence_sha256'])
    if expected['historical_fallback']:
        baseline = verified['baseline']
        historical_config = read(safe(root, f"results/research_v2/{baseline['selected_experiment_id']}/final_fit.json"))['config']
        historical_selector = {'kind': 'single_model', 'experiment_id': baseline['selected_experiment_id'],
            'family': historical_config['family'], 'decoder': historical_config['decoder'], 'member_probability_average': False}
        if selected['selector'] != historical_selector:
            raise ValueError('Historical fallback selector differs from the original fitted configuration')
        expected_components = [{'kind': 'historical_v2', 'experiment_id': baseline['selected_experiment_id'],
            'bundle_subdir': '.', 'freeze_path': 'results/research_v2/V2_FREEZE.json',
            'freeze_sha256': sha(root / 'results/research_v2/V2_FREEZE.json')}]
    else:
        finalist = next(row for row in verified['lock']['finalists'] if row['name'] == expected['name'])
        expected_components = finalist['components']
        if selected['selector'] != finalist['selector']:
            raise ValueError('Selected confidence selector differs from locked finalist')
    if selected['components'] != expected_components:
        raise ValueError('Selected component provenance differs from the locked method')
    # No new fit or changed ensemble manifest is allowed while making this copy.
    old_inventory = audit(safe(root, expected['checkpoint']))
    assets, inventory = asset_hashes(root)
    if {r['path']: r['sha256'] for r in old_inventory['files']} != {r['path']: r['sha256'] for r in inventory['files']}:
        raise ValueError('Deployment copy differs from the selected validation bundle')
    oof = safe(root, selected['oof_directory'])
    if not oof.is_relative_to(root / 'results'):
        raise ValueError('Selected OOF evidence must stay in research results')
    if len(expected_components) == 1:
        component = expected_components[0]
        generation = 'research_v2' if component.get('kind') == 'historical_v2' else 'research_v3'
        if oof != (root / 'results' / generation / component['experiment_id']).resolve():
            raise ValueError('Selected OOF directory differs from the actual fitted component')
    else:
        from reliability_fusion_v3 import load_fusion_oof
        fusion = load_fusion_oof(oof)
        members = [('v2:' if component.get('kind') == 'historical_v2' else 'v3:') + component['experiment_id'] for component in expected_components]
        if fusion['selector'] != selected['selector'] or fusion['selector']['specification']['methods'] != members:
            raise ValueError('Selected fusion OOF differs from the exact ordered deployed components')
    return selected, summary, verified, assets, inventory


def _typed_result(root, owner, key):
    reference = owner.get(key, {})
    if set(reference) != {'path', 'sha256'} or owner['evidence_files'].get(reference['path']) != reference['sha256']:
        raise ValueError('Required typed evidence reference is absent/unbound: ' + key)
    check_hashes(root, {reference['path']: reference['sha256']})
    return read(safe(root, reference['path']))


def _audit_identity(root, report, assets):
    checkpoint = Path(report['checkpoint'])
    checkpoint = checkpoint.resolve() if checkpoint.is_absolute() else (Path(root) / checkpoint).resolve()
    expected_assets = {name.removeprefix(CHECKPOINT + '/'): digest for name, digest in assets.items()}
    actual_assets = {name.replace('\\', '/'): digest for name, digest in report['assets_sha256'].items()}
    if checkpoint != (Path(root) / CHECKPOINT).resolve() or actual_assets != expected_assets:
        raise ValueError('Typed audit report belongs to different selected assets')
    if report.get('status') != 'passed' or report.get('input_source') != 'validation' or report.get('reference_files_read') is not False:
        raise ValueError('Typed audit did not pass full validation-input isolation')


def _gates(root, selected_hash, sources, assets, commit, frozen_at):
    folder = Path(root) / 'submission/v3'
    selected = read(folder / 'SELECTED_MODEL.json')
    review, reproduction = read(folder / 'REVIEW_RESULT.json'), read(folder / 'REPRODUCTION_RESULT.json')
    for record, schema in [(review, 'mp2-v3-review-result-v1'), (reproduction, 'mp2-v3-reproduction-result-v1')]:
        if record.get('schema') != schema or record.get('passed') is not True or record.get('selected_model_sha256') != selected_hash:
            raise ValueError('Passed review/reproduction must bind this exact selected model')
        if record.get('source_files') != sources or record.get('assets') != assets or record.get('method') != METHOD or record.get('checkpoint') != CHECKPOINT:
            raise ValueError('Review/reproduction source or asset inventory differs')
        check_hashes(root, record['evidence_files'])
        if not _instant(selected['created_at_utc']) <= _instant(record['created_at_utc']) <= _instant(frozen_at):
            raise ValueError('Review/reproduction timestamp must follow unique selection and precede freeze')
    dependency = review['dependency_audit']
    if dependency.get('passed') is not True or dependency.get('no_hidden_parameters') is not True or dependency.get('external_runtime_reads') != 'bundle_and_inputs_only' or not dependency.get('inference_dependencies'):
        raise ValueError('Dependency/hidden-asset audit did not pass')
    if (reproduction.get('split') != 'validation' or reproduction.get('fresh_environment') is not True
            or reproduction.get('choices_match') is not True or reproduction.get('official_score_match') is not True
            or reproduction.get('source_commit') != commit):
        raise ValueError('Fresh-environment validation reproduction of the exact commit is required')
    contract, export = (_typed_result(root, review, key) for key in ('contract_result', 'export_result'))
    for report in (contract, export):
        _audit_identity(root, report, assets)
        if _instant(report['timestamp_utc']) > _instant(review['created_at_utc']):
            raise ValueError('Audit evidence postdates the review record')
    required_checks = {'python_int_length_range', 'arbitrary_n_batch_independence', 'id_independence',
        'question_order', 'determinism', 'member_probability_average_and_actual_chosen_confidence',
        'parseability_mask_and_all_none_fallback', 'reject_nonfinite', 'reject_wrong_shape', 'reject_misaligned',
        'missing_bundle_rejected', 'offline_no_references_or_forbidden_access', 'numpy_only_no_torch_import'}
    if (contract.get('n') != 798 or contract.get('method_sha256') != sources[METHOD]
            or not required_checks.issubset(contract.get('checks', {})) or any(value is not True for value in contract['checks'].values())
            or contract.get('candidate_permutation', {}).get('all_mismatches_tie_explained') is not True
            or contract.get('rejected_operations') or contract.get('rejected_input_reads')
            or not 0 <= contract.get('configured_pool_threads_total', 999) <= 4):
        raise ValueError('Actual isolated validation contract report did not satisfy required checks')
    if (export.get('questions') != 798 or export.get('choice_agreement') != {'equal': 798, 'n': 798}
            or export.get('group_probabilities_within_tolerance') is not True or export.get('differing_row_indices') != []
            or export.get('rejected_input_reads') or not export.get('members')
            or any(member.get('status') != 'passed' or member.get('scores_outside_tolerance') != 0 for member in export['members'])):
        raise ValueError('Actual full-input export agreement report did not pass')
    predictions, score, resource, environment = (_typed_result(root, reproduction, key) for key in
        ('predictions_result', 'score_result', 'timing_result', 'environment_result'))
    origin_generation = 'research_v2' if selected['historical_fallback'] else 'research_v3'
    origin = f"results/{origin_generation}/validation/{selected['name']}"
    original_files = {origin + '/validation-predictions.json', origin + '/validation-score.json'}
    if not original_files.issubset(reproduction['evidence_files']):
        raise ValueError('Reproduction must bind the canonical original validation outputs')
    original_prediction, original_score = (read(safe(root, origin + '/validation-' + part + '.json')) for part in ('predictions', 'score'))
    if (predictions.get('protocol') != timing.PROTOCOL or predictions.get('split') != 'validation'
            or predictions.get('ids') != original_prediction.get('ids') or predictions.get('selected') != original_prediction.get('selected')
            or len(predictions['ids']) != 798 or len(set(predictions['ids'])) != 798 or len(predictions['selected']) != 798
            or any(type(choice) is not int or not 0 <= choice < 4 for choice in predictions['selected'])):
        raise ValueError('Fresh validation predictions differ from all 798 locked original choices')
    if (score.get('protocol') != timing.PROTOCOL or score.get('split') != 'validation' or score.get('n') != 798
            or type(score.get('correct')) is not int or not 0 <= score['correct'] <= 798
            or not math.isclose(score['accuracy'], 100 * score['correct'] / 798, rel_tol=0., abs_tol=1e-12)
            or any(score.get(key) != original_score.get(key) for key in ('correct', 'n', 'accuracy'))):
        raise ValueError('Fresh official validation score differs from the locked original result')
    processes, observations = resource.get('processes', []), resource.get('scientific_library_observations', [])
    if (resource.get('status') != 'passed' or resource.get('local_resource_eligible') is not True
            or not 0 <= resource.get('prediction_plus_scoring_seconds', 999) <= 30
            or not 0 <= resource.get('sampled_peak_task_process_tree_threads', 999) <= 4
            or resource.get('scientific_library_thread_cap_passed') is not True
            or len(processes) != 2 or any(p.get('returncode') != 0 or p.get('timed_out') is not False for p in processes)
            or len(observations) != 2 or any(row.get('torch_imported') is not False or any(pool.get('num_threads') != 1 for pool in row['thread_pools'])
                                         or sum(pool['num_threads'] for pool in row['thread_pools']) > 4 for row in observations)):
        raise ValueError('Fresh complete prediction/scoring resource evidence did not pass')
    instants = [_instant(processes[0]['started_at_utc']), _instant(processes[0]['completed_at_utc']),
                _instant(processes[1]['started_at_utc']), _instant(processes[1]['completed_at_utc'])]
    if not _instant(selected['created_at_utc']) <= instants[0] <= instants[1] <= instants[2] <= instants[3] <= _instant(resource['completed_at_utc']) <= _instant(reproduction['created_at_utc']):
        raise ValueError('Fresh official phases overlapped or have inconsistent timestamps')
    for process, expected_script in zip(processes, ('code/predict.py', 'code/score_predictions.py')):
        command = process['command']
        if (len(command) < 8 or command[1:3] != ['-S', '-c'] or command[4].replace('\\', '/') != expected_script
                or command.count('--split') != 1 or command[command.index('--split') + 1] != 'validation'):
            raise ValueError('Fresh resources must describe the direct-base official validation commands')
    prefix, checkout = Path(environment['python_prefix']).resolve(), Path(environment['checkout']).resolve()
    packages = environment.get('packages', [])
    if (environment.get('schema') != 'mp2-v3-clean-environment-v1' or environment.get('source_commit') != commit
            or prefix == (Path(root) / '.venv').resolve() or not prefix.is_relative_to((Path(root) / 'tmp').resolve())
            or checkout == Path(root).resolve() or not checkout.is_relative_to((Path(root) / 'tmp').resolve())
            or not Path(environment['python_executable']).resolve().is_relative_to(prefix)
            or not Path(environment['python_executable']).is_file()
            or environment.get('source_files') != sources or environment.get('assets') != assets
            or not any(name.startswith('numpy==') for name in packages)
            or any(name.lower().startswith(('torch==', 'tensorflow==')) for name in packages)):
        raise ValueError('Fresh environment/checkout/package provenance differs')
    check_hashes(root, {str((checkout / name).relative_to(root)).replace('\\', '/'): digest for name, digest in {**sources, **assets}.items()})
    actual_data = sorted(path.relative_to(checkout / 'code/data').as_posix() for path in (checkout / 'code/data').rglob('*') if path.is_file())
    expected_data = {'candidates/gsm8k_validation.jsonl', 'candidates/gsm8k_validation.npz', 'gsm8k_validation.jsonl', 'manifest.json'}
    if not expected_data.issubset(actual_data) or not set(actual_data).issubset(expected_data | {'LICENSE.txt', 'NOTICE.txt'}) or environment.get('official_data_files') != actual_data:
        raise ValueError('Fresh reproduction checkout must contain validation only, with no train/test data')
    if environment.get('data_manifest_sha256') != sha(Path(root) / 'code/data/manifest.json') or sha(checkout / 'code/data/manifest.json') != environment['data_manifest_sha256']:
        raise ValueError('Fresh reproduction used a different official data manifest')
    for row in observations:
        paths = [Path(value).resolve() for value in (row.get('pythonpath') or '').split(os.pathsep) if value]
        if not paths or any(not value.is_relative_to(prefix) for value in paths):
            raise ValueError('Fresh official workers did not use only the new environment package paths')
    for key in ('captured_at_utc',):
        if not _instant(selected['created_at_utc']) <= _instant(environment[key]) <= _instant(reproduction['created_at_utc']):
            raise ValueError('Fresh environment evidence has inconsistent timestamps')
    return review, reproduction


def training_recipes(root, components):
    recipes = []
    for component in components:
        path = component.get('final_fit_path', f"results/research_v2/{component['experiment_id']}/final_fit.json")
        report = read(safe(root, path))
        config = report['config']
        if config['experiment_id'] != component['experiment_id'] or config['seed'] != SEED:
            raise ValueError('Final fit recipe differs from component identity/seed')
        info = report['info']
        epochs = info.get('epochs_run')
        if config['family'] != 'linear' and (type(epochs) is not int or epochs < 1):
            raise ValueError('Final neural epoch count is missing')
        recipes.append({'experiment_id': component['experiment_id'], 'kind': component.get('kind', 'new'),
            'seed': config['seed'], 'epochs': epochs, 'preprocess': config.get('preprocess', 'standard'),
            'decoder': config['decoder'], 'configuration': config,
            'final_fit_path': path, 'final_fit_sha256': sha(safe(root, path)),
            'evaluation_weights': info.get('evaluation_weights', 'live')})
    return recipes


def freeze(root=ROOT, *, verified=None, source_provider=source_hashes, git_check=commit_bytes, commit=None, now_fn=None):
    root, folder = Path(root).resolve(), Path(root).resolve() / 'submission/v3'
    target = folder / 'FREEZE.json'
    if target.exists():
        return verify_freeze(root, git_check=git_check, source_provider=source_provider)
    if (folder / 'TEST_OPENED.json').exists():
        raise ValueError('A new V3 test was already opened; cannot create another pre-test freeze')
    selected, summary, verified, assets, inventory = validate_selected(root, verified)
    sources = source_provider(root)
    check_hashes(root, sources)
    for path, expected in timing.OFFICIAL_SOURCE.items():
        if sources.get(path) != expected:
            raise ValueError('Official parser/predictor/scorer source must be unchanged')
    commit = commit or subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    git_check(root, commit, {**sources, **assets})
    selected_hash = sha(folder / 'SELECTED_MODEL.json')
    frozen_at = (now_fn() if now_fn else datetime.now(timezone.utc)).isoformat()
    review, reproduction = _gates(root, selected_hash, sources, assets, commit, frozen_at)
    manifest = read(root / 'code/data/manifest.json')
    if manifest['protocol'] != timing.PROTOCOL or manifest['splits']['test'] != 467 or manifest['candidates_per_question'] != 4:
        raise ValueError('Unexpected official protocol/full-test count')
    if sha(root / 'code/data/manifest.json') != verified['lock']['data_manifest_sha256']:
        raise ValueError('Original official manifest differs from finalist lock')
    configuration_files = {path.relative_to(root).as_posix(): read(path) for path in sorted((root / CHECKPOINT).rglob('config.json'))}
    record = {'schema': SCHEMA, 'frozen': True, 'test_evaluated_before_freeze': False,
        'created_at_utc': frozen_at, 'source_commit': commit,
        'method': METHOD, 'checkpoint': CHECKPOINT, 'selected_name': selected['name'],
        'selected_model_sha256': selected_hash, 'source_files': sources, 'assets': assets,
        'finalists_lock_sha256': verified['lock_sha256'], 'validation_summary_sha256': sha(root / 'results/research_v3/validation_summary.json'),
        'search_closed_sha256': sha(root / 'results/research_v3/SEARCH_CLOSED.json'),
        'review_result_sha256': sha(folder / 'REVIEW_RESULT.json'), 'reproduction_result_sha256': sha(folder / 'REPRODUCTION_RESULT.json'),
        'data_manifest_sha256': sha(root / 'code/data/manifest.json'), 'protocol': timing.PROTOCOL, 'test_n': 467,
        'expected_test_file_sha256': {name: value for name, value in manifest['sha256'].items() if 'test' in Path(name).name},
        'deployment_seed': SEED, 'selector': selected['selector'], 'oof_directory': selected['oof_directory'],
        'training_recipes': training_recipes(root, selected['components']), 'bundle_configurations': configuration_files,
        'resource': inventory, 'dependency_audit': review['dependency_audit'],
        'permission_basis': 'User explicitly authorized final once-only official evaluation and formal submission in this V3 project; no separate email approval gate.',
        'known_old_test_aggregate': {'correct': 348, 'n': 467, 'known_before_v3': True,
            'scope': 'Historical V1 aggregate already known; this flag refers only to the new V3 evaluation.'},
        'old_submission_preserved': 'https://github.com/xudongwu-0/xudongwu-0.github.io/issues/294',
        'new_test_evaluated': False, 'formal_test_allowed': True,
        'test_policy': 'One fixed final method, full 467 test questions; no test-driven research or tuning; partial runs are not automatically retried.'}
    write_new(target, record)
    return record


def verify_freeze(root=ROOT, *, git_check=commit_bytes, source_provider=source_hashes):
    root, folder = Path(root).resolve(), Path(root).resolve() / 'submission/v3'
    record = read(folder / 'FREEZE.json')
    if record.get('schema') != SCHEMA or record.get('frozen') is not True or record.get('test_evaluated_before_freeze') is not False or record.get('formal_test_allowed') is not True:
        raise ValueError('Valid V3 pre-test freeze required')
    if record['method'] != METHOD or record['checkpoint'] != CHECKPOINT or record['test_n'] != 467 or record['protocol'] != timing.PROTOCOL:
        raise ValueError('Frozen method/checkpoint/full-test protocol differs')
    if source_provider(root) != record['source_files']:
        raise ValueError('Frozen source closure changed')
    check_hashes(root, record['source_files'])
    assets, inventory = asset_hashes(root)
    if assets != record['assets']:
        raise ValueError('Frozen asset inventory changed')
    if record['resource']['uncompressed_bytes'] != inventory['uncompressed_bytes'] or record['resource']['within_limit'] is not True:
        raise ValueError('Frozen resource inventory differs from actual bundle')
    git_check(root, record['source_commit'], {**record['source_files'], **assets})
    bindings = {'submission/v3/SELECTED_MODEL.json': 'selected_model_sha256',
        'submission/v3/REVIEW_RESULT.json': 'review_result_sha256', 'submission/v3/REPRODUCTION_RESULT.json': 'reproduction_result_sha256',
        'results/research_v3/validation_summary.json': 'validation_summary_sha256',
        'results/research_v3/FINALISTS_LOCK.json': 'finalists_lock_sha256',
        'results/research_v3/SEARCH_CLOSED.json': 'search_closed_sha256', 'code/data/manifest.json': 'data_manifest_sha256'}
    check_hashes(root, {path: record[key] for path, key in bindings.items()})
    _gates(root, record['selected_model_sha256'], record['source_files'], assets, record['source_commit'], record['created_at_utc'])
    for recipe in record['training_recipes']:
        check_hashes(root, {recipe['final_fit_path']: recipe['final_fit_sha256']})
    return record


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    with timing.exclusive_evaluation():
        result = freeze()
    print(json.dumps({'frozen': True, 'source_commit': result['source_commit'], 'selected': result['selected_name'],
                      'new_test_evaluated': False, 'path': 'submission/v3/FREEZE.json'}, indent=2))
