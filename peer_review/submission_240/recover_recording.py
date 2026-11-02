"""Recover only the fixed #240 cache-probe postcheck failure; never execute a model."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import math
import os

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
RUNTIME = ROOT / 'tmp/peer_review/submission_240_7e3a2792'
COMMIT = '7e3a2792b7b4442f5e53705a64b4560a2e365de9'
PROTOCOL = 'dase7506-mp2-open-hard-verifier-v5'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(value if isinstance(value, str) else json.dumps(value, indent=2) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def validate_denials(accesses, runtime):
    official = runtime / 'official'
    source = runtime / 'source/code'
    expected_sources = [[official / 'common.py', source / 'submission.py', source / 'features.py'],
                        [official / 'common.py']]
    reference = str(official / 'data/gsm8k_test.jsonl')
    for phase, access, sources in zip(('prediction', 'scoring'), accesses, expected_sources):
        expected = [{'event': 'open', 'path': str(p.parent / '__pycache__' / (p.stem + '.cpython-312.pyc')),
                     'writing': False} for p in sources]
        require(access['phase'] == phase and access['denied'] == expected, 'Unexpected denied operation')
        require(all(not Path(d['path']).exists() for d in expected), 'Cache file must remain absent')
        require(all(str(p) in access['opened_paths'] for p in sources), 'Pinned source not actually opened')
        require((reference in access['opened_paths']) == (phase == 'scoring'), 'Reference-access boundary differs')
        require(all(d['path'] not in access['opened_paths'] for d in expected), 'Denied bytes were recorded as opened')


def validate_processes(processes, start, finish):
    require(len(processes) == 2, 'Exactly the two original processes required')
    times = []
    for process in processes:
        require(type(process['returncode']) is int and process['returncode'] == 0
                and process['timed_out'] is False, 'Original process did not succeed')
        require(process['command'][1:3] == ['-S', '-c'], 'Direct fresh Python required')
        require(process['command'].count('--split') == 1
                and process['command'][process['command'].index('--split') + 1] == 'test', 'Wrong original split')
        require(0 < process['wall_seconds'] < 30 and 0 < process['sampled_peak_tree_threads'] <= 4,
                'Original resource limits not met')
        require(all(v == '1' for v in process['thread_environment'].values()), 'Thread environment changed')
        times += [datetime.fromisoformat(process['started_at_utc']), datetime.fromisoformat(process['completed_at_utc'])]
    initial, final = datetime.fromisoformat(start), datetime.fromisoformat(finish)
    require(initial <= times[0] <= times[1] <= times[2] <= times[3] <= final, 'Original process chronology differs')
    phase_sum = sum(p['wall_seconds'] for p in processes)
    interval = (times[3] - times[0]).total_seconds()
    conservative = (final - initial).total_seconds()
    require(phase_sum < 30 and 0 < interval <= conservative < 30, 'Recorded original elapsed intervals exceed limit')
    return phase_sum, interval, conservative


def main():
    new_names = ('RECOVERED.json', 'timing.json', 'REPRODUCTION_REPORT.md', 'reason.md', 'COMPLETE.json')
    require(not any((OUT / name).exists() for name in new_names), 'Recovery outputs already exist; never overwrite or rerun')
    original = {p.name: sha(p) for p in OUT.iterdir() if p.is_file()}
    failure, start = read(OUT / 'FAILED.json'), read(OUT / 'STARTED.json')
    require(failure['schema'] == 'mp2-peer-reproduction-failed-v1' and failure['status'] == 'failed'
            and failure['error_type'] == 'ValueError' and failure['error'] == 'Peer attempted a prohibited operation',
            'Only the known postcheck failure is recoverable')
    require(start['peer_commit'] == COMMIT and start['submission'] == 240
            and start['target_sha256'] == sha(OUT / 'target.json'), 'Original target changed')
    review = read(OUT / 'independent_cache_denial_review.json')
    require(review['schema'] == 'mp2-peer-cache-denial-independent-review-v1'
            and review['peer_commit'] == COMMIT and review['peer_submission'] == 240
            and all(review[k] is True for k in ('read_only_review', 'no_unexpected_denials',
                                               'no_prediction_reference_access', 'one_blas_thread_each_process')),
            'Independent narrow recovery review missing')
    required_review_inputs = {'FAILED.json', 'STARTED.json', 'prediction-access.json', 'scoring-access.json',
                             'prediction-libraries.json', 'scoring-libraries.json', 'score.json', 'run_guarded_peer.py'}
    require(set(review['input_sha256']) == required_review_inputs, 'Incomplete independent review inputs')
    for name, expected in review['input_sha256'].items():
        require(sha(OUT / name) == expected, 'Independently reviewed evidence changed')
    download = read(OUT / 'download_complete.json')
    require(download['source_commit'] == COMMIT, 'Downloaded commit mismatch')
    expected_inputs = {str(RUNTIME / 'source' / row['path']): row['sha256'] for row in download['files']}
    expected_inputs.update({row['path']: row['sha256'] for row in read(OUT / 'official_files.json')['files']})
    require(start['inputs_sha256'] == expected_inputs, 'Original complete input inventory differs')
    # This streams bytes to SHA only; it does not parse examples, references, embeddings, or errors.
    for name, expected in expected_inputs.items():
        require(sha(name) == expected, 'Original source, asset, or official data bytes changed')
    accesses = [read(OUT / (phase + '-access.json')) for phase in ('prediction', 'scoring')]
    validate_denials(accesses, RUNTIME)
    asset_dir = RUNTIME / 'source/code/runs/model_bundle'
    require({p.name for p in asset_dir.iterdir()} == {'config.json', 'scaler_mean.npy', 'scaler_std.npy', 'weights.npy'},
            'Exact four submitted assets required')
    require({str(p) for p in asset_dir.iterdir()} <= set(accesses[0]['opened_paths']), 'Silent asset fallback detected')
    assets = read(OUT / 'source_asset_manifest.json')
    asset_bytes = sum(p.stat().st_size for p in asset_dir.iterdir())
    require(asset_bytes == assets['total_asset_bytes'] == 86179 and asset_bytes <= 16 * 1024 ** 2,
            'Checkpoint resource inventory differs')
    libraries = [read(OUT / (phase + '-libraries.json')) for phase in ('prediction', 'scoring')]
    for phase, library in zip(('prediction', 'scoring'), libraries):
        require(library['phase'] == phase and library['torch_imported'] is False
                and library['numpy_version'] == '2.5.3' and library['thread_pools']
                and all(p['num_threads'] == 1 for p in library['thread_pools'])
                and sum(p['num_threads'] for p in library['thread_pools']) <= 4
                and library['pythonpath'] == str(RUNTIME / 'venv/Lib/site-packages'), 'Runtime resource observation differs')
    require(libraries[0]['features_module_file'] == str(RUNTIME / 'source/code/features.py'), 'Unexpected imported peer source')
    require(all(read(OUT / 'guard-self-test.json')['checks'].values()), 'Original isolation selfcheck failed')
    require(read(OUT / 'verified_wheels.json')['runtime_package_and_metadata_bytes_match_public_wheels'] is True,
            'Dependency byte verification missing')
    score, prediction = read(OUT / 'score.json'), read(OUT / 'predictions.json')
    require(score['protocol'] == PROTOCOL and score['split'] == 'test' and score['n'] == 467
            and type(score['correct']) is int and 0 <= score['correct'] <= 467
            and math.isclose(score['accuracy'], 100 * score['correct'] / 467, rel_tol=0, abs_tol=1e-12),
            'Original official aggregate contract differs')
    require(prediction['protocol'] == PROTOCOL and prediction['split'] == 'test'
            and len(prediction['ids']) == len(prediction['selected']) == 467 and len(set(prediction['ids'])) == 467
            and all(type(v) is int and 0 <= v < 4 for v in prediction['selected']), 'Original prediction contract differs')
    processes = failure['processes']
    for phase, process in zip(('prediction', 'scoring'), processes):
        script = 'predict.py' if phase == 'prediction' else 'score_predictions.py'
        require(process['command'][4] == str(RUNTIME / 'official' / script), 'Wrong original official command')
    phase_sum, interval, conservative = validate_processes(processes, start['started_at_utc'], failure['completed_at_utc'])
    target = read(OUT / 'target.json')
    delta = score['accuracy'] - target['selected']['score']
    recovery = {'schema': 'mp2-peer-reproduction-recording-recovery-v1', 'status': 'recovered',
                'created_at_utc': datetime.now(timezone.utc).isoformat(), 'submission': 240, 'peer_commit': COMMIT,
                'reason': 'Overbroad postcheck treated four denied nonexistent read-only import-cache probes as peer violations. Independent review confirms the pinned source fallback and no prediction reference access.',
                'original_failed_record_preserved': True, 'original_evidence_sha256': original,
                'all_original_input_sha256_verified': expected_inputs,
                'exact_benign_denials': {phase: access['denied'] for phase, access in zip(('prediction', 'scoring'), accesses)},
                'independent_review_sha256': sha(OUT / 'independent_cache_denial_review.json'),
                'recovery_source_sha256': sha(__file__), 'prediction_scoring_rerun': False,
                'model_changed': False, 'raw_examples_or_errors_analyzed': False,
                'prediction_contract_checked_without_logging_ids_or_choices': True,
                'all_four_checkpoint_assets_actually_opened': True,
                'missing_measurement': 'Original coordinator combined monotonic elapsed was computed but not serialized before its postcheck raised. It is not reconstructed or reported as measured.',
                'original_phase_monotonic_seconds_sum': phase_sum,
                'original_first_worker_start_to_last_worker_end_utc_seconds': interval,
                'original_started_to_failed_utc_seconds': conservative}
    timing = {'schema': 'mp2-peer-timing-v1', 'status': 'complete', 'recovered_from_postcheck_error': True,
              'prediction_plus_scoring_seconds': None, 'combined_monotonic_elapsed_available': False,
              'sum_of_original_process_monotonic_seconds': phase_sum,
              'prediction_plus_scoring_utc_interval_seconds': interval,
              'original_started_to_failed_utc_seconds': conservative, 'shared_original_deadline_seconds': 30,
              'processes': processes, 'scientific_library_observations': libraries,
              'sampled_peak_task_process_tree_threads': max(p['sampled_peak_tree_threads'] for p in processes),
              'scientific_library_thread_cap_passed': True, 'local_resource_eligible': True, 'asset_bytes': asset_bytes,
              'measurement': 'Original fresh direct CPU predictor and independent scorer, sequential under one 30-second deadline. Saved per-process monotonic walls are exact observations; combined saved UTC intervals include the gap. No timing rerun.',
              'limitation': 'Local Windows i9-12900H, not reference Xeon certification. Combined monotonic elapsed was not serialized. Python audit-hook isolation is application-level, not an OS VM. External coordinator threads are reported separately in each original process record.',
              'recovery_file': 'RECOVERED.json', 'completed_at_utc': failure['completed_at_utc']}
    reason = (f'Reproduced submission #240 from immutable commit {COMMIT} (submission-v1), using its submitted checkpoint without retraining. '
              f'The official v5 full-test scorer returned {score["correct"]}/467 = {score["accuracy"]:.11f}%. '
              f'The published declaration is {target["selected"]["score"]}%, a difference of {delta:.11f} percentage points; 394/467 matches the repository README. No misconduct is inferred from this small discrepancy.\n\n'
              f'One fresh CPU predictor plus separate scorer exited successfully: saved phase walls sum to {phase_sum:.6f}s; the saved first-start/last-end UTC interval is {interval:.6f}s. The combined monotonic value was not serialized. Peak sampled task threads were 4, BLAS threads 1, and assets 86,179 bytes. This is a local Windows i9-12900H check, not Xeon certification.\n\n'
              'Prediction was guarded against references, networking and subprocesses. An overbroad postcheck rejected four read-only probes for nonexistent Python bytecode caches; independent review verified pinned source fallback and no reference access. Original FAILED/logs were preserved and existing outputs recovered without rerunning.\n\n'
              'The first pip install timed out. A fresh isolated environment was reconstructed from exact-version local distributions with RECORD/SHA checks, then runtime files were verified byte-for-byte against anonymously downloaded public pinned wheels (NumPy 2.5.3, threadpoolctl 3.7.0, psutil 7.2.2). This was not a fresh PyPI installation. No peer result was used for our model research.')
    require(len(reason) <= 2000, 'Official reason length exceeded')
    for name, expected in original.items():
        require(sha(OUT / name) == expected, 'Original evidence changed during recovery')
    write(OUT / 'RECOVERED.json', recovery)
    write(OUT / 'timing.json', timing)
    write(OUT / 'reason.md', reason + '\n')
    write(OUT / 'REPRODUCTION_REPORT.md', '# Peer submission 240 checkpoint reproduction\n\n' + reason + '\n\n'
          'Evidence: `RECOVERED.json` binds the untouched `FAILED.json`, original process commands, all source/assets/data hashes, access logs, and independent review. `timing.json` retains both original process records. `COMPLETE.json` binds all files. The original command is preserved in `FAILED.json`; the recording-only recovery command is `python -S peer_review/submission_240/recover_recording.py`. Do not replay the once-only predictor. No report has been posted by this helper.\n')
    complete = {'schema': 'mp2-peer-reproduction-complete-v1', 'status': 'complete', 'submission': 240,
                'target_sha256': sha(OUT / 'target.json'), 'peer_commit': COMMIT, 'official_score_file': 'score.json',
                'official_timing_file': 'timing.json', 'prediction_scoring_success': True, 'local_resource_eligible': True,
                'protocol': PROTOCOL, 'split': 'test', 'n': 467, 'correct': score['correct'], 'accuracy': score['accuracy'],
                'published_declared_score': target['selected']['score'], 'difference_percentage_points': delta,
                'matches_repository_readme': score['correct'] == 394, 'recovered_from_recording_error': True,
                'recovery_file': 'RECOVERED.json', 'original_failed_record_preserved': True,
                'model_retrained': False, 'prediction_scoring_rerun': False, 'own_model_research_updated': False,
                'posted': False, 'completed_at_utc': datetime.now(timezone.utc).isoformat(),
                'output_sha256': {p.name: sha(p) for p in OUT.iterdir() if p.is_file()}}
    write(OUT / 'COMPLETE.json', complete)
    print(json.dumps({k: complete[k] for k in ('status', 'submission', 'correct', 'n', 'accuracy', 'difference_percentage_points', 'local_resource_eligible')}))


if __name__ == '__main__':
    main()
