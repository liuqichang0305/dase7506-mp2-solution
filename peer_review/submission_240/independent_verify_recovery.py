"""Read-only verification of recorded peer recovery; never execute peer code."""
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FOLDER = Path(__file__).resolve().parent
RUNTIME = ROOT / 'tmp/peer_review/submission_240_7e3a2792'
PROTOCOL = 'dase7506-mp2-open-hard-verifier-v5'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def read(name):
    return json.loads((FOLDER / name).read_text(encoding='utf-8'))


def check(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    destination = FOLDER / 'independent_recovery_final_review.json'
    check(not destination.exists(), 'Preserve an existing independent review')
    complete, recovery, timing = read('COMPLETE.json'), read('RECOVERED.json'), read('timing.json')
    start, failed = read('STARTED.json'), read('FAILED.json')
    score, predictions = read('score.json'), read('predictions.json')
    complete_hash = sha(FOLDER / 'COMPLETE.json')
    check(complete['schema'] == 'mp2-peer-reproduction-complete-v1' and complete['status'] == 'complete'
          and complete['recovered_from_recording_error'] is True and complete['prediction_scoring_rerun'] is False
          and complete['model_retrained'] is False and complete['own_model_research_updated'] is False
          and complete['posted'] is False, 'Completion scope differs')
    check(complete['peer_commit'] == recovery['peer_commit'] == start['peer_commit']
          == '7e3a2792b7b4442f5e53705a64b4560a2e365de9', 'Peer source commit differs')
    check(recovery['schema'] == 'mp2-peer-reproduction-recording-recovery-v1' and recovery['status'] == 'recovered'
          and recovery['original_failed_record_preserved'] is True and recovery['prediction_scoring_rerun'] is False
          and recovery['model_changed'] is False and recovery['raw_examples_or_errors_analyzed'] is False,
          'Recovery scope differs')
    required_outputs = {'STARTED.json', 'FAILED.json', 'RECOVERED.json', 'timing.json', 'score.json', 'predictions.json',
        'prediction-access.json', 'scoring-access.json', 'prediction-libraries.json', 'scoring-libraries.json',
        'recover_recording.py', 'run_guarded_peer.py', 'reason.md', 'REPRODUCTION_REPORT.md',
        'independent_cache_denial_review.json', 'independent_recovery_source_review.json'}
    outputs = complete['output_sha256']
    check(required_outputs <= set(outputs), 'Required output evidence omitted')
    for name, expected in outputs.items():
        check(Path(name).name == name and sha(FOLDER / name) == expected, 'Completion output hash differs')
    for name, expected in recovery['original_evidence_sha256'].items():
        check(outputs.get(name) == expected and sha(FOLDER / name) == expected, 'Pre-recovery evidence was altered')
    check(recovery['all_original_input_sha256_verified'] == start['inputs_sha256'], 'Input inventories differ')
    for name, expected in start['inputs_sha256'].items():
        path = Path(name).resolve()
        check(path.is_relative_to(RUNTIME), 'Input inventory leaves pinned peer runtime')
        check(sha(path) == expected, 'Pinned source/asset/official-input hash differs')
    source_review = read('independent_recovery_source_review.json')
    check(source_review['source_review_passed'] is True
          and source_review['source_sha256'] == recovery['recovery_source_sha256'] == sha(FOLDER / 'recover_recording.py'),
          'Recovery source differs from the independent pre-execution review')
    check(recovery['independent_review_sha256'] == sha(FOLDER / 'independent_cache_denial_review.json'),
          'Original denial review changed')
    check(complete['target_sha256'] == start['target_sha256'] == sha(FOLDER / 'target.json'), 'Target changed')
    check(score['protocol'] == predictions['protocol'] == complete['protocol'] == PROTOCOL
          and score['split'] == predictions['split'] == complete['split'] == 'test', 'Protocol/split differs')
    ids, choices = predictions['ids'], predictions['selected']
    check(len(ids) == len(choices) == len(set(ids)) == score['n'] == complete['n'] == 467
          and all(isinstance(v, str) and v for v in ids)
          and all(type(v) is int and 0 <= v < 4 for v in choices), 'Saved467-question prediction contract differs')
    check(type(score['correct']) is int and 0 <= score['correct'] <= 467
          and score['correct'] == complete['correct'] and score['accuracy'] == complete['accuracy']
          and math.isclose(score['accuracy'], 100 * score['correct'] / 467, rel_tol=0, abs_tol=1e-12),
          'Saved aggregate arithmetic differs')
    processes = failed['processes']
    check(timing['processes'] == processes and len(processes) == 2, 'Original process evidence changed')
    check(all(p['returncode'] == 0 and p['timed_out'] is False and p['sampled_peak_tree_threads'] <= 4 for p in processes),
          'Original process outcome/resource differs')
    stamps = [datetime.fromisoformat(v) for v in [start['started_at_utc'], processes[0]['started_at_utc'],
        processes[0]['completed_at_utc'], processes[1]['started_at_utc'], processes[1]['completed_at_utc'], failed['completed_at_utc']]]
    check(all(v.tzinfo is not None for v in stamps) and stamps == sorted(stamps), 'Timestamp order/offset differs')
    phase_sum = sum(p['wall_seconds'] for p in processes)
    interval, conservative = (stamps[4] - stamps[1]).total_seconds(), (stamps[-1] - stamps[0]).total_seconds()
    check(timing['prediction_plus_scoring_seconds'] is None and timing['combined_monotonic_elapsed_available'] is False,
          'Missing combined monotonic measurement was fabricated')
    check(timing['sum_of_original_process_monotonic_seconds'] == recovery['original_phase_monotonic_seconds_sum'] == phase_sum
          and timing['prediction_plus_scoring_utc_interval_seconds'] == recovery['original_first_worker_start_to_last_worker_end_utc_seconds'] == interval
          and timing['original_started_to_failed_utc_seconds'] == recovery['original_started_to_failed_utc_seconds'] == conservative
          and 0 < phase_sum < 30 and 0 < interval <= conservative < 30, 'Derived timing scope/arithmetic differs')
    reference = str(RUNTIME / 'official/data/gsm8k_test.jsonl')
    expected_sources = {'prediction': [RUNTIME / 'official/common.py', RUNTIME / 'source/code/submission.py', RUNTIME / 'source/code/features.py'],
                        'scoring': [RUNTIME / 'official/common.py']}
    assets = list((RUNTIME / 'source/code/runs/model_bundle').iterdir())
    for phase in ('prediction', 'scoring'):
        access, library = read(phase + '-access.json'), read(phase + '-libraries.json')
        expected = [{'event': 'open', 'path': str(path.parent / '__pycache__' / (path.stem + '.cpython-312.pyc')),
                     'writing': False} for path in expected_sources[phase]]
        check(access['denied'] == recovery['exact_benign_denials'][phase] == expected
              and all(not Path(item['path']).exists() for item in expected)
              and all(str(path) in access['opened_paths'] for path in expected_sources[phase])
              and (reference in access['opened_paths']) == (phase == 'scoring'), 'Denied-cache/reference classification differs')
        check(library['torch_imported'] is False and library['thread_pools']
              and all(pool['num_threads'] == 1 for pool in library['thread_pools']), 'Scientific thread observations differ')
        if phase == 'prediction':
            check(all(str(path) in access['opened_paths'] for path in assets), 'Checkpoint fallback occurred')
    asset_bytes = sum(path.stat().st_size for path in assets)
    check(asset_bytes == read('source_asset_manifest.json')['total_asset_bytes'] == timing['asset_bytes']
          and asset_bytes <= 16 * 1024**2, 'Asset-byte count differs')
    check(sha(FOLDER / 'COMPLETE.json') == complete_hash, 'Completion changed during review')
    record = {'schema': 'mp2-peer-recovery-independent-final-review-v1', 'reviewed_at_utc': datetime.now(timezone.utc).isoformat(),
        'passed': True, 'peer_submission': 240, 'peer_commit': complete['peer_commit'],
        'complete_sha256': complete_hash, 'recovery_sha256': sha(FOLDER / 'RECOVERED.json'),
        'timing_sha256': sha(FOLDER / 'timing.json'), 'reviewer_source_sha256': sha(__file__),
        'verified_input_files': len(start['inputs_sha256']), 'verified_output_files': len(outputs),
        'verified_preserved_original_files': len(recovery['original_evidence_sha256']),
        'prediction_contract': {'n': 467, 'unique_nonempty_string_ids': 467, 'legal_integer_choices': 467,
                                'protocol_and_split_match': True, 'individual_ids_and_choices_logged': False},
        'score_aggregate': {key: score[key] for key in ('protocol', 'split', 'correct', 'n', 'accuracy')},
        'timing': {'combined_monotonic_available': False, 'combined_monotonic_seconds': None,
                   'saved_phase_monotonic_sum_seconds': phase_sum, 'saved_first_to_last_utc_seconds': interval,
                   'saved_start_to_failure_utc_seconds': conservative, 'sampled_task_threads': timing['sampled_peak_task_process_tree_threads'],
                   'one_blas_thread_each_process': True, 'asset_bytes': asset_bytes},
        'original_failed_record_preserved': True, 'exact_four_cache_denials_verified': True,
        'no_prediction_reference_access': True, 'all_submitted_assets_actually_opened': True,
        'scope': 'Independent existing-evidence verification only. Official inputs, source and assets were streamed solely for SHA checks. Only saved prediction IDs/choice legality and aggregate score/timing were parsed; no raw examples, references, per-question correctness or error patterns were analyzed.',
        'no_worker_or_scorer_rerun': True, 'no_source_or_model_change': True, 'no_network_or_publication': True,
        'limitation': 'Local sampled resource evidence is not reference-hardware certification; UTC elapsed scopes do not recover the unavailable original combined monotonic timer. Completion/output hashes bind bytes at their recorded recovery/review times, not a claimed earlier unavailable output-hash receipt.'}
    with destination.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(record, stream, indent=2); stream.write('\n')
    print(json.dumps({'passed': True, 'review': str(destination.relative_to(ROOT)), 'input_files_verified': record['verified_input_files'],
        'output_files_verified': record['verified_output_files'], 'n_unique_ids_and_legal_choices': 467,
        'combined_monotonic_seconds': None, 'phase_monotonic_sum': phase_sum, 'first_to_last_utc_interval': interval}, indent=2))


if __name__ == '__main__':
    main()
