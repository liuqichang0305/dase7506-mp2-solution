"""Independent archive review: saved evidence and hashes only, no inference/test."""
from datetime import datetime, timezone
import ast
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/research_v3/independent-selected-archive-review-final'
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'OMP_THREAD_LIMIT'):
    os.environ[key] = '1'
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / 'code/tools'))
import freeze_v3 as freezer

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads((ROOT / path).read_text(encoding='utf-8'))
def now(): return datetime.now(timezone.utc).isoformat()
def write(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False); stream.write('\n')

denied = []
def guard(event, args):
    if event == 'open' and isinstance(args[0], (str, bytes, os.PathLike)):
        path = Path(os.fsdecode(args[0])).resolve()
        if path.is_relative_to(ROOT / 'code/data'):
            name = path.relative_to(ROOT).as_posix()
            if name not in freezer.timing.TRAIN_PATHS:
                denied.append(name); raise PermissionError('Independent review prohibits holdout data: ' + name)
    if event.startswith('socket.') or event in ('subprocess.Popen', 'os.system', 'os.posix_spawn', 'os.exec'):
        denied.append(event); raise PermissionError('Offline independent review')

def main():
    started, start = now(), time.perf_counter()
    OUT.mkdir(exist_ok=False)
    sys.addaudithook(guard)
    sources = freezer.source_hashes(ROOT)
    selected, summary, verified, assets, inventory = freezer.validate_selected(ROOT)
    closed = read('results/research_v3/SEARCH_CLOSED.json')
    selection = read('results/research_v3/FINALISTS_SELECTION.json')
    opened = read('results/research_v3/VALIDATION_OPENED.json')
    assert opened['lock_sha256'] == verified['lock_sha256'] and opened['test_enabled'] is False
    assert opened['candidate_names'] == [row['name'] for row in verified['lock']['finalists']]
    times = [closed['created_at_utc'], selection['created_at_utc'], verified['lock']['created_at_utc'],
             opened['created_at_utc'], summary['completed_at_utc'], selected['created_at_utc']]
    assert list(map(freezer._instant, times)) == sorted(map(freezer._instant, times))
    freezer.timing.check_hashes(selection['evidence_sha256'])
    assert selected['historical_fallback'] is True and selected['name'] == 'deepsets_regularized_s0'
    assert (summary['baseline_validation']['correct'], summary['baseline_validation']['hard_correct']) == (740, 297)
    assert [(v['validation']['correct'], v['validation']['hard_correct']) for v in summary['finalists']] == [(735, 292), (736, 293)]
    assert len(selection['allowed_final_fit_experiments']) == 2
    for name in ('TEST_OPENED.json', 'TEST_COMPLETE.json', 'test-score.json', 'SUBMISSION_SUCCESS.json'):
        assert not (ROOT / 'submission/v3' / name).exists(), name
    block_sha = freezer.research_archive_block(ROOT, now())
    assert block_sha
    snapshots = read('results/research_v3/closure_evidence/SNAPSHOTS.json')
    assert snapshots['search_closed_sha256'] == sha(ROOT / 'results/research_v3/SEARCH_CLOSED.json')
    snapshot_checks = []
    for current, item in snapshots['evidence'].items():
        frozen = (ROOT / item['snapshot']).read_bytes()
        assert hashlib.sha256(frozen).hexdigest() == item['sha256']
        live = (ROOT / current).read_bytes()
        assert live.startswith(frozen), current
        if current not in ('results/research_v3/training_events.jsonl', 'results/research_v3/round_decisions.jsonl'):
            assert live == frozen, current
        snapshot_checks.append({'path': item['snapshot'], 'sha256': item['sha256'], 'current_bytes_appended': len(live)-len(frozen)})
    queue = read('results/research_v3/campaign_queue.json')['jobs']
    complete = [job for job in queue if job['status'] == 'completed']
    incomplete = [job for job in queue if job['status'] != 'completed']
    assert len(complete) == 35 and len(incomplete) == 4
    assert all(job['status'] == 'pruned' and job['config']['seed'] in (3036900958, 3036900959) for job in incomplete)
    baseline = read('results/research_v3/audit_startup_preservation.json')
    preservation = {'frozen_checks': [], 'prior_v1_92_checks': [], 'historical_inventory': []}
    for section in ('frozen_checks', 'prior_v1_92_checks'):
        for row in baseline[section]:
            digest = sha(ROOT / row['path']); assert digest == row['expected'], row['path']
            preservation[section].append({'path': row['path'], 'sha256': digest})
    for name, row in baseline['historical_inventory'].items():
        path = ROOT / name; digest = sha(path)
        assert digest == row['sha256'] and path.stat().st_size == row['bytes'], name
        preservation['historical_inventory'].append({'path': name, 'sha256': digest})
    assert [len(preservation[k]) for k in preservation] == [35, 92, 829]
    preservation['scope'] = read('results/research_v3/audit_preservation_scope.json')
    preservation['no_score_contents_parsed'] = True
    write(OUT / 'preservation.json', preservation)
    contract_path, export_path = ('results/research_v3/selected-contract-validation.json', 'results/research_v3/selected-export-validation.json')
    contract, export = read(contract_path), read(export_path)
    for report in (contract, export): freezer._audit_identity(ROOT, report, assets)
    required = {'python_int_length_range', 'arbitrary_n_batch_independence', 'id_independence', 'question_order',
        'determinism', 'member_probability_average_and_actual_chosen_confidence', 'parseability_mask_and_all_none_fallback',
        'reject_nonfinite', 'reject_wrong_shape', 'reject_misaligned', 'missing_bundle_rejected',
        'offline_no_references_or_forbidden_access', 'numpy_only_no_torch_import'}
    assert contract['n'] == 798 and contract['method_sha256'] == sources[freezer.METHOD]
    assert required <= set(contract['checks']) and all(value is True for value in contract['checks'].values())
    assert contract['candidate_permutation']['all_mismatches_tie_explained'] is True
    assert not contract['rejected_operations'] and not contract['rejected_input_reads']
    assert contract['configured_pool_threads_total'] == 1
    assert export['questions'] == 798 and export['choice_agreement'] == {'equal': 798, 'n': 798}
    assert export['group_probabilities_within_tolerance'] is True and not export['differing_row_indices']
    assert not export['rejected_input_reads'] and all(v['status'] == 'passed' and v['scores_outside_tolerance'] == 0 for v in export['members'])
    imports = {}
    for name in sorted(freezer.timing.INFERENCE_SOURCE):
        tree = ast.parse((ROOT / name).read_text(encoding='utf-8-sig'))
        imports[name] = sorted({node.module or '' for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)} |
                               {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names})
    assert 'torch' not in sys.modules
    dependency = {'passed': True, 'no_hidden_parameters': True, 'external_runtime_reads': 'bundle_and_inputs_only',
        'inference_dependencies': ['Python 3.12 standard library', 'NumPy 2.5.3'],
        'instrumentation_only': ['threadpoolctl 3.7.0', 'psutil 7.2.2'],
        'training_and_export_reference_only': ['PyTorch 2.15.0+cpu; not imported by actual NumPy inference'],
        'inspection': 'Actual fallback loader reads only config.json and numeric allow_pickle=False weights.npz; model arrays and preprocessing scalers are in the complete audited bundle. Fixed feature/decoder code is included in source_files. Runtime code and installed libraries are separate from model data reads.',
        'observed_runtime_evidence': contract_path, 'static_import_inventory': imports,
        'confidence_analysis': 'Post-selection calibration is analysis only; not loaded by solution_v3 or included as a learned inference asset.'}
    review_detail = {'schema': 'mp2-v3-independent-selected-archive-audit-v1', 'passed': True,
        'started_at_utc': started, 'completed_at_utc': now(), 'wall_seconds': time.perf_counter()-start,
        'source_files': sources, 'assets': assets, 'resource': inventory, 'selection_time_chain': times,
        'snapshot_checks': snapshot_checks, 'completed_configurations': 35, 'completed_outer_folds': 105,
        'incomplete_seed_runs': [job['experiment_id'] for job in incomplete], 'incomplete_seed_runs_claimed_complete': False,
        'selected_is_byte_exact_historical_copy': True, 'formal_submission_blocked_sha256': block_sha,
        'actual_export_choice_agreement': export['choice_agreement'], 'candidate_permutation': contract['candidate_permutation'],
        'no_new_inference_training_scoring_or_test': True, 'raw_holdout_data_read': False, 'rejected_operations': denied,
        'fresh_reproduction_reviewed': False,
        'limitations': ['Fresh checkout reproduction is a separate later gate; this review does not claim it completed.',
            'Local CPU resource evidence is not a certification of the course reference Xeon.',
            'Search stopped for expired budget/deadline; four extra-seed fits are incomplete, not evidence of stability or saturation.',
            'Previously used development validation is not an unbiased held-out generalization estimate.',
            'Preservation checks cover 829 historical inventoried files, 35 frozen entries and 92 prior guards, not all repository files.']}
    assert sources == freezer.source_hashes(ROOT) and assets == freezer.asset_hashes(ROOT)[0] and not denied
    write(OUT / 'review.json', review_detail)
    evidence = [contract_path, export_path, 'submission/v3/SELECTED_MODEL.json', 'submission/v3/FORMAL_SUBMISSION_BLOCKED.json',
        'results/research_v3/SEARCH_CLOSED.json', 'results/research_v3/FINALISTS_SELECTION.json', 'results/research_v3/FINALISTS_LOCK.json',
        'results/research_v3/VALIDATION_OPENED.json', 'results/research_v3/validation_summary.json',
        'results/research_v3/closure_evidence/SNAPSHOTS.json', 'results/research_v3/audit_startup_preservation.json',
        'results/research_v3/audit_preservation_scope.json', 'results/research_v3/audit_selected_archive_v3.py',
        'results/research_v3/independent-selected-archive-review-final/preservation.json',
        'results/research_v3/independent-selected-archive-review-final/review.json']
    evidence.extend(item['snapshot'] for item in snapshots['evidence'].values())
    record = {'schema': 'mp2-v3-review-result-v1', 'passed': True, 'created_at_utc': now(),
        'reviewer': '/root/v2_audit_rules; independent from training and deployment-copy author',
        'selected_model_sha256': sha(ROOT / 'submission/v3/SELECTED_MODEL.json'), 'source_files': sources, 'assets': assets,
        'method': freezer.METHOD, 'checkpoint': freezer.CHECKPOINT, 'evidence_files': {name: sha(ROOT/name) for name in evidence},
        'dependency_audit': dependency,
        'contract_result': {'path': contract_path, 'sha256': sha(ROOT/contract_path)},
        'export_result': {'path': export_path, 'sha256': sha(ROOT/export_path)},
        'research_only': True, 'formal_test_allowed': False, 'fresh_reproduction_is_separate_pending_gate': True}
    freezer.check_hashes(ROOT, record['evidence_files'])
    write(ROOT / 'submission/v3/REVIEW_RESULT.json', record)
    print(json.dumps({'passed': True, 'review_sha256': sha(ROOT/'submission/v3/REVIEW_RESULT.json'),
                      'sources': len(sources), 'asset_bytes': inventory['uncompressed_bytes'], 'preserved': [35,92,829]}))

if __name__ == '__main__': main()
