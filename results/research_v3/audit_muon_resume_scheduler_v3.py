"""Read-only actual-resume audit; hash train inputs, never holdouts or checkpoints by deserialization."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import psutil

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/research_v3'
EXPERIMENT='v3_attention_muon_s0'


def sha(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):result.update(block)
    return result.hexdigest()


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def lines(path):return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.startswith('{')]
def instant(value):return datetime.fromisoformat(value.replace('Z','+00:00')).timestamp()
def iso(value):return datetime.fromtimestamp(value,timezone.utc).isoformat()


def main():
    target=OUT/'audit_muon_resume_scheduler_final.json'
    if target.exists():raise FileExistsError('Preserve prior actual audit')
    queue=read(OUT/'campaign_queue.json')
    job=next(row for row in queue['jobs'] if row['experiment_id']==EXPERIMENT)
    summary=read(OUT/EXPERIMENT/'summary.json')
    registration=read(OUT/EXPERIMENT/'registration.json')
    attempts=job['attempts'];assert len(attempts)==2
    source=attempts[0]['provenance']['source_files']
    data=attempts[0]['provenance']['train_and_fold_files']
    comparisons=[attempt['provenance'] for attempt in attempts]+[job['registered_provenance']]+[
        job['repairs'][0][key] for key in ('old_provenance','new_provenance')]
    checks={
        'eleven_training_sources_same_between_attempts_and_repair':len(source)==11 and all(row['source_files']==source for row in comparisons),
        'train_and_folds_same_between_attempts_and_repair':all(row['train_and_fold_files']==data for row in comparisons),
        'actual_source_hashes_match':all(sha(ROOT/name)==digest for name,digest in source.items()),
        'actual_train_manifest_and_fold_hashes_match':all(sha(ROOT/name)==digest for name,digest in data.items()),
        'summary_and_trainer_registration_sources_match':summary['provenance']['source_files']==registration['source_files']==source,
        'summary_and_registration_data_match':summary['provenance']['data_files']==registration['data_files']=={k:v for k,v in data.items() if k!='results/research_v3/common_folds.json'},
        'summary_and_registration_folds_match':summary['provenance']['folds_sha256']==registration['folds_sha256']==data['results/research_v3/common_folds.json'],
        'config_exact':read(ROOT/job['config_path'])==job['config']==summary['config']==registration['config'],
        'canonical_config_hash_matches':hashlib.sha256(json.dumps(job['config'],sort_keys=True,separators=(',',':')).encode()).hexdigest()==job['config_sha256']==registration['config_sha256'],
        'input_configuration_object_hash_matches_registration':hashlib.sha256(json.dumps(read(ROOT/job['config_path']),sort_keys=True,separators=(',',':')).encode()).hexdigest()==registration['input_config_sha256'],
        'complete_three_folds':summary['status']=='complete' and {row['fold'] for row in summary['folds']}=={0,1,2} and all(row['status']=='complete' and row['n']==1585 for row in summary['folds']),
        'one_documented_repair':job['repair_retries']==1 and len(job['repairs'])==1 and bool(job['repairs'][0]['reason']),
        'first_attempt_cap_then_second_success':attempts[0]['stop_reason']=='attempt_wall_budget' and attempts[0]['returncode']!=0 and attempts[1]['returncode']==0 and attempts[1]['stop_reason'] is None,
        'summary_hash_matches_scheduler':sha(OUT/EXPERIMENT/'summary.json')==job['result']['summary_sha256'],
        'oof_hash_matches_summary_and_scheduler':sha(OUT/EXPERIMENT/'oof.npz')==summary['oof_sha256']==job['result']['oof_sha256'],
    }
    folds=[]
    for row in summary['folds']:
        number=row['fold'];outputs={name:sha(OUT/EXPERIMENT/name)==digest for name,digest in row['output_sha256'].items()}
        assets={name:sha(ROOT/row['checkpoint']/name)==digest for name,digest in row['asset_sha256'].items()}
        states={key:sha(row[key]['checkpoint'])==row[key]['checkpoint_sha256'] for key in ('inner_stopping','fit')}
        folds.append({'fold':number,'output_hashes':outputs,'asset_hashes':assets,'checkpoint_file_hashes':states,
            'inner_resumed_from_epoch':row['inner_stopping']['resumed_from_epoch'],
            'inner_epochs':row['inner_stopping']['epochs_run'],'outer_epochs':row['fit']['epochs_run'],
            'fold_record_equal_summary':read(OUT/EXPERIMENT/f'fold{number}.json')==row,
            'record_modified_at_utc':iso((OUT/EXPERIMENT/f'fold{number}.json').stat().st_mtime),
            'record_predates_second_attempt':(OUT/EXPERIMENT/f'fold{number}.json').stat().st_mtime<instant(attempts[1]['started_at_utc'])})
    checks['all_fold_output_asset_checkpoint_hashes_match']=all(all(row[k].values()) for row in folds for k in ('output_hashes','asset_hashes','checkpoint_file_hashes'))
    log1,log2=[lines(ROOT/a['log']) for a in attempts]
    phases1=[row for row in log1 if 'phase' in row];phases2=[row for row in log2 if 'phase' in row]
    log1_inner=[row for row in phases1 if Path(row['phase']).name=='fold2-inner-epochs.json']
    log2_inner=[row for row in phases2 if Path(row['phase']).name=='fold2-inner-epochs.json']
    checks['second_attempt_only_trains_fold2']=bool(phases2) and all(Path(row['phase']).name.startswith('fold2-') for row in phases2)
    checks['exact_epoch_boundary_in_logged_history']=log1_inner[-1]['epoch']==7 and log2_inner[0]['epoch']==8 and summary['folds'][2]['inner_stopping']['resumed_from_epoch']==7
    checks['optimizer_steps_continue_140_to_160']=log1_inner[-1]['optimizer_steps']==140 and log2_inner[0]['optimizer_steps']==160
    checks['completed_fold0_fold1_reused']=all(row['record_predates_second_attempt'] and row['fold_record_equal_summary'] for row in folds[:2])
    history=read(OUT/EXPERIMENT/'fold2-inner-epochs.json')['history']
    checks['inner_history_keeps_all_epochs_once']=[row['epoch'] for row in history]==list(range(1,17))
    events=lines(OUT/'experiment_registry.jsonl')
    training_events=[row for row in lines(OUT/'training_events.jsonl') if row.get('experiment_id')==EXPERIMENT]
    checks['fold0_fold1_running_and_completed_once']=all(sum(row.get('fold')==fold and row.get('status')==status for row in training_events)==1 for fold in (0,1) for status in ('running','completed'))
    old_finish=next(row for row in reversed(events) if row.get('event')=='queue_run_finished' and row['scheduler_pid']==45096)
    new_start=next(row for row in events if row.get('event')=='attempt_started' and row['scheduler_pid']==25488)
    scheduler=ROOT/'scripts/run_campaign_v3.py';new_hash=sha(scheduler)
    process=psutil.Process(25488)
    checks['scheduler_hash_matches_new_process_logged_attempts']=new_hash=='bf2bac3264db9ca7d8cad4a48bbe30fdceb3b2ee17e5f9eec9b14fd45ae23daa' and all(row['attempt']['scheduler_source_sha256']==new_hash for row in events if row.get('event')=='attempt_started' and row['scheduler_pid']==25488)
    checks['scheduler_edit_and_new_process_follow_old_queue_finish']=instant(old_finish['utc'])<scheduler.stat().st_mtime<process.create_time()<instant(new_start['utc'])
    checks['old_scheduler_pid_not_alive']=not psutil.pid_exists(45096)
    test_bytes=(OUT/'scheduler-fusion-budget-tests.log').read_bytes()
    tests=test_bytes.decode('utf-16' if test_bytes[:2] in (b'\xff\xfe',b'\xfe\xff') else 'utf-8-sig')
    checks['saved_scheduler_suite_reports_20_passes']='Ran 20 tests in 0.908s' in tests and tests.rstrip().endswith('OK')
    fusions=lines(OUT/'fusion_registry.jsonl');fusionchecks=[]
    for row in fusions:
        spec=read(ROOT/row['registration'])
        fusionchecks.append(sha(ROOT/row['registration'])==row['registration_sha256'] and spec['experiment_id']==row['experiment_id'] and spec['status']=='registered' and spec['purpose']=='single_fixed_equal_probability_oof')
    checks['two_unique_bound_fusions_counted']=len(fusions)==2 and len({row['experiment_id'] for row in fusions})==2 and all(fusionchecks)
    budget=read(OUT/'campaign_budget.json')
    checks['absolute_clock_unchanged']=budget['started_at_utc']=='2026-10-29T16:22:42Z' and budget['research_end_utc']=='2026-10-30T16:22:42Z' and budget['maximum_wall_seconds']==86400
    total_process_seconds=sum(a['observed_wall_seconds'] for a in attempts)
    training_seconds=sum(row['fit']['seconds']+row['inner_stopping']['seconds'] for row in summary['folds'])
    checks['cumulative_training_seconds_match']=training_seconds==summary['training_seconds']
    closed={str(p.relative_to(ROOT)):p.exists() for p in [OUT/'VALIDATION_OPENED.json',ROOT/'submission/v3/TEST_OPENED.json',ROOT/'submission/v3/TEST_COMPLETE.json']}
    checks['new_validation_and_test_not_opened']=not any(closed.values())
    report={'checked_at_utc':datetime.now(timezone.utc).isoformat(),'scope':'Read-only actual train-resume/log/source/asset/output hashes and process metadata. No training, tests, checkpoint deserialization or validation/test/ranking inputs.',
        'passed':all(checks.values()),'checks':checks,'folds':folds,
        'attempts':[{k:a[k] for k in ['started_at_utc','ended_at_utc','returncode','stop_reason','observed_wall_seconds','scheduler_source_sha256','log']} for a in attempts],
        'repair_reason':job['repairs'][0]['reason'],'process_wall_seconds_all_attempts':total_process_seconds,
        'cumulative_training_seconds':training_seconds,'process_minus_recorded_training_seconds':total_process_seconds-training_seconds,
        'cost_note':'All attempt process wall time remains counted, including capped/interrupted work and startup/teardown; cumulative trainer timings are a different narrower measurement. The absolute campaign clock was never restarted.',
        'metrics':{k:summary[k] for k in ['hard_correct','hard_n','correct','n']},'logged_resume_boundary':{'prior_epoch':log1_inner[-1]['epoch'],'next_epoch':log2_inner[0]['epoch'],'prior_optimizer_steps':log1_inner[-1]['optimizer_steps'],'next_optimizer_steps':log2_inner[0]['optimizer_steps']},
        'scheduler':{'source_sha256':new_hash,'source_modified_at_utc':iso(scheduler.stat().st_mtime),'old_queue_finish':old_finish,'new_process_pid':process.pid,'new_process_created_at_utc':iso(process.create_time()),'new_process_command':process.cmdline(),'first_new_attempt_utc':new_start['utc'],'logged_source_hash':new_start['attempt']['scheduler_source_sha256'],'unique_configurations':len(queue['jobs'])+len(fusions)+1,'cap':budget['maximum_unique_configurations'],'test_execution':'Read original 20-test passing log; no test rerun by auditor.'},
        'closed_markers':closed,'evidence_sha256':{str(path.relative_to(ROOT)):sha(path) for path in [ROOT/a['log'] for a in attempts]+[OUT/EXPERIMENT/'summary.json',OUT/EXPERIMENT/'registration.json',OUT/EXPERIMENT/'fold2-inner-epochs.json',OUT/'scheduler-fusion-budget-tests.log',ROOT/'code/tests/test_scheduler_v3.py',scheduler,OUT/'campaign_budget.json']},
        'limitations':'Epoch/RNG/optimizer exact restoration implementation was covered by prior synthetic tests. This lightweight actual audit verifies recorded continuation/state-file hashes and unchanged trajectory identity without deserializing or rerunning the completed training.',
        'config_hash_semantics':'Trainer input_config_sha256 is the canonical parsed configuration object hash, not raw JSON bytes. The initial audit wrongly compared raw bytes and its failed assertion is preserved in audit_muon_resume_scheduler_verified.json. Current parsed input, registered configuration and summary are exactly equal; raw-file hash is also retained separately.',
        'raw_config_file_sha256':sha(ROOT/job['config_path'])}
    with target.open('x',encoding='utf-8') as stream:json.dump(report,stream,indent=2);stream.write('\n')
    print(json.dumps({'passed':report['passed'],'failed_checks':[key for key,value in checks.items() if not value],'process_seconds':total_process_seconds,'training_seconds':training_seconds,'metrics':report['metrics'],'source':new_hash,'evidence':str(target)},indent=2))


if __name__=='__main__':main()
