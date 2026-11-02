"""Read registered train-CV evidence/configurations only; no model evaluation."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(relative):
    return json.loads((ROOT/relative).read_text(encoding='utf-8'))


def differences(a, b, prefix=''):
    result=[]
    for key in sorted(set(a)|set(b)):
        if key=='experiment_id':
            continue
        path=prefix+key
        if isinstance(a.get(key),dict) and isinstance(b.get(key),dict):
            result.extend(differences(a[key],b[key],path+'.'))
        elif a.get(key)!=b.get(key):
            result.append({'field':path,'control':a.get(key),'variant':b.get(key)})
    return result


def main():
    output=ROOT/'results/research_v3/audit_research_coverage_20261030.json'
    if output.exists():
        raise FileExistsError('Preserve original review evidence')
    planned=[]
    controls={'r05_tabm_optimization_features':'v3_tabm_k4_s0',
              'r06_attention_objectives':'v3_attention_adamw_s0',
              'r09_attention_regularization_capacity':'v3_attention_adamw_s0',
              'r10_realmlp_resolution':'v3_realmlp_robust_all_s0'}
    for name in [*list(controls)[:2],'r07_regularized_interactions',*list(controls)[2:]]:
        relative=f'docs/research_v3/rounds/{name}.json'
        specification=read(relative)
        configurations=specification['configs']
        variants=[]
        for cfg in configurations:
            if name=='r07_regularized_interactions':
                control=next(c for c in configurations if c['family']==cfg['family'] and 'ema_decay' not in c)
                interpretation='within-family EMA contrast; baseline family transfer is not a capacity-matched architecture causal comparison'
            else:
                control=read(f'code/configs/research_v3/{controls[name]}.json')
                interpretation='one registered factor; coupled loss and decoder alpha form the explicit alignment factor'
            variants.append({'experiment_id':cfg['experiment_id'],'control':control['experiment_id'],
                'seed':cfg['seed'],'changes':differences(control,cfg),'interpretation':interpretation})
        planned.append({'round':name,'file':relative,'sha256':sha(ROOT/relative),
            'configuration_count':len(configurations),'two_to_four':2<=len(configurations)<=4,
            'all_fixed_seed':all(c['seed']==3036900957 for c in configurations),'variants':variants})
    queue=read('results/research_v3/campaign_queue.json')
    actual=[]
    for job in queue['jobs']:
        item={'experiment_id':job['experiment_id'],'round':job['round'],'status':job['status'],
              'seed':job['config']['seed'],'repair_retries':job['repair_retries']}
        if job['status']=='completed':
            relative=f"results/research_v3/{job['experiment_id']}/summary.json"
            summary=read(relative)
            item.update(summary_sha256=sha(ROOT/relative),
                complete_three_folds=summary['status']=='complete' and len(summary['folds'])==3
                    and {r['fold'] for r in summary['folds']}=={0,1,2}
                    and all(r['status']=='complete' and r['n']==1585 for r in summary['folds']),
                config_matches_registration=summary['config']==job['config'],
                oof_hash_matches=sha(ROOT/f"results/research_v3/{job['experiment_id']}/oof.npz")==summary['oof_sha256'],
                hard_correct=summary['hard_correct'],hard_n=summary['hard_n'],
                hard_fold_mean=summary['hard_accuracy_fold_mean'],correct=summary['correct'],n=summary['n'])
        actual.append(item)
    historical=read('results/research_v3/analysis/r03_complete/analysis.json')
    reused=[{'id':r['experiment_id'],'seed':r['seed'],'hard_correct':r['hard_correct'],
             'hard_fold_mean':r['hard_fold_mean'],'source_summary_sha256':r['summary_sha256']}
            for r in historical['rows'] if r['version']=='research_v2']
    fusion=read('results/research_v3/fixed_fusion_analysis.json')
    record={'checked_at_utc':datetime.now(timezone.utc).isoformat(),
        'scope':'Read-only train-CV configurations and existing aggregate/OOF hashes; no validation/test/ranking reads or training.',
        'planned_rounds':planned,'queue_snapshot_updated_at':queue['updated_at_utc'],'registered_jobs':actual,
        'historical_common_fold_methods':reused,
        'fusion':{'status':fusion['status'],'configurations':fusion['configurations'],
                  'sha256':sha(ROOT/'results/research_v3/fixed_fusion_analysis.json'),
                  'hard_correct':[r['metrics']['hard']['correct'] for r in fusion['results']]},
        'counter_total_if_all_current_and_planned_complete':len(actual)+sum(r['configuration_count'] for r in planned)+1+2,
        'counter_note':'Includes one timing pilot and two fixed fusion configurations; excludes already completed historical V2 comparisons.',
        'saturation_currently_justified':False,
        'reason':'R04 remains incomplete at review and five documented feasible follow-up hypotheses exist. A zero-gain sequence alone does not meet no-feasible-hypothesis condition.',
        'minimum_stability_recommendation':{'seeds':[3036900958,3036900959],
            'configurations': '2 for one prospective V3 finalist; 4 for two distinct prospective V3 finalists, after broad follow-up results.',
            'deployment_seed':3036900957,
            'reuse':'Do not rerun historical same-recipe complete seeds; do not use extra seeds as alternate deployment candidates.'},
        'stopping_rule':'After mandatory mechanism/optimizer/interaction/TabM-or-RealMLP coverage, three consecutive completed, distinct hypothesis rounds each fail to exceed the fixed prior incumbent by 0.5 common-fold hard percentage points AND no feasible evidence-backed hypothesis remains; otherwise stop only for absolute budget/cap/buffer/real constraint. Stability replications and retries do not manufacture distinct hypotheses.',
        'precision':'Use the preregistered fold-mean metric, not pooled accuracy; about 11 hard questions is only an approximation, not the exact threshold. Chronology uses actual completed-round timestamps, not numeric round labels.'}
    with output.open('x',encoding='utf-8') as stream:
        json.dump(record,stream,indent=2);stream.write('\n')
    print(json.dumps({'output':str(output),'planned_configurations':sum(r['configuration_count'] for r in planned),
        'registered_jobs':len(actual),'completed_jobs':sum(r['status']=='completed' for r in actual),
        'completed_checks_pass':all(all(r.get(k,True) for k in ['complete_three_folds','config_matches_registration','oof_hash_matches']) for r in actual),
        'total_count':record['counter_total_if_all_current_and_planned_complete']},indent=2))


if __name__=='__main__':main()
