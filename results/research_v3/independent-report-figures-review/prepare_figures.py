"""Plot final saved CV/confidence evidence only; no raw banks, scoring or PDF.

Production has four fixed JSON inputs. Missing/incomplete evidence fails before
creating the final figure directory. Temporary synthetic verification is an
internal Python API only and is explicitly watermarked and recorded.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics

ROOT = Path(__file__).resolve().parents[2]
GENERATOR = 'reports/research_v3/prepare_figures.py'
SEED = 3036900957
INPUTS = {
    'analysis':'results/research_v3/analysis/final_search/analysis.json',
    'queue':'results/research_v3/campaign_queue.json',
    'fit':'results/research_v3/confidence_calibration/report.json',
    'validation':'results/research_v3/confidence_validation/report.json',
}
OUTPUT = 'reports/research_v3/figures'
SIZE = (7.1, 3.26)
DPI = 300
HEIGHT = 235


def require(condition, message):
    if not condition:raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def finite(value):
    require(type(value) in (int,float) and math.isfinite(value),'Finite numeric plot evidence is required')
    return float(value)


def integer(value, minimum=0):
    require(type(value) is int and value>=minimum,'Integer evidence count is required')
    return value


def write_new(path,value):
    with Path(path).open('x',encoding='utf-8') as stream:
        json.dump(value,stream,indent=2,ensure_ascii=False,allow_nan=False);stream.write('\n')


def load_inputs(root):
    root=Path(root).resolve()
    paths={key:root/name for key,name in INPUTS.items()}
    for name,path in paths.items():
        require(path.is_file(),'Final figure evidence is not ready: '+INPUTS[name])
        require(not path.is_symlink() and path.resolve()==path.absolute(),'Figure inputs must be fixed local files, not links')
    # No paths inside a report are followed, and no NPZ, raw bank, prediction,
    # reference, leaderboard or test-result file is opened.
    values={key:json.loads(path.read_text(encoding='utf-8-sig')) for key,path in paths.items()}
    return values,{INPUTS[key]:digest(path) for key,path in paths.items()}


def cv_data(analysis,queue):
    require(queue.get('schema')=='mp2-campaign-queue-v3' and isinstance(queue.get('jobs'),list),'Final campaign queue schema differs')
    jobs=queue['jobs'];names=[job['experiment_id'] for job in jobs]
    require(len(names)==len(set(names)) and jobs,'Final campaign jobs must be nonempty and unique')
    require(all(job['status'] in ('completed','failed','pruned') for job in jobs),'Final queue must be terminal; running/pending work cannot be plotted as final')
    require(re.fullmatch('[0-9a-f]{64}',str(analysis.get('folds_sha256'))) is not None,'Common-fold analysis provenance is missing')
    require(re.fullmatch('[0-9a-f]{64}',str(analysis.get('source_sha256'))) is not None,'Analysis source provenance is missing')
    rows=analysis.get('rows');require(isinstance(rows,list) and rows,'Saved final CV table is missing')
    require(len({(r['version'],r['experiment_id']) for r in rows})==len(rows),'Duplicate analysis method/seed rows')
    complete=[];excluded=[]
    for row in rows:
        if row.get('status')!='completed' or row.get('n')!=4755 or len(row.get('hard_fold_accuracies',[]))!=3:
            excluded.append({'version':row.get('version'),'experiment_id':row.get('experiment_id'),'status':row.get('status')})
            continue
        require(row['version'] in ('research_v2','research_v3'),'Unexpected method generation')
        require(type(row['seed']) is int and row['seed'] in (SEED,SEED+1,SEED+2),'Unregistered primary/extra seed')
        require(isinstance(row.get('family'),str) and row['family'],'Method family is missing')
        require(isinstance(row.get('config'),dict) and row['config'].get('seed')==row['seed']
            and row['config'].get('family')==row['family'] and row['config'].get('experiment_id')==row['experiment_id'],
            'Plotted method identity/seed differs from its saved configuration')
        count=integer(row['hard_correct']);n=integer(row['hard_n'],1)
        require(n<=4755 and count<=n,'Invalid hard-subset counts')
        accuracy=finite(row['hard_accuracy'])
        require(math.isclose(accuracy,100*count/n,rel_tol=0,abs_tol=1e-10),'Hard accuracy differs from pooled saved counts')
        folds=[finite(x) for x in row['hard_fold_accuracies']]
        require(all(0<=x<=100 for x in folds),'Fold accuracy must be a percent')
        seconds=finite(row['training_seconds'])
        require(seconds>0,'Every complete method needs positive recorded fit time for the log axis; no imputation')
        require(all(re.fullmatch('[0-9a-f]{64}',str(row.get(key))) is not None for key in ('summary_sha256','oof_sha256')),
                'Completed row requires saved summary/OOF provenance')
        complete.append({**row,'training_seconds':seconds,'hard_accuracy':accuracy})
    require(complete,'No complete common-three-fold methods remain')
    require(len({row['hard_n'] for row in complete})==1,'Common-fold methods disagree on the hard-subset size')
    completed={job['experiment_id'] for job in jobs if job['status']=='completed'}
    require(completed=={r['experiment_id'] for r in complete if r['version']=='research_v3'},
            'Final analysis must contain every and only completed V3 campaign job')
    index={row['experiment_id']:row for row in complete if row['version']=='research_v3'}
    for job in jobs:
        if job['status']=='completed':
            require(index[job['experiment_id']].get('config')==job.get('config'),'Final plotted configuration differs from the registered queue')
    expected_pending={job['experiment_id']:job['status'] for job in jobs if job['status']!='completed'}
    actual_pending={row['experiment_id']:row['status'] for row in analysis.get('noncomplete_jobs',[])}
    require(actual_pending==expected_pending,'Incomplete-job disclosure differs from the terminal campaign queue')
    attempts=[a for job in jobs for a in job.get('attempts',[])]
    failures={'failed_jobs':sum(j['status']=='failed' for j in jobs),
        'pruned_jobs':sum(j['status']=='pruned' for j in jobs),
        'nonzero_exit_attempts':sum(type(a.get('returncode')) is int and a['returncode']!=0 for a in attempts),
        'interrupted_attempts':sum(a.get('status')=='interrupted_process_missing' for a in attempts),
        'excluded_partial_rows':excluded,'total_recorded_attempts':len(attempts),
        'note':'Failed/interrupted work is disclosed separately and is not imputed into complete-method fit times.'}
    return complete,failures


def confidence_metric(metric,n):
    require(metric.get('n')==n,'Confidence report has incomplete question coverage')
    correct=integer(metric['correct']);require(correct<=n,'Invalid confidence event count')
    require(math.isclose(finite(metric['accuracy']),correct/n,rel_tol=0,abs_tol=1e-12),'Confidence accuracy/count mismatch')
    for key in ('brier','ece_10_equal_width','aurc_discrete_mean'):
        require(0<=finite(metric[key])<=1,'Invalid confidence diagnostic')
    bins=metric['bins'];require(len(bins)==10,'All ten equal-width bins must be preserved')
    bin_n=0;bin_correct=0.;ece=0.
    for i,row in enumerate(bins):
        require(row['bin']==i and row['lower']==i/10 and row['upper']==(i+1)/10
            and row['upper_inclusive']==(i==9),'Confidence bin definition changed')
        count=integer(row['n']);bin_n+=count
        if not count:
            require(all(row[key] is None for key in ('accuracy','mean_confidence','absolute_gap')),'Empty bins cannot contain invented points')
            continue
        accuracy=finite(row['accuracy']);mean=finite(row['mean_confidence']);gap=finite(row['absolute_gap'])
        require(0<=accuracy<=1 and row['lower']<=mean<=row['upper'] and (i==9 or mean<row['upper']),
                'Bin empirical accuracy/confidence is out of range')
        require(math.isclose(gap,abs(mean-accuracy),rel_tol=0,abs_tol=1e-12),'Saved bin gap differs')
        bin_correct+=count*accuracy;ece+=count/n*gap
    require(bin_n==n and math.isclose(bin_correct,correct,rel_tol=0,abs_tol=1e-8),'Bin counts/outcomes disagree with selected events')
    require(math.isclose(ece,metric['ece_10_equal_width'],rel_tol=0,abs_tol=1e-12),'ECE differs from saved bins')
    curve=metric['risk_coverage'];retained=curve['retained'];coverage=curve['coverage'];risk=curve['risk'];errors=curve['errors']
    require(retained==list(range(1,n+1)) and all(len(x)==n for x in (coverage,risk,errors,curve['ordered_original_rows'])),
            'The complete saved risk-coverage curve is required')
    require(sorted(curve['ordered_original_rows'])==list(range(n)),'Risk-coverage row order is not a permutation')
    last=0
    for k,c,value,e in zip(retained,coverage,risk,errors):
        require(type(e) is int and e-last in (0,1),'Cumulative errors must be actual binary event counts')
        require(math.isclose(finite(c),k/n,rel_tol=0,abs_tol=1e-12)
            and math.isclose(finite(value),e/k,rel_tol=0,abs_tol=1e-12),'Saved risk-coverage values disagree with counts')
        last=e
    require(last==n-correct and math.isclose(statistics.fmean(risk),metric['aurc_discrete_mean'],rel_tol=0,abs_tol=1e-12),
            'Full-coverage risk/AURC differs from the fixed selected outcomes')
    return metric


def confidence_data(fit,validation):
    for report,split,n in ((fit,'train_oof',4755),(validation,'validation',798)):
        require(report.get('schema')=='mp2-v3-confidence-report-v1' and report.get('split')==split,'Missing completed confidence report: '+split)
        require(report.get('choices_changed')==0 and isinstance(report.get('selector'),dict) and report['selector'],
                'Confidence analysis must preserve the selected method and choices')
        require(re.fullmatch('[0-9a-f]{64}',str(report.get('events_sha256'))) is not None
            and re.fullmatch('[0-9a-f]{64}',str(report.get('analysis_source_sha256'))) is not None,
            'Confidence report source/event provenance is missing')
        require(isinstance(report.get('source_sha256'),dict) and report['source_sha256']
            and all(re.fullmatch('[0-9a-f]{64}',str(x)) for x in report['source_sha256'].values()),'Invalid confidence source hashes')
        for kind in ('raw','calibrated'):confidence_metric(report[kind],n)
        require(report['raw']['correct']==report['calibrated']['correct'],'Calibration changed the chosen-answer event')
    require(fit['selector']==validation['selector'],'Fit and validation confidence refer to different selectors')
    require('IN-SAMPLE' in fit.get('interpretation',''),'Training-OOF calibration must explicitly disclose fitting optimism')
    require('selected_model_json' in fit['source_sha256'] and 'calibration_json' in validation['source_sha256'],
            'Final confidence reports must bind unique selection and the fitted calibration map')
    return fit,validation


def plotting():
    # Keep matplotlib's font/config cache inside the writable project workspace.
    cache=ROOT/'tmp/research_v3/matplotlib-cache'
    cache.mkdir(parents=True,exist_ok=True)
    os.environ['MPLCONFIGDIR']=str(cache)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter,MaxNLocator
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7.2,'axes.titlesize':8.0,
        'axes.labelsize':7.5,'xtick.labelsize':6.8,'ytick.labelsize':6.8,'legend.fontsize':6.7,
        'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white',
        'axes.edgecolor':'#56626D','axes.linewidth':.6,'grid.color':'#D8DEE4','grid.linewidth':.4})
    return plt,PercentFormatter,MaxNLocator,matplotlib.__version__


def watermark(fig,synthetic):
    if synthetic:fig.text(.5,.51,'SYNTHETIC FIXTURE',ha='center',va='center',fontsize=22,color='#B32929',alpha=.19,rotation=12)


def cost_figure(rows,failures,path,synthetic=False):
    plt,_,MaxNLocator,version=plotting()
    fig,ax=plt.subplots(figsize=SIZE)
    fig.subplots_adjust(left=.095,right=.98,bottom=.25,top=.79)
    for generation,color in (('research_v2','#497494'),('research_v3','#C5692A')):
        for primary,marker in ((True,'o'),(False,'^')):
            group=[r for r in rows if r['version']==generation and (r['seed']==SEED)==primary]
            if group:ax.scatter([r['training_seconds'] for r in group],[r['hard_accuracy'] for r in group],
                s=32 if primary else 35,marker=marker,color=color,edgecolors='white',linewidths=.4,alpha=.83,
                label=generation.replace('research_','').upper()+(' fixed seed' if primary else ' extra seeds'),zorder=3)
    ax.set_xscale('log');ax.grid(True,axis='both',alpha=.8)
    ax.set_xlabel('Recorded fit wall time across three CV folds (seconds; log scale)')
    ax.set_ylabel('Hard-subset pooled accuracy (%)')
    ax.yaxis.set_major_locator(MaxNLocator(5))
    fig.text(.095,.95,'Complete common-three-fold train OOF methods',fontweight='bold',fontsize=9.0)
    handles,labels=ax.get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper left',bbox_to_anchor=(.088,.93),ncol=4,frameon=False,
               borderpad=.3,columnspacing=1.2,handletextpad=.35,fontsize=6.5)
    fig.text(.095,.115,f"{len(rows)} complete method/seed rows; fixed seed {SEED}; extra seeds {SEED+1}, {SEED+2}.",fontsize=6.9)
    fig.text(.095,.068,f"Excluded: {failures['failed_jobs']} failed jobs, {failures['pruned_jobs']} pruned jobs, {len(failures['excluded_partial_rows'])} partial rows. "
             f"Recorded nonzero exits: {failures['nonzero_exit_attempts']}; interruptions: {failures['interrupted_attempts']}.",fontsize=6.65)
    fig.text(.095,.023,'Fit cost is not inference latency. Failed/interrupted cost is not imputed; OOF is used for model selection.',fontsize=6.65,color='#4A5662')
    watermark(fig,synthetic);fig.savefig(path,dpi=DPI);plt.close(fig)
    return version


def confidence_figure(fit,validation,path,synthetic=False):
    plt,PercentFormatter,MaxNLocator,version=plotting()
    fig,axes=plt.subplots(2,2,figsize=SIZE)
    fig.subplots_adjust(left=.075,right=.985,bottom=.15,top=.86,wspace=.28,hspace=.63)
    fig.suptitle('Confidence in the actual chosen-answer event; choices unchanged',fontsize=9.0,fontweight='bold',y=.985)
    for row,(report,title) in enumerate(((fit,'Train OOF: map fitted here (optimistic)'),
                                        (validation,'Validation: already-used holdout'))):
        reliability,risk=axes[row]
        reliability.plot([0,1],[0,1],color='#9AA4AD',linewidth=.7,linestyle=':',zorder=1)
        for kind,color,marker,style in (('raw','#235C83','o','-'),('calibrated','#C56729','s','--')):
            metric=report[kind];bins=[b for b in metric['bins'] if b['n']]
            x=[b['mean_confidence'] for b in bins];y=[b['accuracy'] for b in bins]
            reliability.plot(x,y,color=color,linewidth=.9,linestyle=style,zorder=2)
            reliability.scatter(x,y,s=[6+90*b['n']/metric['n'] for b in bins],color=color,marker=marker,
                linewidths=.3,edgecolors='white',zorder=3,label=f"{kind.capitalize()} ECE {metric['ece_10_equal_width']:.3f}")
            curve=metric['risk_coverage']
            risk.plot(curve['coverage'],curve['risk'],color=color,linewidth=1.0 if kind=='raw' else .9,linestyle=style,
                      label=f"{kind.capitalize()} AURC {metric['aurc_discrete_mean']:.3f}")
        reliability.set(xlim=(-.015,1.015),ylim=(-.035,1.035),ylabel='Observed accuracy')
        reliability.set_title(title,loc='left',fontsize=7.4,pad=3)
        reliability.set_xticks([0,.5,1]);reliability.set_yticks([0,.5,1])
        reliability.legend(loc='lower right',fontsize=5.9,framealpha=.85,borderpad=.25,labelspacing=.2,handletextpad=.3)
        risk.set(xlim=(0,1),ylabel='Selective risk')
        risk.set_title(f"Risk–coverage (n={report['raw']['n']:,})",loc='left',fontsize=7.4,pad=3)
        risk.set_ylim(bottom=0);risk.yaxis.set_major_locator(MaxNLocator(3));risk.set_xticks([0,.5,1])
        risk.yaxis.set_major_formatter(PercentFormatter(1,decimals=0))
        risk.legend(loc='upper right',fontsize=5.9,framealpha=.85,borderpad=.25,labelspacing=.2,handlelength=2)
        reliability.grid(True,alpha=.7);risk.grid(True,alpha=.7)
        if row==1:
            reliability.set_xlabel('Mean predicted confidence');risk.set_xlabel('Retained coverage')
    fig.text(.075,.025,'Marker area reflects within-set bin frequency; empty bins omitted. Full curves and saved stable tie order retained.',fontsize=6.35,color='#4A5662')
    watermark(fig,synthetic);fig.savefig(path,dpi=DPI);plt.close(fig)
    return version


def prepare(root=ROOT,*,synthetic=False):
    root=Path(root).resolve()
    require(not synthetic or root!=ROOT.resolve() and 'synthetic' in str(root).lower(),
            'Synthetic fixtures must use a distinct explicitly named synthetic directory')
    require(synthetic or root==ROOT.resolve(),'Production figures use the fixed repository only')
    values,hashes=load_inputs(root)
    rows,failures=cv_data(values['analysis'],values['queue'])
    fit,validation=confidence_data(values['fit'],values['validation'])
    source=root/GENERATOR
    require(source.is_file() and digest(source)==digest(__file__),'Figure generator source must match the running implementation')
    hashes[GENERATOR]=digest(source)
    output=root/OUTPUT
    require(not output.exists(),'Preserve existing figure output and generation records before creating a new reviewed build')
    output.mkdir(parents=True,exist_ok=False)
    scope='synthetic_fixture_only' if synthetic else 'saved_final_evidence_only'
    figures=[]
    definitions=[('cv_fit_cost_vs_hard_accuracy.png',cost_figure,(rows,failures),
        'All complete common-three-fold train-OOF methods and fixed extra seeds. The log x-axis is recorded fitting wall time, not inference latency; failed/pruned work is disclosed separately. OOF results were used for model selection.'),
        ('chosen_event_confidence.png',confidence_figure,(fit,validation),
        'Confidence describes the actual chosen answer. Train-OOF calibration is evaluated on its fitting outcomes and is optimistic. Validation is an already-used holdout. Calibration changes no choices; curves are descriptive, including stable within-tie ordering.')]
    try:
        for filename,renderer,args,caption in definitions:
            path=output/filename;version=renderer(*args,path,synthetic=synthetic)
            subset={key:value for key,value in hashes.items() if key==GENERATOR or key in
                ([INPUTS['analysis'],INPUTS['queue']] if filename.startswith('cv_') else [INPUTS['fit'],INPUTS['validation']])}
            record={'schema':'mp2-v3-figure-generation-v1','scope':scope,'created_at_utc':datetime.now(timezone.utc).isoformat(),
                'path':path.relative_to(root).as_posix(),'sha256':digest(path),'source_sha256':subset,
                'generator_sha256':hashes[GENERATOR],'matplotlib_version':version,'format':'PNG','dpi':DPI,
                'figure_inches':list(SIZE),'report_height_points':HEIGHT,'caption':caption,
                'raw_bank_or_test_read':False,'visual_qa_status':'pending_render_inspection'}
            if filename.startswith('cv_'):
                record.update(plotted_rows=[{k:r[k] for k in ('version','experiment_id','family','seed','training_seconds','hard_accuracy','hard_correct','hard_n')} for r in rows],
                    exclusions=failures,cost_semantics='Saved training_seconds, wall-clock fit cost across three completed CV folds; not inference latency.')
            else:
                record.update(selector=fit['selector'],event_semantics='Actual selected-answer correctness; choices_changed=0',
                    sets={name:{'split':r['split'],'interpretation':r['interpretation'],**{kind:{key:r[kind][key] for key in
                        ('n','correct','brier','ece_10_equal_width','aurc_discrete_mean','bins')} for kind in ('raw','calibrated')}}
                        for name,r in (('train_oof_fit',fit),('used_validation',validation))})
            record_path=path.with_suffix('.generation.json');write_new(record_path,record)
            figures.append({'path':record['path'],'sha256':record['sha256'],'height':HEIGHT,'caption':caption,
                'generation_record':{'path':record_path.relative_to(root).as_posix(),'sha256':digest(record_path)}})
        # Detect saved report/queue changes during rendering without reopening any
        # raw data or following evidence references.
        require(all(digest(root/name)==expected for name,expected in hashes.items()),'Figure inputs changed during rendering')
        manifest={'schema':'mp2-v3-report-figures-v1','scope':scope,'created_at_utc':datetime.now(timezone.utc).isoformat(),
            'evidence_sha256':hashes,'figures':figures,'partial_rows_excluded':len(failures['excluded_partial_rows']),
            'no_pdf_generated':True,'no_raw_bank_or_test_read':True,'visual_qa_status':'pending_final_input_render_inspection'}
        write_new(output/'figure_manifest.json',manifest)
        return manifest
    except BaseException as error:
        write_new(output/'FIGURES_INCOMPLETE.json',{'scope':scope,'created_at_utc':datetime.now(timezone.utc).isoformat(),
            'exception_type':type(error).__name__,'error':str(error),'source_sha256':hashes,
            'status':'Preserved partial build; not usable as final report figures.'})
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.parse_args()
    manifest=prepare()
    print(json.dumps({'manifest':OUTPUT+'/figure_manifest.json','figures':[r['path'] for r in manifest['figures']],
                      'scope':manifest['scope'],'visual_qa_status':manifest['visual_qa_status']},indent=2))


if __name__=='__main__':main()
