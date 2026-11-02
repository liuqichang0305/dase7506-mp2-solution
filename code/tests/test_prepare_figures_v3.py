"""Synthetic saved-summary plots only; no raw course inputs or final outputs."""
import copy
import importlib.util
import json
from pathlib import Path
import shutil
import statistics
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('figures_v3_under_test',ROOT/'reports/research_v3/prepare_figures.py')
f=importlib.util.module_from_spec(spec);spec.loader.exec_module(f)


def synthetic_metrics(n,calibrated=False):
    truth=[int(i%7!=0) for i in range(n)]
    confidence=[.52+.46*(i%100)/99 for i in range(n)]
    if calibrated:confidence=[.8+.12*x for x in confidence]
    bins=[];ece=0.
    for number in range(10):
        indices=[i for i,x in enumerate(confidence) if min(int(x*10),9)==number]
        count=len(indices)
        accuracy=sum(truth[i] for i in indices)/count if count else None
        mean=sum(confidence[i] for i in indices)/count if count else None
        gap=abs(mean-accuracy) if count else None
        ece+=count/n*(gap or 0)
        bins.append({'bin':number,'lower':number/10,'upper':(number+1)/10,'upper_inclusive':number==9,
                     'n':count,'accuracy':accuracy,'mean_confidence':mean,'absolute_gap':gap})
    order=sorted(range(n),key=lambda i:-confidence[i]);errors=[];total=0
    for i in order:total+=1-truth[i];errors.append(total)
    risk=[e/(i+1) for i,e in enumerate(errors)]
    return {'n':n,'correct':sum(truth),'accuracy':sum(truth)/n,'brier':sum((x-y)**2 for x,y in zip(confidence,truth))/n,
        'ece_10_equal_width':ece,'aurc_discrete_mean':statistics.fmean(risk),'bins':bins,
        'risk_coverage':{'retained':list(range(1,n+1)),'coverage':[(i+1)/n for i in range(n)],'risk':risk,'errors':errors,
                         'ordered_original_rows':order,'tie_rule':'Stable original row order; synthetic fixture.'}}


def fixture(root):
    root=Path(root)
    def row(generation,name,seed,seconds,correct):
        return {'version':generation,'experiment_id':name,'family':'mlp','seed':seed,'status':'completed','n':4755,
            'hard_correct':correct,'hard_n':2114,'hard_accuracy':100*correct/2114,
            'hard_fold_accuracies':[100*correct/2114]*3,'training_seconds':seconds,
            'summary_sha256':'a'*64,'oof_sha256':'b'*64,
            'config':{'experiment_id':name,'family':'mlp','seed':seed}}
    rows=[row('research_v2','old_s0',f.SEED,15,1600),row('research_v2','old_s1',f.SEED+1,20,1603),
          row('research_v3','new_s0',f.SEED,130,1640),row('research_v3','new_s1',f.SEED+1,150,1634),
          {'version':'research_v3','experiment_id':'failed_partial','status':'partial','n':1000,'hard_fold_accuracies':[70]}]
    jobs=[{'experiment_id':r['experiment_id'],'status':'completed','config':r['config'],
           'attempts':[{'returncode':0}]} for r in rows[:4] if r['version']=='research_v3']
    jobs+=[{'experiment_id':'failed_partial','status':'failed','config':{},'attempts':[{'returncode':1}]},
           {'experiment_id':'pruned','status':'pruned','config':{},'attempts':[{'status':'interrupted_process_missing'}]}]
    selector={'kind':'single_model','experiment_id':'new_s0','family':'mlp'}
    confidence={}
    for name,split,n in [('fit','train_oof',4755),('validation','validation',798)]:
        confidence[name]={'schema':'mp2-v3-confidence-report-v1','split':split,'selector':selector,'choices_changed':0,
            'events_sha256':'c'*64,'analysis_source_sha256':'d'*64,
            'source_sha256':{'selected_model_json':'e'*64} if name=='fit' else {'calibration_json':'f'*64},
            'interpretation':'IN-SAMPLE synthetic fitting outcomes; optimistic' if name=='fit' else 'Synthetic already-used validation holdout',
            'raw':synthetic_metrics(n),'calibrated':synthetic_metrics(n,True)}
    values={'analysis':{'source_sha256':'a'*64,'folds_sha256':'b'*64,'rows':rows,
        'noncomplete_jobs':[{'experiment_id':j['experiment_id'],'status':j['status']} for j in jobs if j['status']!='completed']},
        'queue':{'schema':'mp2-campaign-queue-v3','jobs':jobs},**confidence}
    for key,name in f.INPUTS.items():
        path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(values[key]),encoding='utf-8')
    source=root/f.GENERATOR;source.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(ROOT/f.GENERATOR,source)
    return values


class FigureTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='mp2-v3-figures-synthetic-')
        self.root=Path(self.tmp.name).resolve();self.values=fixture(self.root)

    def tearDown(self):self.tmp.cleanup()

    def update(self,key):
        (self.root/f.INPUTS[key]).write_text(json.dumps(self.values[key]),encoding='utf-8')

    def test_full_synthetic_plot_manifest_png_and_per_figure_source_hashes(self):
        manifest=f.prepare(self.root,synthetic=True)
        self.assertEqual(manifest['scope'],'synthetic_fixture_only')
        self.assertEqual(manifest['schema'],'mp2-v3-report-figures-v1')
        self.assertEqual(len(manifest['figures']),2)
        self.assertTrue(manifest['no_pdf_generated'])
        self.assertFalse(list((self.root/f.OUTPUT).glob('*.pdf')))
        from PIL import Image
        for figure in manifest['figures']:
            path=self.root/figure['path'];self.assertEqual(f.digest(path),figure['sha256'])
            with Image.open(path) as image:
                self.assertEqual(image.format,'PNG');self.assertGreaterEqual(image.width,2100);self.assertGreaterEqual(image.height,970)
            record_path=self.root/figure['generation_record']['path']
            self.assertEqual(f.digest(record_path),figure['generation_record']['sha256'])
            record=json.loads(record_path.read_text(encoding='utf-8'))
            self.assertEqual(record['scope'],'synthetic_fixture_only')
            self.assertEqual(record['report_height_points'],235)
            for name,digest in record['source_sha256'].items():self.assertEqual(f.digest(self.root/name),digest)
        cost=json.loads((self.root/f.OUTPUT/'cv_fit_cost_vs_hard_accuracy.generation.json').read_text())
        self.assertEqual(len(cost['plotted_rows']),4)
        self.assertEqual(cost['exclusions']['failed_jobs'],1)
        self.assertEqual(cost['exclusions']['pruned_jobs'],1)
        self.assertEqual(cost['exclusions']['nonzero_exit_attempts'],1)
        self.assertEqual(cost['exclusions']['interrupted_attempts'],1)
        self.assertEqual(manifest['partial_rows_excluded'],1)
        with self.assertRaisesRegex(ValueError,'Preserve existing'):f.prepare(self.root,synthetic=True)

    def test_missing_final_evidence_creates_no_output(self):
        (self.root/f.INPUTS['validation']).unlink()
        with self.assertRaisesRegex(ValueError,'not ready'):f.prepare(self.root,synthetic=True)
        self.assertFalse((self.root/f.OUTPUT).exists())

    def test_running_queue_or_missing_completed_method_rejected(self):
        self.values['queue']['jobs'][0]['status']='running';self.update('queue')
        with self.assertRaisesRegex(ValueError,'terminal'):f.prepare(self.root,synthetic=True)
        self.values['queue']['jobs'][0]['status']='completed';self.update('queue')
        self.values['analysis']['rows'].pop(2);self.update('analysis')
        with self.assertRaisesRegex(ValueError,'every and only'):f.prepare(self.root,synthetic=True)

    def test_missing_or_nonpositive_cost_is_not_imputed(self):
        for value in (None,0,-1,float('nan')):
            changed=copy.deepcopy(self.values['analysis']);changed['rows'][0]['training_seconds']=value
            with self.assertRaises(ValueError):f.cv_data(changed,self.values['queue'])

    def test_pooled_hard_counts_and_config_identity_checked(self):
        changed=copy.deepcopy(self.values['analysis']);changed['rows'][0]['hard_accuracy']+=1
        with self.assertRaisesRegex(ValueError,'pooled'):f.cv_data(changed,self.values['queue'])
        changed=copy.deepcopy(self.values['analysis']);changed['rows'][0]['config']['seed']+=1
        with self.assertRaisesRegex(ValueError,'identity'):f.cv_data(changed,self.values['queue'])

    def test_changed_selector_choices_or_synthetic_split_rejected(self):
        for change in ({'selector':{'different':True}},{'choices_changed':1},{'split':'synthetic'}):
            with self.assertRaises(ValueError):f.confidence_data(self.values['fit'],{**self.values['validation'],**change})

    def test_empty_bins_cannot_be_displayed_as_zero_accuracy(self):
        changed=copy.deepcopy(self.values['fit'])
        self.assertEqual(changed['raw']['bins'][0]['n'],0)
        changed['raw']['bins'][0]['accuracy']=0
        with self.assertRaisesRegex(ValueError,'invented'):f.confidence_data(changed,self.values['validation'])

    def test_curve_truncation_and_changed_cumulative_risk_rejected(self):
        changed=copy.deepcopy(self.values['validation'])
        changed['raw']['risk_coverage']['risk'].pop()
        with self.assertRaisesRegex(ValueError,'complete saved'):f.confidence_data(self.values['fit'],changed)
        changed=copy.deepcopy(self.values['validation'])
        changed['calibrated']['risk_coverage']['risk'][0]=.4321
        with self.assertRaisesRegex(ValueError,'disagree'):f.confidence_data(self.values['fit'],changed)

    def test_synthetic_mode_cannot_write_real_final_directory(self):
        with self.assertRaisesRegex(ValueError,'Synthetic fixtures'):f.prepare(ROOT,synthetic=True)
        with self.assertRaisesRegex(ValueError,'fixed repository'):f.prepare(self.root,synthetic=False)


if __name__=='__main__':unittest.main(verbosity=2)
