"""Synthetic only: no public clone, package installation or official data reads."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import sysconfig
import tempfile
import types
import unittest
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('reproduce_v3_under_test', REPO/'scripts/reproduce_selected_v3.py')
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) if not isinstance(value,str) else value,encoding='utf-8')


def synthetic_probe(prefix):
    python = prefix / ('Scripts/python.exe' if os.name=='nt' else 'bin/python')
    put(python,'synthetic executable marker')
    packages = prefix/'Lib/site-packages'
    packages.mkdir(parents=True,exist_ok=True)
    return {'python_prefix':str(prefix),'python_executable':str(python),
        'base_executable':str(Path(getattr(sys,'_base_executable',sys.executable)).resolve()),
        'base_prefix':sys.base_prefix,'site_packages':[str(packages)],
        'runtime_paths':[sysconfig.get_path('stdlib'),str(Path(sys.base_prefix)/'DLLs')],
        'packages':[*r.PACKAGES,'pip==25.0'],'numpy_version':'2.5.3',
        'torch_available':False,'tensorflow_available':False}


def data_fixture(root):
    data=root/'code/data';hashes={}
    for name in r.DATA_FILES[1:]:
        put(data/name,'synthetic bytes for '+name)
        hashes[name]=r.sha(data/name)
    put(data/'manifest.json',{'protocol':r.PROTOCOL,'splits':{'validation':798},'sha256':hashes})


def outputs():
    predictions={'protocol':r.PROTOCOL,'split':'validation','ids':[f'synthetic-{i}' for i in range(798)],
        'selected':[i%4 for i in range(798)],'prediction_seconds':.1}
    score={'protocol':r.PROTOCOL,'split':'validation','correct':700,'n':798,'accuracy':100*700/798,
           'scoring_seconds':.1}
    return predictions,score


class ReproductionTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='mp2-v3-clean-repro-synthetic-')
        self.root=Path(self.tmp.name).resolve()

    def tearDown(self):
        self.tmp.cleanup()

    def test_validation_only_copy_hashes_and_no_overwrite(self):
        data_fixture(self.root)
        checkout=self.root/'tmp/checkout'
        put(checkout/'code/data/NOTICE.txt','synthetic legal notice')
        records=r.copy_validation_data(self.root,checkout)
        self.assertEqual({x['relative_path'] for x in records},set(r.DATA_FILES)|{'NOTICE.txt'})
        self.assertEqual(r.copy_validation_data(self.root,checkout),records)
        put(checkout/'code/data/gsm8k_validation.jsonl','tampered')
        with self.assertRaisesRegex(ValueError,'never overwrite'):
            r.copy_validation_data(self.root,checkout)

    def test_unexpected_train_test_or_extra_data_rejected_before_copy(self):
        data_fixture(self.root)
        for name in ('gsm8k_train.jsonl','candidates/gsm8k_test.npz','other.json'):
            checkout=self.root/'tmp'/name.replace('/','-')
            put(checkout/'code/data'/name,'never opened')
            with self.assertRaisesRegex(ValueError,'unapproved data'):
                r.copy_validation_data(self.root,checkout)
            self.assertFalse((checkout/'code/data/manifest.json').exists())

    def test_wrong_original_manifest_hash_or_count_rejected(self):
        data_fixture(self.root)
        put(self.root/'code/data/candidates/gsm8k_validation.jsonl','changed')
        with self.assertRaisesRegex(ValueError,'fixed manifest'):
            r.copy_validation_data(self.root,self.root/'tmp/checkout')

    def test_exact_fresh_environment_package_and_prefix_contract(self):
        prefix=self.root/'tmp/environment'
        probe=synthetic_probe(prefix)
        r.validate_probe(probe,prefix)
        for change in ({'packages':[*probe['packages'],'torch==2.1']},
                       {'packages':[*probe['packages'],'numpy==2.0']},
                       {'numpy_version':'2.0'}, {'torch_available':True},
                       {'site_packages':[str(self.root/'old-packages')]},
                       {'runtime_paths':[str(self.root/'results')]},
                       {'python_prefix':str(self.root/'old-env')}):
            with self.subTest(change=change),self.assertRaises(ValueError):
                r.validate_probe({**probe,**change},prefix)

    def test_attempts_are_unique_and_failed_retry_needs_reason_same_assets(self):
        identity={'selected_model_sha256':'a'*64,'assets':{'weights':'b'*64}}
        folder,checkout,prefix=r.reserve_attempt(self.root,'first',None,identity)
        self.assertTrue(checkout.is_relative_to(self.root/'tmp'))
        self.assertNotEqual(checkout,prefix)
        with self.assertRaises(ValueError):r.reserve_attempt(self.root,'second',None,identity)
        with self.assertRaises(FileExistsError):r.reserve_attempt(self.root,'first','Fixed a synthetic environment setup error.',identity)
        with self.assertRaises(ValueError):r.reserve_attempt(self.root,'second','Fixed a synthetic environment setup error.',{**identity,'assets':{'weights':'c'*64}})
        r.reserve_attempt(self.root,'second','Fixed a synthetic environment setup error.',identity)
        put(folder/'COMPLETE.json',{'status':'passed'})
        with self.assertRaisesRegex(RuntimeError,'faster repeat'):
            r.reserve_attempt(self.root,'third','Fixed a synthetic environment setup error.',identity)
        with self.assertRaises(ValueError):r.reserve_attempt(self.root,'../escape',None,identity)

    def test_output_comparison_covers_all_798_choices_and_official_counts(self):
        predictions,score=outputs()
        r.compare_outputs(predictions,score,copy.deepcopy(predictions),copy.deepcopy(score))
        changed=copy.deepcopy(predictions);changed['selected'][797]=(changed['selected'][797]+1)%4
        with self.assertRaisesRegex(ValueError,'locked original'):
            r.compare_outputs(changed,score,predictions,score)
        for changed in ({**predictions,'split':'test'},{**predictions,'selected':[False]*798},
                        {**predictions,'ids':['same']*798}):
            with self.assertRaises(ValueError):r.compare_outputs(changed,score,predictions,score)
        with self.assertRaises(ValueError):r.compare_outputs(predictions,{**score,'accuracy':0},predictions,score)

    def test_commands_fixed_direct_base_new_packages_and_validation(self):
        prefix=self.root/'tmp/env';probe=synthetic_probe(prefix)
        timing=types.SimpleNamespace(PHASE_WRAPPER='print("synthetic wrapper")\n')
        for phase,script in [('prediction','code/predict.py'),('scoring','code/score_predictions.py')]:
            command,env=r.phase_command(self.root/'tmp/checkout',prefix,probe,phase,self.root,timing)
            self.assertEqual(command[1:3],['-S','-c'])
            self.assertEqual(command[4],script)
            self.assertEqual(command[command.index('--split')+1],'validation')
            self.assertEqual(command.count('--split'),1)
            self.assertEqual(env['PYTHONPATH'],probe['site_packages'][0])
            self.assertNotIn('test',command[4:])

    def guard_child(self,phase,statement):
        checkout=self.root/'tmp/checkout';prefix=self.root/'tmp/env'
        checkout.mkdir(parents=True,exist_ok=True)
        probe=synthetic_probe(prefix)
        data_fixture(checkout)
        output=self.root/'allowed-output.json';resource=self.root/'allowed-resource.json'
        prediction=self.root/'saved-predictions.json';put(prediction,'synthetic choices')
        guard=r.worker_guard(checkout,prefix,probe,phase,output,resource,prediction)
        env=os.environ.copy();env.update(r.THREAD_ENV);env['PYTHONDONTWRITEBYTECODE']='1'
        result=subprocess.run([sys.executable,'-S','-c',guard+statement],capture_output=True,text=True,env=env,timeout=10)
        return result

    def test_real_guard_prediction_references_history_write_and_spawn_denied(self):
        targets=["Path('code/data/gsm8k_validation.jsonl').read_bytes()",
                 "Path('code/data/gsm8k_test.jsonl').read_bytes()",
                 "Path('results/history.json').read_bytes()",
                 "Path('code/data/manifest.json').write_text('changed')",
                 "import socket;socket.socket()",
                 "import subprocess;subprocess.run([sys.executable,'-c','print(1)'])"]
        for statement in targets:
            with self.subTest(statement=statement):
                result=self.guard_child('prediction',statement)
                self.assertNotEqual(result.returncode,0,result.stdout)
                self.assertIn('PermissionError',result.stderr)

    def test_real_guard_allows_prediction_inputs_and_scoring_gold(self):
        for phase,statement in [('prediction',"assert Path('code/data/candidates/gsm8k_validation.jsonl').read_bytes()"),
                                ('scoring',"assert Path('code/data/gsm8k_validation.jsonl').read_bytes();assert Path(_reproduction_scope['predictions']).read_bytes()")]:
            result=self.guard_child(phase,statement)
            self.assertEqual(result.returncode,0,result.stderr)

    def test_setup_environment_removes_inherited_credentials_paths_and_config(self):
        with patch.dict(os.environ,{'GH_TOKEN':'synthetic','GIT_CONFIG_COUNT':'1','PYTHONPATH':'old','GIT_ASKPASS':'old'}):
            env=r.setup_environment()
        self.assertNotIn('GH_TOKEN',env)
        self.assertNotIn('GIT_CONFIG_COUNT',env)
        self.assertNotIn('PYTHONPATH',env)
        self.assertEqual(env['GIT_CONFIG_GLOBAL'],os.devnull)
        self.assertEqual(env['GIT_TERMINAL_PROMPT'],'0')

    def test_preflight_rejects_open_search_existing_freeze_and_invalid_selection(self):
        closed=self.root/'results/research_v3/SEARCH_CLOSED.json'
        put(closed,{'search_closed':False})
        freezing=types.SimpleNamespace(validate_selected=lambda _:self.fail('Selection read during open search'))
        with self.assertRaisesRegex(ValueError,'closed'):
            r.preflight(self.root,freezing,None)
        put(self.root/'submission/v3/FREEZE.json',{'synthetic':True})
        with self.assertRaisesRegex(ValueError,'precede'):
            r.preflight(self.root,freezing,None)
        (self.root/'submission/v3/FREEZE.json').unlink()
        put(closed,{'search_closed':True})
        def reject_selection(_):raise ValueError('Synthetic selection not uniquely locked')
        freezing.validate_selected=reject_selection
        with self.assertRaisesRegex(ValueError,'uniquely locked'):
            r.preflight(self.root,freezing,None)

    def test_setup_failure_preserves_attempt_marker_and_failure_without_retry(self):
        identity={'selected_model_sha256':'a'*64,'assets':{'weights':'b'*64}}
        with patch.object(r,'preflight',return_value=identity),patch.object(r,'run_setup',side_effect=RuntimeError('synthetic clone failure')):
            with self.assertRaisesRegex(RuntimeError,'synthetic clone failure'):
                r.reproduce(self.root,'failed_setup',freezing=object(),timing=object())
        folder=self.root/'results/research_v3/clean_reproduction/failed_setup'
        self.assertTrue((folder/'STARTED.json').exists())
        failure=r.read(folder/'FAILED.json')
        self.assertEqual(failure['exception_type'],'RuntimeError')
        self.assertEqual(failure['processes'],[])
        self.assertFalse((folder/'COMPLETE.json').exists())

    def test_complete_coordinator_with_fake_setup_and_fake_official_outputs(self):
        data_fixture(self.root)
        predictions,score=outputs()
        origin=self.root/'results/research_v3/validation/fixed_candidate'
        put(origin/'validation-predictions.json',predictions);put(origin/'validation-score.json',score)
        identity={'source_commit':'a'*40,'selected':{'historical_fallback':False,'name':'fixed_candidate'},
            'source_files':{'code/solution_v3.py':'b'*64},'assets':{'artifacts/research_v3/final_selected/config.json':'c'*64},
            'resource':{'within_limit':True},'implementation_sha256':{r.SCRIPT:'d'*64},
            'original_files':{r.relative(self.root,origin/name):r.sha(origin/name) for name in ('validation-predictions.json','validation-score.json')},
            'selected_model_sha256':'e'*64,'search_closed_sha256':'f'*64,'validation_summary_sha256':'1'*64}
        calls=[];context={}
        def setup(command,cwd,folder,stem,**kwargs):
            calls.append(command)
            if stem=='setup-clone':context['checkout']=Path(command[-1]);context['checkout'].mkdir()
            if stem=='setup-venv':context['prefix']=Path(command[-1]);context['probe']=synthetic_probe(context['prefix'])
            if stem=='setup-environment-probe':put(folder/(stem+'.stdout.log'),context['probe'])
            put(folder/(stem+'.command.json'),{'command':command})
        freezing=types.SimpleNamespace(commit_bytes=lambda *a:None,source_hashes=lambda _:identity['source_files'],
            asset_hashes=lambda _:(identity['assets'],identity['resource']))
        def process(command,folder,stem,deadline,environment=None):
            phase='prediction' if command[4]=='code/predict.py' else 'scoring'
            put(folder/('validation-predictions.json' if phase=='prediction' else 'validation-score.json'),predictions if phase=='prediction' else score)
            put(Path(command[5]),{'torch_imported':False,'numpy_version':'2.5.3','thread_pools':[{'num_threads':1}],
                'pythonpath':environment['PYTHONPATH']})
            now=r.utc()
            return {'command':command,'returncode':0,'timed_out':False,'started_at_utc':now,'completed_at_utc':now,
                'sampled_peak_tree_threads':2,'sampled_peak_tree_plus_coordinator_threads':3}
        timing=types.SimpleNamespace(PHASE_WRAPPER='',active_training=lambda:[],run_process=process)
        with patch.object(r,'preflight',return_value=identity),patch.object(r,'run_setup',side_effect=setup),patch.object(r.subprocess,'check_output',return_value='a'*40+'\n'):
            result=r.reproduce(self.root,'synthetic',freezing=freezing,timing=timing)
        self.assertEqual(result['status'],'passed')
        record=r.read(self.root/result['result'])
        self.assertTrue(record['choices_match'])
        for key in ('predictions_result','score_result','timing_result','environment_result'):
            reference=record[key]
            self.assertEqual(reference['sha256'],record['evidence_files'][reference['path']])
        self.assertEqual(len([c for c in calls if 'clone' in c]),1)
        install=next(c for c in calls if 'install' in c)
        self.assertEqual(install[-3:],list(r.PACKAGES))
        self.assertFalse((self.root/'submission/v3/REPRODUCTION_RESULT.json').exists())


if __name__=='__main__':
    unittest.main(verbosity=2)
