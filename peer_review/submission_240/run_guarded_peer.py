"""Once-only, checkpoint-only peer #240 reproduction; no research or publication."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import importlib.util
import json
import math
import os
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
RUNTIME = ROOT / 'tmp/peer_review/submission_240_7e3a2792'
SOURCE = RUNTIME / 'source'
OFFICIAL = RUNTIME / 'official'
ENV = RUNTIME / 'venv'
SITE = ENV / 'Lib/site-packages'
COMMIT = '7e3a2792b7b4442f5e53705a64b4560a2e365de9'
PROTOCOL = 'dase7506-mp2-open-hard-verifier-v5'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2); stream.write('\n')
        stream.flush(); os.fsync(stream.fileno())


def worker_source(phase, resource, access, *, self_test=False):
    inputs = [OFFICIAL / 'data/manifest.json', OFFICIAL / 'data/candidates/gsm8k_test.jsonl',
              OFFICIAL / 'data/candidates/gsm8k_test.npz']
    if phase == 'scoring':
        inputs += [OFFICIAL / 'data/gsm8k_test.jsonl', OUT / 'predictions.json']
    code = [OFFICIAL / name for name in ('common.py', 'predict.py', 'score_predictions.py')]
    if phase == 'prediction':
        code += [SOURCE / 'code/submission.py', SOURCE / 'code/features.py']
        code += sorted((SOURCE / 'code/runs/model_bundle').glob('*'))
    output = OUT / ('predictions.json' if phase == 'prediction' else 'score.json')
    base = Path(sys.base_prefix)
    scope = {'phase': phase, 'read_files': [str(p.resolve()) for p in inputs + code],
             'write_files': [str(p.resolve()) for p in (output, resource, access)],
             'read_directories': sorted({str(p.parent.resolve()) for p in inputs + code}),
             'write_directory': str(OUT.resolve()),
             'library_roots': [str((base / name).resolve()) for name in ('Lib', 'DLLs')]+[str(SITE.resolve())],
             'library_files': [str((base / 'python312.zip').resolve())],
             'cwd': str(OFFICIAL), 'self_test': self_test,
             'reference': str(OFFICIAL / 'data/gsm8k_test.jsonl')}
    prefix = """import os,sys,json,runpy
from pathlib import Path
_scope=json.loads(%r)
os.chdir(_scope['cwd'])
sys.dont_write_bytecode=True
_opened=set();_denied=[]
def _under(path,base):return path==base or path.startswith(base+os.sep)
_read=set(_scope['read_files']);_write=set(_scope['write_files'])
_libraries=_scope['library_roots']
def _audit(event,args):
    if event.startswith('socket.') or event in ('subprocess.Popen','os.system','os.spawn','os.posix_spawn','os.fork','os.forkpty','os.exec','os.startfile','os.startfile/2'):
        _denied.append({'event':event});raise PermissionError('Peer worker is offline and cannot spawn')
    if event in ('os.remove','os.rmdir','os.rename','os.link','os.symlink','os.chmod','os.chown','os.truncate'):
        _denied.append({'event':event});raise PermissionError('Peer worker cannot mutate files')
    if event in ('os.listdir','os.scandir'):
        p=str(Path(os.fsdecode(args[0])).resolve())
        if not any(_under(p,base) for base in _libraries) and p!=_scope['cwd'] and p not in _scope['read_directories']:
            _denied.append({'event':event,'path':p});raise PermissionError('Directory enumeration outside libraries forbidden')
    if event=='os.mkdir' and str(Path(os.fsdecode(args[0])).resolve())!=_scope['write_directory']:
        _denied.append({'event':event});raise PermissionError('Directory creation outside output forbidden')
    if event!='open' or isinstance(args[0],int):return
    p=str(Path(os.fsdecode(args[0])).resolve());mode,flags=args[1],args[2]
    writing=(isinstance(mode,str) and any(c in mode for c in 'wax+')) or flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC)
    permitted=p in _write if writing else p in _read or p in _scope['library_files'] or any(_under(p,b) for b in _libraries)
    if not permitted:
        _denied.append({'event':event,'path':p,'writing':bool(writing)});raise PermissionError('File outside explicit peer phase allowlist')
    _opened.add(p)
sys.addaudithook(_audit)
""" % json.dumps(scope)
    if self_test:
        return prefix + """
checks={}
for label,action in [('reference',lambda:open(_scope['reference'],'rb')),('outside',lambda:open(r'C:\\Users\\Liuye\\.gitconfig','rb')),('network',lambda:__import__('socket').socket()),('spawn',lambda:__import__('subprocess').Popen([sys.executable,'-c','pass']))]:
    try:action();checks[label]=False
    except PermissionError:checks[label]=True
with open(%r,'x',encoding='utf-8') as stream:json.dump({'checks':checks,'denied':_denied},stream,indent=2)
assert all(checks.values())
""" % str(access)
    return prefix + """
from threadpoolctl import threadpool_info
script,resource,access=sys.argv[1:4]
sys.argv=[script]+sys.argv[4:]
sys.path.insert(0,str(Path(script).resolve().parent))
try:
    runpy.run_path(script,run_name='__main__')
finally:
    report={'script':script,'phase':_scope['phase'],'python_executable':sys.executable,
            'pythonpath':os.environ.get('PYTHONPATH'),'numpy_version':getattr(sys.modules.get('numpy'),'__version__',None),
            'torch_imported':'torch' in sys.modules,'thread_pools':threadpool_info(),
            'features_module_file':getattr(sys.modules.get('features'),'__file__',None)}
    with open(resource,'x',encoding='utf-8') as stream:json.dump(report,stream,indent=2)
    with open(access,'x',encoding='utf-8') as stream:json.dump({'phase':_scope['phase'],'opened_paths':sorted(_opened),'denied':_denied,'question_text_logged':False},stream,indent=2)
"""


def environment():
    env = {k: os.environ[k] for k in ('SYSTEMROOT', 'WINDIR', 'COMSPEC', 'PATHEXT') if k in os.environ}
    env.update({'PATH': str(Path(sys.base_prefix)) + os.pathsep + str(Path(os.environ['SYSTEMROOT']) / 'System32'),
                'TEMP': str(RUNTIME / 'temp'), 'TMP': str(RUNTIME / 'temp'),
                'PYTHONPATH': str(SITE), 'PYTHONNOUSERSITE': '1', 'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONUTF8': '1'})
    env.update({k: '1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS',
                              'VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS','OMP_THREAD_LIMIT')})
    return env


def inventory():
    download = read(OUT / 'download_complete.json')
    if download['source_commit'] != COMMIT:
        raise ValueError('Wrong immutable peer commit')
    sources = {}
    for row in download['files']:
        path = SOURCE / row['path']
        if sha(path) != row['sha256']:
            raise ValueError('Downloaded peer file changed')
        sources[str(path)] = row['sha256']
    for row in read(OUT / 'official_files.json')['files']:
        if sha(row['path']) != row['sha256']:
            raise ValueError('Official data/source changed')
        sources[row['path']] = row['sha256']
    safety = read(OUT / 'independent_source_safety_review.json')
    if (safety.get('blocking_source_finding') is not False
            or safety.get('suitable_for_guarded_official_predict') is not True
            or safety.get('source_commit') != COMMIT):
        raise ValueError('Source safety review did not approve this exact scope')
    for name,expected in safety['source_sha256'].items():
        if sha(SOURCE/name)!=expected:
            raise ValueError('Reviewed peer source changed')
    return sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if not SITE.is_dir():
        raise ValueError('Fresh environment not installed')
    base_python = str(Path(sys._base_executable).resolve())
    env = environment()
    if args.self_test:
        access = OUT / 'guard-self-test.json'
        resource = OUT / 'guard-self-test-unused-resource.json'
        code = worker_source('prediction', resource, access, self_test=True)
        result = subprocess.run([base_python, '-S', '-c', code], cwd=RUNTIME, env=env, capture_output=True, timeout=10)
        for suffix, value in [('stdout', result.stdout), ('stderr', result.stderr)]:
            with (OUT / ('guard-self-test.'+suffix+'.log')).open('xb') as stream:stream.write(value)
        print(json.dumps({'self_test_returncode': result.returncode, 'record': str(access)}))
        raise SystemExit(result.returncode)
    if any((OUT / name).exists() for name in ('STARTED.json','COMPLETE.json','FAILED.json','predictions.json','score.json')):
        raise FileExistsError('Prior peer evaluation evidence exists; preserve it, no automatic replay')
    hashes = inventory()
    asset = read(OUT / 'source_asset_manifest.json')
    probe = read(OUT / 'environment.json')
    if probe['packages'].get('numpy') != '2.5.3' or any(p in probe['packages'] for p in ('torch','tensorflow','scipy','scikit-learn')):
        raise ValueError('Expected only the documented inference dependency and instrumentation')
    if not all(read(OUT / 'guard-self-test.json')['checks'].values()):
        raise ValueError('Guard boundary checks did not pass')
    module = ROOT / 'code/tools/confirm_validation_v3.py'
    spec = importlib.util.spec_from_file_location('peer_timing', module)
    timing = importlib.util.module_from_spec(spec); spec.loader.exec_module(timing)
    write(OUT / 'coordinator_provenance.json', {'run_process_source': str(module), 'sha256': sha(module),
          'cooperative_lock': str(ROOT / 'tmp/research_v3/trainer.lock'),
          'worker_environment_keys': sorted(env), 'worker_credentials_inherited': False})
    old_environment = dict(os.environ)
    # run_process starts from its own environment; make it credential-free before copying.
    os.environ.clear(); os.environ.update(env)
    processes = []
    try:
        with timing.exclusive_evaluation():
            started_at = datetime.now(timezone.utc).isoformat()
            write(OUT / 'STARTED.json', {'schema': 'mp2-peer-evaluation-start-v1', 'started_at_utc': started_at,
                  'submission': 240, 'peer_commit': COMMIT, 'target_sha256': sha(OUT / 'target.json'),
                  'inputs_sha256': hashes, 'purpose': 'Independent peer checkpoint reproduction only; no model research or retraining'})
            start = time.perf_counter(); deadline = start + 30.
            for phase in ('prediction','scoring'):
                resource = OUT / (phase+'-libraries.json'); access = OUT / (phase+'-access.json')
                script = OFFICIAL / ('predict.py' if phase=='prediction' else 'score_predictions.py')
                code = worker_source(phase, resource, access)
                command = [base_python, '-S', '-c', code, str(script), str(resource), str(access), '--split','test']
                if phase=='prediction':
                    command += ['--method',str(SOURCE/'code/submission.py'),'--checkpoint',str(SOURCE/'code/runs/model_bundle'),
                                '--output',str(OUT/'predictions.json')]
                else:
                    command += ['--predictions',str(OUT/'predictions.json'),'--output',str(OUT/'score.json')]
                result = timing.run_process(command,OUT,phase,deadline,environment=env)
                processes.append(result)
                if result['returncode'] or result['timed_out']:
                    raise RuntimeError('Peer official process failed or exceeded shared 30-second budget; no fabricated score')
            elapsed = time.perf_counter()-start
            score = read(OUT/'score.json')
            prediction = read(OUT/'predictions.json')
            if (score['protocol']!=PROTOCOL or score['split']!='test' or score['n']!=467
                    or type(score['correct']) is not int or not 0<=score['correct']<=467
                    or not math.isclose(score['accuracy'],100*score['correct']/467,abs_tol=1e-12)
                    or len(prediction['ids'])!=467 or len(prediction['selected'])!=467
                    or any(type(i) is not int or not 0<=i<4 for i in prediction['selected'])):
                raise ValueError('Official result contract mismatch')
            libraries = [read(OUT/(p+'-libraries.json')) for p in ('prediction','scoring')]
            accesses = [read(OUT/(p+'-access.json')) for p in ('prediction','scoring')]
            if any(row['denied'] for row in accesses):
                raise ValueError('Peer attempted a prohibited operation')
            required_assets={str(p.resolve()) for p in (SOURCE/'code/runs/model_bundle').iterdir() if p.is_file()}
            if not required_assets <= set(accesses[0]['opened_paths']):
                raise ValueError('Every submitted checkpoint asset must actually be consumed; no silent fallback')
            if libraries[0]['features_module_file'] != str(SOURCE/'code/features.py'):
                raise ValueError('Peer imported a different features module')
            for path,expected in hashes.items():
                if sha(path)!=expected:raise ValueError('Source/assets/data changed during evaluation')
            pools_ok=all(not row['torch_imported'] and all(p['num_threads']==1 for p in row['thread_pools'])
                         and sum(p['num_threads'] for p in row['thread_pools'])<=4 for row in libraries)
            peak=max(p['sampled_peak_tree_threads'] for p in processes)
            eligible=elapsed<=30 and peak<=4 and pools_ok and asset['total_asset_bytes']<=16*1024**2
            report={'schema':'mp2-peer-timing-v1','status':'complete','prediction_plus_scoring_seconds':elapsed,
                    'processes':processes,'sampled_peak_task_process_tree_threads':peak,
                    'scientific_library_observations':libraries,'scientific_library_thread_cap_passed':pools_ok,
                    'local_resource_eligible':eligible,'asset_bytes':asset['total_asset_bytes'],
                    'hardware':{'platform':platform.platform(),'processor':platform.processor(),'python':sys.version},
                    'measurement':'One fresh direct CPU prediction process plus independent scoring, shared30s deadline. External measurement coordinator disclosed separately.',
                    'limitation':'Local Windows i9-12900H result, not reference Xeon certification. Python audit-hook allowlist and reviewed code provide application-level isolation, not an OS VM.',
                    'completed_at_utc':datetime.now(timezone.utc).isoformat()}
            write(OUT/'timing.json',report)
            target=read(OUT/'target.json')
            complete={'schema':'mp2-peer-reproduction-complete-v1','status':'complete','submission':240,
                      'target_sha256':sha(OUT/'target.json'),'peer_commit':COMMIT,
                      'official_score_file':'score.json','official_timing_file':'timing.json',
                      'prediction_scoring_success':True,'local_resource_eligible':eligible,
                      'protocol':PROTOCOL,'split':'test','n':467,'correct':score['correct'],'accuracy':score['accuracy'],
                      'published_declared_score':target['selected']['score'],
                      'difference_percentage_points':score['accuracy']-target['selected']['score'],
                      'matches_repository_readme':score['correct']==394,
                      'model_retrained':False,'own_model_research_updated':False,'posted':False,
                      'completed_at_utc':datetime.now(timezone.utc).isoformat(),
                      'output_sha256':{p.name:sha(p) for p in OUT.iterdir() if p.is_file()}}
            write(OUT/'COMPLETE.json',complete)
            print(json.dumps({k:complete[k] for k in ('status','submission','correct','n','accuracy','difference_percentage_points','local_resource_eligible')}))
    except BaseException as error:
        write(OUT/'FAILED.json',{'schema':'mp2-peer-reproduction-failed-v1','status':'failed','error_type':type(error).__name__,
              'error':str(error),'processes':processes,'no_score_fabricated':True,'completed_at_utc':datetime.now(timezone.utc).isoformat()})
        raise
    finally:
        os.environ.clear();os.environ.update(old_environment)


if __name__=='__main__':
    main()
