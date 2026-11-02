"""Reproduce the unique closed-search V3 selection in a new public checkout.

No model, checkpoint, seed, split, package, commit or output-root override exists.
This is validation-only engineering verification, never a training/selection job.
Importing this module performs no filesystem/network/model evaluation activity.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
METHOD = 'code/solution_v3.py'
CHECKPOINT = 'artifacts/research_v3/final_selected'
REPOSITORY = 'https://github.com/liuqichang0305/dase7506-mp2-solution.git'
PROTOCOL = 'dase7506-mp2-open-hard-verifier-v5'
PACKAGES = ('numpy==2.5.3', 'threadpoolctl==3.7.0', 'psutil==7.2.2')
DATA_FILES = ('manifest.json', 'gsm8k_validation.jsonl',
              'candidates/gsm8k_validation.jsonl', 'candidates/gsm8k_validation.npz')
OPTIONAL_DATA = {'LICENSE.txt', 'NOTICE.txt'}
THREAD_ENV = {name: '1' for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
    'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS', 'OMP_THREAD_LIMIT')}
SCRIPT = 'scripts/reproduce_selected_v3.py'


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_new(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def within(root, value):
    root, value = Path(root).resolve(), Path(value).resolve()
    if value == root or not value.is_relative_to(root):
        raise ValueError('Path escapes the required task directory')
    return value


def relative(root, path):
    return within(root, path).relative_to(Path(root).resolve()).as_posix()


def runtime():
    sys.path.insert(0, str(ROOT / 'code/tools'))
    import freeze_v3
    import confirm_validation_v3
    return freeze_v3, confirm_validation_v3


def setup_environment():
    # Explicit public HTTPS clone; no inherited helper, token-based URL rewrite,
    # interactive credential request, user git config or Python package path.
    env = os.environ.copy()
    for key in list(env):
        if key.startswith('GIT_CONFIG_') or key in ('PYTHONPATH', 'PYTHONHOME', 'GIT_ASKPASS',
                'SSH_ASKPASS', 'GIT_SSH_COMMAND', 'GITHUB_TOKEN', 'GH_TOKEN'):
            env.pop(key, None)
    env.update(THREAD_ENV)
    env.update(GIT_CONFIG_NOSYSTEM='1', GIT_CONFIG_GLOBAL=os.devnull, GIT_TERMINAL_PROMPT='0',
               GCM_INTERACTIVE='never', PYTHONNOUSERSITE='1', PYTHONUTF8='1', PYTHONDONTWRITEBYTECODE='1')
    return env


def run_setup(command, cwd, folder, stem, *, timeout=600):
    """Raw setup stdout/stderr are separate, immutable, and outside timing."""
    started, tick = utc(), time.perf_counter()
    record = {'command': [str(x) for x in command], 'cwd': str(cwd), 'started_at_utc': started}
    write_new(folder / (stem + '.command.json'), record)
    with (folder / (stem + '.stdout.log')).open('xb') as out, (folder / (stem + '.stderr.log')).open('xb') as err:
        completed = subprocess.run(command, cwd=cwd, env=setup_environment(), stdout=out, stderr=err,
                                   timeout=timeout, check=False)
    record.update(returncode=completed.returncode, completed_at_utc=utc(), wall_seconds=time.perf_counter()-tick)
    write_new(folder / (stem + '.result.json'), record)
    if completed.returncode:
        raise RuntimeError('Clean setup failed; preserve raw logs: ' + stem)
    return record


def preflight(root, freezing, timing):
    root = Path(root).resolve()
    if any((root / 'submission/v3' / name).exists() for name in ('FREEZE.json', 'TEST_OPENED.json', 'TEST_COMPLETE.json')):
        raise ValueError('Clean reproduction must precede the final freeze and new test opening')
    closed = read(root / 'results/research_v3/SEARCH_CLOSED.json')
    if closed.get('search_closed') is not True:
        raise ValueError('Research must be closed before clean reproduction')
    selected, summary, verified, assets, inventory = freezing.validate_selected(root)
    if timing.instant(closed['created_at_utc']) > timing.instant(selected['created_at_utc']):
        raise ValueError('Unique selection must follow search closure')
    if timing.active_training():
        raise RuntimeError('Training or a scheduler is active; reproduction cannot begin')
    sources = freezing.source_hashes(root)
    for name, expected in timing.OFFICIAL_SOURCE.items():
        if sources.get(name) != expected:
            raise ValueError('The official predictor/scorer/parser must remain unchanged')
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    if re.fullmatch('[0-9a-f]{40}', commit) is None:
        raise ValueError('Current HEAD must be a full commit SHA')
    # The new reproducer is also required to be committed, even if it is not
    # part of the model-source closure returned by the freezer.
    implementation = {SCRIPT: sha(root / SCRIPT)}
    freezing.commit_bytes(root, commit, {**sources, **assets, **implementation})
    generation = 'research_v2' if selected['historical_fallback'] else 'research_v3'
    origin = root / 'results' / generation / 'validation' / selected['name']
    originals = {relative(root, origin / ('validation-' + part + '.json')):
                 sha(origin / ('validation-' + part + '.json')) for part in ('predictions', 'score')}
    return {'source_commit': commit, 'selected': selected, 'source_files': sources, 'assets': assets,
        'resource': inventory, 'implementation_sha256': implementation, 'original_files': originals,
        'selected_model_sha256': sha(root / 'submission/v3/SELECTED_MODEL.json'),
        'search_closed_sha256': sha(root / 'results/research_v3/SEARCH_CLOSED.json'),
        'validation_summary_sha256': sha(root / 'results/research_v3/validation_summary.json')}


def reserve_attempt(root, name, reason, identity):
    if not isinstance(name, str) or re.fullmatch('[a-z][a-z0-9_-]{0,47}', name) is None:
        raise ValueError('Attempt name must be a short lowercase identifier')
    base = Path(root) / 'results/research_v3/clean_reproduction'
    base.mkdir(parents=True, exist_ok=True)
    previous = sorted(path for path in base.iterdir() if path.is_dir())
    if any((folder / 'COMPLETE.json').exists() for folder in previous):
        raise RuntimeError('A completed clean reproduction already exists; do not select a faster repeat')
    if previous and (not isinstance(reason, str) or len(reason.strip()) < 20):
        raise ValueError('A new attempt after failure requires an explicit engineering repair reason')
    for folder in previous:
        record = read(folder / 'STARTED.json')
        if record['selected_model_sha256'] != identity['selected_model_sha256'] or record['assets'] != identity['assets']:
            raise ValueError('A reproduction retry cannot change the unique selected model or asset bytes')
    folder = within(base, base / name)
    folder.mkdir(exist_ok=False)
    token = name + '-' + uuid.uuid4().hex[:12]
    checkout = within(Path(root) / 'tmp', Path(root) / 'tmp' / ('reproduce-v3-checkout-' + token))
    environment = within(Path(root) / 'tmp', Path(root) / 'tmp' / ('reproduce-v3-env-' + token))
    if checkout.exists() or environment.exists():
        raise FileExistsError('Fresh checkout/environment paths must not exist')
    write_new(folder / 'STARTED.json', {'schema':'mp2-v3-clean-reproduction-start-v1', 'started_at_utc':utc(),
        'attempt':name, 'engineering_repair_reason':reason, 'prior_attempts':[p.name for p in previous],
        'checkout':str(checkout), 'environment':str(environment), **identity})
    return folder, checkout, environment


def data_inventory(checkout):
    data = Path(checkout) / 'code/data'
    files = sorted(p.relative_to(data).as_posix() for p in data.rglob('*') if p.is_file())
    if any(p.is_symlink() for p in data.rglob('*')):
        raise ValueError('Fresh official data directory must not contain links')
    if not set(files).issubset(set(DATA_FILES) | OPTIONAL_DATA):
        raise ValueError('Fresh checkout contains unapproved data, including possible train/test files')
    return files


def copy_validation_data(root, checkout):
    data_inventory(checkout)  # Reject unexpected data before copying anything.
    source = Path(root) / 'code/data'
    manifest = read(source / 'manifest.json')
    if manifest.get('protocol') != PROTOCOL or manifest.get('splits', {}).get('validation') != 798:
        raise ValueError('Official validation manifest protocol/count differs')
    records = []
    for name in DATA_FILES:
        original, copied = source / name, Path(checkout) / 'code/data' / name
        if original.is_symlink():
            raise ValueError('Official source data cannot be a link')
        digest = sha(original)
        if name != 'manifest.json' and digest != manifest['sha256'].get(name):
            raise ValueError('Official validation file differs from its fixed manifest: ' + name)
        copied.parent.mkdir(parents=True, exist_ok=True)
        if copied.exists():
            if sha(copied) != digest:
                raise ValueError('Existing fresh-checkout data differs; never overwrite it')
        else:
            with original.open('rb') as src, copied.open('xb') as dst:
                shutil.copyfileobj(src, dst)
        if sha(copied) != digest:
            raise ValueError('Copied validation bytes differ')
        records.append({'source':str(original.resolve()), 'destination':str(copied.resolve()),
                        'relative_path':name, 'sha256':digest, 'bytes':copied.stat().st_size})
    for name in data_inventory(checkout):
        if name in OPTIONAL_DATA:
            copied = Path(checkout) / 'code/data' / name
            records.append({'source':'immutable public checkout', 'destination':str(copied.resolve()),
                            'relative_path':name, 'sha256':sha(copied), 'bytes':copied.stat().st_size})
    return records


PROBE = r'''import importlib.metadata,importlib.util,json,sys,sysconfig
from pathlib import Path
import numpy
print(json.dumps({'python_prefix':sys.prefix,'python_executable':sys.executable,
 'base_executable':str(Path(getattr(sys,'_base_executable',sys.executable)).resolve()),
 'base_prefix':sys.base_prefix,'python_version':sys.version,
 'site_packages':sorted({sysconfig.get_paths()[k] for k in ('purelib','platlib')}),
 'runtime_paths':[p for p in sys.path if p and not Path(p).resolve().is_relative_to(Path(sys.prefix).resolve())],
 'packages':sorted(d.metadata['Name'].lower().replace('_','-')+'=='+d.version for d in importlib.metadata.distributions()),
 'numpy_version':numpy.__version__,'torch_available':importlib.util.find_spec('torch') is not None,
 'tensorflow_available':importlib.util.find_spec('tensorflow') is not None},indent=2))
'''


def validate_probe(probe, prefix):
    prefix = Path(prefix).resolve()
    executable = Path(probe['python_executable']).resolve()
    if Path(probe['python_prefix']).resolve() != prefix or not executable.is_relative_to(prefix) or not executable.is_file():
        raise ValueError('Probe did not run in the newly created Python environment')
    packages = probe.get('packages', [])
    names = [item.split('==')[0] for item in packages]
    if len(names) != len(set(names)) or not set(PACKAGES).issubset(packages):
        raise ValueError('Fresh environment must contain all exact pinned packages')
    if any(item.split('==')[0] not in {'numpy','threadpoolctl','psutil','pip','setuptools'} for item in packages):
        raise ValueError('Fresh environment contains an unapproved extra dependency')
    if probe.get('numpy_version') != '2.5.3' or probe.get('torch_available') is not False or probe.get('tensorflow_available') is not False:
        raise ValueError('Fresh environment must be NumPy-only with no Torch/TensorFlow')
    if not probe.get('site_packages') or any(not Path(p).resolve().is_relative_to(prefix) for p in probe['site_packages']):
        raise ValueError('Fresh package paths escape the new environment')
    if not Path(probe['base_executable']).is_file() or Path(probe['base_executable']).resolve().is_relative_to(prefix):
        raise ValueError('A direct base executable is required, avoiding the Windows venv launcher')
    base_prefix = Path(probe['base_prefix']).resolve()
    if base_prefix == prefix or not probe.get('runtime_paths') or any(
            not Path(p).resolve().is_relative_to(base_prefix) for p in probe['runtime_paths']):
        raise ValueError('Python standard-library paths must belong to the recorded base installation')
    return probe


def worker_guard(checkout, prefix, probe, phase, output, resource, predictions):
    if phase not in ('prediction', 'scoring'):
        raise ValueError('Only fixed validation prediction/scoring phases exist')
    checkout = Path(checkout).resolve()
    data = checkout / 'code/data'
    allowed = [data/'manifest.json',data/'candidates/gsm8k_validation.jsonl',data/'candidates/gsm8k_validation.npz']
    if phase == 'scoring':
        allowed.append(data/'gsm8k_validation.jsonl')
    scope = {'checkout':str(checkout),'data':str(data),'allowed_data':[str(p) for p in allowed],
        'code':str(checkout/'code'),'assets':str(checkout/CHECKPOINT),
        'runtime':[str(Path(p).resolve()) for p in [*probe['runtime_paths'],*probe['site_packages']]],
        'outputs':[str(Path(p).resolve()) for p in (output,resource)],
        'predictions':str(Path(predictions).resolve()),'phase':phase}
    return """import os,sys,json
from pathlib import Path
_reproduction_scope=json.loads(%r)
_reproduction_denied=[]
os.chdir(_reproduction_scope['checkout'])
def _reproduction_guard(event,args):
    if event.startswith('socket.') or event in ('subprocess.Popen','os.system','os.spawn','os.posix_spawn','os.fork','os.forkpty','os.exec'):
        _reproduction_denied.append(event)
        raise PermissionError('Clean reproduction worker is offline and cannot spawn')
    if event!='open' or isinstance(args[0],int):return
    p=Path(os.fsdecode(args[0])).resolve()
    mode,flags=args[1],args[2]
    writing=(isinstance(mode,str) and any(c in mode for c in 'wax+')) or flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC)
    allowed=str(p) in _reproduction_scope['outputs']
    if not writing:
        if p.is_relative_to(Path(_reproduction_scope['data'])):
            allowed=str(p) in _reproduction_scope['allowed_data']
        elif _reproduction_scope['phase']=='scoring' and str(p)==_reproduction_scope['predictions']:
            allowed=True
        elif p.is_relative_to(Path(_reproduction_scope['code'])):
            allowed=p.suffix.lower() in ('.py','.pyc','.pyd','.dll')
        elif p.is_relative_to(Path(_reproduction_scope['assets'])):
            allowed=True
        elif any(p==Path(base) or p.is_relative_to(Path(base)) for base in _reproduction_scope['runtime']):
            allowed=True
    if not allowed:
        _reproduction_denied.append('open:'+str(p))
        raise PermissionError('Clean worker cannot read references/history/outside files or modify inputs')
sys.addaudithook(_reproduction_guard)
""" % json.dumps(scope)


def phase_command(checkout, prefix, probe, phase, folder, timing):
    predictions = folder / 'validation-predictions.json'
    resource = folder / ('validation-' + phase + '-thread-pools.json')
    if phase == 'prediction':
        script, output = 'code/predict.py', predictions
        arguments = ['--method',METHOD,'--checkpoint',CHECKPOINT,'--split','validation','--output',str(output)]
    elif phase == 'scoring':
        script, output = 'code/score_predictions.py', folder / 'validation-score.json'
        arguments = ['--split','validation','--predictions',str(predictions),'--output',str(output)]
    else:
        raise ValueError('Unapproved reproduction phase')
    wrapper = worker_guard(checkout,prefix,probe,phase,output,resource,predictions) + timing.PHASE_WRAPPER
    wrapper += "\nif _reproduction_denied: raise PermissionError('Worker attempted a forbidden operation')\n"
    environment = {**THREAD_ENV,'PYTHONPATH':os.pathsep.join(probe['site_packages']),
                   'PYTHONNOUSERSITE':'1','PYTHONDONTWRITEBYTECODE':'1','PYTHONHOME':''}
    return [probe['base_executable'],'-S','-c',wrapper,script,str(resource),*arguments], environment


def compare_outputs(predictions, score, original_predictions, original_score):
    for row in (predictions, score, original_predictions, original_score):
        if row.get('protocol') != PROTOCOL or row.get('split') != 'validation':
            raise ValueError('Only the official full-validation output protocol is permitted')
    if (len(predictions.get('ids', [])) != 798 or len(set(predictions['ids'])) != 798
            or len(predictions.get('selected', [])) != 798
            or any(type(value) is not int or not 0 <= value < 4 for value in predictions['selected'])):
        raise ValueError('Exactly 798 ordered IDs and Python-int candidate choices are required')
    if predictions['ids'] != original_predictions.get('ids') or predictions['selected'] != original_predictions.get('selected'):
        raise ValueError('Fresh predictions differ from the unique locked original choices')
    if (type(score.get('n')) is not int or score['n'] != 798 or type(score.get('correct')) is not int
            or not 0 <= score['correct'] <= 798 or not isinstance(score.get('accuracy'), (float,int))
            or not math.isfinite(score['accuracy'])
            or not math.isclose(score['accuracy'],100*score['correct']/798,rel_tol=0.,abs_tol=1e-12)
            or any(score.get(key) != original_score.get(key) for key in ('correct','n','accuracy'))):
        raise ValueError('Fresh official score differs from the unique original score')


def reproduce(root, attempt, reason=None, *, freezing=None, timing=None):
    root = Path(root).resolve()
    if freezing is None or timing is None:
        freezing, timing = runtime()
    # Caller holds the shared root resource lock for the entire setup/run.
    identity = preflight(root, freezing, timing)
    folder, checkout, prefix = reserve_attempt(root, attempt, reason, identity)
    processes = []
    try:
        (root/'tmp').mkdir(exist_ok=True)
        git_options = ['git','-c','credential.helper=','-c','core.askPass=','-c','http.extraHeader=',
                       '-c','protocol.file.allow=never','-c','submodule.recurse=false']
        run_setup([*git_options,'clone','--no-checkout','--no-local',REPOSITORY,str(checkout)],root,folder,'setup-clone')
        run_setup([*git_options,'checkout','--detach',identity['source_commit']],checkout,folder,'setup-checkout')
        actual = subprocess.check_output(['git','rev-parse','HEAD'],cwd=checkout,text=True,env=setup_environment()).strip()
        if actual != identity['source_commit']:
            raise ValueError('Fresh public checkout is not the exact original HEAD')
        freezing.commit_bytes(checkout,actual,{**identity['source_files'],**identity['assets'],**identity['implementation_sha256']})
        if freezing.source_hashes(checkout) != identity['source_files'] or freezing.asset_hashes(checkout)[0] != identity['assets']:
            raise ValueError('Fresh public source/asset inventory differs from the selected committed bytes')
        data_inventory(checkout)
        base_python = str(Path(getattr(sys,'_base_executable',sys.executable)).resolve())
        run_setup([base_python,'-I','-m','venv',str(prefix)],root,folder,'setup-venv')
        python = prefix / ('Scripts/python.exe' if os.name=='nt' else 'bin/python')
        run_setup([str(python),'-I','-m','pip','--isolated','--disable-pip-version-check','install','--no-input',
            '--only-binary=:all:','--no-deps','--index-url','https://pypi.org/simple',*PACKAGES],root,folder,'setup-packages')
        run_setup([str(python),'-I','-c',PROBE],root,folder,'setup-environment-probe')
        probe = validate_probe(read(folder/'setup-environment-probe.stdout.log'),prefix)
        data = copy_validation_data(root,checkout)
        write_new(folder/'validation-data-files.json',data)
        environment = {'schema':'mp2-v3-clean-environment-v1','captured_at_utc':utc(),
            'source_commit':actual,'checkout':str(checkout),'source_files':identity['source_files'],
            'assets':identity['assets'],'data_manifest_sha256':sha(root/'code/data/manifest.json'),
            'official_data_files':data_inventory(checkout),'official_data_hashes':{row['relative_path']:row['sha256'] for row in data},
            'data_file_records':data,'public_clone_url':REPOSITORY,'setup_package_pins':list(PACKAGES),**probe}
        write_new(folder/'environment.json',environment)
        if timing.active_training():
            raise RuntimeError('Training appeared before reproduction timing')
        started = time.perf_counter()
        deadline = started+30.
        for phase in ('prediction','scoring'):
            if time.perf_counter() >= deadline:
                raise RuntimeError('The shared 30-second budget expired before the next phase')
            command, env = phase_command(checkout,prefix,probe,phase,folder,timing)
            result = timing.run_process(command,folder,'validation-'+phase,deadline,environment=env)
            processes.append(result)
            if result['returncode'] != 0 or result['timed_out']:
                raise RuntimeError('Fresh official phase failed or timed out; preserve this attempt')
        elapsed = time.perf_counter()-started
        libraries = [read(folder/('validation-'+phase+'-thread-pools.json')) for phase in ('prediction','scoring')]
        cap = all(row.get('torch_imported') is False and row.get('numpy_version')=='2.5.3'
            and all(pool.get('num_threads')==1 for pool in row['thread_pools'])
            and sum(pool.get('num_threads',999) for pool in row['thread_pools'])<=4 for row in libraries)
        peak = max(row['sampled_peak_tree_threads'] for row in processes)
        eligible = elapsed<=30. and 1<=peak<=4 and cap and identity['resource']['within_limit']
        resource = {'schema':'mp2-v3-clean-reproduction-timing-v1','status':'passed' if eligible else 'failed',
            'completed_at_utc':utc(),'prediction_plus_scoring_seconds':elapsed,'processes':processes,
            'sampled_peak_task_process_tree_threads':peak,
            'sampled_peak_worker_tree_plus_coordinator_threads':max(row['sampled_peak_tree_plus_coordinator_threads'] for row in processes),
            'scientific_library_observations':libraries,'scientific_library_thread_cap_passed':cap,
            'local_resource_eligible':eligible,'resource':identity['resource'],
            'hardware':{'platform':platform.platform(),'processor':platform.processor(),'python':sys.version},
            'guard_policy':'Offline direct workers; no child processes, prior results, non-validation data, or input/asset writes; prediction cannot read gold.',
            'measurement':'One shared 30-second deadline includes both sequential fresh official processes, startup, imports, input/assets, output, scoring and wrappers; setup excluded.',
            'limitation':'Local measurement only, not certification on the reference Xeon hardware.'}
        write_new(folder/'validation-timing.json',resource)
        if not eligible:
            raise RuntimeError('Clean reproduction violates measured local resource constraints')
        generation='research_v2' if identity['selected']['historical_fallback'] else 'research_v3'
        original=root/'results'/generation/'validation'/identity['selected']['name']
        compare_outputs(read(folder/'validation-predictions.json'),read(folder/'validation-score.json'),
                        read(original/'validation-predictions.json'),read(original/'validation-score.json'))
        after=preflight(root,freezing,timing)
        if after != identity or freezing.source_hashes(checkout)!=identity['source_files'] or freezing.asset_hashes(checkout)[0]!=identity['assets']:
            raise ValueError('Selection or source/assets changed during reproduction')
        if data_inventory(checkout)!=environment['official_data_files'] or any(sha(Path(row['destination']))!=row['sha256'] for row in data):
            raise ValueError('Fresh official data bytes changed during reproduction')
        evidence={relative(root,path):sha(path) for path in sorted(folder.iterdir()) if path.is_file()}
        evidence.update(identity['original_files'])
        typed={key:{'path':relative(root,folder/name),'sha256':sha(folder/name)} for key,name in (
            ('predictions_result','validation-predictions.json'),('score_result','validation-score.json'),
            ('timing_result','validation-timing.json'),('environment_result','environment.json'))}
        record={'schema':'mp2-v3-reproduction-result-v1','passed':True,'created_at_utc':utc(),
            'method':METHOD,'checkpoint':CHECKPOINT,'selected_model_sha256':identity['selected_model_sha256'],
            'source_files':identity['source_files'],'assets':identity['assets'],'source_commit':actual,
            'split':'validation','fresh_environment':True,'choices_match':True,'official_score_match':True,
            'evidence_files':evidence,**typed}
        write_new(folder/'reproduction-result.json',record)
        write_new(folder/'COMPLETE.json',{'schema':'mp2-v3-clean-reproduction-complete-v1','status':'passed',
            'completed_at_utc':utc(),'result_file':'reproduction-result.json','result_sha256':sha(folder/'reproduction-result.json')})
        return {'status':'passed','questions':798,'result':relative(root,folder/'reproduction-result.json'),
                'prediction_plus_scoring_seconds':elapsed,'test_evaluated':False}
    except BaseException as error:
        if not (folder/'FAILED.json').exists():
            write_new(folder/'FAILED.json',{'schema':'mp2-v3-clean-reproduction-failed-v1','failed_at_utc':utc(),
                'exception_type':type(error).__name__,'reason':str(error),'processes':processes,
                'recovery':'Preserve all files. Any new attempt requires an explicit engineering repair reason; model/asset changes and choosing a faster repeat are forbidden.'})
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attempt',required=True,help='New lowercase evidence-directory name; never reused')
    parser.add_argument('--engineering-reason',help='Required explanation for a new attempt after an earlier engineering failure')
    args=parser.parse_args()
    freezing,timing=runtime()
    with timing.exclusive_evaluation():
        result=reproduce(ROOT,args.attempt,args.engineering_reason,freezing=freezing,timing=timing)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
