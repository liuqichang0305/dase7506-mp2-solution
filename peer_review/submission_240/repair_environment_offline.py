"""Preserve failed network setup and copy only verified inference distributions."""
from pathlib import Path
from datetime import datetime, timezone
import base64
import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
RUNTIME=ROOT/'tmp/peer_review/submission_240_7e3a2792'
DEST=RUNTIME/'venv/Lib/site-packages'
SOURCE=ROOT/'.venv/Lib/site-packages'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):
    with path.open('x',encoding='utf-8') as stream:json.dump(value,stream,indent=2);stream.write('\n')


def main():
    if (OUT/'environment.json').exists():raise FileExistsError('Environment already recorded')
    records=[]
    for name,version in [('numpy','2.5.3'),('threadpoolctl','3.7.0'),('psutil','7.2.2')]:
        dist=importlib.metadata.distribution(name)
        if dist.version!=version:raise ValueError('Source dependency version mismatch')
        for item in dist.files:
            path=Path(dist.locate_file(item)).resolve()
            if not path.is_relative_to(SOURCE.resolve()) or '__pycache__' in path.parts:
                continue  # CLI entrypoints and regenerable pyc files are not inference requirements.
            relative=path.relative_to(SOURCE.resolve())
            destination=DEST/relative
            if destination.exists():raise FileExistsError('Do not overwrite any dependency')
            content=path.read_bytes()
            if item.hash:
                if item.hash.mode!='sha256':raise ValueError('Unexpected wheel RECORD digest')
                actual=base64.urlsafe_b64encode(hashlib.sha256(content).digest()).decode().rstrip('=')
                if actual!=item.hash.value:raise ValueError('Installed dependency differs from wheel RECORD: '+str(relative))
            destination.parent.mkdir(parents=True,exist_ok=True)
            with destination.open('xb') as stream:stream.write(content)
            records.append({'distribution':name,'version':version,'path':relative.as_posix(),
                'source':str(path),'destination':str(destination),'sha256':sha(destination),
                'wheel_record_hash_verified':item.hash is not None,'bytes':len(content)})
    write(OUT/'environment_repair.json',{'schema':'mp2-peer-environment-repair-v1','status':'complete',
        'created_at_utc':datetime.now(timezone.utc).isoformat(),
        'reason':'First anonymous binary-only pip install timed out after120s with no output. Original setup logs retained; no peer code had run.',
        'repair':'Fresh venv retained; copied only three already-installed exact-version inference/instrumentation distributions, checking each available wheel RECORD hash and every destination SHA256. No training dependencies, model weights or cached predictions copied.',
        'limitation':'Dependency bytes came from the existing trusted local installation; this is offline environment reconstruction, not proof of a fresh PyPI download.',
        'files':records})
    env={k:os.environ[k] for k in ('SYSTEMROOT','WINDIR','PATH') if k in os.environ}
    env.update({'PYTHONPATH':str(DEST),'PYTHONNOUSERSITE':'1','PYTHONDONTWRITEBYTECODE':'1',
                'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1'})
    code="import sys,json,importlib.metadata,numpy,threadpoolctl,psutil;print(json.dumps({'python_executable':sys.executable,'python_prefix':sys.prefix,'python_version':sys.version,'packages':{d.metadata['Name'].lower():d.version for d in importlib.metadata.distributions()},'numpy_file':numpy.__file__,'numpy_version':numpy.__version__,'thread_pools':threadpoolctl.threadpool_info()}))"
    command=[str(Path(sys._base_executable).resolve()),'-S','-c',code]
    result=subprocess.run(command,cwd=RUNTIME,env=env,capture_output=True,timeout=20)
    for label,value in [('stdout',result.stdout),('stderr',result.stderr)]:
        with (OUT/('environment_probe.'+label+'.log')).open('xb') as stream:stream.write(value)
    if result.returncode:raise RuntimeError('Repaired environment probe failed')
    probe=json.loads(result.stdout)
    probe.update(schema='mp2-peer-environment-v1',created_at_utc=datetime.now(timezone.utc).isoformat(),
        venv_directory=str(RUNTIME/'venv'),pythonpath=str(DEST),command=command,
        reconstruction='See environment_repair.json; direct base -S imports only the explicit fresh venv site-packages.')
    write(OUT/'environment.json',probe)
    print(json.dumps({'status':'repaired','copied_files':len(records),'packages':probe['packages']}))


if __name__=='__main__':main()
