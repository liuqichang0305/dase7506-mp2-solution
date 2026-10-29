"""Recover exact upstream git bytes and record stage-zero evidence."""
from pathlib import Path
import subprocess, json, hashlib, sys, platform, os, shutil
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent
UP = ROOT / 'upstream'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
files = subprocess.check_output(['git','ls-tree','-r','--name-only','HEAD'],cwd=UP,text=True).splitlines()
records=[]
for name in files:
    if name == '.gitignore': continue
    content=subprocess.check_output(['git','show',f'HEAD:{name}'],cwd=UP)
    target=ROOT/name
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_bytes(content)
    records.append({'path':name,'sha256':sha(target),'bytes':len(content)})
commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=UP,text=True).strip()
(ROOT/'results/official_files.json').write_text(json.dumps({'upstream_commit':commit,'obtained_at':datetime.now(timezone.utc).isoformat(),'source':'git object bytes (Windows checkout CRLF avoided)','files':records},indent=2),encoding='utf-8')
env=os.environ.copy()
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS']: env[key]='1'
cmd=[sys.executable,'check_data.py']
p=subprocess.run(cmd,cwd=ROOT/'code',env=env,capture_output=True,text=True)
for stream in ['stdout','stderr']:(ROOT/f'logs/check_data_raw.{stream}.txt').write_text(getattr(p,stream),encoding='utf-8')
(ROOT/'logs/check_data_raw.run.json').write_text(json.dumps({'command':cmd,'cwd':str(ROOT/'code'),'returncode':p.returncode,'time':datetime.now(timezone.utc).isoformat()},indent=2),encoding='utf-8')
print(p.stdout)
print(p.stderr)
print('exit=',p.returncode)
sys.exit(p.returncode)
