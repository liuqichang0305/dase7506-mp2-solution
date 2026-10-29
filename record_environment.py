from pathlib import Path
import os, json, platform, sys, subprocess, psutil, hashlib
from datetime import datetime, timezone
root=Path(__file__).resolve().parent
pkgs=subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True)
(root/'requirements-lock.txt').write_text(pkgs,encoding='utf-8')
(root/'requirements-inference.txt').write_text('numpy==2.5.3\n',encoding='utf-8')
(root/'requirements-training.txt').write_text('numpy==2.5.3\nscipy==1.18.1\n',encoding='utf-8')
record={'captured_at':datetime.now(timezone.utc).isoformat(),'python':sys.version,'executable':sys.executable,'os':platform.platform(),'processor':platform.processor(),'cpu_model':'12th Gen Intel(R) Core(TM) i9-12900H (Windows registry)','logical_cpus':os.cpu_count(),'physical_cpus':psutil.cpu_count(logical=False),'ram_bytes':psutil.virtual_memory().total,'disk':psutil.disk_usage(str(root))._asdict(),'training_device':'CPU','inference_device':'CPU','thread_policy':'one BLAS/OpenMP thread; sequential child processes','reference_machine':False,'gpu_detected':'NVIDIA GeForce RTX 3070 Ti Laptop GPU, 8192 MiB; not used','upstream_commit':'bb801f4f2e7629d270df765ab12c333b2dc616bb'}
(root/'results/environment.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
print(json.dumps(record,indent=2))
