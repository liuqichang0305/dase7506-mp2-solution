"""Record non-secret local resource and package metadata."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
import json
import platform
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone
import importlib.metadata
import psutil
import torch
import winreg
ROOT = Path(__file__).resolve().parents[2]
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r'HARDWARE\DESCRIPTION\System\CentralProcessor\0') as key:
    processor = winreg.QueryValueEx(key, 'ProcessorNameString')[0]
gpu = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader'], capture_output=True, text=True)
record = {'utc': datetime.now(timezone.utc).isoformat(), 'platform': platform.platform(),
          'python': sys.version, 'processor': processor.strip(), 'physical_cores': psutil.cpu_count(False),
          'logical_cores': psutil.cpu_count(), 'ram_bytes': psutil.virtual_memory().total,
          'gpu_inventory': gpu.stdout.strip(), 'training_device': 'cpu',
          'torch_threads': torch.get_num_threads(), 'torch_interop_threads': torch.get_num_interop_threads(),
          'torch_cuda_available': torch.cuda.is_available(), 'inference_device': 'cpu',
          'packages': {name: importlib.metadata.version(name) for name in ('numpy', 'scipy', 'scikit-learn', 'torch', 'threadpoolctl', 'psutil')},
          'reference_machine_certification': False, 'paid_compute': False}
(ROOT / 'results/research_v2/environment.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
print(json.dumps(record))
