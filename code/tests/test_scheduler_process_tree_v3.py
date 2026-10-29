"""Actual Windows venv process-tree termination, without task data or training."""
import importlib.util
import subprocess
from pathlib import Path
import time
import unittest
from unittest.mock import patch

import psutil

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('tree_scheduler', ROOT / 'scripts/run_campaign_v3.py')
scheduler = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scheduler)


class ProcessTreeTest(unittest.TestCase):
    def test_actual_venv_launcher_and_sleep_descendants_are_stopped(self):
        code = ('import subprocess,sys,time; '
                'subprocess.Popen([sys.executable,"-c","import time;time.sleep(120)"]); '
                'time.sleep(120) # v3_tree_termination_test')
        parent = subprocess.Popen([str(ROOT / '.venv/Scripts/python.exe'), '-c', code], cwd=ROOT,
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        owner = psutil.Process(parent.pid)
        identity = {'pid': owner.pid, 'create_time': owner.create_time(),
                    'command_fragment': 'v3_tree_termination_test'}
        records, captured = [], []
        try:
            until = time.monotonic() + 5
            while time.monotonic() < until:
                captured = owner.children(recursive=True)
                if len(captured) >= 3:
                    break
                time.sleep(.05)
            self.assertGreaterEqual(len(captured), 1)
            with patch.object(scheduler, 'event', side_effect=lambda kind, **kw: records.append((kind, kw))):
                scheduler.terminate_owned(identity)
            self.assertIsNotNone(parent.poll())
            self.assertTrue(all(not p.is_running() for p in captured))
            self.assertEqual(records[-1][1]['remaining_pids'], [])
            self.assertGreaterEqual(len(records[-1][1]['processes']), 2)
            print({'synthetic_processes_stopped': len(records[-1][1]['processes']),
                   'real_data_read': False, 'training_started': False})
        finally:
            for process in list(reversed(captured)) + [owner]:
                try:
                    if process.is_running():
                        process.kill()
                except psutil.NoSuchProcess:
                    pass
            parent.wait(timeout=5)


if __name__ == '__main__':
    unittest.main()
