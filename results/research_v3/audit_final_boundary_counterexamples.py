"""Independent temporary-fixture counterexamples; no real evaluation or HTTP."""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / 'results/research_v3/independent-final-boundary-initial'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    OUTPUT.mkdir(exist_ok=False)
    sources = ['code/tools/freeze_v3.py', 'submission/v3/evaluate_final.py',
               'submission/v3/submit_final.py']
    before = {name: sha(ROOT / name) for name in sources}
    for name in sources:
        (OUTPUT / (Path(name).stem + '.source.py')).write_bytes((ROOT / name).read_bytes())
    fixture = load('synthetic_freeze_fixture', ROOT / 'code/tests/test_final_freeze_v3.py')
    submission = load('synthetic_submit_fixture', ROOT / 'code/tests/test_submit_final_v3.py')
    rows = []

    def check(name, operation, fixture_class=fixture.FinalFreezeTests):
        case = fixture_class()
        case.setUp()
        try:
            operation(case)
            rows.append({'case': name, 'unexpectedly_accepted': True})
        except Exception as error:
            rows.append({'case': name, 'unexpectedly_accepted': False,
                         'exception': type(error).__name__, 'message': str(error)})
        finally:
            case.tearDown()

    def wrong_oof(case):
        case.selected['oof_directory'] = 'results/research_v3/not_the_selected_model'
        case.save('submission/v3/SELECTED_MODEL.json', case.selected)
        fixture.f.validate_selected(case.root, case.verified)

    def text_only_gates(case):
        # Existing fixture intentionally has only docs/synthetic-audit.txt.
        assets, _ = fixture.f.asset_hashes(case.root)
        fixture.f._gates(case.root, fixture.f.sha(case.folder / 'SELECTED_MODEL.json'),
                        case.source_provider(case.root), assets, case.commit)

    def completion_subset(case):
        case.freeze()
        case.run_pipeline(case.fake_runner)
        complete = fixture.f.read(case.folder / 'TEST_COMPLETE.json')
        complete['output_sha256'] = {'submission/v3/test-score.json':
                                    fixture.f.sha(case.folder / 'test-score.json')}
        case.save('submission/v3/TEST_COMPLETE.json', complete)
        fixture.e.verify_complete(case.root, case.verify(case.root))

    def completion_wrong_scope(case):
        case.freeze()
        case.run_pipeline(case.fake_runner)
        complete = fixture.f.read(case.folder / 'TEST_COMPLETE.json')
        complete.update(method='changed.py', split='validation', n=1)
        case.save('submission/v3/TEST_COMPLETE.json', complete)
        fixture.e.verify_complete(case.root, case.verify(case.root))

    def worker_child(case):
        gold = case.root / 'code/data/gsm8k_test.jsonl'
        case.put('code/data/gsm8k_test.jsonl', b'generated synthetic sentinel only')
        hook = fixture.e.worker_guard(case.root, 'prediction', case.folder / 'test-predictions.json',
                                      case.folder / 'pool.json')
        child = 'from pathlib import Path; Path(' + repr(str(gold)) + ').read_bytes()'
        code = hook + '\nimport subprocess,sys\nsubprocess.run([sys.executable,"-S","-c",' + repr(child) + '],check=True)\n'
        process = subprocess.run([sys.executable, '-S', '-c', code], capture_output=True, text=True, timeout=10)
        if process.returncode:
            raise PermissionError(process.stderr[-1000:])

    def worker_socket_event(case):
        hook = fixture.e.worker_guard(case.root, 'prediction', case.folder / 'test-predictions.json',
                                      case.folder / 'pool.json')
        # Audit event only, no network packet or socket creation occurs.
        code = hook + '\nimport sys\nsys.audit("socket.sendto",None,("127.0.0.1",9))\n'
        process = subprocess.run([sys.executable, '-S', '-c', code], capture_output=True, text=True, timeout=10)
        if process.returncode:
            raise PermissionError(process.stderr[-1000:])

    def submit_subset(case):
        completion = submission.s.read(case.folder / 'TEST_COMPLETE.json')
        completion['output_sha256'] = {'submission/v3/test-score.json':
                                       submission.s.digest(case.folder / 'test-score.json')}
        case.save('TEST_COMPLETE.json', completion)
        submission.s.validate_inputs(case.root, submission.NOW, git_check=lambda *a: None)

    for name, operation in [('wrong_selected_oof_directory', wrong_oof),
                            ('arbitrary_text_only_review_and_reproduction', text_only_gates),
                            ('completion_output_inventory_subset', completion_subset),
                            ('completion_wrong_method_split_count', completion_wrong_scope),
                            ('prediction_child_can_read_generated_gold', worker_child),
                            ('unblocked_socket_sendto_audit_event', worker_socket_event)]:
        check(name, operation)
    check('submit_accepts_score_only_completion', submit_subset, submission.SubmissionTests)
    result = {'checked_at_utc': datetime.now(timezone.utc).isoformat(),
              'scope': 'Generated temporary files and fake process outputs only; no official validation/test, credentials, HTTP, submission or network packet',
              'source_sha256_before': before,
              'source_sha256_after': {name: sha(ROOT / name) for name in sources}, 'cases': rows}
    (OUTPUT / 'counterexamples.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
