"""One final authorized V3 course issue, with immutable evidence and recovery.

Importing this module performs no I/O. Only the explicit --submit CLI calls the
network. Credentials come from the existing Git helper and are never persisted.
No training or research module imports this post-freeze aggregate-score tool.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import urllib.error
import urllib.parse
import urllib.request
import uuid

ROOT = Path(__file__).resolve().parents[2]
ACCOUNT = 'liuqichang0305'
STUDENT = '3036900957'
REPOSITORY = ACCOUNT + '/dase7506-mp2-solution'
COURSE = '/repos/xudongwu-0/xudongwu-0.github.io/issues'
DEADLINE = datetime(2026, 10, 31, 16, tzinfo=timezone.utc)
DEADLINE_TEXT = '2026-10-31T16:00:00Z'


def utcnow():
    return datetime.now(timezone.utc)


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def digest(path):
    return sha_bytes(Path(path).read_bytes())


def timestamp(value):
    parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if parsed.tzinfo is None:
        raise ValueError('Evidence timestamps must have a timezone')
    return parsed.astimezone(timezone.utc)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_new(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def inside(root, relative):
    if not isinstance(relative, str) or '\\' in relative or PurePosixPath(relative).is_absolute() or '..' in PurePosixPath(relative).parts:
        raise ValueError('Unsafe relative evidence path')
    target = (Path(root) / relative).resolve()
    if target == Path(root).resolve() or not target.is_relative_to(Path(root).resolve()):
        raise ValueError('Evidence path escapes its expected directory')
    return target


def checked_hash(path, expected):
    if not isinstance(expected, str) or re.fullmatch('[0-9a-f]{64}', expected) is None or digest(path) != expected:
        raise ValueError('Evidence hash mismatch: ' + Path(path).name)


def verify_git(root, freeze, commit):
    """Verify a real ancestor relation and the bytes at the submitted commit."""
    def git(*args):
        result = subprocess.run(['git', *args], cwd=root, capture_output=True, timeout=60)
        if result.returncode:
            raise ValueError('Frozen/publication Git commit verification failed')
        return result.stdout
    git('merge-base', '--is-ancestor', freeze['source_commit'], commit)
    for relative, expected in {**freeze['source_files'], **freeze['assets']}.items():
        if sha_bytes(git('show', f'{commit}:{relative}')) != expected:
            raise ValueError('Submitted immutable commit differs from frozen bytes: ' + relative)
    for relative, expected in freeze['source_files'].items():
        if sha_bytes(git('show', f"{freeze['source_commit']}:{relative}")) != expected:
            raise ValueError('Freeze source commit differs from frozen source bytes: ' + relative)


def official_phase(root, generated, snapshot, now):
    dependencies = snapshot['published_module_graph'][snapshot['published_entry_path']]
    arena = [name for name in dependencies if re.search(r'/arena\.[0-9a-f]+\.mjs$', name)]
    if len(arena) != 1:
        raise ValueError('Reviewed published arena module is missing/ambiguous')
    item = next(record for record in snapshot['files'] if record['path'] == arena[0])
    module = inside(Path(root) / 'tmp/form-source-v3', item.get('local_path', PurePosixPath(item['path']).name))
    script = ("const m=await import(process.argv[1]); const publication=JSON.parse(process.argv[2]); "
              "console.log(JSON.stringify({phase:m.submissionPhase('mp2',publication,Date.parse(process.argv[3])),"
              "deadline:m.PROJECTS.mp2.score_deadline,repository:m.REPO}));")
    result = subprocess.run(['node', '--input-type=module', '-e', script, module.as_uri(),
                             json.dumps(generated['captured_board_publication']), now.isoformat()],
                            cwd=root, capture_output=True, text=True, timeout=30)
    if result.returncode:
        raise ValueError('Local official phase evaluation failed; inspect reviewed source before retrying')
    phase = json.loads(result.stdout)
    if phase['phase'] != 'scores' or timestamp(phase['deadline']) != DEADLINE or phase['repository'] != COURSE.split('/repos/')[1].removesuffix('/issues'):
        raise ValueError('Current official form phase/deadline/repository differs; no new issue permitted')


def validate_inputs(root, now, *, for_post=False, git_check=verify_git, phase_check=official_phase):
    root, folder = Path(root).resolve(), Path(root).resolve() / 'submission/v3'
    if not (root / 'results/research_v3/SEARCH_CLOSED.json').is_file():
        raise ValueError('Research must be closed before publication')
    generated = read(folder / 'GENERATED_FORM.json')
    freeze = read(folder / 'FREEZE.json')
    if freeze.get('frozen') is not True or freeze.get('test_evaluated_before_freeze') is not False:
        raise ValueError('A valid pre-test final freeze is required')
    if any(not isinstance(commit, str) or re.fullmatch('[0-9a-f]{40}', commit) is None for commit in
           (freeze.get('source_commit'), generated.get('final_code_commit'))):
        raise ValueError('Full immutable source/publication commits are required')
    if generated.get('status') != 'prepared_not_submitted' or generated.get('student_id') != STUDENT or generated.get('intended_github_account') != ACCOUNT or generated.get('expected_issue_author_type') != 'User':
        raise ValueError('Official generated form identity/status mismatch')
    if generated.get('declaration_checked') is not True or generated.get('phase') != 'scores' or generated.get('official_phase_badge') != 'Score submissions open':
        raise ValueError('Official form declaration/score phase is missing')
    checked_hash(folder / 'FREEZE.json', generated['freeze_file_sha256'])
    if generated['freeze_source_commit'] != freeze['source_commit']:
        raise ValueError('Generated form differs from frozen source commit')
    if not freeze.get('source_files') or not freeze.get('assets') or set(freeze['source_files']) & set(freeze['assets']):
        raise ValueError('Distinct nonempty frozen source and asset maps are required')
    frozen_files = {**freeze['source_files'], **freeze['assets']}
    for relative, expected in frozen_files.items():
        if relative in freeze['assets'] and not relative.startswith('artifacts/research_v3/final_selected/'):
            raise ValueError('Frozen assets must be the selected V3 inference bundle')
        checked_hash(inside(root, relative), expected)
    git_check(root, freeze, generated['final_code_commit'])
    expected_code = f"https://github.com/{REPOSITORY}/tree/{generated['final_code_commit']}"
    if generated['code_url'] != expected_code or generated['checkpoint_url'] != expected_code + '/artifacts/research_v3/final_selected':
        raise ValueError('Code and asset links must use the same approved immutable commit')
    title_bytes, body_bytes = ((folder / name).read_bytes() for name in ('issue-title.txt', 'issue-body.txt'))
    title, body = title_bytes.decode('utf-8'), body_bytes.decode('utf-8')
    if title != generated['issue_title'] or body != generated['issue_body']:
        raise ValueError('Issue text differs from the exact official generated output')
    checked_hash(folder / 'issue-title.txt', generated['issue_title_sha256'])
    checked_hash(folder / 'issue-body.txt', generated['issue_body_sha256'])
    checked_hash(folder / 'issue-url.txt', generated['issue_url_sha256'])
    if (folder / 'issue-url.txt').read_text(encoding='utf-8') != generated['issue_url']:
        raise ValueError('Issue URL file differs from official generated output')
    issue_url = urllib.parse.urlsplit(generated['issue_url'])
    query = urllib.parse.parse_qs(issue_url.query)
    if (issue_url.scheme, issue_url.netloc, issue_url.path) != ('https', 'github.com', '/xudongwu-0/xudongwu-0.github.io/issues/new') or query != {'title': [title], 'body': [body]}:
        raise ValueError('Official generated issue URL does not round-trip exact text')
    blocks = re.findall(r'```json\s*\n([\s\S]*?)\n```', body)
    if len(blocks) != 1 or blocks[0] != generated['official_payload_json_text'] or json.loads(blocks[0]) != generated['official_payload']:
        raise ValueError('Official payload text is inconsistent')
    payload = generated['official_payload']
    if payload.get('schema') != 'dase7506/submission-v1' or payload.get('project') != 'mp2' or payload.get('student_id') != STUDENT:
        raise ValueError('Official payload schema/project/student mismatch')
    if generated['score_file'] != 'submission/v3/test-score.json':
        raise ValueError('Only the final V3 aggregate score artifact is accepted')
    checked_hash(folder / 'test-score.json', generated['score_file_sha256'])
    completion = read(folder / 'TEST_COMPLETE.json')
    if (completion.get('schema') != 'mp2-v3-test-complete-v1' or completion.get('status') != 'passed'
            or completion.get('local_resource_eligible') is not True
            or completion.get('freeze_sha256') != generated['freeze_file_sha256']
            or completion.get('method') != 'code/solution_v3.py'
            or completion.get('checkpoint') != 'artifacts/research_v3/final_selected'
            or completion.get('n') != 467 or completion.get('split') != 'test'):
        raise ValueError('A passing once-only complete frozen test evaluation is required')
    if completion['output_sha256'].get('submission/v3/test-score.json') != generated['score_file_sha256']:
        raise ValueError('Official score differs from the once-only test completion')
    for relative, expected in completion['output_sha256'].items():
        checked_hash(inside(root, relative), expected)
    score = read(folder / 'test-score.json')
    if score != generated['official_score'] or score.get('split') != 'test' or score.get('protocol') != payload.get('protocol'):
        raise ValueError('Frozen aggregate score differs from official generated form')
    if type(score.get('n')) is not int or score['n'] != completion['n'] or type(score.get('correct')) is not int or not 0 <= score['correct'] <= score['n']:
        raise ValueError('Invalid aggregate score counts')
    if not math.isfinite(score['accuracy']) or abs(score['accuracy'] - 100 * score['correct'] / score['n']) > 1e-12 or payload['score'] != score['accuracy']:
        raise ValueError('Aggregate score and payload percentage differ')
    if timestamp(generated['generated_at_utc']) < timestamp(freeze['created_at_utc']):
        raise ValueError('Generated form predates freeze')
    snapshot_path = folder / 'FORM_SOURCE_SNAPSHOT.json'
    checked_hash(snapshot_path, generated['official_source_snapshot_sha256'])
    snapshot = read(snapshot_path)
    if snapshot['official_site_commit'] != generated['official_source_commit'] or snapshot['files'] != generated['official_source_files']:
        raise ValueError('Official source snapshot differs from executed form evidence')
    if re.fullmatch('[0-9a-f]{40}', snapshot['official_site_commit']) is None or generated['official_executed_entry'] != snapshot['published_entry_path']:
        raise ValueError('Official immutable source commit/entry differs')
    if timestamp(snapshot['retrieved_at_utc']) < timestamp(freeze['created_at_utc']):
        raise ValueError('Official form sources must be refreshed after final freeze')
    for item in snapshot['files']:
        if item['url'] != f"https://raw.githubusercontent.com/xudongwu-0/xudongwu-0.github.io/{snapshot['official_site_commit']}/{item['path']}":
            raise ValueError('Official source URL is not the verified course commit')
        path = inside(root / 'tmp/form-source-v3', item.get('local_path', PurePosixPath(item['path']).name))
        checked_hash(path, item['sha256'])
        if path.stat().st_size != item['bytes']:
            raise ValueError('Official source snapshot byte count mismatch')
    publication = read(folder / 'PUBLICATION_VERIFICATION.json')
    if publication.get('schema') != 'mp2-v3-publication-verification-v1' or publication.get('passed') is not True:
        raise ValueError('Passed public artifact verification is required')
    for key in ('final_code_commit', 'code_url', 'checkpoint_url', 'freeze_file_sha256'):
        if publication.get(key) != generated[key]:
            raise ValueError('Public verification differs from generated form: ' + key)
    for key in ('source_commit', 'source_files', 'assets'):
        if publication.get(key) != freeze[key]:
            raise ValueError('Public verification differs from frozen files: ' + key)
    if timestamp(publication['verified_at_utc']) < timestamp(freeze['created_at_utc']):
        raise ValueError('Public verification predates final freeze')
    checks = publication['anonymous_file_checks']
    if len(checks) != len(frozen_files) or {item['path'] for item in checks} != set(frozen_files):
        raise ValueError('Anonymous public checks must cover every frozen source and asset')
    for item in checks:
        _check_public_file(root, generated['final_code_commit'], item, frozen_files[item['path']])
    report = publication['report_check']
    if not report['path'].startswith('reports/') or not report['path'].endswith('.pdf') or type(report['pages']) is not int or not 1 <= report['pages'] <= 10:
        raise ValueError('A public report PDF of at most ten pages is required')
    _check_public_file(root, generated['final_code_commit'], report, report['sha256'])
    if for_post:
        if now >= DEADLINE:
            raise ValueError('New-score deadline has passed; no new issue permitted')
        rules = read(folder / 'RULES_RECHECK.json')
        checked = timestamp(rules['checked_at_utc'])
        retrieved = timestamp(rules['retrieved_at_utc'])
        if not timestamp(freeze['created_at_utc']) <= retrieved <= checked <= now or now - retrieved > timedelta(hours=1):
            raise ValueError('Final public-rule recheck must be after freeze and within one hour')
        if rules.get('no_conflicting_notice_found') is not True or rules.get('phase') != 'scores' or rules.get('deadline_utc') != DEADLINE_TEXT:
            raise ValueError('Final public-rule recheck does not permit a new score')
        for key, value in [('official_site_commit', generated['official_source_commit']),
                           ('official_source_snapshot_sha256', generated['official_source_snapshot_sha256'])]:
            if rules.get(key) != value:
                raise ValueError('Final public-rule evidence differs from the official form source')
        if not rules.get('evidence_sources'):
            raise ValueError('Bounded checked public sources must be recorded')
        for item in rules['evidence_sources']:
            url = urllib.parse.urlsplit(item['url'])
            prefixes = {'github.com': '/xudongwu-0/xudongwu-0.github.io/',
                        'raw.githubusercontent.com': '/xudongwu-0/xudongwu-0.github.io/',
                        'api.github.com': '/repos/xudongwu-0/xudongwu-0.github.io/',
                        'xudongwu-0.github.io': '/courses/dase7506/'}
            if url.scheme != 'https' or url.netloc not in prefixes or not url.path.startswith(prefixes[url.netloc]):
                raise ValueError('Rule-check source must be an official public HTTPS source')
            if not retrieved <= timestamp(item['retrieved_at_utc']) <= checked:
                raise ValueError('Rule-check evidence retrieval timestamp differs')
        phase_check(root, generated, snapshot, now)
    return {'title': title, 'body': body, 'title_sha256': sha_bytes(title_bytes), 'body_sha256': sha_bytes(body_bytes),
            'generated_form_sha256': digest(folder / 'GENERATED_FORM.json'), 'freeze_sha256': digest(folder / 'FREEZE.json'),
            'publication_verification_sha256': digest(folder / 'PUBLICATION_VERIFICATION.json'),
            'test_completion_sha256': digest(folder / 'TEST_COMPLETE.json'),
            'score_file_sha256': generated['score_file_sha256'], 'final_code_commit': generated['final_code_commit']}


def _check_public_file(root, commit, item, expected):
    path = inside(root, item['path'])
    checked_hash(path, expected)
    if item['sha256'] != expected or item.get('http_status') != 200 or item['bytes'] != path.stat().st_size or item['url'] != f"https://raw.githubusercontent.com/{REPOSITORY}/{commit}/{item['path']}":
        raise ValueError('Anonymous immutable public file verification differs')


class TransportFailure(RuntimeError):
    def __init__(self, status=None):
        self.http_status = status
        super().__init__('GitHub request failed; credentials and remote error body are withheld')


class GitHubTransport:
    def __init__(self, root):
        env = os.environ.copy()
        env.update(GCM_INTERACTIVE='never', GIT_TERMINAL_PROMPT='0')
        try:
            result = subprocess.run(['git', 'credential', 'fill'], input='protocol=https\nhost=github.com\n\n',
                                    cwd=root, capture_output=True, text=True, env=env, timeout=30)
        except Exception:
            raise RuntimeError('Existing Git credentials unavailable; no credential contents logged') from None
        credential = dict(line.split('=', 1) for line in result.stdout.splitlines() if '=' in line)
        if result.returncode or not credential.get('password'):
            raise RuntimeError('Existing Git credentials unavailable; no credential contents logged')
        self._token = credential['password']
        class NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, *args, **kwargs):
                raise TransportFailure()
        self._opener = urllib.request.build_opener(NoRedirect())

    def call(self, path, method='GET', data=None):
        query_prefix = COURSE + '?state=all&creator=' + ACCOUNT + '&per_page=100&page='
        allowed_get = path == '/user' or re.fullmatch(re.escape(COURSE) + r'/[1-9][0-9]*', path) or (
            path.startswith(query_prefix) and path[len(query_prefix):].isdigit())
        if not (method == 'GET' and allowed_get and data is None) and not (method == 'POST' and path == COURSE and set(data or {}) == {'title', 'body'}):
            raise ValueError('Request is outside the fixed personal course-submission scope')
        request = urllib.request.Request('https://api.github.com' + path, method=method,
            data=None if data is None else json.dumps(data, ensure_ascii=False).encode('utf-8'),
            headers={'Authorization': 'Bearer ' + self._token, 'Accept': 'application/vnd.github+json',
                     'Content-Type': 'application/json', 'User-Agent': 'DASE7506-final-v3-submission'})
        try:
            with self._opener.open(request, timeout=60) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            raise TransportFailure(error.code) from None
        except Exception:
            raise TransportFailure() from None


def validate_issue(issue, payload):
    if issue.get('user', {}).get('login') != ACCOUNT or issue['user'].get('type') != 'User' or 'pull_request' in issue:
        raise ValueError('Course issue author must be the existing personal account')
    if issue.get('title') != payload['title'] or sha_bytes((issue.get('body') or '').encode('utf-8')) != payload['body_sha256']:
        raise ValueError('Course issue does not exactly match the official title/body')
    number = issue.get('number')
    if type(number) is not int or number <= 0 or issue.get('html_url') != f'https://github.com/xudongwu-0/xudongwu-0.github.io/issues/{number}':
        raise ValueError('Unexpected course issue identity/URL')
    if timestamp(issue['created_at']) >= DEADLINE:
        raise ValueError('Course issue was created after the actual score deadline')
    return {key: issue[key] for key in ('number', 'html_url', 'title', 'body', 'user', 'created_at', 'state')}


def find_existing(transport, payload):
    for page in range(1, 10001):
        issues = transport.call(COURSE + f'?state=all&creator={ACCOUNT}&per_page=100&page={page}')
        if not isinstance(issues, list):
            raise ValueError('Invalid course issue list response; no POST is permitted')
        for item in issues:
            if item.get('title') == payload['title'] and sha_bytes((item.get('body') or '').encode('utf-8')) == payload['body_sha256']:
                validate_issue(item, payload)
                return item['number']
        if len(issues) < 100:
            return None
    raise ValueError('Course issue pagination limit reached; no POST is permitted')


@contextmanager
def submission_lock(folder):
    import psutil
    path = Path(folder) / 'submit.lock'
    owner = {'pid': os.getpid(), 'create_time': psutil.Process().create_time(), 'nonce': uuid.uuid4().hex}
    try:
        write_new(path, owner)
    except FileExistsError:
        old = read(path)
        try:
            live = abs(psutil.Process(old['pid']).create_time() - old['create_time']) < .01
        except psutil.NoSuchProcess:
            live = False
        if live:
            raise RuntimeError('Another final submission process owns the lock')
        path.rename(path.with_name('submit.stale-' + uuid.uuid4().hex + '.json'))
        write_new(path, owner)
    try:
        yield
    finally:
        if path.exists() and read(path) == owner:
            path.unlink()


def submit(root=ROOT, *, transport=None, now_fn=utcnow, git_check=verify_git, phase_check=official_phase):
    root, folder = Path(root).resolve(), Path(root).resolve() / 'submission/v3'
    with submission_lock(folder):
        payload = validate_inputs(root, now_fn(), git_check=git_check, phase_check=phase_check)
        transport = transport if transport is not None else GitHubTransport(root)
        identity = transport.call('/user')
        if identity.get('login') != ACCOUNT or identity.get('type') != 'User':
            raise ValueError('Personal GitHub identity mismatch; no issue sent')
        success_path = folder / 'SUBMISSION_SUCCESS.json'
        if success_path.exists():
            success = read(success_path)
            if success['body_sha256'] != payload['body_sha256'] or success['generated_form_sha256'] != payload['generated_form_sha256']:
                raise ValueError('Existing V3 submission receipt belongs to different generated content')
            checked_hash(inside(folder, success['readback_path']), success['readback_sha256'])
            validate_issue(read(inside(folder, success['readback_path'])), payload)
            issue = validate_issue(transport.call(COURSE + '/' + str(success['issue_number'])), payload)
            return {'status': 'existing_success_verified', 'issue_url': issue['html_url'], 'issue_number': issue['number']}
        attempt_root = folder / 'submission-attempts'
        attempt_root.mkdir(exist_ok=True)
        attempt = attempt_root / (now_fn().strftime('%Y%m%dT%H%M%S%fZ') + '-' + uuid.uuid4().hex[:8])
        attempt.mkdir()
        existing = find_existing(transport, payload)
        if existing is not None:
            issue = validate_issue(transport.call(COURSE + '/' + str(existing)), payload)
            return _save_success(folder, attempt, issue, payload, now_fn(), 'existing_remote_issue_reconciled')
        unresolved = list(attempt_root.glob('*/before-post.json'))
        if unresolved:
            write_new(attempt / 'unresolved-prior-attempt.json', {'checked_at_utc': now_fn().isoformat(),
                'body_sha256': payload['body_sha256'], 'prior_attempts': [path.relative_to(folder).as_posix() for path in unresolved],
                'status': 'No exact remote issue found. An earlier POST may have succeeded; automatic repost is prohibited.'})
            raise RuntimeError('Unresolved earlier POST; exact remote reconciliation is required before any further action')
        # Recheck every local binding, official phase and the deadline immediately before POST.
        checked = validate_inputs(root, now_fn(), for_post=True, git_check=git_check, phase_check=phase_check)
        if checked != payload:
            raise ValueError('Publication inputs changed during remote duplicate checking')
        write_new(attempt / 'before-post.json', {'before_post_utc': now_fn().isoformat(), 'destination': COURSE,
            'author': ACCOUNT, 'author_type': 'User', **{key: value for key, value in payload.items() if key not in ('title', 'body')},
            'rules_recheck_sha256': digest(folder / 'RULES_RECHECK.json')})
        if now_fn() >= DEADLINE:
            write_new(attempt / 'deadline-reached.json', {'status': 'POST not attempted: deadline reached'})
            raise ValueError('Deadline reached immediately before POST; no issue sent')
        try:
            created = transport.call(COURSE, 'POST', {'title': payload['title'], 'body': payload['body']})
            created = validate_issue(created, payload)
            write_new(attempt / 'post-response.json', created)
            readback = validate_issue(transport.call(COURSE + '/' + str(created['number'])), payload)
            return _save_success(folder, attempt, readback, payload, now_fn(), 'created_and_read_back')
        except Exception as error:
            write_new(attempt / 'uncertain-post-outcome.json', {'recorded_at_utc': now_fn().isoformat(),
                'exception_type': type(error).__name__, 'http_status': getattr(error, 'http_status', None),
                'status': 'May have been created. Retry performs identity/list/readback reconciliation, never a blind POST.'})
            raise RuntimeError('Final POST/readback outcome uncertain; preserved attempt evidence, reconcile before retrying') from None


def _save_success(folder, attempt, issue, payload, now, status):
    write_new(attempt / 'readback.json', issue)
    success = {'schema': 'mp2-v3-submission-success-v1', 'status': status, 'verified_at_utc': now.isoformat(),
               'issue_url': issue['html_url'], 'issue_number': issue['number'], 'created_at_utc': issue['created_at'],
               'author': ACCOUNT, 'author_type': 'User', 'student_id': STUDENT, 'before_deadline': True,
               'readback_path': (attempt / 'readback.json').relative_to(folder).as_posix(),
               'readback_sha256': digest(attempt / 'readback.json'),
               **{key: value for key, value in payload.items() if key not in ('title', 'body')}}
    write_new(folder / 'SUBMISSION_SUCCESS.json', success)
    return {'status': status, 'issue_url': issue['html_url'], 'issue_number': issue['number']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--submit', action='store_true', help='Execute the already-authorized final course submission')
    args = parser.parse_args()
    if not args.submit:
        parser.error('This script does nothing without --submit')
    try:
        result = submit()
    except Exception as error:
        # All external exceptions are wrapped without credentials/response bodies.
        raise SystemExit(str(error)) from None
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
