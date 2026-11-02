"""Only generated fixtures and fake transports; no real data, credentials or HTTP."""
from datetime import datetime, timedelta, timezone
import base64
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import urllib.parse

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('submit_final_v3', ROOT / 'submission/v3/submit_final.py')
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)
NOW = datetime(2026, 10, 31, 12, tzinfo=timezone.utc)


class FakeTransport:
    def __init__(self, issue, pages=None, ambiguous=False, identity=None):
        self.issue, self.pages, self.ambiguous = issue, pages or [[]], ambiguous
        self.identity = identity or {'login': s.ACCOUNT, 'type': 'User'}
        self.calls = []

    def call(self, path, method='GET', data=None):
        self.calls.append((method, path, data))
        if path == '/user':
            return self.identity
        if method == 'POST':
            if self.ambiguous:
                raise s.TransportFailure()
            return self.issue
        if '?' in path:
            page = int(urllib.parse.parse_qs(urllib.parse.urlsplit(path).query)['page'][0])
            return self.pages[page - 1]
        return self.issue


class SubmissionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=ROOT / 'tmp', prefix='submit-v3-synthetic-')
        self.root = Path(self.temporary.name)
        self.folder = self.root / 'submission/v3'
        self.folder.mkdir(parents=True)
        self.commit, self.official_commit = 'a' * 40, 'b' * 40
        self.source = self.put('code/synthetic_predict.py', b'# synthetic only\n')
        self.asset = self.put('artifacts/research_v3/final_selected/config.json', b'{"synthetic":true}\n')
        self.report = self.put('reports/synthetic.pdf', b'%PDF synthetic test fixture only')
        self.put('results/research_v3/SEARCH_CLOSED.json', b'{"synthetic":true}')
        self.freeze = {'frozen': True, 'test_evaluated_before_freeze': False, 'source_commit': self.commit,
                       'created_at_utc': (NOW - timedelta(minutes=30)).isoformat(),
                       'source_files': {self.source: s.digest(self.root / self.source)},
                       'assets': {self.asset: s.digest(self.root / self.asset)}}
        self.save('FREEZE.json', self.freeze)
        # A made-up aggregate only; no official test or score artifact is opened.
        score = {'split': 'test', 'protocol': s.SCORE_PROTOCOL, 'correct': 350, 'n': 467, 'accuracy': 100 * 350 / 467}
        self.save('test-score.json', score)
        opening = {'method': 'code/solution_v3.py', 'checkpoint': 'artifacts/research_v3/final_selected', 'n': 467, 'split': 'test',
                   'freeze_sha256': s.digest(self.folder / 'FREEZE.json'), 'opened_at_utc': (NOW - timedelta(minutes=20)).isoformat()}
        self.save('TEST_OPENED.json', opening)
        for relative in s.TEST_OUTPUTS:
            if not (self.root / relative).exists():
                self.put(relative, b'synthetic official-output placeholder')
        self.save('TEST_COMPLETE.json', {'schema': 'mp2-v3-test-complete-v1', 'status': 'passed',
            'local_resource_eligible': True, 'freeze_sha256': s.digest(self.folder / 'FREEZE.json'),
            'method': 'code/solution_v3.py', 'checkpoint': 'artifacts/research_v3/final_selected', 'n': 467, 'split': 'test',
            'completed_at_utc': (NOW - timedelta(minutes=19)).isoformat(),
            'output_sha256': {relative: s.digest(self.root / relative) for relative in s.TEST_OUTPUTS}})
        source_bytes = b'''// Synthetic module only; no official data or network.
export const PROTOCOL='7506-submissions-v1';
export const REPO='xudongwu-0/xudongwu-0.github.io';
export const PROJECTS={mp2:{score_deadline:'2026-11-01T00:00:00+08:00'}};
export function submissionPhase(){return 'scores'}
export function validateSubmission(p){return p.protocol===PROTOCOL}
export function validSealedArtifacts(a){return a.algorithm==='RSA-OAEP-256+AES-256-GCM'&&a.key_id==='7506-2026-v1'}
export function makeIssueURL(p){const q=new URLSearchParams({title:'[DASE7506] MP2 synthetic test',body:'Synthetic fixture only.\\n```json\\n'+JSON.stringify(p,null,2)+'\\n```'}); return 'https://github.com/'+REPO+'/issues/new?'+q.toString()}
'''
        source_path = 'courses/dase7506/assets/arena.1234.mjs'
        self.put('tmp/form-source-v3/served/' + source_path, source_bytes)
        source_record = {'path': source_path, 'local_path': 'served/' + source_path,
                         'sha256': s.sha_bytes(source_bytes), 'bytes': len(source_bytes),
                         'url': f'https://raw.githubusercontent.com/xudongwu-0/xudongwu-0.github.io/{self.official_commit}/{source_path}'}
        entry_record = {**source_record, 'path': 'courses/dase7506/assets/mp2.1234.mjs',
                        'local_path': 'served/courses/dase7506/assets/mp2.1234.mjs',
                        'url': f'https://raw.githubusercontent.com/xudongwu-0/xudongwu-0.github.io/{self.official_commit}/courses/dase7506/assets/mp2.1234.mjs'}
        self.put('tmp/form-source-v3/' + entry_record['local_path'], source_bytes)
        snapshot = {'official_site_commit': self.official_commit, 'retrieved_at_utc': (NOW - timedelta(minutes=10)).isoformat(),
                    'published_entry_path': 'courses/dase7506/assets/mp2.1234.mjs', 'files': [source_record, entry_record],
                    'published_module_graph': {'courses/dase7506/assets/mp2.1234.mjs': [source_path]}}
        self.save('FORM_SOURCE_SNAPSHOT.json', snapshot)
        plain = json.dumps({'schema': 'dase7506/artifact-links-v1', 'project': 'mp2', 'submission': None,
            'code_url': f'https://github.com/{s.REPOSITORY}/tree/{self.commit}',
            'checkpoint_url': f'https://github.com/{s.REPOSITORY}/tree/{self.commit}/artifacts/research_v3/final_selected'}, separators=(',', ':'))
        payload = {'schema': 'dase7506/submission-v1', 'project': 'mp2', 'student_id': s.STUDENT,
                   'score': score['accuracy'], 'protocol': '7506-submissions-v1', 'artifacts': {
                       'algorithm': 'RSA-OAEP-256+AES-256-GCM', 'key_id': '7506-2026-v1',
                       'wrapped_key': 'A' * 512, 'iv': 'A' * 16,
                       'ciphertext': base64.b64encode(b'\x00' * (len(plain.encode()) + 16)).decode()}}
        payload_text = json.dumps(payload, indent=2)
        self.title = '[DASE7506] MP2 synthetic test'
        self.body = 'Synthetic fixture only.\n```json\n' + payload_text + '\n```'
        url = 'https://github.com/xudongwu-0/xudongwu-0.github.io/issues/new?' + urllib.parse.urlencode({'title': self.title, 'body': self.body})
        for name, value in [('issue-title.txt', self.title), ('issue-body.txt', self.body), ('issue-url.txt', url)]:
            (self.folder / name).write_text(value, encoding='utf-8', newline='')
        self.generated = {'status': 'prepared_not_submitted', 'student_id': s.STUDENT,
            'intended_github_account': s.ACCOUNT, 'expected_issue_author_type': 'User',
            'declaration_checked': True, 'phase': 'scores', 'official_phase_badge': 'Score submissions open',
            'freeze_file_sha256': s.digest(self.folder / 'FREEZE.json'), 'freeze_source_commit': self.commit,
            'final_code_commit': self.commit, 'code_url': f'https://github.com/{s.REPOSITORY}/tree/{self.commit}',
            'checkpoint_url': f'https://github.com/{s.REPOSITORY}/tree/{self.commit}/artifacts/research_v3/final_selected',
            'issue_title': self.title, 'issue_body': self.body, 'issue_url': url,
            'issue_title_sha256': s.sha_bytes(self.title.encode()), 'issue_body_sha256': s.sha_bytes(self.body.encode()),
            'issue_url_sha256': s.sha_bytes(url.encode()), 'official_payload_json_text': payload_text, 'official_payload': payload,
            'score_file': 'submission/v3/test-score.json', 'score_file_sha256': s.digest(self.folder / 'test-score.json'),
            'official_score': score, 'generated_at_utc': (NOW - timedelta(minutes=5)).isoformat(),
            'official_source_snapshot_sha256': s.digest(self.folder / 'FORM_SOURCE_SNAPSHOT.json'),
            'official_source_commit': self.official_commit, 'official_source_files': [source_record, entry_record],
            'official_executed_entry': snapshot['published_entry_path'], 'captured_board_publication': {}}
        self.put('submission/v3/prepare_official_form.mjs', b'// synthetic adapter fixture')
        self.put('submission/v3/artifact_binding.mjs', b'// synthetic observer fixture')
        binding = {'schema': 'mp2-v3-artifact-binding-v1', 'plaintext_exact': plain, 'plaintext_sha256': s.sha_bytes(plain.encode()),
            'adapter_sha256': s.digest(self.folder / 'prepare_official_form.mjs'),
            'observer_sha256': s.digest(self.folder / 'artifact_binding.mjs'), 'official_entry_sha256': entry_record['sha256'],
            **{key + '_sha256': s.sha_bytes(base64.b64decode(payload['artifacts'][key])) for key in ('iv', 'ciphertext', 'wrapped_key')}}
        self.save('ARTIFACT_BINDING.json', binding)
        self.generated.update({'artifact_binding': binding, 'artifact_binding_file': 'ARTIFACT_BINDING.json',
                               'artifact_binding_file_sha256': s.digest(self.folder / 'ARTIFACT_BINDING.json')})
        self.save('GENERATED_FORM.json', self.generated)
        def public_check(relative):
            return {'path': relative, 'sha256': s.digest(self.root / relative), 'http_status': 200,
                    'bytes': (self.root / relative).stat().st_size,
                    'url': f'https://raw.githubusercontent.com/{s.REPOSITORY}/{self.commit}/{relative}'}
        self.publication = {'schema': 'mp2-v3-publication-verification-v1', 'passed': True,
            'verified_at_utc': (NOW - timedelta(minutes=4)).isoformat(), 'source_commit': self.commit,
            **{key: self.generated[key] for key in ('final_code_commit', 'code_url', 'checkpoint_url', 'freeze_file_sha256')},
            'source_files': self.freeze['source_files'], 'assets': self.freeze['assets'],
            'anonymous_file_checks': [public_check(self.source), public_check(self.asset)],
            'report_check': {**public_check(self.report), 'pages': 7}}
        self.save('PUBLICATION_VERIFICATION.json', self.publication)
        retrieved = (NOW - timedelta(minutes=2)).isoformat()
        self.rules = {'checked_at_utc': (NOW - timedelta(minutes=1)).isoformat(), 'retrieved_at_utc': retrieved,
            'no_conflicting_notice_found': True, 'phase': 'scores', 'deadline_utc': s.DEADLINE_TEXT,
            'official_site_commit': self.official_commit, 'official_source_snapshot_sha256': self.generated['official_source_snapshot_sha256'],
            'evidence_sources': [{'url': 'https://xudongwu-0.github.io/courses/dase7506/mp2/', 'retrieved_at_utc': retrieved}]}
        self.save('RULES_RECHECK.json', self.rules)
        self.issue = {'number': 999, 'html_url': 'https://github.com/xudongwu-0/xudongwu-0.github.io/issues/999',
                      'title': self.title, 'body': self.body, 'user': {'login': s.ACCOUNT, 'type': 'User'},
                      'created_at': NOW.isoformat(), 'state': 'open'}

    def tearDown(self):
        self.temporary.cleanup()

    def put(self, relative, data):
        target = self.root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        return relative

    def save(self, name, data):
        (self.folder / name).write_text(json.dumps(data, indent=2), encoding='utf-8')

    def run_submit(self, transport, now=NOW, phase_check=lambda *args: None):
        return s.submit(self.root, transport=transport, now_fn=lambda: now,
                        git_check=lambda *args: None, phase_check=phase_check)

    def test_new_issue_exact_post_readback_and_local_success_retry(self):
        api = FakeTransport(self.issue)
        result = self.run_submit(api)
        self.assertEqual(result['status'], 'created_and_read_back')
        posts = [call for call in api.calls if call[0] == 'POST']
        self.assertEqual(posts, [('POST', s.COURSE, {'title': self.title, 'body': self.body})])
        saved = s.read(self.folder / 'SUBMISSION_SUCCESS.json')
        self.assertEqual(saved['body_sha256'], self.generated['issue_body_sha256'])
        self.assertTrue(saved['before_deadline'])
        retry = FakeTransport(self.issue)
        # Existing verified success can be checked after the deadline, without POST.
        result = self.run_submit(retry, s.DEADLINE + timedelta(days=1))
        self.assertEqual(result['status'], 'existing_success_verified')
        self.assertFalse(any(call[0] == 'POST' for call in retry.calls))

    def test_paginated_duplicate_found_before_post(self):
        other = {**self.issue, 'title': 'another issue', 'body': 'different'}
        api = FakeTransport(self.issue, pages=[[other] * 100, [self.issue]])
        result = self.run_submit(api)
        self.assertEqual(result['status'], 'existing_remote_issue_reconciled')
        self.assertTrue(any('page=2' in call[1] for call in api.calls))
        self.assertFalse(any(call[0] == 'POST' for call in api.calls))

    def test_ambiguous_post_then_remote_reconciliation_without_second_post(self):
        first = FakeTransport(self.issue, ambiguous=True)
        with self.assertRaisesRegex(RuntimeError, 'uncertain'):
            self.run_submit(first)
        self.assertEqual(sum(call[0] == 'POST' for call in first.calls), 1)
        self.assertEqual(len(list((self.folder / 'submission-attempts').glob('*/uncertain-post-outcome.json'))), 1)
        second = FakeTransport(self.issue, pages=[[self.issue]])
        result = self.run_submit(second)
        self.assertEqual(result['status'], 'existing_remote_issue_reconciled')
        self.assertFalse(any(call[0] == 'POST' for call in second.calls))

    def test_ambiguous_post_unresolved_blocks_blind_repost(self):
        with self.assertRaises(RuntimeError):
            self.run_submit(FakeTransport(self.issue, ambiguous=True))
        retry = FakeTransport(self.issue)
        with self.assertRaisesRegex(RuntimeError, 'Unresolved earlier POST'):
            self.run_submit(retry)
        self.assertFalse(any(call[0] == 'POST' for call in retry.calls))

    def test_identity_deadline_and_official_phase_block_post(self):
        api = FakeTransport(self.issue, identity={'login': s.ACCOUNT, 'type': 'Organization'})
        with self.assertRaisesRegex(ValueError, 'identity mismatch'):
            self.run_submit(api)
        self.assertEqual(len(api.calls), 1)
        api = FakeTransport(self.issue)
        with self.assertRaisesRegex(ValueError, 'deadline'):
            self.run_submit(api, s.DEADLINE)
        self.assertFalse(any(call[0] == 'POST' for call in api.calls))
        def reject(*args):
            raise ValueError('Official phase closed')
        api = FakeTransport(self.issue)
        with self.assertRaisesRegex(ValueError, 'Official phase closed'):
            self.run_submit(api, phase_check=reject)
        self.assertFalse(any(call[0] == 'POST' for call in api.calls))

    def test_changed_asset_text_report_or_rules_rejected(self):
        api = FakeTransport(self.issue)
        (self.root / self.asset).write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            self.run_submit(api)
        self.assertFalse(api.calls)
        (self.root / self.asset).write_bytes(b'{"synthetic":true}\n')
        self.publication['report_check']['pages'] = 11
        self.save('PUBLICATION_VERIFICATION.json', self.publication)
        with self.assertRaisesRegex(ValueError, 'ten pages'):
            self.run_submit(api)
        self.publication['report_check']['pages'] = 7
        self.save('PUBLICATION_VERIFICATION.json', self.publication)
        self.rules['no_conflicting_notice_found'] = False
        self.save('RULES_RECHECK.json', self.rules)
        with self.assertRaisesRegex(ValueError, 'does not permit'):
            self.run_submit(api)
        self.assertFalse(any(call[0] == 'POST' for call in api.calls))

    def test_freshness_and_public_commit_mismatch_rejected(self):
        self.rules['retrieved_at_utc'] = (NOW - timedelta(hours=2)).isoformat()
        self.save('RULES_RECHECK.json', self.rules)
        with self.assertRaisesRegex(ValueError, 'within one hour'):
            self.run_submit(FakeTransport(self.issue))
        self.publication['final_code_commit'] = 'c' * 40
        self.save('PUBLICATION_VERIFICATION.json', self.publication)
        with self.assertRaisesRegex(ValueError, 'Public verification differs'):
            self.run_submit(FakeTransport(self.issue))

    def test_transport_allowlist_and_credential_errors_do_not_expose_secret(self):
        api = object.__new__(s.GitHubTransport)
        api._token = 'synthetic-secret-do-not-log'
        with self.assertRaisesRegex(ValueError, 'outside'):
            api.call('/repos/other/repo/issues', 'POST', {'title': 'x', 'body': 'y'})
        with patch.object(s.subprocess, 'run', side_effect=RuntimeError('synthetic-secret-do-not-log')):
            with self.assertRaises(RuntimeError) as error:
                s.GitHubTransport(self.root)
        self.assertNotIn('synthetic-secret', str(error.exception))
        class BrokenOpener:
            def open(self, *args, **kwargs):
                raise RuntimeError('synthetic-secret-do-not-log')
        api._opener = BrokenOpener()
        with self.assertRaises(s.TransportFailure) as error:
            api.call('/user')
        self.assertNotIn('synthetic-secret', str(error.exception))

    def test_changed_official_title_fails_before_any_network(self):
        with (self.folder / 'issue-title.txt').open('a', encoding='utf-8') as stream:
            stream.write('\n')
        api = FakeTransport(self.issue)
        with self.assertRaisesRegex(ValueError, 'exact official generated'):
            self.run_submit(api)
        self.assertFalse(api.calls)

    def test_git_verifier_checks_ancestor_and_committed_bytes(self):
        calls = []
        def fake_run(command, **kwargs):
            calls.append(command)
            value = b''
            if command[1] == 'show':
                relative = command[2].split(':', 1)[1]
                value = (self.root / relative).read_bytes()
            return SimpleNamespace(returncode=0, stdout=value)
        with patch.object(s.subprocess, 'run', side_effect=fake_run):
            s.verify_git(self.root, self.freeze, self.commit)
        self.assertEqual(calls[0][1:3], ['merge-base', '--is-ancestor'])
        self.assertEqual(sum(command[1] == 'show' for command in calls), 3)
        with patch.object(s.subprocess, 'run', return_value=SimpleNamespace(returncode=0, stdout=b'wrong bytes')):
            with self.assertRaisesRegex(ValueError, 'frozen bytes'):
                s.verify_git(self.root, self.freeze, self.commit)

    def test_official_phase_wrapper_checks_phase_deadline_repository(self):
        snapshot = s.read(self.folder / 'FORM_SOURCE_SNAPSHOT.json')
        result = {'phase': 'scores', 'deadline': '2026-11-01T00:00:00+08:00',
                  'repository': 'xudongwu-0/xudongwu-0.github.io', 'protocol': '7506-submissions-v1',
                  'payload_valid': True, 'issue_url': self.generated['issue_url']}
        with patch.object(s.subprocess, 'run', return_value=SimpleNamespace(returncode=0, stdout=json.dumps(result))) as call:
            s.official_phase(self.root, self.generated, snapshot, NOW)
        self.assertEqual(call.call_args.args[0][0], 'node')
        self.assertEqual(call.call_args.args[0][-2], NOW.isoformat())
        result['phase'] = 'artifacts'
        with patch.object(s.subprocess, 'run', return_value=SimpleNamespace(returncode=0, stdout=json.dumps(result))):
            with self.assertRaisesRegex(ValueError, 'phase/deadline/repository'):
                s.official_phase(self.root, self.generated, snapshot, NOW)

    def test_live_submission_lock_blocks_second_process(self):
        with s.submission_lock(self.folder):
            with self.assertRaisesRegex(RuntimeError, 'owns the lock'):
                with s.submission_lock(self.folder):
                    self.fail('Second submit acquired the live lock')

    def test_completion_subset_cannot_authorize_submission(self):
        completion = s.read(self.folder / 'TEST_COMPLETE.json')
        completion['output_sha256'] = {'submission/v3/test-score.json': s.digest(self.folder / 'test-score.json')}
        self.save('TEST_COMPLETE.json', completion)
        api = FakeTransport(self.issue)
        with self.assertRaisesRegex(ValueError, 'not a subset'):
            self.run_submit(api)
        self.assertFalse(api.calls)

    def test_form_and_score_protocols_are_distinct_and_official_payload_must_validate(self):
        self.assertNotEqual(self.generated['official_score']['protocol'], self.generated['official_payload']['protocol'])
        api = FakeTransport(self.issue)
        self.assertEqual(self.run_submit(api)['status'], 'created_and_read_back')
        snapshot = s.read(self.folder / 'FORM_SOURCE_SNAPSHOT.json')
        invalid = {'phase': 'scores', 'deadline': '2026-11-01T00:00:00+08:00',
                   'repository': 'xudongwu-0/xudongwu-0.github.io', 'protocol': '7506-submissions-v1',
                   'payload_valid': False, 'issue_url': self.generated['issue_url']}
        with patch.object(s.subprocess, 'run', return_value=SimpleNamespace(returncode=0, stdout=json.dumps(invalid))):
            with self.assertRaisesRegex(ValueError, 'payload/envelope'):
                s.official_phase(self.root, self.generated, snapshot, NOW)

    def test_observed_encryption_binding_must_match_public_links_and_envelope(self):
        snapshot = s.read(self.folder / 'FORM_SOURCE_SNAPSHOT.json')
        changed = json.loads(json.dumps(self.generated))
        changed['official_payload']['artifacts']['ciphertext'] = base64.b64encode(b'changed ciphertext').decode()
        with self.assertRaisesRegex(ValueError, 'observed encryption output'):
            s.verify_artifact_binding(self.root, self.folder, changed, snapshot)
        binding = s.read(self.folder / 'ARTIFACT_BINDING.json')
        plain = json.loads(binding['plaintext_exact'])
        plain['code_url'] = plain['code_url'].replace(self.commit, 'd' * 40)
        binding['plaintext_exact'] = json.dumps(plain, separators=(',', ':'))
        binding['plaintext_sha256'] = s.sha_bytes(binding['plaintext_exact'].encode())
        self.save('ARTIFACT_BINDING.json', binding)
        self.generated['artifact_binding'] = binding
        self.generated['artifact_binding_file_sha256'] = s.digest(self.folder / 'ARTIFACT_BINDING.json')
        self.save('GENERATED_FORM.json', self.generated)
        api = FakeTransport(self.issue)
        with self.assertRaisesRegex(ValueError, 'verified public artifact links'):
            self.run_submit(api)
        self.assertFalse(api.calls)

    def test_reconciliation_rejects_malformed_protocol_and_envelope_before_network(self):
        original = json.loads(json.dumps(self.generated))
        for malformed in ('protocol', 'algorithm'):
            self.generated = json.loads(json.dumps(original))
            payload = self.generated['official_payload']
            if malformed == 'protocol':
                payload['protocol'] = 'invalid-form-protocol'
            else:
                payload['artifacts']['algorithm'] = 'invalid-algorithm'
            payload_text = json.dumps(payload, indent=2)
            body = 'Synthetic fixture only.\n```json\n' + payload_text + '\n```'
            url = 'https://github.com/xudongwu-0/xudongwu-0.github.io/issues/new?' + urllib.parse.urlencode({'title': self.title, 'body': body})
            self.generated.update(official_payload_json_text=payload_text, issue_body=body, issue_url=url,
                issue_body_sha256=s.sha_bytes(body.encode()), issue_url_sha256=s.sha_bytes(url.encode()))
            (self.folder / 'issue-body.txt').write_text(body, encoding='utf-8', newline='')
            (self.folder / 'issue-url.txt').write_text(url, encoding='utf-8', newline='')
            self.save('GENERATED_FORM.json', self.generated)
            api = FakeTransport({**self.issue, 'body': body}, pages=[[{**self.issue, 'body': body}]])
            with self.subTest(malformed=malformed), self.assertRaisesRegex(ValueError, 'payload/envelope'):
                self.run_submit(api, s.DEADLINE + timedelta(days=1))
            self.assertFalse(api.calls)

    def test_public_file_paths_are_url_encoded(self):
        relative = self.put('reports/synthetic report 中.pdf', b'%PDF synthetic')
        item = {'path': relative, 'sha256': s.digest(self.root / relative), 'bytes': (self.root / relative).stat().st_size,
                'http_status': 200, 'url': f'https://raw.githubusercontent.com/{s.REPOSITORY}/{self.commit}/' + urllib.parse.quote(relative, safe='/')}
        s._check_public_file(self.root, self.commit, item, item['sha256'])
        item['url'] = item['url'].replace('%20', ' ')
        with self.assertRaisesRegex(ValueError, 'public file'):
            s._check_public_file(self.root, self.commit, item, item['sha256'])


if __name__ == '__main__':
    unittest.main()
