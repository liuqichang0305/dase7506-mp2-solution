"""Synthetic HTTP only; no official score/data, network, credentials or issues."""
import copy
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import tempfile
import types
import unittest
from unittest.mock import patch
import urllib.error
import urllib.request

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('publication_v3_under_test',ROOT/'submission/v3/verify_publication.py')
p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)


def put(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(value if isinstance(value,bytes) else (json.dumps(value) if not isinstance(value,str) else value).encode('utf-8'))


class FakeHTTP:
    def __init__(self,files,overrides=None):self.files,self.overrides,self.calls=files,overrides or {},[]
    def fetch(self,url,max_bytes):
        self.calls.append((url,max_bytes))
        if url in self.overrides:
            override=self.overrides[url]
            if isinstance(override,Exception):raise override
            return override
        return {'http_status':200,'url':url,'body':self.files[url],'headers':{'ETag':'synthetic'}}


class PublicationTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory(prefix='mp2-v3-publication-synthetic-')
        self.root=Path(self.temporary.name).resolve()
        self.folder=self.root/'submission/v3'
        self.source_name='code/example space-é.py'
        self.asset_name=p.CHECKPOINT+'/weights.npz'
        put(self.root/self.source_name,'# synthetic public source\n')
        put(self.root/self.asset_name,b'synthetic numeric asset bytes')
        from pypdf import PdfWriter
        writer=PdfWriter();writer.add_blank_page(width=100,height=100);writer.add_blank_page(width=100,height=100)
        report=self.root/p.REPORT;report.parent.mkdir(parents=True,exist_ok=True)
        with report.open('wb') as f:writer.write(f)
        self.freeze={'schema':'dase7506-mp2-final-freeze-v3','frozen':True,'test_evaluated_before_freeze':False,
            'source_commit':'1'*40,'method':p.METHOD,'checkpoint':p.CHECKPOINT,
            'source_files':{self.source_name:p.sha(self.root/self.source_name)},
            'assets':{self.asset_name:p.sha(self.root/self.asset_name)}}
        put(self.folder/'FREEZE.json',self.freeze)
        self.complete={'schema':'mp2-v3-test-complete-v1','status':'passed','local_resource_eligible':True,
            'freeze_sha256':p.sha(self.folder/'FREEZE.json'),'method':p.METHOD,'checkpoint':p.CHECKPOINT,
            'split':'test','n':467,'output_sha256':{'submission/v3/test-score.json':'d'*64}}
        put(self.folder/'TEST_COMPLETE.json',self.complete)
        self.generated={'status':'prepared_not_submitted','freeze_file_sha256':p.sha(self.folder/'FREEZE.json'),
            'freeze_source_commit':'1'*40,'student_id':'3036900957','intended_github_account':'liuqichang0305',
            'score_file':'submission/v3/test-score.json','score_file_sha256':'d'*64,'final_code_commit':'2'*40,
            'code_url':f'https://github.com/{p.REPOSITORY}/tree/'+('2'*40),
            'checkpoint_url':f'https://github.com/{p.REPOSITORY}/tree/'+('2'*40)+'/'+p.CHECKPOINT}
        put(self.folder/'GENERATED_FORM.json',self.generated)
        self.files={p.raw_url('2'*40,name):(self.root/name).read_bytes() for name in (self.source_name,self.asset_name,p.REPORT)}
        self.git=lambda root,freeze,commit,report_hash:{'passed':True,'source_commit':freeze['source_commit'],
            'final_code_commit':commit,'report_sha256':report_hash,'commands':[]}

    def tearDown(self):self.temporary.cleanup()

    def run_fake(self,attempt='synthetic',overrides=None):
        transport=FakeHTTP(self.files,overrides)
        result=p.verify(self.root,attempt,transport=transport,git_check=self.git)
        return result,transport

    def test_success_exact_schema_maps_anonymous_urls_cache_and_real_pdf_pages(self):
        result,http=self.run_fake()
        self.assertEqual(result['report_pages'],2)
        self.assertEqual(len(http.calls),3)
        record=p.read(self.folder/'PUBLICATION_VERIFICATION.json')
        self.assertTrue(record['passed'])
        self.assertEqual(record['schema'],'mp2-v3-publication-verification-v1')
        self.assertEqual(record['source_files'],self.freeze['source_files'])
        self.assertEqual(record['assets'],self.freeze['assets'])
        self.assertEqual(record['code_url'],self.generated['code_url'])
        for row in [*record['anonymous_file_checks'],record['report_check']]:
            self.assertEqual(row['http_status'],200)
            self.assertTrue(row['anonymous'])
            self.assertFalse(row['authorization_header_sent'])
            self.assertEqual(p.sha(self.root/row['download_path']),row['sha256'])
            self.assertEqual(row['bytes'],(self.root/row['path']).stat().st_size)
        self.assertIn('example%20space-%C3%A9.py',record['anonymous_file_checks'][0]['url'] if record['anonymous_file_checks'][0]['path']==self.source_name else record['anonymous_file_checks'][1]['url'])
        with self.assertRaises(FileExistsError):self.run_fake('second')

    def test_missing_prerequisite_never_constructs_or_calls_http(self):
        (self.folder/'TEST_COMPLETE.json').unlink()
        with patch.object(p,'AnonymousTransport',side_effect=AssertionError('network construction forbidden')):
            with self.assertRaisesRegex(ValueError,'must all exist'):p.verify(self.root,'missing')
        self.assertFalse((self.folder/'publication-attempts').exists())

    def test_actual_score_path_is_never_opened(self):
        score_path=self.folder/'test-score.json'
        put(score_path,'must not be read')
        original=Path.open
        def guarded(path,*args,**kwargs):
            if path.resolve()==score_path.resolve():raise AssertionError('Actual score file opened')
            return original(path,*args,**kwargs)
        with patch.object(Path,'open',guarded):self.run_fake()

    def test_http_404_keeps_failed_attempt_and_explicit_retry_succeeds(self):
        url=p.raw_url('2'*40,self.source_name)
        http=FakeHTTP(self.files,{url:urllib.error.HTTPError(url,404,'synthetic missing',{},None)})
        with self.assertRaisesRegex(RuntimeError,'checks failed'):
            p.verify(self.root,'first',transport=http,git_check=self.git)
        self.assertEqual(len(http.calls),3)
        failed=p.read(self.folder/'publication-attempts/first/FAILED.json')
        self.assertFalse(failed['passed'])
        self.assertEqual(next(row for row in failed['anonymous_file_checks'] if row['path']==self.source_name)['http_status'],404)
        self.assertFalse((self.folder/'PUBLICATION_VERIFICATION.json').exists())
        self.run_fake('second')
        self.assertTrue((self.folder/'publication-attempts/first/FAILED.json').exists())

    def test_corrupt_hash_size_or_redirect_never_marks_passed(self):
        url=p.raw_url('2'*40,self.asset_name)
        for i,change in enumerate(({'body':b'corrupted'}, {'url':'https://example.com/not-approved'}, {'http_status':302})):
            override={'http_status':200,'url':url,'body':self.files[url],**change}
            with self.assertRaises(RuntimeError):self.run_fake('failed-'+str(i),{url:override})
            self.assertFalse((self.folder/'PUBLICATION_VERIFICATION.json').exists())

    def test_timeout_and_local_git_failure_are_retained_without_false_success(self):
        url=p.raw_url('2'*40,self.asset_name)
        with self.assertRaises(RuntimeError):self.run_fake('timeout',{url:TimeoutError('synthetic timeout')})
        with self.assertRaisesRegex(ValueError,'synthetic git failure'):
            p.verify(self.root,'git-failure',transport=FakeHTTP(self.files),
                git_check=lambda *args:(_ for _ in ()).throw(ValueError('synthetic git failure')))
        self.assertTrue((self.folder/'publication-attempts/git-failure/FAILED.json').exists())

    def test_changed_form_repo_commit_or_completion_rejected_before_http(self):
        for i,change in enumerate(({'code_url':'https://github.com/another/repo'},
                                   {'final_code_commit':'main'}, {'score_file_sha256':None})):
            put(self.folder/'GENERATED_FORM.json',{**self.generated,**change})
            http=FakeHTTP(self.files)
            with self.assertRaises(ValueError):p.verify(self.root,'bad-form-'+str(i),transport=http,git_check=self.git)
            self.assertEqual(http.calls,[])
        put(self.folder/'GENERATED_FORM.json',self.generated)
        put(self.folder/'TEST_COMPLETE.json',{**self.complete,'status':'failed'})
        with self.assertRaisesRegex(ValueError,'completion'):self.run_fake('bad-completion')

    def test_changed_source_or_asset_bytes_rejected_before_http(self):
        put(self.root/self.asset_name,b'changed')
        with self.assertRaisesRegex(ValueError,'bytes changed'):self.run_fake()
        self.assertFalse((self.folder/'PUBLICATION_VERIFICATION.json').exists())

    def test_report_page_count_is_measured_and_eleven_pages_fail(self):
        self.assertEqual(p.count_pdf_pages(self.root/p.REPORT),2)
        from pypdf import PdfWriter
        writer=PdfWriter()
        for _ in range(11):writer.add_blank_page(width=10,height=10)
        with (self.root/p.REPORT).open('wb') as f:writer.write(f)
        http=FakeHTTP(self.files)
        with self.assertRaisesRegex(ValueError,'ten actual PDF pages'):
            p.verify(self.root,'long-pdf',transport=http,git_check=self.git)
        self.assertEqual(http.calls,[])

    def test_git_verifies_ancestor_and_both_freeze_final_blobs_and_report(self):
        calls=[]
        def fake_run(command,**kwargs):
            calls.append(command)
            if command[1]=='merge-base':body=b''
            else:body=(self.root/command[2].split(':',1)[1]).read_bytes()
            return types.SimpleNamespace(returncode=0,stdout=body,stderr=b'')
        with patch.object(p.subprocess,'run',side_effect=fake_run):
            record=p.verify_commit(self.root,self.freeze,'2'*40,p.sha(self.root/p.REPORT))
        self.assertTrue(record['passed'])
        self.assertEqual(calls[0],['git','merge-base','--is-ancestor','1'*40,'2'*40])
        self.assertEqual(len(calls),6)

    def test_anonymous_transport_scope_no_auth_and_no_redirect(self):
        transport=p.AnonymousTransport();requests=[]
        class Response:
            status=200;headers={'ETag':'synthetic'}
            def __enter__(self):return self
            def __exit__(self,*args):pass
            def read(self,n):return b'abc'[:n]
            def geturl(self):return p.raw_url('2'*40,p.REPORT)
        def open_fake(request,**kwargs):requests.append(request);return Response()
        transport.opener=types.SimpleNamespace(open=open_fake)
        result=transport.fetch(p.raw_url('2'*40,p.REPORT),3)
        self.assertEqual(result['body'],b'abc')
        headers={k.lower():v for k,v in requests[0].header_items()}
        self.assertNotIn('authorization',headers);self.assertNotIn('proxy-authorization',headers)
        for url in ('http://raw.githubusercontent.com/'+p.REPOSITORY+'/x',
                    'https://raw.githubusercontent.com/other/repo/x',
                    p.raw_url('2'*40,p.REPORT)+'?token=synthetic'):
            with self.assertRaises(ValueError):transport.fetch(url,3)
        self.assertEqual(len(requests),1)
        request=urllib.request.Request(p.raw_url('2'*40,p.REPORT))
        with self.assertRaises(urllib.error.HTTPError):p.NoRedirect().redirect_request(request,None,302,'redirect',{},'https://example.com')

    def test_attempt_and_artifact_paths_cannot_escape(self):
        for name in ('../escape','/absolute','bad\\path','code/../secret.py'):
            with self.assertRaises(ValueError):p.inside(self.root,name)
        with self.assertRaises(ValueError):p.verify(self.root,'../escape',transport=FakeHTTP(self.files),git_check=self.git)


if __name__=='__main__':unittest.main(verbosity=2)
