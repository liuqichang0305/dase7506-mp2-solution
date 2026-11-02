"""Synthetic peer evidence plus unchanged local official handler; no real IO online."""
from datetime import datetime, timezone
import base64
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest

REPO=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('peer_report',REPO/'peer_review/tools/peer_report.py')
p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)

class FakeTransport:
    def __init__(self,test): self.test=test;self.calls=[];self.issues=[];self.posts=0;self.fail_post=False;self.identity=p.ACCOUNT;self.changed_source=False
    def call(self,path,method='GET',data=None):
        self.calls.append((path,method))
        if path=='/user': return {'login':self.identity,'type':'User'}
        if method=='POST':
            self.posts+=1
            if self.fail_post: raise p.TransportFailure(502)
            issue=self.test.issue();self.issues.append(issue);return issue
        if '?state=all' in path: return self.issues
        if path.startswith(p.COURSE+'/'):
            return next(v for v in self.issues if str(v['number'])==path.rsplit('/',1)[1])
        if '/contents/' in path:
            name=path.split('/contents/')[1].split('?')[0]
            if name.endswith('publication.json'):
                target=p.read(self.test.folder/'target.json')
                publication={'projects':{'mp2':{**{k:target[k] for k in ('published_at','review_ends_at')},'entries':[target['selected']]}}}
                return {'sha':'b'*40,'encoding':'base64','content':base64.b64encode(json.dumps(publication).encode()).decode()}
            return {'sha':'0'*40 if self.changed_source else p.BLOBS[name]}
        raise AssertionError(path)

class PeerReportTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='synthetic-peer-report-');self.root=Path(self.temp.name)
        shutil.copytree(REPO/p.SOURCE,self.root/p.SOURCE)
        for name in p.TOOLS:
            dest=self.root/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(REPO/name,dest)
        self.relative='peer_review/submission_240';self.folder=self.root/self.relative;self.folder.mkdir(parents=True)
        self.output=self.folder/'report_form'
        window=p.read(self.root/p.SNAPSHOT)['publication']['projects']['mp2']
        target={'schema':'mp2-peer-review-target-v1','selected':{'submission':240,'author':'SyntheticPeer','score':80.0,
            'code_url':'https://github.com/example/peer/tree/'+('c'*40),'checkpoint_url':'https://github.com/example/peer/tree/'+('c'*40)+'/assets'},
            'issue_url':f'https://github.com/{p.COURSE_REPO}/issues/240','publication_git_commit':p.COMMIT,
            'peer_commit':{'sha':'c'*40},**window}
        self.put('target.json',target)
        self.put('score.json',{'protocol':p.PROTOCOL,'split':'test','n':467,'correct':394,'accuracy':100*394/467,'scoring_seconds':.1})
        self.put('timing.json',{'synthetic':True,'prediction_plus_scoring_seconds':1.})
        self.put('predictions.json',{'synthetic':True,'no_course_data':True})
        complete={'status':'complete','target_sha256':p.digest(self.folder/'target.json'),'peer_commit':'c'*40,
            'prediction_scoring_success':True,'protocol':p.PROTOCOL,'split':'test','n':467,
            'official_score_file':'score.json','official_timing_file':'timing.json',
            'output_sha256':{name:p.digest(self.folder/name) for name in ['score.json','timing.json','predictions.json']}}
        self.put('COMPLETE.json',complete)
        (self.folder/'report.md').write_text('Synthetic fixture. Reproduced with the submitted checkpoint; no misconduct claim.',encoding='utf-8')
    def tearDown(self): self.temp.cleanup()
    def put(self,name,value): (self.folder/name).write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
    def prepare(self): return p.prepare(self.root,self.relative,self.relative+'/report.md')
    def approve(self):
        form=p.read(self.output/'GENERATED_FORM.json')
        p.write_new(self.output/'REVIEW_APPROVAL.json',{'schema':'mp2-peer-review-approval-v1','approved':True,'reviewer':'root',
            'approved_at_utc':p.now().isoformat(),'generated_form_sha256':p.digest(self.output/'GENERATED_FORM.json'),
            'report_input_sha256':p.digest(self.output/'FORM_INPUT.json'),'complete_sha256':p.digest(self.folder/'COMPLETE.json'),
            'target_sha256':p.digest(self.folder/'target.json'),'tool_sources':p.tool_sources(self.root)})
    def issue(self):
        form=p.read(self.output/'GENERATED_FORM.json')
        return {'number':999,'title':form['issue_title'],'body':form['issue_body'],'html_url':f'https://github.com/{p.COURSE_REPO}/issues/999',
            'user':{'login':p.ACCOUNT,'type':'User'},'created_at':p.now().isoformat(),'state':'open'}
    def test_actual_published_peer_handler_prepares_one_target_without_network(self):
        result=self.prepare();self.assertEqual(result['status'],'prepared_not_submitted')
        form=p.read(self.output/'GENERATED_FORM.json');self.assertEqual(form['official_payload']['schema'],'dase7506/peer-review-v1')
        self.assertEqual(form['official_payload']['submission'],240);self.assertAlmostEqual(form['official_payload']['reproduced_score'],100*394/467)
        self.assertEqual(form['official_payload']['evidence_url'],None);self.assertEqual(form['official_phase'],'review')
        self.assertTrue(all(x['network_used'] is False for x in form['fetch_log']))
        self.assertNotIn('student_id',form['official_payload']);self.assertNotIn('artifacts',form['official_payload'])
        p.validate_prepared(self.root,self.relative,require_approval=False)
        with self.assertRaises(FileExistsError): self.prepare()
    def test_incomplete_actual_reproduction_prevents_preparation(self):
        value=p.read(self.folder/'COMPLETE.json');value['prediction_scoring_success']=False;self.put('COMPLETE.json',value)
        with self.assertRaises(ValueError): self.prepare()
        self.assertFalse(self.output.exists())
    def test_wrong_or_tampered_official_score_is_rejected(self):
        value=p.read(self.folder/'score.json');value['accuracy']+=1;self.put('score.json',value)
        receipt=p.read(self.folder/'COMPLETE.json');receipt['output_sha256']['score.json']=p.digest(self.folder/'score.json');self.put('COMPLETE.json',receipt)
        with self.assertRaisesRegex(ValueError,'inconsistent'): self.prepare()
    def test_no_self_review_or_cross_target(self):
        target=p.read(self.folder/'target.json');target['selected']['author']=p.ACCOUNT;self.put('target.json',target)
        with self.assertRaises(ValueError): self.prepare()
    def test_closed_window_long_reason_and_credentials_url_rejected(self):
        with self.assertRaisesRegex(ValueError,'not currently open'):
            p.prepare(self.root,self.relative,self.relative+'/report.md',now_fn=lambda:datetime(2026,11,10,tzinfo=timezone.utc))
        (self.folder/'report.md').write_text('x'*2001)
        with self.assertRaisesRegex(ValueError,'2000'): self.prepare()
        (self.folder/'report.md').write_text('Valid report')
        with self.assertRaises(ValueError): p.prepare(self.root,self.relative,self.relative+'/report.md','https://u:password@example.com/x')
    def test_changed_official_source_fails_before_handler(self):
        source=self.root/p.SOURCE/list(p.BLOBS)[1];source.write_text(source.read_text(encoding='utf-8')+'\n//changed\n',encoding='utf-8')
        with self.assertRaisesRegex(ValueError,'source bytes differ'): self.prepare()
    def test_reconcile_is_read_only_and_new_publish_requires_review(self):
        self.prepare();transport=FakeTransport(self)
        result=p.publish(self.root,self.relative,transport=transport)
        self.assertFalse(result['posted']);self.assertEqual(transport.posts,0)
        with self.assertRaises(FileNotFoundError): p.publish(self.root,self.relative,allow_post=True,transport=transport)
        self.assertEqual(transport.posts,0)
    def test_one_publish_readback_then_success_reuse(self):
        self.prepare();self.approve();transport=FakeTransport(self)
        result=p.publish(self.root,self.relative,allow_post=True,transport=transport)
        self.assertEqual(result['status'],'created_and_read_back');self.assertEqual(transport.posts,1)
        result=p.publish(self.root,self.relative,allow_post=True,transport=transport)
        self.assertEqual(result['status'],'existing_success_verified');self.assertEqual(transport.posts,1)
    def test_existing_remote_report_reconciles_without_post(self):
        self.prepare();transport=FakeTransport(self);transport.issues=[self.issue()]
        result=p.publish(self.root,self.relative,transport=transport)
        self.assertEqual(result['status'],'existing_remote_report_reconciled');self.assertEqual(transport.posts,0)
    def test_ambiguous_post_never_blindly_retries(self):
        self.prepare();self.approve();transport=FakeTransport(self);transport.fail_post=True
        with self.assertRaisesRegex(RuntimeError,'uncertain'): p.publish(self.root,self.relative,allow_post=True,transport=transport)
        self.assertEqual(transport.posts,1);transport.fail_post=False
        with self.assertRaisesRegex(RuntimeError,'unresolved'): p.publish(self.root,self.relative,allow_post=True,transport=transport)
        self.assertEqual(transport.posts,1)
        transport.issues=[self.issue()]
        self.assertEqual(p.publish(self.root,self.relative,transport=transport)['status'],'existing_remote_report_reconciled')
        self.assertEqual(transport.posts,1)
    def test_wrong_identity_and_new_official_source_block_post(self):
        self.prepare();self.approve();transport=FakeTransport(self);transport.identity='wrong-user'
        with self.assertRaisesRegex(ValueError,'identity mismatch'): p.publish(self.root,self.relative,allow_post=True,transport=transport)
        transport.identity=p.ACCOUNT;transport.changed_source=True
        with self.assertRaisesRegex(ValueError,'source changed'): p.publish(self.root,self.relative,allow_post=True,transport=transport)
        self.assertEqual(transport.posts,0)
    def test_conflicting_existing_peer_report_blocks_duplicates(self):
        self.prepare();self.approve();transport=FakeTransport(self);issue=self.issue();issue['body']=issue['body'].replace('Synthetic fixture','Different fixture')
        transport.issues=[issue]
        with self.assertRaisesRegex(ValueError,'different/duplicate'): p.publish(self.root,self.relative,allow_post=True,transport=transport)
        self.assertEqual(transport.posts,0)
    def test_malformed_review_protocol_rejected_before_remote_reads(self):
        self.prepare();form=p.read(self.output/'GENERATED_FORM.json');form['official_payload']['protocol']='wrong'
        (self.output/'GENERATED_FORM.json').write_text(json.dumps(form),encoding='utf-8');transport=FakeTransport(self)
        with self.assertRaises(ValueError): p.publish(self.root,self.relative,transport=transport)
        self.assertEqual(transport.calls,[])
    def test_incomplete_preparation_receipt_blocks_even_reconciliation(self):
        self.prepare();run=p.read(self.output/'PREPARATION_RUN.json');run['exit_code']=1
        (self.output/'PREPARATION_RUN.json').write_text(json.dumps(run),encoding='utf-8');transport=FakeTransport(self)
        with self.assertRaisesRegex(ValueError,'preparation receipt'): p.publish(self.root,self.relative,transport=transport)
        self.assertEqual(transport.calls,[])

if __name__=='__main__': unittest.main(verbosity=2)
