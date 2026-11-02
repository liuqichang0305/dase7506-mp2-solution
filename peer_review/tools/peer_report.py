"""Prepare/reconcile/publish one evidence-bound official peer report.

Preparation is offline. Publication requires the coordinator's completed review
record and an explicit --publish invocation; this is not a score submission.
"""
from __future__ import annotations
import argparse
import base64
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import urllib.error
import urllib.parse
import urllib.request
import uuid

ROOT = Path(__file__).resolve().parents[2]
ACCOUNT = 'liuqichang0305'
COURSE_REPO = 'xudongwu-0/xudongwu-0.github.io'
COURSE = '/repos/' + COURSE_REPO + '/issues'
COMMIT = '65c16a3eef424e30b254ff58804dffe0fbda8793'
SOURCE = 'peer_review/official_form/' + COMMIT
SNAPSHOT = SOURCE + '/FORM_SOURCE_SNAPSHOT.json'
TOOLS = ['peer_review/tools/peer_report.py', 'peer_review/tools/prepare_official_peer_form.mjs']
BLOBS = {
    'courses/dase7506/mp2/index.html': '53b5ad66a5f8b1069a55cf83633a4bf0c278b548',
    'courses/dase7506/assets/mp2.9a439c7af66b.mjs': 'b891777d9602c10b99f5ba5b5402974c08f9dad8',
    'courses/dase7506/assets/arena.4f1cb17a638d.mjs': 'a53803cc06195ba44d02a2d97dd91ffe28494aed',
    'courses/dase7506/assets/projects.ae4b9dc4b02c.mjs': 'bc0ea31136171598c902c7ad30c84636b162ead2',
}
PROTOCOL = 'dase7506-mp2-open-hard-verifier-v5'

def now(): return datetime.now(timezone.utc)
def timestamp(value):
    value = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if value.tzinfo is None: raise ValueError('UTC offset required')
    return value
def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def sha(value): return hashlib.sha256(value.encode('utf-8')).hexdigest()
def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def write_new(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False); stream.write('\n')
        stream.flush(); os.fsync(stream.fileno())
def inside(root, relative):
    root=Path(root).resolve(); name=str(relative).replace('\\','/')
    path=(root/name).resolve()
    if Path(name).is_absolute() or not path.is_relative_to(root) or path.relative_to(root).as_posix()!=name:
        raise ValueError('Noncanonical evidence path')
    if any(p.is_symlink() for p in [root/name, *(root/name).parents] if p.is_relative_to(root)):
        raise ValueError('Symlinked evidence is not accepted')
    return path
def verify_map(root, values):
    if not isinstance(values,dict) or not values: raise ValueError('Empty hash mapping')
    for path, expected in values.items():
        if digest(inside(root,path))!=expected: raise ValueError('Changed evidence: '+path)
def tool_sources(root): return {name:digest(inside(root,name)) for name in TOOLS}
def peer_folder(root, relative):
    if not re.fullmatch(r'peer_review/submission_[1-9][0-9]*',relative): raise ValueError('One canonical peer evidence directory required')
    return inside(root,relative)

def source_snapshot(root):
    snapshot=read(inside(root,SNAPSHOT))
    if snapshot.get('schema')!='mp2-peer-official-form-source-v1' or snapshot.get('official_site_commit')!=COMMIT or snapshot.get('source_root')!=SOURCE:
        raise ValueError('Wrong official source snapshot')
    files=snapshot['files']
    if len(files)!=4 or {r['path']:r['git_blob_sha'] for r in files}!=BLOBS: raise ValueError('Official source graph differs')
    for item in files:
        raw=inside(root,SOURCE+'/'+item['path']).read_bytes()
        blob=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
        if blob!=BLOBS[item['path']] or hashlib.sha256(raw).hexdigest()!=item['sha256'] or len(raw)!=item['bytes']:
            raise ValueError('Official source bytes differ')
    expected={'published_html':list(BLOBS)[0], 'published_entry':list(BLOBS)[1],
              'published_arena':list(BLOBS)[2], 'published_projects':list(BLOBS)[3]}
    if any(snapshot[k]!=v for k,v in expected.items()): raise ValueError('Official entry identity differs')
    return snapshot

def evidence(root, relative):
    folder=peer_folder(root,relative); target=read(folder/'target.json'); complete=read(folder/'COMPLETE.json')
    selected=target['selected']; number=selected['submission']
    if (target.get('schema')!='mp2-peer-review-target-v1' or type(number)is not int or number<=0 or
        relative!=f'peer_review/submission_{number}' or selected['author'].lower()==ACCOUNT.lower() or
        target['issue_url']!=f'https://github.com/{COURSE_REPO}/issues/{number}' or
        target['publication_git_commit']!=COMMIT or not re.fullmatch('[0-9a-f]{40}',target['peer_commit']['sha'])):
        raise ValueError('Invalid peer identity/publication provenance')
    if not (isinstance(selected['score'],(int,float)) and not isinstance(selected['score'],bool) and math.isfinite(selected['score']) and 0<=selected['score']<=100):
        raise ValueError('Invalid published target score')
    if (complete.get('status')!='complete' or complete.get('target_sha256')!=digest(folder/'target.json') or
        complete.get('peer_commit')!=target['peer_commit']['sha'] or complete.get('prediction_scoring_success')is not True or
        complete.get('protocol')!=PROTOCOL or complete.get('split')!='test' or complete.get('n')!=467):
        raise ValueError('Actual completed matching peer reproduction is required')
    outputs=complete['output_sha256']; verify_map(folder,outputs)
    score_file=complete['official_score_file']; timing_file=complete['official_timing_file']
    if score_file not in outputs or timing_file not in outputs or not any('prediction' in name and name.endswith('.json') for name in outputs):
        raise ValueError('Full scoring/prediction/timing evidence missing')
    score=read(inside(folder,score_file))
    if (score.get('protocol')!=PROTOCOL or score.get('split')!='test' or score.get('n')!=467 or
        type(score.get('correct'))is not int or not 0<=score['correct']<=467 or
        not isinstance(score.get('accuracy'),(int,float)) or not math.isfinite(score['accuracy']) or
        not math.isclose(score['accuracy'],100*score['correct']/467,abs_tol=1e-10,rel_tol=0)):
        raise ValueError('Actual official full-test score is inconsistent')
    inputs={relative+'/target.json':digest(folder/'target.json'),relative+'/COMPLETE.json':digest(folder/'COMPLETE.json')}
    inputs.update({relative+'/'+name:value for name,value in outputs.items()})
    if (folder/'metadata_binding.json').exists(): inputs[relative+'/metadata_binding.json']=digest(folder/'metadata_binding.json')
    return target,complete,score,inputs

def prepare(root, relative, reason_file, evidence_url=None, now_fn=now):
    root=Path(root).resolve(); folder=peer_folder(root,relative); out=folder/'report_form'
    target,complete,score,inputs=evidence(root,relative); snapshot=source_snapshot(root)
    reason_path=inside(root,reason_file)
    if not reason_path.is_relative_to(folder): raise ValueError('Report prose must be in the selected peer evidence directory')
    reason=reason_path.read_text(encoding='utf-8').strip()
    if len(reason)>2000: raise ValueError('Markdown report exceeds official 2000-character limit')
    inputs[reason_file]=digest(reason_path)
    window={key:target[key] for key in ('published_at','review_ends_at')}
    if window!=snapshot['publication']['projects']['mp2'] or not timestamp(window['published_at'])<=now_fn()<timestamp(window['review_ends_at']):
        raise ValueError('Official peer review is not currently open')
    if evidence_url is not None:
        u=urllib.parse.urlsplit(evidence_url)
        if u.scheme!='https' or not u.hostname or '.' not in u.hostname or u.username or u.password or len(evidence_url)>500 or any(x.isspace() for x in evidence_url):
            raise ValueError('Invalid optional evidence URL')
    if out.exists(): raise FileExistsError('Prepared/partial report files are immutable; inspect the existing attempt')
    out.mkdir()
    selected=target['selected']
    row={'number':selected['submission'],'author':selected['author'],'project':'mp2','score':selected['score'],
         'student_id':target.get('student_id','ID pending'),'code_url':selected['code_url'],'checkpoint_url':selected['checkpoint_url'],
         'url':target['issue_url'],'status':'self-reported','created_at':target['published_at'],'artifact_status':'published'}
    record={'schema':'mp2-peer-report-input-v1','created_at_utc':now_fn().isoformat(),'intended_account':ACCOUNT,
            'peer_directory':relative,'target':row,'window':window,'fields':{'reproduced_score':score['accuracy'],'reason':reason,'evidence_url':evidence_url},
            'input_sha256':inputs,'tool_sources':tool_sources(root),
            'official_snapshot':{'path':SNAPSHOT,'sha256':digest(inside(root,SNAPSHOT))},
            'scope':'Independent peer reproduction only; not an own-model score update or artifact supplement.'}
    write_new(out/'FORM_INPUT.json',record)
    command=['node',str(root/TOOLS[1]),'--root',str(root),'--input',relative+'/report_form/FORM_INPUT.json','--output',relative+'/report_form']
    result=subprocess.run(command,cwd=root,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=15)
    (out/'adapter.stdout.log').write_text(result.stdout,encoding='utf-8');(out/'adapter.stderr.log').write_text(result.stderr,encoding='utf-8')
    write_new(out/'PREPARATION_RUN.json',{'command':command,'exit_code':result.returncode,'completed_at_utc':now_fn().isoformat(),
        'output_sha256':{p.name:digest(p) for p in sorted(out.iterdir()) if p.is_file()}})
    if result.returncode: raise RuntimeError('Official peer handler failed; partial output and logs preserved')
    validate_prepared(root,relative,require_approval=False,now_fn=now_fn)
    return {'status':'prepared_not_submitted','target_submission':selected['submission'],'directory':relative+'/report_form'}

def official_check(root, snapshot, form, publication, at):
    script="const fs=require('fs');(async()=>{const a=await import(process.argv[1]);const d=JSON.parse(fs.readFileSync(0,'utf8'));console.log(JSON.stringify({valid:a.validatePeerReviewReport(d.payload),url:a.makeIssueURL(d.payload),phase:a.submissionPhase('mp2',d.publication,Date.parse(d.now))}));})().catch(()=>process.exit(2));"
    uri=inside(root,SOURCE+'/'+snapshot['published_arena']).as_uri()
    r=subprocess.run(['node','-e',script,uri],input=json.dumps({'payload':form['official_payload'],'publication':publication,'now':at.isoformat()}),
                     capture_output=True,text=True,encoding='utf-8',timeout=10,cwd=root)
    if r.returncode: raise ValueError('Official payload validation failed')
    check=json.loads(r.stdout)
    if check.get('valid')is not True or check['url']!=form['issue_url']: raise ValueError('Official peer formatter validation differs')
    return check

def validate_prepared(root, relative, *, require_approval, now_fn=now):
    folder=peer_folder(root,relative); out=folder/'report_form'
    target,complete,score,current_inputs=evidence(root,relative); snapshot=source_snapshot(root)
    inputs=read(out/'FORM_INPUT.json'); form=read(out/'GENERATED_FORM.json')
    run=read(out/'PREPARATION_RUN.json')
    expected_preparation={'FORM_INPUT.json','GENERATED_FORM.json','issue-title.txt','issue-body.txt','issue-url.txt','adapter.stdout.log','adapter.stderr.log'}
    expected_command=['node',str(Path(root).resolve()/TOOLS[1]),'--root',str(Path(root).resolve()),
                      '--input',relative+'/report_form/FORM_INPUT.json','--output',relative+'/report_form']
    if run.get('exit_code')!=0 or run.get('command')!=expected_command or set(run.get('output_sha256',{}))!=expected_preparation:
        raise ValueError('Incomplete official-handler preparation receipt')
    verify_map(out,run['output_sha256'])
    if inputs.get('schema')!='mp2-peer-report-input-v1' or inputs.get('peer_directory')!=relative or inputs.get('intended_account')!=ACCOUNT:
        raise ValueError('Prepared report input scope differs')
    verify_map(root,inputs['input_sha256']);verify_map(root,inputs['tool_sources'])
    if inputs['tool_sources']!=tool_sources(root) or any(inputs['input_sha256'].get(k)!=v for k,v in current_inputs.items()): raise ValueError('Evidence/tool closure differs')
    if inputs['official_snapshot']!={'path':SNAPSHOT,'sha256':digest(inside(root,SNAPSHOT))}: raise ValueError('Official snapshot changed')
    payload=form['official_payload']; fields=inputs['fields'];selected=target['selected']
    expected={'schema':'dase7506/peer-review-v1','protocol':'7506-submissions-v1','project':'mp2','submission':selected['submission'],
              'reason':fields['reason'],'reproduced_score':score['accuracy'],'evidence_url':fields['evidence_url']}
    if payload!=expected or set(payload)!=set(expected) or fields['reproduced_score']!=score['accuracy']: raise ValueError('Report payload differs from actual peer reproduction')
    window={k:target[k] for k in ('published_at','review_ends_at')}
    if inputs['window']!=window or form['captured_publication']!={'projects':{'mp2':window}}: raise ValueError('Published review window differs')
    if (form.get('schema')!='mp2-peer-generated-form-v1' or form.get('status')!='prepared_not_submitted' or
        form.get('intended_account')!=ACCOUNT or form.get('expected_author_type')!='User' or
        form.get('target_submission')!=selected['submission'] or form.get('official_source_commit')!=COMMIT or
        form.get('report_input_sha256')!=digest(out/'FORM_INPUT.json') or form.get('official_source_snapshot_sha256')!=digest(inside(root,SNAPSHOT)) or
        form.get('adapter_sha256')!=digest(root/TOOLS[1]) or form.get('official_executed_entry')!=snapshot['published_entry']):
        raise ValueError('Generated-form provenance changed')
    for kind in ('title','body','url'):
        value=form['issue_'+kind]
        if sha(value)!=form['issue_'+kind+'_sha256'] or (out/('issue-'+kind+'.txt')).read_text(encoding='utf-8')!=value:
            raise ValueError('Official issue text changed')
    u=urllib.parse.urlsplit(form['issue_url']);q=urllib.parse.parse_qs(u.query)
    if u.scheme!='https' or u.netloc!='github.com' or u.path!='/'+COURSE_REPO+'/issues/new' or q!={'title':[form['issue_title']],'body':[form['issue_body']]}:
        raise ValueError('Official issue URL differs')
    official_check(root,snapshot,form,form['captured_publication'],now_fn())
    if require_approval:
        approval=read(out/'REVIEW_APPROVAL.json')
        required={'schema':'mp2-peer-review-approval-v1','approved':True,'reviewer':'root',
                  'generated_form_sha256':digest(out/'GENERATED_FORM.json'),'report_input_sha256':digest(out/'FORM_INPUT.json'),
                  'complete_sha256':digest(folder/'COMPLETE.json'),'target_sha256':digest(folder/'target.json'),
                  'tool_sources':tool_sources(root)}
        if any(approval.get(k)!=v for k,v in required.items()) or timestamp(approval['approved_at_utc'])<timestamp(form['generated_at_utc']):
            raise ValueError('Completed coordinator review does not bind this report')
    return {'title':form['issue_title'],'body':form['issue_body'],'body_sha256':form['issue_body_sha256'],
            'generated_form_sha256':digest(out/'GENERATED_FORM.json'),'report_input_sha256':digest(out/'FORM_INPUT.json'),
            'target_submission':selected['submission'],'window':window,'official_payload':payload}

class TransportFailure(RuntimeError):
    def __init__(self,status=None):
        self.http_status=status;super().__init__('GitHub request failed; credentials and response body withheld')

class GitHubTransport:
    def __init__(self,root):
        env={**os.environ,'GCM_INTERACTIVE':'never','GIT_TERMINAL_PROMPT':'0'}
        try:
            r=subprocess.run(['git','credential','fill'],input='protocol=https\nhost=github.com\n\n',cwd=root,
                             capture_output=True,text=True,env=env,timeout=20)
            credential=dict(line.split('=',1) for line in r.stdout.splitlines() if '=' in line)
        except Exception: raise RuntimeError('Existing personal Git credentials unavailable; no contents logged') from None
        if r.returncode or not credential.get('password'): raise RuntimeError('Existing personal Git credentials unavailable; no contents logged')
        self._token=credential['password']
        class NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self,*args,**kwargs): raise TransportFailure()
        self._opener=urllib.request.build_opener(NoRedirect())
    def call(self,url,method='GET',data=None):
        prefix=COURSE+'?state=all&creator='+ACCOUNT+'&per_page=100&page='
        contents='/repos/'+COURSE_REPO+'/contents/'
        allowed_files=set(BLOBS)|{'courses/dase7506/data/publication.json'}
        get=(url=='/user' or bool(re.fullmatch(re.escape(COURSE)+r'/[1-9][0-9]*',url)) or
             (url.startswith(prefix) and url[len(prefix):].isdigit()) or
             any(url==contents+name+'?ref=main' for name in allowed_files))
        if not (method=='GET' and get and data is None) and not (method=='POST' and url==COURSE and set(data or {})=={'title','body'}):
            raise ValueError('Request outside personal peer-report scope')
        request=urllib.request.Request('https://api.github.com'+url,method=method,
            data=None if data is None else json.dumps(data,ensure_ascii=False).encode(),
            headers={'Authorization':'Bearer '+self._token,'Accept':'application/vnd.github+json','Content-Type':'application/json','User-Agent':'DASE7506-peer-reproduction'})
        try:
            with self._opener.open(request,timeout=20) as response: return json.load(response)
        except urllib.error.HTTPError as error: raise TransportFailure(error.code) from None
        except Exception: raise TransportFailure() from None

def fresh_rules(transport,root,relative,form,at):
    snapshot=source_snapshot(root)
    for item in snapshot['files']:
        value=transport.call('/repos/'+COURSE_REPO+'/contents/'+item['path']+'?ref=main')
        if value.get('sha')!=item['git_blob_sha']: raise ValueError('Published official source changed; source review required')
    value=transport.call('/repos/'+COURSE_REPO+'/contents/courses/dase7506/data/publication.json?ref=main')
    if value.get('encoding')!='base64': raise ValueError('Unexpected publication encoding')
    publication=json.loads(base64.b64decode(value['content']))
    release=publication['projects']['mp2'];target=read(peer_folder(root,relative)/'target.json')['selected']
    matches=[r for r in release['entries'] if r.get('submission')==target['submission'] and r.get('author','').lower()==target['author'].lower()]
    if len(matches)!=1 or any(matches[0].get(k)!=target.get(k) for k in ('score','code_url','checkpoint_url')):
        raise ValueError('Published peer target changed')
    dates={k:release[k] for k in ('published_at','review_ends_at')}
    if dates!=form['captured_publication']['projects']['mp2']: raise ValueError('Review window changed; review current notice')
    if official_check(root,snapshot,form,{'projects':{'mp2':dates}},at)['phase']!='review': raise ValueError('Peer-review window closed')
    return {'checked_at_utc':at.isoformat(),'publication_blob_sha':value['sha'],'phase':'review','dates':dates,
            'source_blobs':BLOBS,'target_submission':target['submission'],'selected_entry_sha256':sha(json.dumps(matches[0],sort_keys=True,separators=(',',':'))),
            'other_entries_retained':False}

def validate_issue(issue,payload):
    if (issue.get('user',{}).get('login','').lower()!=ACCOUNT.lower() or issue['user'].get('type')!='User' or
        'pull_request' in issue or issue.get('title')!=payload['title'] or sha(issue.get('body') or '')!=payload['body_sha256']):
        raise ValueError('Peer-report issue identity/content differs')
    n=issue.get('number')
    if type(n)is not int or n<=0 or issue.get('html_url')!=f'https://github.com/{COURSE_REPO}/issues/{n}': raise ValueError('Unexpected issue URL/number')
    created=timestamp(issue['created_at'])
    if not timestamp(payload['window']['published_at'])<=created<timestamp(payload['window']['review_ends_at']): raise ValueError('Report outside published review window')
    return {k:issue[k] for k in ('number','html_url','title','body','user','created_at','state')}

def find_existing(transport,payload):
    exact=[];conflicts=[]
    for page in range(1,11):
        issues=transport.call(COURSE+f'?state=all&creator={ACCOUNT}&per_page=100&page={page}')
        if not isinstance(issues,list): raise ValueError('Invalid personal issue response')
        for issue in issues:
            if issue.get('user',{}).get('login','').lower()!=ACCOUNT.lower(): continue
            blocks=re.findall(r'```json\s*\n([\s\S]*?)\n```',issue.get('body') or '')
            try: data=json.loads(blocks[0]) if len(blocks)==1 else {}
            except ValueError: continue
            if data.get('schema')!='dase7506/peer-review-v1' or data.get('project')!='mp2' or data.get('submission')!=payload['target_submission']: continue
            if issue.get('title')==payload['title'] and sha(issue.get('body') or '')==payload['body_sha256']:
                exact.append(validate_issue(issue,payload)['number'])
            else: conflicts.append(issue.get('number'))
        if len(issues)<100: break
    else: raise ValueError('Personal issue pagination cap reached')
    if conflicts or len(exact)>1: raise ValueError('Existing different/duplicate report for this peer requires reconciliation; no new issue')
    return exact[0] if exact else None

@contextmanager
def report_lock(folder):
    path=folder/'report.lock'; identity={'pid':os.getpid(),'nonce':uuid.uuid4().hex,'created_at_utc':now().isoformat()}
    write_new(path,identity)  # Preserve a stale lock for explicit inspection; never steal it.
    try: yield
    finally:
        if path.exists() and read(path)==identity: path.unlink()

def publish(root,relative,*,allow_post=False,transport=None,now_fn=now):
    root=Path(root).resolve();folder=peer_folder(root,relative)/'report_form'
    with report_lock(folder):
        payload=validate_prepared(root,relative,require_approval=allow_post,now_fn=now_fn)
        transport=transport or GitHubTransport(root)
        identity=transport.call('/user')
        if identity.get('login','').lower()!=ACCOUNT.lower() or identity.get('type')!='User': raise ValueError('Personal account identity mismatch')
        success_file=folder/'REPORT_SUCCESS.json'
        if success_file.exists():
            success=read(success_file)
            if any(success.get(k)!=payload[k] for k in ('body_sha256','generated_form_sha256','report_input_sha256','target_submission')): raise ValueError('Existing success belongs to another report')
            if digest(inside(folder,success['readback_path']))!=success['readback_sha256']: raise ValueError('Saved readback changed')
            validate_issue(read(inside(folder,success['readback_path'])),payload)
            issue=validate_issue(transport.call(COURSE+'/'+str(success['issue_number'])),payload)
            return {'status':'existing_success_verified','issue_url':issue['html_url']}
        attempts=folder/'publish-attempts';attempts.mkdir(exist_ok=True)
        attempt=attempts/(now_fn().strftime('%Y%m%dT%H%M%S%fZ')+'-'+uuid.uuid4().hex[:8]);attempt.mkdir()
        existing=find_existing(transport,payload)
        if existing is not None:
            issue=validate_issue(transport.call(COURSE+'/'+str(existing)),payload)
            return save_success(folder,attempt,issue,payload,now_fn(),'existing_remote_report_reconciled')
        if list(attempts.glob('*/before-post.json')):
            write_new(attempt/'unresolved.json',{'status':'Earlier POST outcome unknown; no automatic repost','checked_at_utc':now_fn().isoformat()})
            raise RuntimeError('Earlier POST remains unresolved; only remote reconciliation is permitted')
        if not allow_post: return {'status':'no_existing_report','posted':False}
        checked=validate_prepared(root,relative,require_approval=True,now_fn=now_fn)
        if checked!=payload: raise ValueError('Report changed during duplicate check')
        form=read(folder/'GENERATED_FORM.json');rules=fresh_rules(transport,root,relative,form,now_fn())
        write_new(attempt/'rules-recheck.json',rules)
        write_new(attempt/'before-post.json',{'status':'one POST about to be attempted','at_utc':now_fn().isoformat(),'destination':COURSE,
            'body_sha256':payload['body_sha256'],'generated_form_sha256':payload['generated_form_sha256'],
            'approval_sha256':digest(folder/'REVIEW_APPROVAL.json'),'rules_sha256':digest(attempt/'rules-recheck.json')})
        if not timestamp(payload['window']['published_at'])<=now_fn()<timestamp(payload['window']['review_ends_at']): raise ValueError('Review window ended immediately before POST')
        try:
            issue=validate_issue(transport.call(COURSE,'POST',{'title':payload['title'],'body':payload['body']}),payload)
            write_new(attempt/'post-response.json',issue)
            issue=validate_issue(transport.call(COURSE+'/'+str(issue['number'])),payload)
            return save_success(folder,attempt,issue,payload,now_fn(),'created_and_read_back')
        except Exception as error:
            write_new(attempt/'uncertain.json',{'status':'POST/readback may have succeeded. Never blindly repost.',
                'exception_type':type(error).__name__,'http_status':getattr(error,'http_status',None),'at_utc':now_fn().isoformat()})
            raise RuntimeError('Peer-report outcome uncertain; preserved evidence, reconcile only') from None

def save_success(folder,attempt,issue,payload,at,status):
    write_new(attempt/'readback.json',issue)
    record={'schema':'mp2-peer-report-success-v1','status':status,'verified_at_utc':at.isoformat(),
            'issue_number':issue['number'],'issue_url':issue['html_url'],'author':ACCOUNT,'author_type':'User',
            'readback_path':(attempt/'readback.json').relative_to(folder).as_posix(),'readback_sha256':digest(attempt/'readback.json'),
            **{k:payload[k] for k in ('body_sha256','generated_form_sha256','report_input_sha256','target_submission')}}
    write_new(folder/'REPORT_SUCCESS.json',record)
    return {'status':status,'issue_url':issue['html_url']}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['prepare','check','reconcile','publish'])
    parser.add_argument('--peer',required=True);parser.add_argument('--reason-file');parser.add_argument('--evidence-url')
    args=parser.parse_args()
    if args.mode=='prepare':
        if not args.reason_file: parser.error('--reason-file is required for preparation')
        result=prepare(ROOT,args.peer,args.reason_file,args.evidence_url)
    elif args.mode=='check': result=validate_prepared(ROOT,args.peer,require_approval=False)
    else: result=publish(ROOT,args.peer,allow_post=args.mode=='publish')
    print(json.dumps({k:v for k,v in result.items() if k not in ('body','title','official_payload')},indent=2))
if __name__=='__main__': main()
