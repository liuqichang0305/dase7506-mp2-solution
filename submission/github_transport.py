"""Authenticated publication transport; uses existing Git helper, never logs credentials.

Explicit --request JSON supplies method/path/data; writes a sanitized GitHub response.
Only this coursework repository and the official course issue collection are allowed.
"""
import argparse, subprocess, os, json, urllib.request, re, hashlib
from datetime import datetime,timezone
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--request',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
if a.output.exists():raise SystemExit('Preserve earlier publication records; choose a new output')
request=json.loads(a.request.read_text(encoding='utf-8'))
path=request['path']; method=request.get('method','GET')
ready_query='mutation($id:ID!){markPullRequestReadyForReview(input:{pullRequestId:$id}){pullRequest{url isDraft}}}'
allowed=path.startswith('/repos/liuqichang0305/dase7506-mp2-solution/') or path=='/repos/xudongwu-0/xudongwu-0.github.io/issues' or (path=='/graphql' and method=='POST' and request.get('data',{}).get('query')==ready_query)
if not allowed or method not in ('GET','POST','PATCH','PUT'):raise SystemExit('Publication endpoint is outside authorized coursework scope')
env=os.environ.copy();env.update(GCM_INTERACTIVE='never',GIT_TERMINAL_PROMPT='0')
r=subprocess.run(['git','credential','fill'],input='protocol=https\nhost=github.com\n\n',capture_output=True,text=True,env=env,timeout=30)
if r.returncode:raise SystemExit('Existing Git credentials unavailable; no credential contents logged')
credential=dict(x.split('=',1) for x in r.stdout.splitlines() if '=' in x)
token=credential.get('password')
if not token:raise SystemExit('Existing Git credential has no authentication secret')
def call(path,method='GET',data=None):
    q=urllib.request.Request('https://api.github.com'+path,method=method,
        data=None if data is None else json.dumps(data,ensure_ascii=False).encode('utf-8'),
        headers={'Authorization':'Bearer '+token,'Accept':'application/vnd.github+json','Content-Type':'application/json','User-Agent':'DASE7506-coursework'})
    with urllib.request.urlopen(q,timeout=60) as response:return json.load(response)
identity=call('/user')
if identity.get('login')!='liuqichang0305' or identity.get('type')!='User':raise SystemExit('Personal GitHub identity mismatch')
if path=='/repos/xudongwu-0/xudongwu-0.github.io/issues' and method=='POST':
    folder=Path(__file__).resolve().parent
    body=(folder/'issue-body.txt').read_text(encoding='utf-8')
    title=(folder/'issue-title.txt').read_text(encoding='utf-8')
    if request.get('data',{}).get('body')!=body or request.get('data',{}).get('title')!=title:raise SystemExit('Course issue must preserve exact official generated title and body')
    instant=datetime.now(timezone.utc)
    if instant>=datetime(2026,10,31,16,tzinfo=timezone.utc):raise SystemExit('New-score deadline has passed; no issue sent')
    (folder/'SUBMISSION_ACTION_TIME.json').write_text(json.dumps({'before_post_utc':instant.isoformat(),'author':identity['login'],'author_type':identity['type'],'body_sha256':hashlib.sha256(body.encode()).hexdigest(),'destination':path},indent=2),encoding='utf-8')
result=call(path,method,request.get('data'))
if result.get('errors'):raise SystemExit('GitHub GraphQL request failed: '+json.dumps(result['errors']))
a.output.parent.mkdir(parents=True,exist_ok=True)
a.output.write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
print(json.dumps({k:result.get(k) for k in ['html_url','number','state','draft','merged','sha','message']}))
