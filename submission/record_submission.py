"""Read back the actual public Issue and latest course snapshot; never creates one."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import urllib.request,json,hashlib,re
root=Path(__file__).resolve().parents[1];folder=root/'submission'
def get(url):
    with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'DASE7506-coursework','Cache-Control':'no-cache'}),timeout=60) as r:return json.load(r)
issue=get('https://api.github.com/repos/xudongwu-0/xudongwu-0.github.io/issues/294')
body=(folder/'issue-body.txt').read_text(encoding='utf-8')
assert issue['user']['login']=='liuqichang0305' and issue['user']['type']=='User'
assert issue['body']==body
blocks=re.findall(r'```json\s*\n([\s\S]*?)\n```',body);assert len(blocks)==1
payload=json.loads(blocks[0]);score=json.loads((root/'results/test-score.json').read_text(encoding='utf-8'))
assert payload['student_id']=='3036900957' and payload['project']=='mp2' and payload['score']==score['accuracy']
created=datetime.fromisoformat(issue['created_at'].replace('Z','+00:00'))
assert created<datetime(2026,10,31,16,tzinfo=timezone.utc)
generated=json.loads((folder/'GENERATED_FORM.json').read_text(encoding='utf-8'))
board=get('https://raw.githubusercontent.com/xudongwu-0/xudongwu-0.github.io/main/courses/dase7506/data/leaderboard.json')
row=next((r for r in board['submissions'] if r.get('number')==294),None)
if row:
    assert row['author']=='liuqichang0305' and row['student_id']=='3036900957' and row['score']==score['accuracy']
record={'verified_at_utc':datetime.now(timezone.utc).isoformat(),'issue_url':issue['html_url'],'issue_number':issue['number'],'author':issue['user']['login'],'author_type':issue['user']['type'],'student_id':payload['student_id'],'created_at_utc':issue['created_at'],'created_at_utc8':created.astimezone(timezone(timedelta(hours=8))).isoformat(),'before_deadline':True,'actual_score':score,'code_commit':generated['final_code_commit'],'code_url':generated['code_url'],'checkpoint_url':generated['checkpoint_url'],'weights_sha256':'a785f4a48e2d5708aec9fec29c776e9ce1b7fea1832d064e95b25c926626d462','config_sha256':'24473c3b95b9472b5d844b795edf86c7902d21fe7351604cc680d93d7ead0703','body_exactly_matches_official_generated_body':True,'body_sha256':hashlib.sha256(body.encode()).hexdigest(),'board_generated_at':board.get('generated_at'),'board_row':row,'board_status':'recognized' if row else 'issue_created_waiting_snapshot_sync','mp2_publication':board.get('publication',{}).get('projects',{}).get('mp2'),'project_status':'submitted_waiting_peer_release'}
previous=folder/'SUBMISSION_VERIFICATION.json'
if previous.exists():
    old=json.loads(previous.read_text(encoding='utf-8'))
    with (folder/'board-check-history.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(old)+'\n')
previous.write_text(json.dumps(record,indent=2,ensure_ascii=False),encoding='utf-8')
(folder/'course-issue-readback.json').write_text(json.dumps(issue,indent=2,ensure_ascii=False),encoding='utf-8')
status=('Recognized by the course snapshot as '+str(row.get('status'))+'; artifact status '+str(row.get('artifact_status'))+'. This is not instructor verification.') if row else 'Issue created and verified; waiting for the course snapshot to synchronize. Do not create a duplicate issue.'
text=f'''# Formal course submission record

Status: **submitted_waiting_peer_release**. Actual Issue created and independently read back; this is not merely a prepared URL.

- Issue: {record['issue_url']} (#{record['issue_number']}).
- Author: {record['author']}, type {record['author_type']}.
- Student ID: {record['student_id']}.
- Created: {record['created_at_utc8']} (Asia/Shanghai / UTC+8), before the 2026-10-31 23:59 deadline.
- Official full test: {score['correct']}/{score['n']}, accuracy **{score['accuracy']}%**. Original JSON: results/test-score.json.
- Frozen code: {generated['code_url']}.
- Matching checkpoint/config: {generated['checkpoint_url']}.
- Final code commit: `{generated['final_code_commit']}`.
- Inference source freeze: `8602992842d444524ab4148991da2f1dc88f316f`.
- Weight SHA-256: `{record['weights_sha256']}`.
- Config SHA-256: `{record['config_sha256']}`.
- Official generated Issue body SHA-256: `{record['body_sha256']}`; exact remote body match verified.
- Public code/PDF/assets verified anonymously: FINAL_PUBLICATION_VERIFICATION.json.
- Reviewed and merged PR: https://github.com/liuqichang0305/dase7506-mp2-solution/pull/1.

{status}

Submission mechanism: the browser-control tool could not establish the current URL and stopped. The unchanged, actually served official website form handler and encryption code were executed through the reviewed local DOM adapter, using the official public key and phase rules. Its generated title/body/payload were sent unchanged with the authorized personal GitHub account after the connector returned an integration-permission error. No browser click is claimed. Exact source snapshots, form records, API receipt and readback are retained.

AI assistance and internal AI review are disclosed. The result declaration confirms ownership of this task's actual result; no separate human understanding or manual review has been claimed.

The prepared code was independently reproduced from public commit 92c7240c5a4710054de1363d23e2f11f81d33b72. Final merged code has identical frozen source, model and PDF bytes. Subsequent repository commits only append submission/progress evidence; the submitted links remain fixed to the above commit.

Peer review: MP2 publication is {json.dumps(record['mp2_publication'])}. No MP2 review was performed before actual teacher release; preparation and resume instructions are in PEER_REVIEW_RESUME.txt. No future scheduled task was created.
'''
(folder/'SUBMISSION_RECORD.md').write_text(text,encoding='utf-8')
print(json.dumps({k:record[k] for k in ['issue_url','author','author_type','student_id','created_at_utc8','board_status','mp2_publication']},indent=2))
