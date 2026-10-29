"""Anonymous download/hash verification of the immutable final publication."""
from pathlib import Path
from datetime import datetime,timezone
import urllib.request,json,hashlib,subprocess
root=Path(__file__).resolve().parents[1]
commit='419e1158ba2f83b8c08796429e635f1bf8ab2977'
reproduced='92c7240c5a4710054de1363d23e2f11f81d33b72'
freeze=json.loads((root/'submission/PRE_TEST_FREEZE.json').read_text(encoding='utf-8'))
for relative,expected in {**freeze['source_files'],**freeze['assets']}.items():
    b=subprocess.check_output(['git','show',f'{commit}:{relative}'],cwd=root)
    assert hashlib.sha256(b).hexdigest()==expected, relative
    assert b==subprocess.check_output(['git','show',f'{reproduced}:{relative}'],cwd=root),relative
records=[]
for relative in ['code/solution.py','README.md','reports/report.pdf','artifacts/final_bundle/config.json','artifacts/final_bundle/weights.npz']:
    url=f'https://raw.githubusercontent.com/liuqichang0305/dase7506-mp2-solution/{commit}/{relative}'
    with urllib.request.urlopen(url,timeout=60) as response:
        data=response.read();status=response.status
    expected=subprocess.check_output(['git','show',f'{commit}:{relative}'],cwd=root)
    assert data==expected,relative
    records.append({'url':url,'http_status':status,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'matches_committed_bytes':True})
record={'checked_at_utc':datetime.now(timezone.utc).isoformat(),'code_commit':commit,'reproduced_commit':reproduced,'all_13_frozen_sources_and_2_assets_byte_identical':True,'anonymous_downloads':records,'status':'passed'}
(root/'submission/FINAL_PUBLICATION_VERIFICATION.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
print(json.dumps(record,indent=2))
