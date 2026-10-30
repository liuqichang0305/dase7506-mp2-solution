"""Snapshot public official form code for exact payload generation, without credentials."""
from pathlib import Path, PurePosixPath
from datetime import datetime, timezone
import urllib.request, json, hashlib, posixpath, re, argparse
root=Path(__file__).resolve().parents[2]
base='https://api.github.com/repos/xudongwu-0/xudongwu-0.github.io'
def read(url):
    with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'DASE7506-coursework'}),timeout=60) as f:return f.read()
dest=root/'tmp/form-source-v3'
dest.mkdir(parents=True,exist_ok=True)
manifest=root/'submission/v3/FORM_SOURCE_SNAPSHOT.json'
parser=argparse.ArgumentParser()
parser.add_argument('--extend-published',action='store_true',help='Keep the existing commit and add the exact index-referenced module/import graph')
parser.add_argument('--restore',action='store_true',help='Restore every exact hashed snapshot file in a clean checkout without changing the snapshot record')
args=parser.parse_args()
if not (root/'submission/v3/FREEZE.json').exists() or not (root/'results/research_v3/SEARCH_CLOSED.json').exists():
    raise SystemExit('V3 form snapshot is fetched only after research closure and final freeze')
if args.restore:
    if args.extend_published: parser.error('--restore and --extend-published are mutually exclusive')
    record=json.loads(manifest.read_text(encoding='utf-8'))
    for item in record['files']:
        local=dest/item.get('local_path',PurePosixPath(item['path']).name)
        content=read(item['url'])
        if hashlib.sha256(content).hexdigest()!=item['sha256'] or len(content)!=item['bytes']:
            raise ValueError(f'Restored snapshot hash mismatch: {item["path"]}')
        if local.exists() and local.read_bytes()!=content: raise ValueError(f'Refusing to overwrite changed snapshot: {local}')
        local.parent.mkdir(parents=True,exist_ok=True)
        local.write_bytes(content)
    print(json.dumps({'restored_commit':record['official_site_commit'],'files':len(record['files'])},indent=2))
    raise SystemExit(0)
if args.extend_published:
    record=json.loads(manifest.read_text(encoding='utf-8'))
    commit=record['official_site_commit']
    records=record['files']
    for item in records:
        local=dest/item.get('local_path',PurePosixPath(item['path']).name)
        if hashlib.sha256(local.read_bytes()).hexdigest()!=item['sha256']:
            raise ValueError(f'Snapshot hash mismatch: {local}')
    html=(dest/'index.html').read_text(encoding='utf-8')
    scripts=re.findall(r'<script\b[^>]*\btype="module"[^>]*\bsrc="([^"]+)"',html)
    if len(scripts)!=1: raise ValueError('Expected exactly one published module entry')
    entry=posixpath.normpath(posixpath.join('courses/dase7506/mp2',scripts[0]))
    queue=[entry]; seen=set(); graph={}
    while queue:
        path=queue.pop(0)
        if path in seen: continue
        if not path.startswith('courses/dase7506/') or not path.endswith('.mjs'):
            raise ValueError(f'Unexpected official module path: {path}')
        url=f'https://raw.githubusercontent.com/xudongwu-0/xudongwu-0.github.io/{commit}/{path}'
        content=read(url)
        local=dest/'served'/path
        local.parent.mkdir(parents=True,exist_ok=True)
        if local.exists() and local.read_bytes()!=content: raise ValueError('Refusing to overwrite different snapshot bytes')
        local.write_bytes(content)
        item={'path':path,'local_path':local.relative_to(dest).as_posix(),'sha256':hashlib.sha256(content).hexdigest(),'bytes':len(content),'url':url}
        existing=next((r for r in records if r['path']==path),None)
        if existing is not None and existing['sha256']!=item['sha256']: raise ValueError('Conflicting fixed-commit source hash')
        if existing is None: records.append(item)
        imports=re.findall(r'\b(?:import|export)\b[^;]*?\bfrom\s*[\'\"]([^\'\"]+)[\'\"]|\bimport\s*[\'\"]([^\'\"]+)[\'\"]',content.decode('utf-8'))
        dependencies=[]
        for pair in imports:
            specifier=next(v for v in pair if v)
            if not specifier.startswith('.'): raise ValueError(f'Non-relative import requires review: {specifier}')
            dependency=posixpath.normpath(posixpath.join(posixpath.dirname(path),specifier))
            dependencies.append(dependency); queue.append(dependency)
        graph[path]=dependencies; seen.add(path)
    record['published_entry_path']=entry
    record['published_module_graph']=graph
    record['published_graph_retrieved_at_utc']=datetime.now(timezone.utc).isoformat()
    # Compare authored and deployed modules after removing only relative import path spelling.
    normalize=lambda text: re.sub(r"(\bfrom\s*['\"])(\.[^'\"]+)(['\"])",r'\1RELATIVE_MODULE\3',text)
    comparisons=[]
    for path in sorted(seen):
        basename=PurePosixPath(path).name.split('.')[0]+'.mjs'
        authored=dest/basename
        if authored.exists():
            published=dest/'served'/path
            comparisons.append({'authored':basename,'published':path,'equal_except_relative_import_paths':normalize(authored.read_text(encoding='utf-8'))==normalize(published.read_text(encoding='utf-8'))})
    record['authored_published_comparison']=comparisons
else:
    if manifest.exists(): raise FileExistsError('Snapshot already exists; use --extend-published to preserve its fixed commit')
    commit=json.loads(read(base+'/commits/main'))['sha']
    paths=['courses/dase7506/mp2.mjs','courses/dase7506/arena.mjs','courses/dase7506/projects.mjs','courses/dase7506/data/student-id-public.pem','courses/dase7506/data/leaderboard.json','courses/dase7506/mp2/index.html']
    records=[]
    for path in paths:
        url=f'https://raw.githubusercontent.com/xudongwu-0/xudongwu-0.github.io/{commit}/{path}'
        content=read(url)
        (dest/PurePosixPath(path).name).write_bytes(content)
        records.append({'path':path,'sha256':hashlib.sha256(content).hexdigest(),'bytes':len(content),'url':url})
    record={'retrieved_at_utc':datetime.now(timezone.utc).isoformat(),'official_site_commit':commit,'files':records,'purpose':'Run unchanged official submission-form handler locally because browser automation cannot establish URL. No hand-written replacement submission schema. No issue created by this script.'}
manifest.write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'commit':commit,'files':len(records),'published_entry':record.get('published_entry_path'),'comparisons':record.get('authored_published_comparison')},indent=2))
