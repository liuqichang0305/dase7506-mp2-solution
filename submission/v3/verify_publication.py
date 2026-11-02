"""Read-only anonymous verification of the exact V3 official-form artifact URLs.

Runs only after final freeze, once-only test completion, and form generation.
No score file, raw holdout, credential helper, GitHub issue API, or upload is used.
Importing this module has no I/O. Failed attempts are retained; none is overwritten.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import urllib.error
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
REPOSITORY = 'liuqichang0305/dase7506-mp2-solution'
METHOD = 'code/solution_v3.py'
CHECKPOINT = 'artifacts/research_v3/final_selected'
REPORT = 'reports/research_v3/report_v3.pdf'
REQUIRED = ('FREEZE.json', 'TEST_COMPLETE.json', 'GENERATED_FORM.json')


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_new(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())


def inside(root, name):
    if (not isinstance(name, str) or not name or '\\' in name or PurePosixPath(name).is_absolute()
            or '..' in PurePosixPath(name).parts or PurePosixPath(name).as_posix() != name):
        raise ValueError('A canonical repository-relative POSIX path is required')
    root = Path(root).resolve()
    target = (root/name).resolve()
    if target == root or not target.is_relative_to(root) or (root/name).is_symlink():
        raise ValueError('Artifact/evidence path escapes the repository or is a link')
    return target


def raw_url(commit, name):
    if re.fullmatch('[0-9a-f]{40}', str(commit)) is None:
        raise ValueError('Full immutable commit SHA required')
    # Validate lexical path independently of whether this file exists yet.
    inside(ROOT, name)
    return f'https://raw.githubusercontent.com/{REPOSITORY}/{commit}/' + urllib.parse.quote(name, safe='/')


def source_name(name):
    return (name.endswith('.py') and name.startswith(('code/', 'scripts/', 'submission/'))
            and not name.startswith('code/data/')) or bool(re.fullmatch(r'(?:code/)?requirements[^/]*\.txt', name))


def validate_records(root):
    folder = Path(root)/'submission/v3'
    freeze, complete, generated = (read(folder/name) for name in REQUIRED)
    if (freeze.get('schema') != 'dase7506-mp2-final-freeze-v3' or freeze.get('frozen') is not True
            or freeze.get('test_evaluated_before_freeze') is not False
            or freeze.get('method') != METHOD or freeze.get('checkpoint') != CHECKPOINT):
        raise ValueError('A valid fixed-method V3 pre-test freeze is required')
    freeze_hash = sha(folder/'FREEZE.json')
    if (complete.get('schema') != 'mp2-v3-test-complete-v1' or complete.get('status') != 'passed'
            or complete.get('local_resource_eligible') is not True or complete.get('freeze_sha256') != freeze_hash
            or complete.get('method') != METHOD or complete.get('checkpoint') != CHECKPOINT
            or complete.get('split') != 'test' or type(complete.get('n')) is not int or complete['n'] != 467):
        raise ValueError('The once-only test completion must accept this exact freeze')
    if (generated.get('status') != 'prepared_not_submitted' or generated.get('freeze_file_sha256') != freeze_hash
            or generated.get('freeze_source_commit') != freeze.get('source_commit')
            or generated.get('student_id') != '3036900957' or generated.get('intended_github_account') != 'liuqichang0305'
            or generated.get('score_file') != 'submission/v3/test-score.json'
            or re.fullmatch('[0-9a-f]{64}',str(generated.get('score_file_sha256'))) is None
            or complete.get('output_sha256', {}).get('submission/v3/test-score.json') != generated.get('score_file_sha256')):
        raise ValueError('Generated form is not bound to this completed frozen result')
    commit = generated.get('final_code_commit')
    if any(re.fullmatch('[0-9a-f]{40}', str(value)) is None for value in (commit,freeze.get('source_commit'))):
        raise ValueError('Frozen and submitted commits must be full immutable SHAs')
    expected = f'https://github.com/{REPOSITORY}/tree/{commit}'
    if generated.get('code_url') != expected or generated.get('checkpoint_url') != expected+'/'+CHECKPOINT:
        raise ValueError('Form links differ from the fixed public repository and selected checkpoint')
    sources, assets = freeze.get('source_files'), freeze.get('assets')
    if not isinstance(sources, dict) or not sources or not isinstance(assets, dict) or not assets or set(sources)&set(assets):
        raise ValueError('Complete distinct source and asset maps are required')
    for name,digest in {**sources,**assets}.items():
        if (name in sources and not source_name(name)) or (name in assets and not name.startswith(CHECKPOINT+'/')):
            raise ValueError('Frozen mapping names a file outside source/selected-asset scope')
        path=inside(root,name)
        if re.fullmatch('[0-9a-f]{64}',str(digest)) is None or sha(path)!=digest:
            raise ValueError('Local frozen artifact bytes changed: '+name)
    return freeze,complete,generated


def verify_commit(root, freeze, commit, report_hash):
    commands=[]
    def git(*args):
        command=['git',*args]
        result=subprocess.run(command,cwd=root,capture_output=True,timeout=60)
        commands.append({'command':command,'returncode':result.returncode,
                         'stdout_sha256':sha_bytes(result.stdout),'stderr_sha256':sha_bytes(result.stderr)})
        if result.returncode:
            raise ValueError('Local Git ancestry/blob verification failed')
        return result.stdout
    git('merge-base','--is-ancestor',freeze['source_commit'],commit)
    for revision in dict.fromkeys((freeze['source_commit'],commit)):
        for name,expected in {**freeze['source_files'],**freeze['assets']}.items():
            if sha_bytes(git('show',f'{revision}:{name}'))!=expected:
                raise ValueError('Committed source/asset differs from frozen bytes: '+name)
    if sha_bytes(git('show',f'{commit}:{REPORT}'))!=report_hash:
        raise ValueError('Report differs from the submitted public commit')
    return {'source_commit':freeze['source_commit'],'final_code_commit':commit,'passed':True,'commands':commands}


def count_pdf_pages(path):
    from pypdf import PdfReader
    reader=PdfReader(str(path),strict=True)
    if reader.is_encrypted:
        raise ValueError('The public report must be readable without a PDF password')
    pages=len(reader.pages)
    if not 1<=pages<=10:
        raise ValueError('The public report must have one to ten actual PDF pages')
    return pages


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,*args,**kwargs):
        raise urllib.error.HTTPError(args[0].full_url,args[2],'Redirects are not accepted for immutable raw artifacts',args[4],None)


class AnonymousTransport:
    def __init__(self):
        # Disable inherited proxies and all credential/cookie/auth handlers.
        self.opener=urllib.request.build_opener(urllib.request.ProxyHandler({}),NoRedirect())

    def fetch(self,url,max_bytes):
        parsed=urllib.parse.urlsplit(url)
        if (parsed.scheme!='https' or parsed.netloc!='raw.githubusercontent.com' or parsed.query or parsed.fragment
                or not parsed.path.startswith('/'+REPOSITORY+'/')):
            raise ValueError('Only fixed-repository anonymous raw HTTPS downloads are allowed')
        request=urllib.request.Request(url,headers={'User-Agent':'DASE7506-V3-public-artifact-verification'},method='GET')
        with self.opener.open(request,timeout=30) as response:
            body=response.read(max_bytes+1)
            return {'http_status':response.status,'url':response.geturl(),'body':body,
                'headers':{name:response.headers[name] for name in ('Date','ETag','Last-Modified','Content-Length') if name in response.headers}}


def download_check(root, folder, commit, name, expected, transport, *, pdf_pages=None, page_counter=count_pdf_pages):
    source=inside(root,name);url=raw_url(commit,name)
    record={'path':name,'url':url,'expected_sha256':expected,'expected_bytes':source.stat().st_size,
            'started_at_utc':utc(),'anonymous':True,'authorization_header_sent':False}
    try:
        response=transport.fetch(url,record['expected_bytes'])
        body=response['body']
        if not isinstance(body,bytes):raise ValueError('HTTP transport must return raw bytes')
        record.update(http_status=response['http_status'],response_url=response['url'],
                      sha256=sha_bytes(body),bytes=len(body),response_headers=response.get('headers',{}))
        cache=inside(folder,'downloads/'+name)
        cache.parent.mkdir(parents=True,exist_ok=True)
        with cache.open('xb') as stream:stream.write(body)
        record['download_path']=cache.relative_to(Path(root).resolve()).as_posix()
        if response['http_status']!=200 or response['url']!=url or len(body)!=record['expected_bytes'] or record['sha256']!=expected:
            raise ValueError('Anonymous HTTP status, final URL, hash or byte count differs')
        if pdf_pages is not None:
            record['pages']=page_counter(cache)
            if record['pages']!=pdf_pages:raise ValueError('Downloaded report PDF page count differs')
        record['passed']=True
    except Exception as error:
        record.update(passed=False,exception_type=type(error).__name__,error=str(error))
        if isinstance(error,urllib.error.HTTPError):record['http_status']=error.code
    record['completed_at_utc']=utc()
    write_new(folder/('check-'+hashlib.sha256(name.encode()).hexdigest()[:16]+'.json'),record)
    return record


def verify(root=ROOT,attempt=None,*,transport=None,git_check=verify_commit,page_counter=count_pdf_pages):
    root=Path(root).resolve();folder=root/'submission/v3'
    if not all((folder/name).is_file() for name in REQUIRED):
        raise ValueError('FREEZE, TEST_COMPLETE and GENERATED_FORM must all exist before publication verification')
    if (folder/'PUBLICATION_VERIFICATION.json').exists():
        raise FileExistsError('Preserve the unique existing PUBLICATION_VERIFICATION.json')
    if re.fullmatch('[a-z][a-z0-9_-]{0,47}',str(attempt)) is None:
        raise ValueError('A new short lowercase attempt name is required')
    parent=folder/'publication-attempts';parent.mkdir(exist_ok=True)
    evidence=inside(parent,attempt);evidence.mkdir(exist_ok=False)
    inputs={name:sha(folder/name) for name in REQUIRED}
    write_new(evidence/'STARTED.json',{'schema':'mp2-v3-publication-attempt-v1','started_at_utc':utc(),
        'attempt':attempt,'input_sha256':inputs,'fixed_repository':REPOSITORY,
        'operation':'Anonymous GET and local byte checks only; no score file opened, no issue posted.'})
    checks=[];report=None
    try:
        freeze,complete,generated=validate_records(root)
        report_path=inside(root,REPORT);report_hash=sha(report_path);pages=page_counter(report_path)
        git_result=git_check(root,freeze,generated['final_code_commit'],report_hash)
        write_new(evidence/'git-verification.json',git_result)
        transport=transport if transport is not None else AnonymousTransport()
        for name,expected in sorted({**freeze['source_files'],**freeze['assets']}.items()):
            checks.append(download_check(root,evidence,generated['final_code_commit'],name,expected,transport))
        report=download_check(root,evidence,generated['final_code_commit'],REPORT,report_hash,transport,
                              pdf_pages=pages,page_counter=page_counter)
        if any(not item['passed'] for item in [*checks,report]):
            raise RuntimeError('One or more anonymous public artifact checks failed; preserve this attempt')
        # Bind final publication evidence to unchanged generated/frozen inputs
        # and recheck local source, assets and report after network operations.
        if any(sha(folder/name)!=digest for name,digest in inputs.items()):
            raise ValueError('Form/freeze/test-completion evidence changed during anonymous verification')
        validate_records(root)
        if sha(report_path)!=report_hash:raise ValueError('Local report changed during anonymous verification')
        verified_at=utc()
        record={'schema':'mp2-v3-publication-verification-v1','passed':True,'verified_at_utc':verified_at,
            'final_code_commit':generated['final_code_commit'],'code_url':generated['code_url'],
            'checkpoint_url':generated['checkpoint_url'],'freeze_file_sha256':inputs['FREEZE.json'],
            'source_commit':freeze['source_commit'],'source_files':freeze['source_files'],'assets':freeze['assets'],
            'anonymous_file_checks':checks,'report_check':report,
            'generated_form_sha256':inputs['GENERATED_FORM.json'],'test_complete_sha256':inputs['TEST_COMPLETE.json'],
            'attempt_directory':evidence.relative_to(root).as_posix(),
            'evidence_sha256':{path.relative_to(root).as_posix():sha(path) for path in sorted(evidence.rglob('*')) if path.is_file()},
            'network_scope':'Anonymous fixed-repository raw.githubusercontent.com GET only; no credentials, proxies, redirects or issue API.'}
        write_new(evidence/'PASSED.json',record)
        write_new(folder/'PUBLICATION_VERIFICATION.json',record)
        return {'status':'passed','files':len(checks),'report_pages':pages,
                'record':'submission/v3/PUBLICATION_VERIFICATION.json','attempt':attempt}
    except BaseException as error:
        if not (evidence/'FAILED.json').exists():
            write_new(evidence/'FAILED.json',{'schema':'mp2-v3-publication-attempt-v1','passed':False,
                'failed_at_utc':utc(),'input_sha256':inputs,'exception_type':type(error).__name__,
                'error':str(error),'anonymous_file_checks':checks,'report_check':report,
                'recovery':'Preserve this evidence. A new named attempt may repeat read-only verification after a transient HTTP/setup failure; never manufacture a passing record.'})
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attempt',required=True,help='New lowercase read-only verification attempt name')
    args=parser.parse_args()
    print(json.dumps(verify(attempt=args.attempt),indent=2))


if __name__=='__main__':
    main()
