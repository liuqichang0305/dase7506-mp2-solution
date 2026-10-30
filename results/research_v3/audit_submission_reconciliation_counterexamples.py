"""Generated fixture and fake transport only; never performs HTTP or real evaluation."""
import hashlib
import importlib.util
import json
from pathlib import Path
import urllib.parse

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('synthetic_submit_fixture', ROOT/'code/tests/test_submit_final_v3.py')
t = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t)


def run(folder_name):
    folder = ROOT/'results/research_v3'/folder_name
    folder.mkdir(exist_ok=False)
    source = ROOT/'submission/v3/submit_final.py'
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    (folder/'submit_final.py.snapshot.txt').write_bytes(source.read_bytes())
    cases = []
    for kind in ('wrong_form_protocol', 'wrong_envelope_algorithm'):
        fixture = t.SubmissionTests()
        fixture.setUp()
        try:
            payload = fixture.generated['official_payload']
            if kind == 'wrong_form_protocol':
                payload['protocol'] = 'synthetic-invalid-protocol'
            else:
                payload['artifacts']['algorithm'] = 'synthetic-invalid-algorithm'
            payload_text = json.dumps(payload, indent=2)
            body = 'Synthetic fixture only.\n```json\n' + payload_text + '\n```'
            url = 'https://github.com/xudongwu-0/xudongwu-0.github.io/issues/new?' + urllib.parse.urlencode({'title': fixture.title, 'body': body})
            for name, value in [('issue-body.txt', body), ('issue-url.txt', url)]:
                (fixture.folder/name).write_text(value, encoding='utf-8', newline='')
            fixture.generated.update(issue_body=body, issue_url=url, official_payload_json_text=payload_text,
                issue_body_sha256=t.s.sha_bytes(body.encode()), issue_url_sha256=t.s.sha_bytes(url.encode()))
            fixture.save('GENERATED_FORM.json', fixture.generated)
            issue = {**fixture.issue, 'body':body}
            api = t.FakeTransport(issue, pages=[[issue]])
            try:
                # Stub Git transport only; official payload check must be real when present.
                value = t.s.submit(fixture.root, transport=api, now_fn=lambda:t.NOW,
                    git_check=lambda *args:None, phase_check=lambda *args:None)
                result = {'unexpectedly_accepted':True, 'result':value}
            except Exception as error:
                result = {'unexpectedly_accepted':False, 'exception':type(error).__name__, 'message':str(error)}
            cases.append({'case':kind, 'fake_network_calls':len(api.calls),
                          'post_calls':sum(c[0]=='POST' for c in api.calls), **result})
        finally:
            fixture.tearDown()
    result = {'scope':'Generated fixtures and fake transport only; no official data/score, credentials, HTTP or submission',
        'source_sha256_before':before, 'source_sha256_after':hashlib.sha256(source.read_bytes()).hexdigest(), 'cases':cases}
    (folder/'review.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    import sys
    run(sys.argv[1])
