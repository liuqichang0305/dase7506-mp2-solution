"""Anonymous exact-commit peer snapshot; never execute downloaded code."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = Path(__file__).resolve().parent
DEST = ROOT / 'tmp/peer_review/submission_240_7e3a2792/source'
COMMIT = '7e3a2792b7b4442f5e53705a64b4560a2e365de9'
BASE = 'https://raw.githubusercontent.com/Dylan0528/DASE7506-Project2/' + COMMIT + '/'
FILES = ['README.md', '.gitattributes', 'code/submission.py', 'code/features.py', 'code/model.py',
         'code/train.py', 'code/requirements.txt', 'code/requirements-train.txt', 'code/common.py',
         'code/predict.py', 'code/score_predictions.py', 'code/PACKAGE_MANIFEST.json',
         'code/runs/model_bundle/config.json', 'code/runs/model_bundle/scaler_mean.npy',
         'code/runs/model_bundle/scaler_std.npy', 'code/runs/model_bundle/weights.npy',
         'code/data/LICENSE.txt', 'code/data/NOTICE.txt']


def main():
    if DEST.exists():
        raise FileExistsError('Preserve the previous exact snapshot or partial download')
    DEST.mkdir(parents=True)
    target = json.loads((EVIDENCE / 'target.json').read_text(encoding='utf-8'))
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    records = []
    for name in FILES:
        url = BASE + name
        request = urllib.request.Request(url, headers={'User-Agent': 'DASE7506-independent-peer-reproduction'})
        with opener.open(request, timeout=45) as response:
            content = response.read()
            if response.status != 200:
                raise ValueError('Non-200 download')
        blob = hashlib.sha1(b'blob ' + str(len(content)).encode() + b'\0' + content).hexdigest()
        expected = target['expected_files'].get(name)
        if expected and (expected['git_blob_sha'] != blob or expected['bytes'] != len(content)):
            raise ValueError('Downloaded file differs from GitHub immutable tree metadata: ' + name)
        path = DEST / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as stream:
            stream.write(content)
        row = {'path': name, 'url': url, 'http_status': 200, 'bytes': len(content),
               'sha256': hashlib.sha256(content).hexdigest(), 'git_blob_sha': blob,
               'git_blob_metadata_verified': expected is not None}
        records.append(row)
        with (EVIDENCE / 'download_progress.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row) + '\n')
    result = {'schema': 'mp2-peer-download-v1', 'status': 'complete', 'source_commit': COMMIT,
              'created_at_utc': datetime.now(timezone.utc).isoformat(), 'destination': str(DEST),
              'anonymous': True, 'authentication_headers_sent': False, 'downloaded_code_executed': False,
              'raw_task_data_downloaded': False, 'files': records}
    with (EVIDENCE / 'download_complete.json').open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2); stream.write('\n')
    print(json.dumps({'status': 'complete', 'files': len(records), 'destination': str(DEST)}))


if __name__ == '__main__':
    main()
