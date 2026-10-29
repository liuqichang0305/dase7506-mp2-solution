"""Record a one-time pre-test freeze from the reviewed committed source."""
from pathlib import Path
from datetime import datetime, timezone
import subprocess, hashlib, json
ROOT=Path(__file__).resolve().parents[2]
target=ROOT/'submission/PRE_TEST_FREEZE.json'
if target.exists():raise SystemExit('Freeze already exists; never overwrite it')
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
source=[ROOT/'code/solution.py',ROOT/'code/train.py',*sorted((ROOT/'code/mp2').glob('*.py')),
        *[ROOT/'code'/name for name in ['common.py','predict.py','score_predictions.py','check_data.py']],
        ROOT/'code/configs/recommended.json',ROOT/'requirements-inference.txt',ROOT/'requirements-training.txt']
commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
for p in source:
    data=subprocess.check_output(['git','show',f'{commit}:{p.relative_to(ROOT).as_posix()}'],cwd=ROOT)
    if hashlib.sha256(data).hexdigest()!=sha(p):
        raise SystemExit(f'Uncommitted or newline-different source: {p}')
summary=json.loads((ROOT/'results/development/search_summary.json').read_text())
chosen=next(x for x in summary['validation'] if x['experiment_id']==summary['recommended_learned'])
record={'schema':'dase7506-mp2-freeze-v1','frozen':True,'created_at_utc':datetime.now(timezone.utc).isoformat(),
        'source_commit':commit,'upstream_commit':'bb801f4f2e7629d270df765ab12c333b2dc616bb',
        'source_files':{p.relative_to(ROOT).as_posix():sha(p) for p in source},
        'assets':{p.relative_to(ROOT).as_posix():sha(p) for p in sorted((ROOT/'artifacts/final_bundle').glob('*')) if p.is_file()},
        'uncompressed_asset_bytes':json.loads((ROOT/'results/asset-audit.json').read_text())['uncompressed_bytes'],
        'seed':3036900957,'training_split':'train only','configuration':summary['recommended_config'],
        'calibration':'Train-only three-fold out-of-fold logistic calibrator; does not change choices',
        'validation':chosen,'validation_selection_reason':'Highest complete validation accuracy among the prespecified three learned family winners and two baselines. Improvement over embedding BCE is uncertain (paired bootstrap interval crosses zero). No additional search; no test score has been seen.',
        'test_evaluated_before_freeze':False,'inference_dependencies':['numpy==2.5.3'],'threads':1,
        'audit_evidence':['docs/CODE_REVIEW.md','results/contract-validation.json','results/contract-synthetic.json','results/official-hashes.json','results/asset-audit.json']}
target.write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'freeze':str(target),'source_commit':commit,'files':len(source),'asset_bytes':record['uncompressed_asset_bytes']},indent=2))
