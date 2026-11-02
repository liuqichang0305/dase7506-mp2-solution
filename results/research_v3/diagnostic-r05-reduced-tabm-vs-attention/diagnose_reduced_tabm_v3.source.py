"""One fixed R05 train-OOF diagnostic: feature ablation and one reciprocal gate.

No fitting, calibration, fusion, raw gold, validation, test, or ranking access.
The existing immutable diagnostic guard is installed before official input I/O.
"""
from __future__ import annotations
import os
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code/tools'))
import diagnose_oof_v3 as diagnostic
from analyze_oof_v2 import paired_comparison

REDUCED='v3:v3_tabm_k4_embedding_stats_s0'
FULL='v3:v3_tabm_k4_s0'
ATTENTION='v3:v3_attention_ema_s0'
METHODS=[REDUCED,FULL,ATTENTION]
OUTPUT=ROOT/'results/research_v3/diagnostic-r05-reduced-tabm-vs-attention'


def main():
    if len(sys.argv)!=1:raise ValueError('This fixed diagnostic has no model/seed/fusion overrides')
    started=time.perf_counter()
    result=diagnostic.diagnose(METHODS,[[REDUCED,ATTENTION]],OUTPUT)
    if len(result['complementarity'])!=1 or result['complementarity'][0]['methods']!=[REDUCED,ATTENTION] or result['fusion']['status']!='not_run':
        raise ValueError('Only the requested reduced-TabM/Attention gate may be evaluated')
    manifest=json.loads((diagnostic.DATA/'manifest.json').read_text(encoding='utf-8'))
    records={key:diagnostic.load_method(key,manifest) for key in METHODS}
    first=records[REDUCED]
    for record in records.values():
        if any(not np.array_equal(first[key],record[key]) for key in ('labels','groups','fold')):
            raise ValueError('Question-level alignment changed after the diagnostic')
    left,right=records[REDUCED]['config'],records[FULL]['config']
    differences={key:{'reduced':left.get(key),'full':right.get(key)} for key in sorted(set(left)|set(right)) if left.get(key)!=right.get(key)}
    if set(differences)!={'experiment_id','input_features'} or left['input_features']!='embedding_stats' or right.get('input_features','relation')!='relation':
        raise ValueError('Matched TabM recipes differ beyond the declared feature subset')
    bank=diagnostic.load_bank('train')
    slices,hard,definitions=diagnostic.fixed_slices(bank,first['labels'],first['groups'])
    paired=paired_comparison(first['correct'],records[FULL]['correct'],hard,5000,3036901059)
    single=first['labels'].sum(1)==1
    paired['single_positive_counts']={'n':int(single.sum()),'reduced_correct':int(first['correct'][single].sum()),
        'full_correct':int(records[FULL]['correct'][single].sum()),
        'attention_correct':int(records[ATTENTION]['correct'][single].sum())}
    slice_deltas=[]
    for dimension,bucket,mask in slices:
        reduced=first['correct'][mask];full=records[FULL]['correct'][mask];attention=records[ATTENTION]['correct'][mask]
        slice_deltas.append({'dimension':dimension,'bucket':bucket,'n':int(mask.sum()),
            'reduced_correct':int(reduced.sum()),'full_correct':int(full.sum()),'attention_correct':int(attention.sum()),
            'reduced_minus_full':int(reduced.sum())-int(full.sum()),
            'reduced_only_vs_attention':int((reduced & ~attention).sum()),
            'attention_only_vs_reduced':int((attention & ~reduced).sum()),
            'both_wrong_reduced_attention':int((~reduced & ~attention).sum())})
    input_names={'code/data/manifest.json','code/data/candidates/gsm8k_train.jsonl'}
    source_names={'code/tools/diagnose_reduced_tabm_v3.py','code/tools/diagnose_oof_v3.py',
        'code/tools/analyze_oof_v2.py','code/tools/audit_assets.py','code/common.py',
        'code/mp2/selection_v2.py','code/mp2/research_v3/selection.py'}
    for record in records.values():
        folder=record['folder']
        input_names.update((folder/name).relative_to(ROOT).as_posix() for name in ('registration.json','summary.json','oof.npz'))
        input_names.add((folder.parent/'common_folds.json').relative_to(ROOT).as_posix())
        source_names.update(record['registration']['source_files'])
        for fold_report in record['summary']['folds']:
            input_names.update((folder/name).relative_to(ROOT).as_posix() for name in fold_report.get('output_sha256',{}))
            input_names.add((folder/f"fold{fold_report['fold']}-scores.npy").relative_to(ROOT).as_posix())
            if (folder/f"fold{fold_report['fold']}-raw-scores.npy").exists():
                input_names.add((folder/f"fold{fold_report['fold']}-raw-scores.npy").relative_to(ROOT).as_posix())
    source_hashes={name:diagnostic.digest(ROOT/name) for name in sorted(source_names)}
    input_hashes={name:diagnostic.digest(ROOT/name) for name in sorted(input_names)}
    report={'schema':'mp2-v3-reduced-tabm-diagnostic-v1','status':'complete','created_at_utc':datetime.now(timezone.utc).isoformat(),
        'methods':METHODS,'matched_feature_ablation':{'first':REDUCED,'second':FULL,'configuration_differences':differences,
            'input_dimensions':{'reduced':1040,'full':2064},'paired_comparison':paired},
        'only_reciprocal_gate':result['complementarity'][0],'fixed_slice_deltas':slice_deltas,
        'fixed_slice_definitions':result['slice_definitions'],'folds_sha256':result['folds_sha256'],
        'input_sha256':input_hashes,'source_sha256':source_hashes,'diagnostic_sha256':diagnostic.digest(OUTPUT/'diagnostic.json'),
        'numpy_version':np.__version__,'command':'.venv/Scripts/python.exe code/tools/diagnose_reduced_tabm_v3.py',
        'bootstrap':{'function':'analyze_oof_v2.paired_comparison','repetitions':5000,'base_seed':3036901059,
            'actual_overall_seed':paired['overall']['bootstrap_seed'],'actual_hard_seed':paired['hard']['bootstrap_seed']},
        'fusion':{'status':'not_run'},'calibration_fit':False,'validation_test_ranking_read':False,
        'question_text_or_answers_output':False,'elapsed_seconds':time.perf_counter()-started,
        'limitations':['The paired interval is conditional on completed training OOF predictions; search-selection optimism remains.',
            'The matched recipe changes feature subset and hence first-layer parameter count; it does not isolate information removal from capacity/regularization effects.',
            'Fixed descriptive slices overlap and are not separate confirmatory significance tests.',
            'Passing a reciprocal/oracle gate does not demonstrate any fused-model improvement. No probabilities are averaged in this run.']}
    diagnostic.write_new(OUTPUT/'paired-and-slices.json',report)
    with (OUTPUT/'diagnose_reduced_tabm_v3.source.py').open('xb') as out:out.write(Path(__file__).read_bytes())
    with (OUTPUT/'analyze_oof_v2.source.py').open('xb') as out:out.write((ROOT/'code/tools/analyze_oof_v2.py').read_bytes())
    print(json.dumps({'status':'complete','output':str(OUTPUT),'matched_paired':paired,
        'only_gate':result['complementarity'][0],'fusion':'not_run','calibration_fit':False},indent=2))


if __name__=='__main__':main()
