from pathlib import Path
import json

root=Path(__file__).resolve().parent
d=json.loads((root/'remote_evidence_bundle.json').read_text(encoding='utf-8-sig'))
for path,rec in d['log_summaries'].items():
    print('\nLOG',path)
    for line in rec['signals']: print(line)
    print('LAST', json.dumps({k: (rec.get('last_train_record') or {}).get(k) for k in ['epoch','step','total']}))
for path,rec in d['metric_summaries'].items():
    a=rec['last_train'] or {};v=rec['last_validation'] or {};b=rec['best_validation'] or {}
    print('METRIC',path, json.dumps({'counts':rec['record_counts'],'last_epoch':a.get('epoch'),'last_step':a.get('step'),'last_train_total':a.get('total'),'last_val_step':v.get('step'),'last_val_total':v.get('total'),'best_val_step':b.get('step'),'best_val_total':b.get('total'),'best_physical_score':b.get('physical_score'),'resets':rec['nonincreasing_step_events']}))
for p,rec in d['files'].items():
    name=Path(p).name
    if name=='strict_test_metrics.json':
        print('\nSTRICT',p,json.dumps(json.loads(rec['text'])['global']))
    if name in {'phase.txt','metadata.json','calibration_summary.json','selected_lambdas.json','selection_summary.json','training_latest.state.json','V7_CONFIG.json','V7F_CONFIG.json','V7F_UNSEEN_TRAJECTORY_CONFIG.json'}:
        print('\nFILE',p,rec['text'][:5000])
