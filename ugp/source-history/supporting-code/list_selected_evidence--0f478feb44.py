from pathlib import Path
import json
import csv

root=Path(__file__).resolve().parent
d=json.loads((root/'remote_evidence_bundle.json').read_text(encoding='utf-8-sig'))
for path,rec in d['files'].items():
    if 'merged_feelwm_v7_runs' in path and Path(path).suffix in {'.json','.csv','.yaml'}:
        print(path,rec['size'])
for path,rec in d['files'].items():
    if 'merged_feelwm_v7_runs' in path and Path(path).name in {'resolved_config.yaml','index_metadata.json','selection_metadata.json','test_selection.json','selected_schedule.json','calibration_results.json','strict_test_summary.json'}:
        print('\nCONTENT',path,'\n',rec['text'][:8000])
