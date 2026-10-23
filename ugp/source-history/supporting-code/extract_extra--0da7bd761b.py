import json
from pathlib import Path
root=Path(__file__).resolve().parent
d=json.loads((root/'remote_extra_evidence.json').read_text(encoding='utf-8-sig'))
for p,r in d['files'].items():
    dest=root/'server_snapshot'/p.lstrip('/');dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(r['text'],encoding='utf-8')
(root/'feelwm_checkpoint_metadata.json').write_text(json.dumps(d['checkpoint_metadata'],indent=2),encoding='utf-8')
print('SNAPSHOT',d['snapshot_utc'],'RUNTIME',d.get('runtime'),'ERRORS',d['errors'])
print('MANIFEST_COUNTS',json.dumps(d['manifest_counts'],indent=2))
print('ADDITIONAL_ROOT',len(d['extra_inventory']))
for k,r in d['checkpoint_metadata'].items():
    print(k,json.dumps({n:r[n] for n in ['epoch','step','best_val','state_numel','num_state_tensors','optimizer_groups']}))
print('EXTRA_FILES',list(d['files']))
