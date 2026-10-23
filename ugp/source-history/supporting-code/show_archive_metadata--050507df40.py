from pathlib import Path
import json
d=json.loads((Path(__file__).parent/'remote_evidence_bundle.json').read_text(encoding='utf-8-sig'))
for p,a in d['archives'].items():
    print('ARCHIVE',p,'SIZE',a['size'])
    for x in a['members']:
        if any(s in x['name'].lower() for s in ['metric','manifest','readme','source','metadata','summary','prediction']): print(x)
    if 'p2000' in p:
        for name,rec in a['selected_files'].items():
            if any(s in name.lower() for s in ['metric','readme','metadata','summary']): print(name,rec['text'][:7000])
