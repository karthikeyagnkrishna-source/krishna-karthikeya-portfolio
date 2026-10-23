from pathlib import Path
import json
import hashlib

ROOT = Path(__file__).resolve().parent
d = json.loads((ROOT/'remote_evidence_bundle.json').read_text(encoding='utf-8-sig'))
out = ROOT/'server_snapshot'
manifest=[]
for remote,rec in d['files'].items():
    local=out/remote.lstrip('/')
    local.parent.mkdir(parents=True,exist_ok=True)
    data=rec['text'].encode('utf-8')
    local.write_bytes(data)
    manifest.append({'remote':remote,'local':str(local),'remote_sha256':rec['sha256'],'local_sha256':hashlib.sha256(data).hexdigest(),'remote_size':rec['size']})
for remote,arc in d['archives'].items():
    dest=ROOT/'archive_snapshot'/Path(remote).stem
    for member,rec in arc['selected_files'].items():
        if '..' in Path(member).parts: raise ValueError(member)
        local=dest/member
        local.parent.mkdir(parents=True,exist_ok=True)
        local.write_text(rec['text'],encoding='utf-8')
        manifest.append({'archive':remote,'member':member,'local':str(local),'remote_sha256':rec['sha256']})
(ROOT/'evidence_manifest.json').write_text(json.dumps({'snapshot_utc':d['snapshot_utc'],'files':manifest,'errors':d['errors']},indent=2),encoding='utf-8')
(ROOT/'run_summaries.json').write_text(json.dumps({k:d[k] for k in ['snapshot_utc','log_summaries','metric_summaries']},indent=2),encoding='utf-8')
print(json.dumps({'snapshot_utc':d['snapshot_utc'],'n_files':len(d['files']),'n_archives':len(d['archives']),'n_logs':len(d['log_summaries']),'n_metrics':len(d['metric_summaries']),'errors':d['errors']},indent=2))
for path,rec in d['metric_summaries'].items():
    print(path, 'FIRST',rec['first_train'],'LAST',rec['last_train'],'VAL',rec['last_validation'])
