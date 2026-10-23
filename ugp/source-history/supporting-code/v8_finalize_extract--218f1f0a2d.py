import csv, hashlib, json, math, statistics, difflib
from collections import Counter, defaultdict
from pathlib import Path

root = Path(__file__).parent
base = root / 'server_snapshot/path/to/ugp/V7F_4080_Run'
nb_path = base / 'V7F_2xRTX4080_EXPANDED_28TRAIN_4STRICT_HOLDOUT_CHECKPOINTED_FINAL.ipynb'
ref_path = Path(r'/path/to/ugp/local')
nb = json.loads(nb_path.read_text(encoding='utf-8'))
ref = json.loads(ref_path.read_text(encoding='utf-8'))
sources = [''.join(c.get('source', [])) for c in nb['cells']]
ref_sources = [''.join(c.get('source', [])) for c in ref['cells']]
result = {'notebook_sha256': hashlib.sha256(nb_path.read_bytes()).hexdigest(), 'cells': []}
extract = []
for idx, (cell, source) in enumerate(zip(nb['cells'], sources)):
    if any(s in source for s in ['V8-A', 'V8-B', 'V8-C', 'V8-D', 'V8-E', 'V8-F']):
        digest = hashlib.sha256(source.encode('utf-8')).hexdigest()
        matches = [i for i, s in enumerate(ref_sources) if hashlib.sha256(s.encode('utf-8')).hexdigest() == digest]
        result['cells'].append({'index': idx, 'type': cell['cell_type'], 'execution_count': cell.get('execution_count'), 'outputs': len(cell.get('outputs', [])), 'lines': len(source.splitlines()), 'sha256': digest, 'identical_reference_cells': matches, 'heading': source.splitlines()[0] if source.splitlines() else ''})
        extract += [f'===== DEPLOYED CELL {idx} ====='] + [f'C{idx}:L{i} {line}' for i, line in enumerate(source.splitlines(), 1)]
(root / 'v8_rtx_deployed_source.txt').write_text('\n'.join(extract) + '\n', encoding='utf-8')
result['csv'] = {}
for path in sorted((base / 'storage/V8_contactmap').rglob('*.csv')):
    with path.open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        result['csv'][str(path.relative_to(base / 'storage/V8_contactmap'))] = {'rows': len(rows), 'columns': reader.fieldnames, 'head': rows[:2]}
        if '16TEST_SHADOW_PLANS' in path.name or 'SAFE_CURIOSITY' in path.name or 'CONTACT_TIMING' in path.name:
            result['csv'][str(path.relative_to(base / 'storage/V8_contactmap'))]['all'] = rows
result['memory_meta'] = {p.name: json.loads(p.read_text(encoding='utf-8')) for p in (base / 'storage/V8_contactmap/A_memory').glob('*META.json')}
(root / 'v8_rtx_inspection.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
diff = '\n'.join(difflib.unified_diff(ref_sources[22].splitlines(), sources[21].splitlines(), fromfile='reference_cell22', tofile='deployed_cell21'))
(root / 'v8_rtx_f_source_diff.txt').write_text(diff + '\n', encoding='utf-8')
print(json.dumps({'notebook_sha256':result['notebook_sha256'],'cells':result['cells'],'memory_meta':result['memory_meta']}, indent=2))
print(diff)
