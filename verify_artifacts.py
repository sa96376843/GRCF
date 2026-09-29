"""Verify uploaded data/code/results using Python's standard library."""
from pathlib import Path
import ast,json,hashlib
ROOT=Path(__file__).resolve().parent
manifest=json.loads((ROOT/'ARTIFACTS_SHA256.json').read_text(encoding='utf-8'))
for rel,expected in manifest.items():
    p=ROOT/rel
    assert p.is_file(),f'Missing: {rel}'
    data=p.read_bytes()
    assert len(data)==expected['bytes'],f'Size differs: {rel}'
    assert hashlib.sha256(data).hexdigest()==expected['sha256'],f'Hash differs: {rel}'
    if p.suffix=='.py':ast.parse(data.decode('utf-8'),filename=rel)
main=json.loads((ROOT/'reproducibility/results/main.json').read_text())
assert set(main)=={'XTX','MARS'}
methods={'LR','RF','MLP','WideDeep','NCF','LightGCN','SGL','GRA-MLOF','GRCF'}
for runs in main.values():
    assert set(runs)=={'0','1','2'}
    for methods_run in runs.values():assert set(methods_run)==methods
ablation=json.loads((ROOT/'reproducibility/results/ablation.json').read_text())
assert len(ablation)==16
print(f'PASS: {len(manifest)} files; 54 main runs; 16 ablation entries.')
