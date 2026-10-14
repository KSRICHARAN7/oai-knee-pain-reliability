#!/usr/bin/env python3
"""Fail closed on changed/unlisted release files and record-like CSV fields."""
from pathlib import Path
import csv,hashlib,json,re
ROOT=Path(__file__).resolve().parents[1]
IGNORED={'.git','.venv','__pycache__','local_runs'}
def files():
    for p in ROOT.rglob('*'):
        rel=p.relative_to(ROOT)
        if any(part in IGNORED for part in rel.parts):continue
        if len(rel.parts)>1 and rel.parts[0]=='analysis' and rel.parts[1] in {'outputs','outputs_repeat'}:continue
        if p.is_file() and rel.as_posix()!='PUBLIC_MANIFEST.json':yield p

def main():
    manifest=json.loads((ROOT/'PUBLIC_MANIFEST.json').read_text())
    actual={p.relative_to(ROOT).as_posix():p for p in files()}
    expected={x['path']:x for x in manifest['files']}
    if set(actual)!=set(expected):raise SystemExit(f'Allowlist mismatch: extra={sorted(set(actual)-set(expected))}, missing={sorted(set(expected)-set(actual))}')
    forbidden={'id','participant_key','knee_key','predicted_womac_48m','observed_womac_48m','uncertainty_score'}
    for name,p in actual.items():
        if hashlib.sha256(p.read_bytes()).hexdigest()!=expected[name]['sha256']:raise SystemExit(f'Hash mismatch: {name}')
        if p.suffix.lower() in {'.zip','.gz','.parquet','.pkl','.joblib','.xlsx','.pdf'}:raise SystemExit(f'Forbidden public binary: {name}')
        if p.suffix=='.csv':
            with p.open(newline='') as f:
                reader=csv.reader(f);header=next(reader)
                if forbidden & {s.lower().strip() for s in header}:raise SystemExit(f'Record-level header: {name}')
                rows=list(reader)
                if len(rows)>1000:raise SystemExit(f'Unexpectedly large aggregate table: {name}')
        if p.suffix in {'.py','.md','.json','.yml','.txt','.csv'}:
            text=p.read_text()
            if re.search(r'(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|AKIA[A-Z0-9]{16})',text):raise SystemExit(f'Possible credential: {name}')
    print(f'PASS: {len(actual)} allowlisted public files; hashes and privacy checks passed')
if __name__=='__main__':main()
