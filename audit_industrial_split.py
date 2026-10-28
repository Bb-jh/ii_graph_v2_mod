#!/usr/bin/env python3
"""Audit prepared Amazon 2023 dataset and optional CLIP item IDs (read-only)."""
import argparse, json, sqlite3, tempfile
from pathlib import Path


def ingest(db, name, path):
    rows=[]; count=0
    with path.open(encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            x=json.loads(line)
            rows.append((name,x['user_id'],x['parent_asin'],int(x['timestamp'])))
            count+=1
            if len(rows)==20000:
                db.executemany('INSERT INTO events VALUES (?,?,?,?)',rows); db.commit();rows=[]
    if rows: db.executemany('INSERT INTO events VALUES (?,?,?,?)',rows); db.commit()
    return count

def n(db,sql): return db.execute(sql).fetchone()[0]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--processed-dir',type=Path,default=Path.home()/'datasets/amazon2023/processed/5core/Industrial_and_Scientific')
    ap.add_argument('--embedding',type=Path)
    a=ap.parse_args()
    with tempfile.TemporaryDirectory(prefix='audit_split_') as t:
        db=sqlite3.connect(str(Path(t)/'audit.sqlite'))
        db.execute('CREATE TABLE events(split TEXT, uid TEXT, iid TEXT, ts INT)')
        counts={s:ingest(db,s,a.processed_dir/f'{s}.jsonl') for s in ('full','train','valid','test')}
        db.execute('CREATE INDEX ix1 ON events(split,uid,iid)');db.execute('CREATE INDEX ix2 ON events(split,iid)')
        print('File counts:', counts)
        print('Check full = train + valid + test:',counts['full']==sum(counts[s] for s in ('train','valid','test')))
        print('5-core min user degree:',n(db,"SELECT MIN(cnt) FROM (SELECT COUNT(*) cnt FROM events WHERE split='full' GROUP BY uid)"))
        print('5-core min item degree:',n(db,"SELECT MIN(cnt) FROM (SELECT COUNT(*) cnt FROM events WHERE split='full' GROUP BY iid)"))
        print('Repeated user-item pairs (full):',n(db,"SELECT COUNT(*) FROM (SELECT 1 FROM events WHERE split='full' GROUP BY uid,iid HAVING COUNT(*)>1)"))
        print('Valid/test items absent from train:',n(db,"SELECT COUNT(*) FROM events a WHERE a.split IN ('valid','test') AND NOT EXISTS(SELECT 1 FROM events b WHERE b.split='train' AND b.iid=a.iid)"))
        print('Valid/test distinct items absent from train:',n(db,"SELECT COUNT(DISTINCT a.iid) FROM events a WHERE a.split IN ('valid','test') AND NOT EXISTS(SELECT 1 FROM events b WHERE b.split='train' AND b.iid=a.iid)"))
        print('Users with !=1 validation:',n(db,"SELECT COUNT(*) FROM (SELECT uid,COUNT(*) c FROM events WHERE split='valid' GROUP BY uid HAVING c!=1)"))
        print('Users with !=1 test:',n(db,"SELECT COUNT(*) FROM (SELECT uid,COUNT(*) c FROM events WHERE split='test' GROUP BY uid HAVING c!=1)"))
        print('Chronology violations train > valid:',n(db,"SELECT COUNT(*) FROM events v WHERE v.split='valid' AND EXISTS(SELECT 1 FROM events tr WHERE tr.split='train' AND tr.uid=v.uid AND tr.ts>v.ts)"))
        print('Chronology violations valid > test:',n(db,"SELECT COUNT(*) FROM events v JOIN events te ON v.uid=te.uid WHERE v.split='valid' AND te.split='test' AND v.ts>te.ts"))
        mp=a.processed_dir/'item_mapping.tsv'
        with mp.open(encoding='utf-8') as f:
            next(f)
            ids={line.rstrip('\n').split('\t',1)[1] for line in f if line.strip()}
        print('Mapping items:',len(ids),'Full unique items:',n(db,"SELECT COUNT(DISTINCT iid) FROM events WHERE split='full'"))
        if a.embedding:
            import numpy as np
            with np.load(a.embedding,allow_pickle=False) as npz:
                keys=list(npz.files)
                print('CLIP NPZ keys:',keys)
                for k in keys:
                    if k in ('item_ids','parent_asins','asins','item_id','ids'):
                        emb_ids=set(str(x) for x in npz[k].tolist())
                        print(f'CLIP ID key={k}, ids={len(emb_ids)}, missing mapping IDs={len(ids-emb_ids)}')
                        print('Missing examples:',list(sorted(ids-emb_ids))[:5])
                        break
                else:
                    print('CLIP ID matching skipped: recognized ID field not found; inspect keys above.')
        db.close()
if __name__=='__main__': main()
