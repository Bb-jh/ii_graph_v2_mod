#!/usr/bin/env python3
"""Amazon 2023 common dataset preparation (CPU, stdlib-only).

Policy: keep earliest user-item review, iterative 5-core, chronological
leave-last-two-out; short (<3) users in full are train-only.
No evaluation items are removed to force warm-start. Audit cold targets.
Output is intentionally versioned (processed_v2), never overwriting v1.
"""
import argparse
import gzip
import json
import sqlite3
import tempfile
from pathlib import Path

CATEGORIES = ('Industrial_and_Scientific', 'Musical_Instruments', 'Office_Products')


def open_db(path):
    db = sqlite3.connect(str(path))
    db.execute('PRAGMA journal_mode=DELETE')
    db.execute('PRAGMA temp_store=FILE')
    db.execute('PRAGMA busy_timeout=30000')
    db.execute('CREATE TABLE events (event_id INTEGER PRIMARY KEY, user_id TEXT NOT NULL, item_id TEXT NOT NULL, ts INTEGER NOT NULL)')
    return db


def ingest(db, review_file, batch_size=20000):
    buf, read = [], 0
    with gzip.open(review_file, 'rt', encoding='utf-8') as stream:
        for line in stream:
            if not line.strip():
                continue
            r = json.loads(line)
            u = r.get('user_id') or r.get('reviewerID')
            i = r.get('parent_asin') or r.get('asin')
            if not u or not i:
                continue
            ts = int(r.get('timestamp') or r.get('unixReviewTime') or 0)
            if ts >= 100_000_000_000:
                ts //= 1000
            buf.append((str(u), str(i), ts))
            read += 1
            if len(buf) >= batch_size:
                db.executemany('INSERT INTO events (user_id,item_id,ts) VALUES (?,?,?)', buf)
                db.commit()
                buf.clear()
                if read % 1_000_000 < batch_size:
                    print(f'  loaded {read:,} interactions', flush=True)
    if buf:
        db.executemany('INSERT INTO events (user_id,item_id,ts) VALUES (?,?,?)', buf)
        db.commit()
    print(f'  loaded total {read:,}', flush=True)


def restrict_metadata(db, meta_file):
    db.execute('CREATE TABLE meta_items (item_id TEXT PRIMARY KEY) WITHOUT ROWID')
    buf = []
    with gzip.open(meta_file, 'rt', encoding='utf-8') as stream:
        for line in stream:
            if not line.strip():
                continue
            r = json.loads(line)
            i = r.get('parent_asin') or r.get('asin')
            if i:
                buf.append((str(i),))
            if len(buf) >= 20000:
                db.executemany('INSERT OR IGNORE INTO meta_items VALUES (?)', buf)
                db.commit()
                buf.clear()
    if buf:
        db.executemany('INSERT OR IGNORE INTO meta_items VALUES (?)', buf)
        db.commit()
    before = db.execute('SELECT COUNT(*) FROM events').fetchone()[0]
    db.execute('DELETE FROM events WHERE item_id NOT IN (SELECT item_id FROM meta_items)')
    db.commit()
    db.execute('DROP TABLE meta_items')
    db.commit()
    after = db.execute('SELECT COUNT(*) FROM events').fetchone()[0]
    print(f'  metadata restriction: {before - after:,} interactions excluded', flush=True)


def deduplicate(db):
    """Retain earliest timestamp per (user, item), breaking ties by event_id."""
    before = db.execute('SELECT COUNT(*) FROM events').fetchone()[0]
    db.execute('CREATE INDEX idx_pair_order ON events(user_id, item_id, ts, event_id)')
    db.execute('''DELETE FROM events WHERE event_id NOT IN (
        SELECT MIN(event_id) FROM events e
        WHERE NOT EXISTS (
            SELECT 1 FROM events e2
            WHERE e2.user_id=e.user_id AND e2.item_id=e.item_id
              AND (e2.ts < e.ts OR (e2.ts=e.ts AND e2.event_id < e.event_id))
        ) GROUP BY user_id, item_id
    )''')
    db.commit()
    db.execute('DROP INDEX idx_pair_order')
    db.commit()
    after = db.execute('SELECT COUNT(*) FROM events').fetchone()[0]
    print(f'  deduplicated {before - after:,} interactions; kept {after:,}', flush=True)
    return before - after


def stats(db):
    n = db.execute('SELECT COUNT(*) FROM events').fetchone()[0]
    u = db.execute('SELECT COUNT(DISTINCT user_id) FROM events').fetchone()[0]
    i = db.execute('SELECT COUNT(DISTINCT item_id) FROM events').fetchone()[0]
    return {'interactions': n, 'users': u, 'items': i}


def five_core(db):
    iteration = 0
    while True:
        iteration += 1
        db.execute('CREATE TEMP TABLE low_users AS SELECT user_id FROM events GROUP BY user_id HAVING COUNT(*) < 5')
        low_u = db.execute('SELECT COUNT(*) FROM low_users').fetchone()[0]
        db.execute('DELETE FROM events WHERE user_id IN (SELECT user_id FROM low_users)')
        db.execute('DROP TABLE low_users')
        db.execute('CREATE TEMP TABLE low_items AS SELECT item_id FROM events GROUP BY item_id HAVING COUNT(*) < 5')
        low_i = db.execute('SELECT COUNT(*) FROM low_items').fetchone()[0]
        db.execute('DELETE FROM events WHERE item_id IN (SELECT item_id FROM low_items)')
        db.execute('DROP TABLE low_items')
        db.commit()
        count = db.execute('SELECT COUNT(*) FROM events').fetchone()[0]
        print(f'  iteration {iteration}: low_users={low_u:,}, low_items={low_i:,}, remaining={count:,}', flush=True)
        if low_u == 0 and low_i == 0:
            return iteration


def write_variant(db, outdir, variant, duplicate_removed, core_iterations):
    if outdir.exists():
        if any(outdir.iterdir()):
            raise FileExistsError(f'Refusing to overwrite nonempty output directory: {outdir}')
    outdir.mkdir(parents=True, exist_ok=True)
    with (outdir / 'item_mapping.tsv').open('w', encoding='utf-8') as f:
        f.write('item_id\tparent_asin\n')
        for idx, (item,) in enumerate(db.execute('SELECT DISTINCT item_id FROM events ORDER BY item_id')):
            f.write(f'{idx}\t{item}\n')

    files = {k: (outdir / f'{k}.jsonl').open('w', encoding='utf-8') for k in ('full','train','valid','test')}
    counts = {'full': 0, 'train': 0, 'valid': 0, 'test': 0, 'eligible_users': 0,
              'train_only_short_users': 0}
    train_items, evaluated_items = set(), set()
    cold_events = 0

    def emit(k, uid, item, ts):
        files[k].write(json.dumps({'user_id':uid,'parent_asin':item,'timestamp':ts}, ensure_ascii=False)+'\n')
        counts[k] += 1
        if k == 'train':
            train_items.add(item)
        elif k in ('valid','test'):
            evaluated_items.add(item)

    def flush_user(uid, seq):
        if uid is None:
            return
        for item, ts in seq:
            emit('full', uid, item, ts)
        if len(seq) < 3:
            counts['train_only_short_users'] += 1
            for item, ts in seq:
                emit('train', uid, item, ts)
            return
        counts['eligible_users'] += 1
        for item,ts in seq[:-2]:
            emit('train', uid, item, ts)
        emit('valid', uid, *seq[-2])
        emit('test', uid, *seq[-1])

    db.execute('CREATE INDEX IF NOT EXISTS idx_events_user_time ON events(user_id,ts,event_id)')
    db.commit()
    curr, seq = None, []
    try:
        for uid,item,ts in db.execute('SELECT user_id,item_id,ts FROM events ORDER BY user_id,ts,event_id'):
            if uid != curr:
                flush_user(curr,seq)
                curr,seq = uid, []
            seq.append((item,ts))
        flush_user(curr,seq)
    finally:
        for f in files.values():
            f.close()

    cold = evaluated_items - train_items
    # Count held-out events with globally train-absent item, exact and audit-friendly.
    for part in ('valid','test'):
        with (outdir / f'{part}.jsonl').open('r',encoding='utf-8') as f:
            for line in f:
                if json.loads(line)['parent_asin'] in cold:
                    cold_events += 1

    result = {
        'variant': variant, 'statistics': stats(db), 'output_counts': counts,
        'deduplication': {'policy': 'one event per (user,item); earliest timestamp, tie original file order',
                          'events_removed_prior_to_core': duplicate_removed},
        'five_core': {'policy': 'iterative degree >= 5 on deduplicated interactions, before split',
                      'iterations': core_iterations},
        'split_policy': 'chronological leave-last-two-out for users with >=3 events; short users train-only',
        'cold_start_audit': {'valid_test_events_item_absent_from_train':cold_events,
                             'distinct_valid_test_items_absent_from_train':len(cold),
                             'policy':'not removed; decide evaluation protocol before final baseline runs'},
        'graph_policy': 'build behavior graph using train.jsonl only; never validation/test interactions',
    }
    (outdir/'manifest.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(f'  {variant}: {result["statistics"]}; split={counts}; cold={result["cold_start_audit"]}',flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset',required=True,choices=CATEGORIES)
    p.add_argument('--raw-dir',type=Path,default=Path.home()/'datasets/amazon2023/raw')
    p.add_argument('--output-dir',type=Path,default=Path.home()/'datasets/amazon2023/processed_v2')
    p.add_argument('--work-dir',type=Path,default=Path.home()/'datasets/amazon2023/work')
    args = p.parse_args()
    rev = args.raw_dir/f'{args.dataset}.jsonl.gz'
    meta = args.raw_dir/f'meta_{args.dataset}.jsonl.gz'
    for f in (rev,meta):
        if not f.is_file():
            p.error(f'Missing input file: {f}')
    for variant in ('full','5core'):
        outdir = args.output_dir/variant/args.dataset
        if outdir.exists() and any(outdir.iterdir()):
            p.error(f'Output exists: {outdir} (use another --output-dir to preserve results)')
    args.work_dir.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f'amazon23_{args.dataset}_',dir=args.work_dir) as tmp:
        db = open_db(Path(tmp)/'working.sqlite')
        try:
            print('Ingesting raw reviews...',flush=True)
            ingest(db,rev)
            print('Restricting to metadata items...',flush=True)
            restrict_metadata(db,meta)
            print('Deduplicating (user,item), keeping earliest...',flush=True)
            removed = deduplicate(db)
            print('Writing full variant...',flush=True)
            write_variant(db,args.output_dir/'full'/args.dataset,'full',removed,None)
            print('Iterative 5-core...',flush=True)
            iterations = five_core(db)
            print('Writing 5core variant...',flush=True)
            write_variant(db,args.output_dir/'5core'/args.dataset,'5core',removed,iterations)
        finally:
            db.close()
    print('SUCCESS',flush=True)

if __name__=='__main__':
    main()
