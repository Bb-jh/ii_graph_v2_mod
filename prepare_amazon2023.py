#!/usr/bin/env python3
"""Disk-backed Amazon Reviews 2023 Full/iterative-5-core preparation.

Split: chronological leave-last-two-out for users with >=3 interactions.
Raw JSONL.GZ and CLIP embeddings are never modified. Requires Python stdlib only.
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
    buffer = []
    read = 0
    with gzip.open(review_file, 'rt', encoding='utf-8') as stream:
        for line in stream:
            if not line.strip():
                continue
            r = json.loads(line)
            u, i = r.get('user_id') or r.get('reviewerID'), r.get('parent_asin') or r.get('asin')
            if not u or not i:
                continue
            ts = int(r.get('timestamp') or r.get('unixReviewTime') or 0)
            if ts >= 100_000_000_000:
                ts //= 1000
            buffer.append((str(u), str(i), ts))
            read += 1
            if len(buffer) >= batch_size:
                db.executemany('INSERT INTO events (user_id,item_id,ts) VALUES (?,?,?)', buffer)
                db.commit()
                buffer.clear()
                if read % 1_000_000 < batch_size:
                    print(f'  loaded {read:,} interactions', flush=True)
    if buffer:
        db.executemany('INSERT INTO events (user_id,item_id,ts) VALUES (?,?,?)', buffer)
        db.commit()
    print(f'  loaded total {read:,}', flush=True)


def save_metadata_ids(db, meta_file):
    db.execute('CREATE TABLE meta_items (item_id TEXT PRIMARY KEY) WITHOUT ROWID')
    buf = []
    with gzip.open(meta_file, 'rt', encoding='utf-8') as f:
        for line in f:
            if line.strip():
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
    db.execute('DELETE FROM events WHERE item_id NOT IN (SELECT item_id FROM meta_items)')
    db.commit()
    db.execute('DROP TABLE meta_items')
    db.commit()


def stats(db):
    n = db.execute('SELECT COUNT(*) FROM events').fetchone()[0]
    u = db.execute('SELECT COUNT(DISTINCT user_id) FROM events').fetchone()[0]
    i = db.execute('SELECT COUNT(DISTINCT item_id) FROM events').fetchone()[0]
    return {'interactions': n, 'users': u, 'items': i}


def five_core(db):
    iteration = 0
    while True:
        iteration += 1
        # Delete all currently under-5 users, then all under-5 items.
        # Repeat until both sides satisfy the criterion.
        db.execute('DROP TABLE IF EXISTS low_users')
        db.execute('CREATE TEMP TABLE low_users AS SELECT user_id FROM events GROUP BY user_id HAVING COUNT(*) < 5')
        low_u = db.execute('SELECT COUNT(*) FROM low_users').fetchone()[0]
        db.execute('DELETE FROM events WHERE user_id IN (SELECT user_id FROM low_users)')
        db.commit()
        db.execute('DROP TABLE low_users')
        db.execute('DROP TABLE IF EXISTS low_items')
        db.execute('CREATE TEMP TABLE low_items AS SELECT item_id FROM events GROUP BY item_id HAVING COUNT(*) < 5')
        low_i = db.execute('SELECT COUNT(*) FROM low_items').fetchone()[0]
        db.execute('DELETE FROM events WHERE item_id IN (SELECT item_id FROM low_items)')
        db.commit()
        db.execute('DROP TABLE low_items')
        count = db.execute('SELECT COUNT(*) FROM events').fetchone()[0]
        print(f'  5-core iteration {iteration}: low_users={low_u:,}, low_items={low_i:,}, remaining={count:,}', flush=True)
        if low_u == 0 and low_i == 0:
            break
    return iteration


def write_variant(db, outdir, variant):
    outdir.mkdir(parents=True, exist_ok=True)
    # Map IDs based on deterministic lexical ordering; identical across train/valid/test of a variant.
    map_path = outdir / 'item_mapping.tsv'
    with map_path.open('w', encoding='utf-8') as m:
        m.write('item_id\tparent_asin\n')
        for idx, (item,) in enumerate(db.execute('SELECT DISTINCT item_id FROM events ORDER BY item_id')):
            m.write(f'{idx}\t{item}\n')

    files = {k: (outdir / f'{k}.jsonl').open('w', encoding='utf-8') for k in ('full', 'train', 'valid', 'test')}
    counts = {'full': 0, 'train': 0, 'valid': 0, 'test': 0, 'eligible_users': 0, 'excluded_short_users': 0}
    def emit(k, uid, item, ts):
        files[k].write(json.dumps({'user_id': uid, 'parent_asin': item, 'timestamp': ts}, ensure_ascii=False) + '\n')
        counts[k] += 1

    def flush_user(uid, events):
        if uid is None:
            return
        for item,ts in events:
            emit('full', uid, item, ts)
        if len(events) < 3:
            counts['excluded_short_users'] += 1
            return
        counts['eligible_users'] += 1
        for item,ts in events[:-2]:
            emit('train', uid, item, ts)
        emit('valid', uid, *events[-2])
        emit('test', uid, *events[-1])

    # Sorted index supports deterministic chronological user sequences.
    db.execute('CREATE INDEX IF NOT EXISTS idx_events_user_time ON events(user_id, ts, event_id)')
    db.commit()
    current = None
    sequence = []
    for uid, item, ts in db.execute('SELECT user_id,item_id,ts FROM events ORDER BY user_id,ts,event_id'):
        if uid != current:
            flush_user(current, sequence)
            current, sequence = uid, []
        sequence.append((item,ts))
    flush_user(current, sequence)
    for f in files.values():
        f.close()
    result = {'variant': variant, 'statistics': stats(db), 'output_counts': counts,
              'split_policy': 'chronological leave-last-two-out per user; users with <3 interactions excluded from splits; full retains all',
              'five_core_policy': 'iteratively filter users and items with <5 interactions; performed before split (transductive dataset filtering)',
              'duplicates': 'all review events retained; equal timestamps ordered by original file order',
              'graph_policy': 'construct behavior graph from train.jsonl only; no validation/test edges'}
    (outdir / 'manifest.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(f'  {variant}: {json.dumps(result["statistics"])}; split={counts}', flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True, choices=CATEGORIES)
    p.add_argument('--raw-dir', type=Path, default=Path.home() / 'datasets/amazon2023/raw')
    p.add_argument('--output-dir', type=Path, default=Path.home() / 'datasets/amazon2023/processed')
    p.add_argument('--work-dir', type=Path, default=None)
    args = p.parse_args()
    rev = args.raw_dir / f'{args.dataset}.jsonl.gz'
    meta = args.raw_dir / f'meta_{args.dataset}.jsonl.gz'
    for f in (rev, meta):
        if not f.is_file():
            p.error(f'Missing input file: {f}')

    args.work_dir.mkdir(parents=True, exist_ok=True)
    # Temporary SQL database stored on disk. Never modify original downloads.
    with tempfile.TemporaryDirectory(prefix=f'amazon23_{args.dataset}_', dir=args.work_dir) as tmp:
        db = open_db(Path(tmp) / 'working.sqlite')
        try:
            print('Ingesting raw reviews...', flush=True)
            ingest(db, rev)
            print('Restricting to items with metadata...', flush=True)
            save_metadata_ids(db, meta)
            print('Exporting full and chronological splits...', flush=True)
            write_variant(db, args.output_dir / 'full' / args.dataset, 'full')
            print('Iterative 5-core filtering...', flush=True)
            five_core(db)
            print('Exporting 5-core and chronological splits...', flush=True)
            write_variant(db, args.output_dir / '5core' / args.dataset, '5core')
        finally:
            db.close()
    print('SUCCESS', flush=True)


if __name__ == '__main__':
    main()
