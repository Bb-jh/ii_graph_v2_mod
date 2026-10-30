#!/usr/bin/env python3
"""Convert a prepare_amazon2023_v2 5-core split + aligned CLIP NPZ into pipeline input.

Input  (--src): train/valid/test.jsonl {user_id, parent_asin, timestamp}, item_mapping.tsv,
                clip_vit_b32_aligned.npz (item_ids in item_mapping order).
Output (--out): {name}.train.inter  prefix samples of each user's chronological train sequence
                {name}.valid.inter  history = train sequence,          target = valid item
                {name}.test.inter   history = train sequence + valid,  target = test item
                {name}.item.json, {name}.emb-text.npy, {name}.emb-image.npy,
                {name}.text-mask.npy, {name}.image-mask.npy (False = embedding missing)

Run from the repository root:
  python -m scripts.data_prep.convert_to_pipeline --src shared_data/Industrial_and_Scientific/5core_v2 \
      --out data/Industrial --name Industrial
"""
import argparse
import json
from pathlib import Path

import numpy as np

HEADER = "user_id:token\titem_id_list:token_seq\titem_id:token\n"


def read_split(path, index):
    events = {}
    with path.open(encoding="utf-8") as stream:
        for order, line in enumerate(stream):
            r = json.loads(line)
            events.setdefault(r["user_id"], []).append((int(r["timestamp"]), order, index[r["parent_asin"]]))
    return {u: [item for _, _, item in sorted(e)] for u, e in events.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--name", default="Industrial")
    parser.add_argument("--embedding", default="clip_vit_b32_aligned.npz")
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        parser.error(f"Refusing to overwrite nonempty {args.out}")
    with (args.src / "item_mapping.tsv").open(encoding="utf-8") as stream:
        next(stream)
        rows = [line.rstrip("\n").split("\t") for line in stream if line.strip()]
    if [int(i) for i, _ in rows] != list(range(len(rows))):
        raise ValueError("item_mapping.tsv indices must be 0..n-1 in order")
    asins = [a for _, a in rows]
    index = {a: i for i, a in enumerate(asins)}
    with np.load(args.src / args.embedding, allow_pickle=False) as npz:
        if npz["item_ids"].tolist() != asins:
            raise ValueError("Embedding item_ids do not match item_mapping order")
        arrays = {k: npz[k] for k in ("text_embeddings", "image_embeddings", "text_valid_mask", "image_valid_mask")}
    for key in ("text_embeddings", "image_embeddings"):
        if not np.isfinite(arrays[key]).all():
            raise ValueError(f"{key} contains non-finite values")
    train, valid, test = (read_split(args.src / f"{s}.jsonl", index) for s in ("train", "valid", "test"))
    if set(valid) != set(test) or not set(valid) <= set(train):
        raise ValueError("Every evaluated user needs train, valid and test events")
    if any(len(v) != 1 for v in (*valid.values(), *test.values())):
        raise ValueError("Expected exactly one valid and one test event per user")
    args.out.mkdir(parents=True, exist_ok=True)
    counts = dict(train=0, valid=0, test=0)
    with (args.out / f"{args.name}.train.inter").open("w") as tr, \
         (args.out / f"{args.name}.valid.inter").open("w") as va, \
         (args.out / f"{args.name}.test.inter").open("w") as te:
        for stream in (tr, va, te):
            stream.write(HEADER)
        for u, seq in train.items():
            for t in range(1, len(seq)):
                tr.write(f"{u}\t{' '.join(map(str, seq[:t]))}\t{seq[t]}\n")
                counts["train"] += 1
            if u in valid:
                va.write(f"{u}\t{' '.join(map(str, seq))}\t{valid[u][0]}\n")
                te.write(f"{u}\t{' '.join(map(str, seq + valid[u]))}\t{test[u][0]}\n")
                counts["valid"] += 1
                counts["test"] += 1
    np.save(args.out / f"{args.name}.emb-text.npy", arrays["text_embeddings"].astype(np.float32))
    np.save(args.out / f"{args.name}.emb-image.npy", arrays["image_embeddings"].astype(np.float32))
    np.save(args.out / f"{args.name}.text-mask.npy", arrays["text_valid_mask"].astype(bool))
    np.save(args.out / f"{args.name}.image-mask.npy", arrays["image_valid_mask"].astype(bool))
    (args.out / f"{args.name}.item.json").write_text(json.dumps({str(i): {"parent_asin": a} for i, a in enumerate(asins)}))
    summary = dict(source=str(args.src), items=len(asins), users=len(train), samples=counts,
                   text_missing=int((~arrays["text_valid_mask"]).sum()),
                   image_missing=int((~arrays["image_valid_mask"]).sum()))
    (args.out / "conversion.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
