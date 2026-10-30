"""MACRec-format dataset loading and a synthetic dataset for smoke runs.

Expected files in data_dir (item IDs are 0..n-1 and equal embedding rows):
  {dataset}.train.inter / .valid.inter / .test.inter
      TSV with header; columns user_id, item_id_list (space separated history), item_id (target).
      train.inter holds prefix samples; valid/test are leave-one-out samples.
  {dataset}.item.json   (optional) {item_id: {...}} — defines n_items when present
  text / image embedding .npy files, [n_items, d].
  optional {dataset}.text-mask.npy / .image-mask.npy, bool [n_items] (False = missing embedding).
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from scipy import sparse


@dataclass
class Sample:
    user: str
    history: list
    target: int


@dataclass
class Dataset:
    n_items: int
    train: list
    valid: list
    test: list
    text: np.ndarray
    image: np.ndarray
    interactions: sparse.csr_matrix  # users x items, train split only
    train_items: np.ndarray  # bool [n_items]: appears in train split (warm)
    text_mask: np.ndarray  # bool [n_items]: embedding present
    image_mask: np.ndarray


def read_inter(path):
    samples = []
    with open(path, encoding="utf-8") as stream:
        for line in stream:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3 or not parts[2].strip().lstrip("-").isdigit():
                continue  # header or malformed line
            history = [int(token) for token in parts[1].split()]
            samples.append(Sample(parts[0], history, int(parts[2])))
    if not samples:
        raise ValueError(f"No samples parsed from {path}")
    return samples


def load_dataset(cfg) -> Dataset:
    root = Path(cfg.data_dir)
    name = cfg.dataset
    text = np.load(root / cfg.text_emb.format(dataset=name), mmap_mode="r")
    image = np.load(root / cfg.image_emb.format(dataset=name), mmap_mode="r")
    meta = root / f"{name}.item.json"
    n_items = len(json.loads(meta.read_text(encoding="utf-8"))) if meta.exists() else len(text)
    if text.shape[0] != n_items or image.shape[0] != n_items:
        raise ValueError(f"Embedding rows {text.shape[0]}/{image.shape[0]} != n_items {n_items}")
    splits = {split: read_inter(root / f"{name}.{split}.inter") for split in ("train", "valid", "test")}
    for split, samples in splits.items():
        top = max(max([s.target, *s.history]) for s in samples)
        low = min(min([s.target, *s.history]) for s in samples)
        if low < 0 or top >= n_items:
            raise ValueError(f"{split} item IDs outside [0, {n_items})")
    # Train interactions: union of every train prefix (history + target) per user.
    users, items = {}, ([], [])
    for s in splits["train"]:
        u = users.setdefault(s.user, len(users))
        for item in (*s.history, s.target):
            items[0].append(u)
            items[1].append(item)
    a = sparse.csr_matrix((np.ones(len(items[0]), dtype=np.float32), items), shape=(len(users), n_items))
    a.sum_duplicates()
    a.data[:] = 1
    train_items = np.asarray(a.sum(0)).ravel() > 0
    masks = []
    for pattern in (cfg.text_mask, cfg.image_mask):
        path = root / pattern.format(dataset=name)
        mask = np.load(path).astype(bool) if path.exists() else np.ones(n_items, dtype=bool)
        if mask.shape != (n_items,):
            raise ValueError(f"{path} must have shape ({n_items},)")
        masks.append(mask)
    return Dataset(n_items, splits["train"], splits["valid"], splits["test"],
                   np.array(text, dtype=np.float32), np.array(image, dtype=np.float32), a, train_items, *masks)


def make_synthetic_dataset(root, name="Synth", *, n_items=240, n_users=300, seed=0):
    """Hierarchically clustered items; users mostly stay inside one Local cluster.

    A few items never occur in train (cold). For pipeline smoke tests only.
    """
    rng = np.random.default_rng(seed)
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    g, l, f = rng.integers(0, 3, n_items), rng.integers(0, 2, n_items), rng.integers(0, 2, n_items)
    centers = [rng.normal(size=(n, 16)) for n in (3, 6, 12)]
    latent = (centers[0][g] + 0.7*centers[1][g*2+l] + 0.5*centers[2][(g*2+l)*2+f]
              + 0.3*rng.normal(size=(n_items, 16)))
    text = (latent @ rng.normal(size=(16, 48)) + 0.5*rng.normal(size=(n_items, 48))).astype(np.float32)
    image = (latent @ rng.normal(size=(16, 24)) + 0.8*rng.normal(size=(n_items, 24))).astype(np.float32)
    np.save(root / f"{name}.emb-text.npy", text)
    np.save(root / f"{name}.emb-image.npy", image)
    (root / f"{name}.item.json").write_text(json.dumps({str(i): {"title": f"item {i}"} for i in range(n_items)}))
    cold = set(rng.choice(n_items, 6, replace=False).tolist())
    local = g*2 + l
    pools = {c: np.flatnonzero((local == c) & ~np.isin(np.arange(n_items), list(cold))) for c in np.unique(local)}
    header = "user_id:token\titem_id_list:token_seq\titem_id:token\n"
    rows = {"train": [header], "valid": [header], "test": [header]}
    for u in range(n_users):
        pool = pools[rng.choice(list(pools))]
        seq = rng.choice(pool, int(rng.integers(5, 11)), replace=False).tolist()
        seq[int(rng.integers(len(seq)-2))] = int(rng.choice(pools[rng.choice(list(pools))]))  # cross-cluster noise
        if u % 25 == 0:
            seq[-1] = sorted(cold)[u // 25 % len(cold)]  # occasional cold test target
        for t in range(1, len(seq)-2):
            rows["train"].append(f"u{u}\t{' '.join(map(str, seq[:t]))}\t{seq[t]}\n")
        rows["valid"].append(f"u{u}\t{' '.join(map(str, seq[:-2]))}\t{seq[-2]}\n")
        rows["test"].append(f"u{u}\t{' '.join(map(str, seq[:-1]))}\t{seq[-1]}\n")
    for split, lines in rows.items():
        (root / f"{name}.{split}.inter").write_text("".join(lines))
    return dict(data_dir=str(root), dataset=name, text_emb="{dataset}.emb-text.npy", image_emb="{dataset}.emb-image.npy")
