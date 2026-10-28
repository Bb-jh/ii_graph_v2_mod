from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components


def load_item_meta(path: Path):
    with path.open("r", encoding="utf-8") as f:
        meta = json.load(f)

    ids = sorted(int(k) for k in meta.keys())
    n_items = max(ids) + 1 if ids else 0
    titles = {
        int(k): str(v.get("title", ""))
        for k, v in meta.items()
    }
    return n_items, titles


def load_reconstructed_train(train_path: Path, n_items: int):
    # train.inter는 prefix training sample이므로
    # user별로 등장한 모든 item을 합쳐 원래 train interaction set을 복원
    user_items = defaultdict(set)

    with train_path.open("r", encoding="utf-8") as f:
        next(f, None)  # header skip

        for lineno, line in enumerate(f, start=2):
            parts = line.rstrip("\n").split("\t")

            if len(parts) < 2:
                continue

            user = parts[0].strip()
            if not user:
                continue

            # item_id_list
            tokens = parts[1].strip().split()

            # 마지막 target item_id가 있으면 포함
            if len(parts) >= 3 and parts[2].strip():
                tokens.append(parts[2].strip())

            for tok in tokens:
                try:
                    item = int(tok)
                except ValueError:
                    continue

                if 0 <= item < n_items:
                    user_items[user].add(item)

            if lineno % 200000 == 0:
                print(f"[load] {lineno:,} lines processed")

    rows = []
    cols = []

    for user_idx, items in enumerate(user_items.values()):
        rows.extend([user_idx] * len(items))
        cols.extend(items)

    data = np.ones(len(rows), dtype=np.int32)

    A = sparse.csr_matrix(
        (data, (rows, cols)),
        shape=(len(user_items), n_items),
        dtype=np.int32,
    )

    A.sum_duplicates()
    A.data[:] = 1

    return A, len(user_items)


def build_behavior_graph(
    A: sparse.csr_matrix,
    min_cooc: int,
    top_k: int,
    symmetrize: str,
):
    print("[graph] computing A.T @ A ...")

    C = (A.T @ A).tocsr()

    # self-edge 제거
    C.setdiag(0)
    C.eliminate_zeros()

    # 최소 co-occurrence
    if min_cooc > 1:
        C.data[C.data < min_cooc] = 0
        C.eliminate_zeros()

    print(
        f"[graph] candidate directed edges after "
        f"min_cooc={min_cooc}: {C.nnz:,}"
    )

    # item별 Top-K
    indptr = [0]
    indices = []
    data = []

    for i in range(C.shape[0]):
        start = C.indptr[i]
        end = C.indptr[i + 1]

        row_idx = C.indices[start:end]
        row_val = C.data[start:end]

        if len(row_val) > top_k:
            keep = np.argpartition(
                row_val, -top_k
            )[-top_k:]

            order = keep[
                np.argsort(row_val[keep])[::-1]
            ]

        else:
            order = np.argsort(row_val)[::-1]

        indices.extend(row_idx[order].tolist())
        data.extend(row_val[order].tolist())
        indptr.append(len(indices))

        if (i + 1) % 10000 == 0:
            print(
                f"[top-k] {i + 1:,}/{C.shape[0]:,} items"
            )

    S = sparse.csr_matrix(
        (
            np.asarray(data, dtype=np.int32),
            np.asarray(indices, dtype=np.int32),
            np.asarray(indptr, dtype=np.int64),
        ),
        shape=C.shape,
    )

    # undirected graph
    if symmetrize == "union":
        G = S.maximum(S.T)

    elif symmetrize == "mutual":
        G = S.minimum(S.T)

    else:
        raise ValueError(
            "symmetrize must be union or mutual"
        )

    G.setdiag(0)
    G.eliminate_zeros()

    return G


def graph_statistics(
    G,
    n_users,
    n_interactions,
):
    degrees = np.diff(G.indptr)

    n_nodes = G.shape[0]
    n_edges = sparse.triu(
        G, k=1
    ).nnz

    n_components, labels = connected_components(
        G,
        directed=False,
        return_labels=True,
    )

    component_sizes = (
        np.bincount(
            labels,
            minlength=n_components,
        )
        if n_nodes
        else np.array([], dtype=int)
    )

    edge_weights = sparse.triu(
        G, k=1
    ).data

    return {
        "users": int(n_users),
        "items": int(n_nodes),
        "reconstructed_train_interactions":
            int(n_interactions),

        "nodes": int(n_nodes),
        "edges": int(n_edges),

        "avg_degree":
            float(degrees.mean())
            if n_nodes else 0.0,

        "max_degree":
            int(degrees.max())
            if n_nodes else 0,

        "density":
            float(
                2 * n_edges
                / (n_nodes * (n_nodes - 1))
            )
            if n_nodes > 1 else 0.0,

        "isolated_nodes":
            int(np.sum(degrees == 0)),

        "connected_components":
            int(n_components),

        "largest_component":
            int(component_sizes.max())
            if component_sizes.size
            else 0,

        "weight_mean":
            float(edge_weights.mean())
            if edge_weights.size
            else 0.0,

        "weight_max":
            int(edge_weights.max())
            if edge_weights.size
            else 0,
    }


def top_neighbors(
    G,
    item,
    titles,
    top_n=10,
):
    start = G.indptr[item]
    end = G.indptr[item + 1]

    neighbors = G.indices[start:end]
    weights = G.data[start:end]

    order = np.argsort(
        weights
    )[::-1][:top_n]

    return [
        {
            "item_id": int(neighbors[j]),
            "title": titles.get(
                int(neighbors[j]), ""
            ),
            "cooccurrence":
                int(weights[j]),
        }
        for j in order
    ]


def make_case_study(
    G,
    titles,
    top_n=10,
):
    degrees = np.diff(G.indptr)

    # 일단 degree가 높은 item 3개
    candidates = np.argsort(
        degrees
    )[::-1]

    chosen = [
        int(x)
        for x in candidates
        if degrees[x] > 0
    ][:3]

    result = {}

    for item in chosen:
        result[str(item)] = {
            "title":
                titles.get(item, ""),

            "degree":
                int(degrees[item]),

            "neighbors":
                top_neighbors(
                    G,
                    item,
                    titles,
                    top_n,
                ),
        }

    return result


def export_top_edges(
    G,
    titles,
    path: Path,
    limit=500,
):
    U = sparse.triu(
        G, k=1
    ).tocoo()

    if U.nnz == 0:
        return

    if U.nnz > limit:
        keep = np.argpartition(
            U.data, -limit
        )[-limit:]

        keep = keep[
            np.argsort(
                U.data[keep]
            )[::-1]
        ]

    else:
        keep = np.argsort(
            U.data
        )[::-1]

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "item_i",
            "title_i",
            "item_j",
            "title_j",
            "cooccurrence",
        ])

        for k in keep:
            i = int(U.row[k])
            j = int(U.col[k])
            c = int(U.data[k])

            writer.writerow([
                i,
                titles.get(i, ""),
                j,
                titles.get(j, ""),
                c,
            ])


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--items",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "outputs/legacy/Toys_behavior"
        ),
    )

    parser.add_argument(
        "--min-cooc",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--symmetrize",
        choices=[
            "union",
            "mutual",
        ],
        default="union",
    )

    args = parser.parse_args()

    args.output.mkdir(
        parents=True,
        exist_ok=True,
    )

    n_items, titles = load_item_meta(
        args.items
    )

    print(
        f"[meta] items={n_items:,}"
    )

    A, n_users = load_reconstructed_train(
        args.train,
        n_items,
    )

    print(
        f"[train] users={n_users:,}"
    )

    print(
        "[train] reconstructed "
        f"user-item interactions={A.nnz:,}"
    )

    G = build_behavior_graph(
        A,
        min_cooc=args.min_cooc,
        top_k=args.top_k,
        symmetrize=args.symmetrize,
    )

    stats = graph_statistics(
        G,
        n_users,
        A.nnz,
    )

    cases = make_case_study(
        G,
        titles,
        top_n=10,
    )

    sparse.save_npz(
        args.output
        / "behavior_graph.npz",
        G,
    )

    (
        args.output
        / "graph_analysis.json"
    ).write_text(
        json.dumps(
            stats,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    (
        args.output
        / "case_study.json"
    ).write_text(
        json.dumps(
            cases,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    export_top_edges(
        G,
        titles,
        args.output
        / "top_edges_preview.csv",
    )

    print(
        "\n===== GRAPH STATS ====="
    )

    for key, value in stats.items():
        print(
            f"{key}: {value}"
        )

    print(
        "\n===== CASE STUDY ====="
    )

    for item, info in cases.items():
        print(
            f"\nTarget {item}: "
            f"{info['title']} "
            f"(degree={info['degree']})"
        )

        for neighbor in (
            info["neighbors"][:5]
        ):
            print(
                "  -> "
                f"{neighbor['item_id']} "
                f"| cooc="
                f"{neighbor['cooccurrence']} "
                f"| "
                f"{neighbor['title'][:100]}"
            )

    print(
        f"\nSaved to: {args.output}"
    )


if __name__ == "__main__":
    main()
