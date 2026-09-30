"""Read-only diagnostics of archived Toys results; no encoder or training runs.

Step 1 optionally needs original embeddings. Step 2 needs only edge_evidence.csv.
Outputs are written to a separate directory; original graph files are untouched.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def describe(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if not len(x):
        return {"n": 0}
    q = np.quantile(x, [0, .1, .25, .5, .75, .9, 1])
    return dict(n=len(x), mean=float(x.mean()), std=float(x.std()),
                **dict(zip(["min", "p10", "p25", "median", "p75", "p90", "max"], map(float, q))))


def correlation(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 2 or np.std(a[ok]) == 0 or np.std(b[ok]) == 0:
        return None
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def write_csv(path, rows):
    rows = iter(rows)
    first = next(rows, None)
    if first is None:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(first))
        writer.writeheader()
        writer.writerow(first)
        writer.writerows(rows)


def save_plot(fig, out, name):
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(out / f"{name}.{suffix}", dpi=170)
    plt.close(fig)


def transform(x, mode):
    if mode == "clip01":
        return np.clip(x, 0, 1)
    if mode == "shift01":
        return np.clip((x + 1) / 2, 0, 1)
    raise ValueError(f"Unsupported transform: {mode}")


def scores(b, text, image, cfg):
    """Match the archived implementation, including missing-modality fallbacks."""
    t, i = transform(text, cfg["cosine_transform"]), transform(image, cfg["cosine_transform"])
    both = np.isfinite(t) & np.isfinite(i)
    strength = (t + i) / 2
    consistency = 1 - np.abs(t - i)
    mean_rel = np.full(len(b), np.nan)
    penalized_rel = mean_rel.copy()
    mean_rel[both] = strength[both]
    penalized_rel[both] = strength[both] * consistency[both]
    one = np.isfinite(t) ^ np.isfinite(i)
    policy = cfg["missing_modality"]
    if policy == "single":
        mean_rel[one] = np.where(np.isfinite(t), t, i)[one]
        penalized_rel[one] = mean_rel[one]
    elif policy == "zero":
        mean_rel[one] = penalized_rel[one] = 0
    elif policy != "behavior_only":
        raise ValueError(f"Unsupported missing modality policy: {policy}")
    lam = cfg["lambda_multimodal"]
    if not 0 <= lam <= 1:
        raise ValueError("lambda must be in [0,1]")
    combine = lambda m: np.where(np.isfinite(m), (1-lam)*b + lam*m, b)
    return combine(mean_rel), combine(penalized_rel), strength, consistency


def compare_ranks(edges, reference, candidate, top_k=10):
    adjacency = defaultdict(list)
    for k, (u, v) in enumerate(edges):
        adjacency[int(u)].append((int(v), k))
        adjacency[int(v)].append((int(u), k))
    counts = defaultdict(int)
    per_node = []
    for node, entries in sorted(adjacency.items()):
        ids = np.array([e[0] for e in entries])
        ix = np.array([e[1] for e in entries])
        a, b = reference[ix], candidate[ix]
        old = np.lexsort((ids, -a))
        new = np.lexsort((ids, -b))
        k = min(top_k, len(ids))
        overlap = len(set(ids[old[:k]]) & set(ids[new[:k]])) / k
        tied_top = np.abs(a - a.max()) <= 1e-12
        top_changed = not bool(tied_top[new[0]])
        x, y = np.triu_indices(len(ids), 1)
        da, db = a[x] - a[y], b[x] - b[y]
        strict = np.abs(da) > 1e-12
        inversions = int(np.sum(strict & (np.abs(db) > 1e-12) & (da*db < 0)))
        if len(ids) >= 2:
            counts["nodes_degree_ge2"] += 1
            counts["top_outside_reference_top_tie"] += int(top_changed)
            counts["nodes_with_strict_inversion"] += int(inversions > 0)
            if tied_top.sum() == 1:
                counts["unique_reference_top_nodes"] += 1
                counts["unique_reference_top_changed"] += int(top_changed)
        counts["strict_neighbor_pairs"] += int(strict.sum())
        counts["reference_tied_neighbor_pairs"] += int((~strict).sum())
        counts["strict_inversions"] += inversions
        per_node.append(dict(item_id=node, degree=len(ids), reference_top=int(ids[old[0]]),
                             candidate_top=int(ids[new[0]]), top_outside_reference_top_tie=top_changed,
                             strict_inversions=inversions, top_k_retention=overlap))
    result = dict(counts)
    for numerator, denominator in [("unique_reference_top_changed", "unique_reference_top_nodes"),
                                   ("nodes_with_strict_inversion", "nodes_degree_ge2"),
                                   ("strict_inversions", "strict_neighbor_pairs")]:
        result[numerator + "_rate"] = counts[numerator] / counts[denominator] if counts[denominator] else None
    eligible = [r["top_k_retention"] for r in per_node if r["degree"] > top_k]
    result["top_k"] = top_k
    result["top_k_retention_degree_gt_k"] = describe(eligible)
    result["tie_policy"] = "Equal within 1e-12 excluded from strict inversions; displayed top-k uses numeric ID tie-break."
    return result, per_node, adjacency


def sample_pairs(ids, count, seed):
    """Uniform distinct unordered pairs; behavior edges are NOT excluded."""
    ids = np.asarray(ids, dtype=np.int64)
    total = len(ids) * (len(ids)-1) // 2
    if count < 1 or total < 1:
        raise ValueError("Need a positive sample size and at least two jointly valid items")
    count = min(count, total)
    rng = np.random.default_rng(seed)
    if count > total // 2 and total <= 2_000_000:
        u, v = np.triu_indices(len(ids), 1)
        ix = rng.choice(total, count, replace=False)
        return np.sort(np.column_stack((ids[u[ix]], ids[v[ix]])), axis=1)
    chosen = set()
    while len(chosen) < count:
        batch = rng.choice(ids, (min(200_000, 2*(count-len(chosen))+100), 2))
        for u, v in batch:
            if u != v:
                chosen.add((min(int(u), int(v)), max(int(u), int(v))))
            if len(chosen) == count:
                break
    return np.array(sorted(chosen), dtype=np.int64)


def load_features(args, n):
    masks = [np.ones(n, dtype=bool), np.ones(n, dtype=bool)]
    if args.embedding_npz:
        with np.load(args.embedding_npz, allow_pickle=False) as data:
            rows = np.arange(n)
            if "item_ids" in data:
                ids = [str(x.decode() if isinstance(x, bytes) else x) for x in data["item_ids"]]
                if len(set(ids)) != len(ids):
                    raise ValueError("Duplicate embedding item IDs")
                index = {x: k for k, x in enumerate(ids)}
                missing = [str(k) for k in range(n) if str(k) not in index]
                if missing:
                    raise ValueError(f"Embedding IDs missing: {missing[:5]}")
                rows = np.array([index[str(k)] for k in range(n)])
            features = [np.asarray(data[key][rows], dtype=np.float32)
                        for key in ("text_embeddings", "image_embeddings")]
            for k, name in enumerate(("text_valid_mask", "image_valid_mask")):
                if name in data:
                    masks[k] = np.asarray(data[name][rows], dtype=bool)
    else:
        features = [np.load(p, mmap_mode="r", allow_pickle=False)
                    for p in (args.text_npy, args.image_npy)]
    normalized, valid = [], []
    for matrix, mask in zip(features, masks):
        if matrix.ndim != 2 or matrix.shape[0] != n or mask.shape != (n,):
            raise ValueError(f"Expected exactly {n} rows (ID alignment required), got {matrix.shape}")
        z = np.zeros(matrix.shape, dtype=np.float32)
        good = mask.copy()
        for start in range(0, n, 2048):
            x = np.asarray(matrix[start:start+2048], dtype=np.float32)
            norms = np.linalg.norm(x, axis=1)
            ok = np.all(np.isfinite(x), axis=1) & np.isfinite(norms) & (norms > 0)
            good[start:start+len(x)] &= ok
            np.divide(x, norms[:, None], out=z[start:start+len(x)], where=ok[:, None])
        normalized.append(z)
        valid.append(good)
    return normalized, valid


def cosine_pairs(z, pairs):
    out = np.empty(len(pairs), dtype=float)
    for start in range(0, len(pairs), 2048):
        part = pairs[start:start+2048]
        out[start:start+len(part)] = np.einsum("ij,ij->i", z[part[:, 0]], z[part[:, 1]])
    return out


def distribution_comparison(behavior, random):
    ordered = np.sort(random)
    lo = np.searchsorted(ordered, behavior, side="left")
    hi = np.searchsorted(ordered, behavior, side="right")
    return {"behavior": describe(behavior), "random": describe(random),
            "prob_behavior_gt_random_plus_half_ties": float(np.mean((lo+hi)/2/len(random)))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("outputs/Toys_multimodal_mutual"))
    parser.add_argument("--output", type=Path, default=Path("outputs/Toys_relation_diagnostics"))
    parser.add_argument("--embedding-npz", type=Path)
    parser.add_argument("--text-npy", type=Path)
    parser.add_argument("--image-npy", type=Path)
    parser.add_argument("--random-pairs", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=20261001)
    args = parser.parse_args()
    if args.output.resolve() == args.results.resolve():
        parser.error("Output must differ from the original results directory")
    if args.embedding_npz and (args.text_npy or args.image_npy):
        parser.error("Choose NPZ or two NPY files, not both")
    if bool(args.text_npy) != bool(args.image_npy):
        parser.error("Both --text-npy and --image-npy are required")
    cfg = json.loads((args.results / "run_config.json").read_text(encoding="utf-8"))
    if cfg["reliability_mode"] != "strength_x_consistency":
        raise ValueError("Archive must be a strength_x_consistency run for this controlled ablation")
    evidence_path = args.results / "edge_evidence.csv"
    with evidence_path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError("Empty evidence file")
    edges = np.array([(int(r["item_i"]), int(r["item_j"])) for r in rows])
    canonical = {tuple(sorted(map(int, e))) for e in edges}
    if len(canonical) != len(edges) or np.any(edges[:, 0] == edges[:, 1]):
        raise ValueError("Duplicate or self edges in evidence")
    col = lambda key: np.array([float(r[key]) if r.get(key) else np.nan for r in rows])
    b, t, i, archived = map(col, ["normalized_behavior_score", "text_similarity", "image_similarity", "final_score"])
    if not np.all(np.isfinite(b)) or not np.all(np.isfinite(archived)):
        raise ValueError("Nonfinite archived weights")
    mean, penalized, strength, consistency = scores(b, t, i, cfg)
    error = float(np.max(np.abs(penalized - archived)))
    if error > 1e-7:
        raise ValueError(f"Recomputed weights disagree with archive: {error}")
    if np.any(mean < penalized - 1e-12):
        raise AssertionError("Removing consistency should never lower a score")
    graph_stats = json.loads((args.results / "graph_analysis.json").read_text(encoding="utf-8"))
    n = graph_stats["graphs"]["collaborative"]["nodes"]
    if edges.min() < 0 or edges.max() >= n or cfg["behavior_edges"] != len(edges):
        raise ValueError("Archived graph metadata and evidence disagree")
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    report = {"settings": cfg, "items": n, "edges": len(edges),
              "evidence_sha256": hashlib.sha256(evidence_path.read_bytes()).hexdigest(),
              "source_results": str(args.results.resolve()), "archive_max_error": error,
              "behavior_edge_distributions": {"text": describe(t), "image": describe(i),
                                               "pearson": correlation(t, i)},
              "step1": {"status": "blocked_missing_embeddings"},
              "step2": {"status": "complete", "weights": {"behavior": describe(b), "mean": describe(mean),
                                                            "consistency": describe(penalized)},
                        "consistency_penalty": describe(mean-penalized),
                        "mean_above_behavior": int(np.sum(mean > b+1e-12)),
                        "consistency_above_behavior": int(np.sum(penalized > b+1e-12)),
                        "rank_comparisons": {}}}
    weights = {"behavior": b, "mean": mean, "consistency": penalized}
    print(f"Validated {len(edges):,} archived edges; computing controlled ablation...", flush=True)
    for a, c in [("behavior", "mean"), ("behavior", "consistency"), ("mean", "consistency")]:
        name = f"{a}_vs_{c}"
        result, node_rows, adjacency = compare_ranks(edges, weights[a], weights[c])
        report["step2"]["rank_comparisons"][name] = result
        write_csv(out / f"ranks_{name}.csv", node_rows)
    edge_rows = [dict(item_i=int(u), item_j=int(v), text_similarity=float(t[k]), image_similarity=float(i[k]),
                      normalized_behavior=float(b[k]), mean_weight=float(mean[k]),
                      consistency_weight=float(penalized[k]), penalty=float(mean[k]-penalized[k]))
                 for k, (u, v) in enumerate(edges)]
    write_csv(out / "ablation_edges.csv", edge_rows)
    write_csv(out / "largest_consistency_penalties.csv", sorted(edge_rows, key=lambda r: -r["penalty"])[:100])
    cases_path = args.results / "case_study.json"
    cases = json.loads(cases_path.read_text(encoding="utf-8")) if cases_path.exists() else {}
    titles = {}
    for item, case in cases.items():
        titles[int(item)] = case.get("title", "")
        for name in ("behavior_neighbors", "multimodal_neighbors"):
            for neighbor in case.get(name, []):
                titles[int(neighbor["item_id"])] = neighbor.get("title", "")
    case_rows = []
    for item in cases:
        entries = adjacency.get(int(item), [])
        ranks = {name: {v: rank for rank, (v, k) in enumerate(sorted(entries, key=lambda e: (-w[e[1]], e[0])), 1)}
                 for name, w in weights.items()}
        for v, k in entries:
            case_rows.append(dict(item_id=int(item), title=titles.get(int(item), ""), neighbor=v,
                                  neighbor_title=titles.get(v, ""),
                                  **{name+"_rank": ranks[name][v] for name in weights}, **edge_rows[k]))
    write_csv(out / "case_rank_changes.csv", case_rows)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name, w in weights.items():
        axes[0].hist(w, bins=np.linspace(0, max(1, float(b.max())), 81), histtype="step", density=True, label=name)
    axes[0].set(xlabel="Weight", ylabel="Density", title="Same edges and lambda; reliability rule changes")
    axes[0].legend()
    axes[1].hist(mean-penalized, bins=60, color="#7656a6")
    axes[1].set(xlabel="W_mean - W_consistency", ylabel="Edges", title="Additional reduction from consistency")
    save_plot(fig, out, "consistency_ablation")
    random_scores = None
    if args.embedding_npz or args.text_npy:
        print("Loading embeddings and checking archived cosines...", flush=True)
        features, masks = load_features(args, n)
        jointly_valid = masks[0] & masks[1]
        valid_edges = jointly_valid[edges[:, 0]] & jointly_valid[edges[:, 1]]
        if not valid_edges.any():
            raise ValueError("No behavior edges with two valid modalities")
        errors = {}
        for name, z, mask, saved in zip(("text", "image"), features, masks, (t, i)):
            valid = mask[edges[:, 0]] & mask[edges[:, 1]] & np.isfinite(saved)
            if not valid.any():
                raise ValueError(f"No comparable archived {name} scores")
            err = float(np.max(np.abs(cosine_pairs(z, edges[valid]) - saved[valid])))
            errors[name] = err
            if err > 1e-4:
                raise ValueError(f"{name} embeddings do not reproduce archive ({err}); check item IDs/model")
        pairs = sample_pairs(np.flatnonzero(jointly_valid), args.random_pairs, args.seed)
        random_scores = [cosine_pairs(z, pairs) for z in features]
        membership = [tuple(sorted(map(int, e))) in canonical for e in pairs]
        step1 = {"status": "complete", "seed": args.seed, "sampling": "Uniform unique unordered pairs from jointly valid items; behavior edges included, not treated as known negatives",
                 "jointly_valid_items": int(jointly_valid.sum()), "behavior_edges_jointly_valid": int(valid_edges.sum()),
                 "excluded_behavior_edges": int((~valid_edges).sum()), "sampled_pairs": len(pairs),
                 "sampled_behavior_edges": sum(membership), "embedding_archive_max_errors": errors,
                 "features": {name: {"shape": list(z.shape), "valid_items": int(mask.sum())}
                              for name, z, mask in zip(("text", "image"), features, masks)},
                 "inputs": [str(p.resolve()) for p in (args.embedding_npz, args.text_npy, args.image_npy) if p],
                 "text": distribution_comparison(t[valid_edges], random_scores[0]),
                 "image": distribution_comparison(i[valid_edges], random_scores[1]),
                 "random_text_image_pearson": correlation(*random_scores)}
        report["step1"] = step1
        write_csv(out / "random_pair_scores.csv", (dict(item_i=int(u), item_j=int(v), text_similarity=float(random_scores[0][k]),
                                                        image_similarity=float(random_scores[1][k]), is_behavior_edge=membership[k])
                                                    for k, (u, v) in enumerate(pairs)))
        behavior_plot = [t[valid_edges], i[valid_edges]]
    else:
        behavior_plot = [t[np.isfinite(t)], i[np.isfinite(i)]]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for k, name in enumerate(("Text", "Image")):
        series = [("Behavior edges", behavior_plot[k])]
        if random_scores is not None:
            series.append(("Uniform random pairs", random_scores[k]))
        for label, values in series:
            axes[0, k].hist(values, bins=np.linspace(-1, 1, 101), density=True, histtype="step", label=label)
            ordered = np.sort(values)
            axes[1, k].plot(ordered, np.arange(1, len(ordered)+1)/len(ordered), label=label)
        axes[0, k].set(title=f"{name}: raw cosine", xlabel="Cosine", ylabel="Density")
        axes[1, k].set(xlabel="Cosine", ylabel="Empirical CDF", ylim=(0, 1))
        axes[0, k].legend()
    fig.suptitle("Behavior vs random pairs" if random_scores is not None else "Behavior edges only — random baseline pending", y=1.02)
    save_plot(fig, out, "similarity_distributions")
    (out / "diagnostics.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    penalty = report["step2"]["consistency_penalty"]
    lines = ["# Toys 관계 진단", "", f"분석 간선: {len(edges):,}, 상품: {n:,}. 기존 파일은 변경하지 않았습니다.", "",
             "## 1. 행동 간선과 무작위 상품 쌍", ""]
    if random_scores is None:
        lines += ["**원본 임베딩 부재로 무작위 비교 미실행.** 행동 간선 분포만 산출했습니다.",
                  "`--text-npy`와 `--image-npy` 또는 `--embedding-npz`를 지정하면 전체 비교를 실행합니다."]
    else:
        lines += [f"공통 유효 상품 {report['step1']['jointly_valid_items']:,}개에서 서로 다른 무방향 상품 쌍 {len(pairs):,}개를 균등 추출했습니다.",
                  "무작위 쌍은 알려진 비관련/음성 정답이 아닙니다. 결과는 선택된 행동 간선과의 기술적 비교이며 유의성 검정이 아닙니다.", "",
                  "| 모달리티 | 행동 평균 | 무작위 평균 | P(행동 > 무작위) + 0.5 P(동점) |", "|---|---:|---:|---:|"]
        for name in ("text", "image"):
            d = report["step1"][name]
            lines += [f"| {name} | {d['behavior']['mean']:.4f} | {d['random']['mean']:.4f} | {d['prob_behavior_gt_random_plus_half_ties']:.4f} |"]
    lines += ["", "## 2. Consistency 유무 비교", "", "간선, 정규화 행동 점수 B, 임베딩 점수, λ는 고정했습니다.",
              "- mean: W = (1-λ)B + λS", "- consistency: W = (1-λ)B + λS(1-|t-i|)",
              f"- 저장 점수 재현 최대 오차: {error:.3g}",
              f"- consistency 추가 감소량: 평균 {penalty['mean']:.6f}, 중앙값 {penalty['median']:.6f}, 최대 {penalty['max']:.6f}", "",
              "| 비교 (기준 → 후보) | 단독 1위 변경률 | 엄격한 이웃 쌍 순위 역전율 |", "|---|---:|---:|"]
    for name, r in report["step2"]["rank_comparisons"].items():
        fmt = lambda x: f"{100*x:.2f}%" if x is not None else "N/A"
        lines += [f"| {name} | {fmt(r['unique_reference_top_changed_rate'])} | {fmt(r['strict_inversions_rate'])} |"]
    lines += ["", "| 가중치 | 평균 |", "|---|---:|"]
    for name, values in weights.items():
        lines += [f"| {name} | {float(values.mean()):.6f} |"]
    lines += ["", "단독 1위 변경률은 이웃이 2개 이상이고 기준 방식의 1위가 동점이 아닌 상품만 분모로 사용합니다.",
              "Behavior 기준 두 비교는 같은 분모이며, mean 기준 비교는 분모가 다릅니다. 정확한 개수는 diagnostics.json에 있습니다.",
              "현재 consistency 항은 mean보다 각 간선 점수를 낮추거나 유지합니다. 중요한 것은 절대 감소뿐 아니라 이웃 간 차등 감소에 따른 순위 변화입니다.",
              "", "### 기존 사례의 변화", "", "| 기준 상품 ID | 이웃 | Behavior 순위 | Mean 순위 | Consistency 순위 |", "|---|---|---:|---:|---:|"]
    for row in case_rows:
        if row["neighbor_title"] and min(row["mean_rank"], row["consistency_rank"]) <= 3:
            title = row["neighbor_title"].replace("|", "/")
            lines += [f"| {row['item_id']} | {title} | {row['behavior_rank']} | {row['mean_rank']} | {row['consistency_rank']} |"]
    lines += ["", "동점 이웃은 엄격한 역전 계산에서 제외합니다. Top-10 및 사례 표시에는 상품 ID로 동점을 해소하므로 해당 결과는 동점 처리에 영향을 받습니다.",
              "두 방식의 점수 차이는 penalty의 효과를 분리하지만, 추천/quantization 품질의 우열은 평가하지 않습니다.", "",
              "## 산출물", "", "- similarity_distributions.png / .pdf: 원시 유사도 히스토그램·누적분포",
              "- consistency_ablation.png / .pdf: 가중치 분포와 추가 penalty",
              "- diagnostics.json: 설정·검증·전체 수치", "- ablation_edges.csv: 간선별 세 점수",
              "- ranks_*.csv: 상품별 순위 변화", "- case_rank_changes.csv: 기존 사례의 전체 이웃 순위",
              "- largest_consistency_penalties.csv: 추가 감소량 상위 100개"]
    (out / "REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(json.dumps({"output": str(out.resolve()), "step1": report["step1"]["status"], "step2": "complete"}))


if __name__ == "__main__":
    main()
