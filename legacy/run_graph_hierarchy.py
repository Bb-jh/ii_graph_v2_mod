"""Connect an existing weighted I-I graph to the supplied hierarchical pseudo code.

No graph reconstruction or text/image encoder runs. Source files stay unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from old_hierarchy.pseudo_bridge import dump_json, export_training_labels, prepare_graph, read_graph

ROOT = Path(__file__).resolve().parent
PSEUDO_SRC = ROOT / "pseudo" / "hierarchical-pseudo-label" / "src"


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_stage(script, arguments, output, env):
    command = [sys.executable, "-u", str(script), *map(str, arguments)]
    print(f"Running {script.name}", flush=True)
    with (output / f"{script.stem}.log").open("w", encoding="utf-8") as log:
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, encoding="utf-8", env=env) as process:
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)
            if process.wait():
                raise RuntimeError(f"{script.name} failed; inspect {log.name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, default=Path("outputs/legacy/Toys_multimodal_mutual/multimodal_behavior_graph.npz"))
    parser.add_argument("--output", type=Path, default=Path("outputs/legacy/Toys_multimodal_hierarchy"))
    parser.add_argument("--item-ids", type=Path, help="Optional row-to-item ID NPY; defaults to numeric row indices")
    parser.add_argument("--metadata", type=Path, help="Optional MACRec Toys.item.json for case study titles")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--reg", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--global-k", type=int, default=32)
    parser.add_argument("--local-k", type=int, default=8)
    parser.add_argument("--fine-k", type=int, default=4)
    parser.add_argument("--min-cluster-size", type=int, default=10)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--cpu", action="store_true", help="Force CPU; otherwise upstream selects CUDA when available")
    parser.add_argument("--prepare-only", action="store_true", help="Only export the legacy graph format and LCC")
    parser.add_argument("--smoke", action="store_true", help="One training epoch; label artifacts explicitly marked as integration smoke output")
    args = parser.parse_args()
    for name in ("epochs", "dim", "batch_size", "global_k", "local_k", "fine_k", "min_cluster_size", "threads"):
        if getattr(args, name) < 1:
            parser.error(f"{name} must be positive")
    if args.layers < 0 or args.lr <= 0 or args.reg < 0:
        parser.error("layers/reg must be nonnegative and lr positive")
    if args.smoke:
        args.epochs = 1
    source, output = args.graph.resolve(), args.output.resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error("Use a new or empty output directory; existing results will not be overwritten")
    if args.metadata and not args.metadata.is_file():
        parser.error("Metadata path does not exist")
    scripts = [PSEUDO_SRC / name for name in ("structural_embedding.py", "recursive_kmeans.py", "evaluate_hierarchy.py")]
    if not args.prepare_only:
        for script in scripts:
            if not script.is_file():
                parser.error(f"Missing supplied pseudo source: {script}")
    graph, ids = read_graph(source, args.item_ids)
    # Validate all inputs before writing outputs as far as possible.
    if args.metadata:
        meta = json.loads(args.metadata.read_text(encoding="utf-8"))
        if not isinstance(meta, dict):
            parser.error("Metadata must be keyed by item ID")
    started = time.monotonic()
    stats = prepare_graph(graph, ids, output)
    manifest = {"status": "prepared", "source_graph": str(source), "source_sha256": file_hash(source),
                "parameters": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "run_kind": "integration_smoke_not_quality_evaluation" if args.smoke else "experiment",
                "graph": stats, "source_modules": {p.name: file_hash(p) for p in scripts if p.exists()}}
    dump_json(output / "hierarchy_run.json", manifest)
    print(json.dumps(stats), flush=True)
    if args.prepare_only:
        print(f"Prepared input format at {output}; no training or labels generated.")
        return
    try:
        manifest["versions"] = {p: version(p) for p in ("numpy", "scipy", "torch", "pandas", "scikit-learn")}
        env = os.environ.copy()
        env.update(PYTHONIOENCODING="utf-8", OMP_NUM_THREADS=str(args.threads), MKL_NUM_THREADS=str(args.threads),
                   OPENBLAS_NUM_THREADS=str(args.threads), PYTHONHASHSEED=str(args.seed))
        if args.cpu:
            env["CUDA_VISIBLE_DEVICES"] = "-1"
        training_args = ["--graph-dir", output]
        for option in ("dim", "layers", "epochs", "batch_size", "lr", "reg", "seed"):
            training_args.extend(["--"+option.replace("_", "-"), getattr(args, option)])
        manifest["status"] = "training"
        dump_json(output / "hierarchy_run.json", manifest)
        run_stage(scripts[0], training_args, output, env)
        manifest["status"] = "clustering"
        dump_json(output / "hierarchy_run.json", manifest)
        clustering_args = ["--graph-dir", output]
        for option in ("global_k", "local_k", "fine_k", "min_cluster_size", "seed"):
            clustering_args.extend(["--"+option.replace("_", "-"), getattr(args, option)])
        run_stage(scripts[1], clustering_args, output, env)
        labels = export_training_labels(output, graph, ids, args.metadata)
        run_stage(scripts[2], ["--graph-dir", output], output, env)
        manifest.update(status="complete", labels=labels, elapsed_seconds=time.monotonic()-started)
        dump_json(output / "hierarchy_run.json", manifest)
        note = "**1 epoch 통합 검증 결과입니다. 라벨 품질/수렴을 검증한 결과가 아닙니다.**" if args.smoke else "그래프 기반 pseudo label 실험 결과이며 의미적 정답 또는 추천 성능 검증이 아닙니다."
        report = ["# I-I Graph → Hierarchical Pseudo Labels", "", note, "",
                  f"입력: `{source}`", f"전체 상품 {stats['nodes']:,}, 간선 {stats['edges']:,}.",
                  f"최대 연결 성분 {labels['valid_items']:,}개 상품에 라벨 생성 ({100*stats['labeled_fraction']:.2f}%).",
                  f"나머지 {labels['unlabeled_items']:,}개는 `valid_mask=False`, 라벨 `-1`.",
                  f"Global/Local/Fine 군집 수: {labels['label_counts']}.", "",
                  "`pseudo_labels.npz`의 labels는 각 계층에서 전역적으로 고유한 군집 ID입니다.",
                  "원본 parent-relative 라벨은 path_labels에 보존합니다. local_parent/fine_parent가 계층 연결을 나타냅니다.",
                  "학습 시 valid_mask로 먼저 필터링하세요. 서로 다른 실행의 군집 ID 숫자는 직접 비교할 수 없습니다.", "",
                  "원본 pseudo의 LightGCN/BPR 및 재귀 K-means를 수정 없이 호출했습니다.",
                  "간선 가중치는 정규화 인접행렬 전파에 반영되지만 BPR loss 자체에는 간선별 가중치가 곱해지지 않습니다.",
                  "min_cluster_size는 분할 K를 정하는 휴리스틱이며 실제 군집의 최소 크기를 보장하지 않습니다.",
                  "hierarchy_evaluation.json은 학습에 사용한 그래프 내부 통계이며 held-out 성능이 아닙니다.", "",
                  "출력: pseudo_labels.npz, all_item_labels.csv, label_schema.json, hierarchical_labels.csv,",
                  "structural_embeddings.npy, cluster_stats.json, hierarchy_evaluation.json, hierarchy_run.json 및 단계별 로그."]
        if args.metadata:
            report += ["hierarchy_cases.csv에는 큰 Fine 군집 10개의 내부 연결 강도 상위 상품 제목을 담았습니다."]
        (output / "REPORT.md").write_text("\n".join(report)+"\n", encoding="utf-8")
        print(f"Complete: {output}\npseudo_labels.npz: {labels}", flush=True)
    except Exception as exc:
        manifest.update(status="failed", error=str(exc), elapsed_seconds=time.monotonic()-started)
        dump_json(output / "hierarchy_run.json", manifest)
        raise


if __name__ == "__main__":
    main()
