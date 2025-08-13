import json
import os
import time
import csv
from typing import List, Dict
from core import retrieve
from config import RETRIEVAL_MODE


def load_questions(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_retrieval(dataset_path: str, k: int = 4, mode: str = RETRIEVAL_MODE, use_reranker: bool = False, use_mmr: bool = False) -> Dict:
    data = load_questions(dataset_path)
    total = len(data)
    hits_at_k = 0
    mrr = 0.0
    latencies = []

    for item in data:
        q = item["question"]
        refs = set(item.get("references", []))  # 期望的文件名或 (file:page) 标识
        start = time.time()
        docs = retrieve(q, k=k, mode=mode, use_reranker=use_reranker, use_mmr=use_mmr)
        latencies.append(time.time() - start)

        found = False
        rank_pos = None
        for idx, d in enumerate(docs, start=1):
            src = os.path.basename(d.metadata.get("source") or d.metadata.get("file_path") or "unknown")
            page = d.metadata.get("page")
            tags = {src}
            if page is not None:
                tags.add(f"{src}:{page}")
            if refs & tags:
                found = True
                rank_pos = idx
                break

        if found:
            hits_at_k += 1
            mrr += 1.0 / rank_pos

    return {
        "n": total,
        "hit_rate@k": hits_at_k / total if total else 0.0,
        "mrr": mrr / total if total else 0.0,
        "avg_latency_s": sum(latencies) / len(latencies) if latencies else 0.0,
    }


def run_grid(dataset_path: str, k: int, out_csv: str) -> Dict:
    modes = ["dense", "bm25", "hybrid"]
    reranks = [False, True]
    mmrs = [False, True]

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    rows = []

    # baseline for对比
    baseline = evaluate_retrieval(dataset_path, k=k, mode="dense", use_reranker=False, use_mmr=False)

    for m in modes:
        for r in reranks:
            for mm in mmrs:
                metrics = evaluate_retrieval(dataset_path, k=k, mode=m, use_reranker=r, use_mmr=mm)
                row = {
                    "mode": m,
                    "rerank": int(r),
                    "mmr": int(mm),
                    "hit_rate@k": round(metrics["hit_rate@k"], 4),
                    "mrr": round(metrics["mrr"], 4),
                    "avg_latency_s": round(metrics["avg_latency_s"], 4),
                    "delta_hit": round(metrics["hit_rate@k"] - baseline["hit_rate@k"], 4),
                    "delta_mrr": round(metrics["mrr"] - baseline["mrr"], 4),
                    "delta_latency": round(metrics["avg_latency_s"] - baseline["avg_latency_s"], 4),
                }
                rows.append(row)

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # 选择最优（优先 hit，其次 mrr，再看延迟低）
    best = max(rows, key=lambda r: (r["hit_rate@k"], r["mrr"], -r["avg_latency_s"]))
    return {"baseline": baseline, "best": best, "rows": rows, "csv": out_csv}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="JSON 文件，包含 [{question, references}] 数组")
    parser.add_argument("-k", type=int, default=4)
    parser.add_argument("--mode", default=RETRIEVAL_MODE, choices=["dense", "bm25", "hybrid"])
    parser.add_argument("--rerank", action="store_true")
    parser.add_argument("--mmr", action="store_true")
    parser.add_argument("--grid", action="store_true", help="运行多配置对比并导出 CSV")
    parser.add_argument("--out", default="reports/metrics.csv", help="CSV 输出路径")
    args = parser.parse_args()

    if args.grid:
        summary = run_grid(args.dataset, k=args.k, out_csv=args.out)
        print(json.dumps({
            "csv": summary["csv"],
            "baseline": summary["baseline"],
            "best": summary["best"],
        }, ensure_ascii=False, indent=2))
    else:
        base = evaluate_retrieval(args.dataset, k=args.k, mode="dense", use_reranker=False, use_mmr=False)
        strong = evaluate_retrieval(args.dataset, k=args.k, mode=args.mode, use_reranker=args.rerank, use_mmr=args.mmr)
        print(json.dumps({
            "baseline_dense": base,
            "tuned": strong,
        }, ensure_ascii=False, indent=2))


