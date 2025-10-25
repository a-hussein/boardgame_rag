"""
runs queries, compares to gold answers and produced metrics
"""


from __future__ import annotations
import argparse, json, pathlib, statistics
from typing import List, Dict
from .retriever import HybridRetriever

def recall_at_k(gold:List[str], pred:List[str], k:int)->float:
    return 1.0 if any(g in pred[:k] for g in gold) else 0.0

def mrr_at_k(gold:List[str], pred:List[str], k:int)->float:
    for r, doc in enumerate(pred[:k], start=1):
        if doc in gold: return 1.0 / r
    return 0.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=False, default="eval/gold.jsonl")
    ap.add_argument("--indices", required=True)
    ap.add_argument("--report", required=False, default="eval/report.md")
    ap.add_argument("--alpha", type=float, default=.5)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--q")
    args = ap.parse_args()

    retr = HybridRetriever(args.indices, args.alpha)

    if args.demo and args.q:
        hits = retr.search(args.q, k=args.k)
        for h in hits: print(h)
        return

    gold_path = pathlib.Path(args.gold)
    items = [json.loads(l) for l in gold_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    recs, mrrs = [], []
    for ex in items:
        hits = retr.search(ex["query"], k=args.k)
        pred_ids = [h["doc_id"] for h in hits]
        recs.append(recall_at_k(ex["gold_doc_ids"], pred_ids, args.k))
        mrrs.append(mrr_at_k(ex["gold_doc_ids"], pred_ids, args.k))

    report = f"# Eval Report\n\nN={len(items)}  \nRecall@{args.k}: {statistics.mean(recs):.3f}  \nMRR@{args.k}: {statistics.mean(mrrs):.3f}\n"
    pathlib.Path(args.report).write_text(report, encoding="utf-8")
    print(report)

if __name__ == "__main__":
    main()
