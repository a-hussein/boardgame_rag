"""
The HybridRetriever allows us to answer:
Which board game docs are most relevant to this query?
...using both lexical and semantic matching together.
"""


from __future__ import annotations
import argparse, pickle, pathlib
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import faiss

def tokenize(text:str):
    return [t.lower() for t in text.split()]

class HybridRetriever:
    def __init__(self, indices_dir: str, alpha: float = 0.5):
        p = pathlib.Path(indices_dir)
        with open(p/"bm25.pkl","rb") as f:
            obj = pickle.load(f)
        self.bm25: BM25Okapi = obj["bm25"]
        self.tokenized = obj["tokenized"]
        self.doc_ids = obj["doc_ids"]

        self.faiss_index = faiss.read_index(str(p/"faiss.index"))
        with open(p/"faiss_meta.pkl","rb") as f:
            meta = pickle.load(f)
        self.embed_model = SentenceTransformer(meta["model"])
        self.alpha = alpha

    def search(self, query: str, k: int = 10, kb: int = 50, kv: int = 50):
        # BM25
        bm_scores = self.bm25.get_scores(tokenize(query))
        top_bm_idx = np.argsort(bm_scores)[::-1][:kb]
        # FAISS
        q = self.embed_model.encode([query])
        faiss.normalize_L2(q)
        sims, idxs = self.faiss_index.search(q.astype(np.float32), kv)
        top_vec_idx = idxs[0]; vec_scores = sims[0]

        # gather candidates
        cand_idx = np.unique(np.concatenate([top_bm_idx, top_vec_idx]))
        bm = bm_scores[cand_idx]
        # lookup vec scores for those cands
        vec_map = {int(i): float(s) for i, s in zip(top_vec_idx, vec_scores)}
        vec = np.array([vec_map.get(int(i), 0.0) for i in cand_idx])

        # z-normalize per channel
        def z(x):
            mu, sd = float(np.mean(x)), float(np.std(x) + 1e-9)
            return (x - mu) / sd
        fused = self.alpha * z(bm) + (1 - self.alpha) * z(vec)

        order = np.argsort(fused)[::-1][:k]
        results = []
        for o in order:
            i = int(cand_idx[o])
            results.append({"doc_id": self.doc_ids[i], "score": float(fused[o]),
                            "bm25": float(bm[cand_idx==i][0]), "vec": float(vec[cand_idx==i][0])})
        return results

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indices", required=True)
    ap.add_argument("--q", required=True)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--alpha", type=float, default=0.5)
    args = ap.parse_args()
    retriever = HybridRetriever(args.indices, alpha=args.alpha)
    hits = retriever.search(args.q, k=args.k)
    for h in hits:
        print(h)

if __name__ == "__main__":
    main()
