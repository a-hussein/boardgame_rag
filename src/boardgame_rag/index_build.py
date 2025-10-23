from __future__ import annotations
import argparse, pathlib, pickle
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi # lexical ranking (read more here: https://pypi.org/project/rank-bm25/)
from sentence_transformers import SentenceTransformer # embeddings (read more here: https://sbert.net/)
import faiss # ANN (read more here: https://python.langchain.com/docs/integrations/vectorstores/faiss/)

def tokenize(text:str):
    return [t.lower() for t in text.split()]

def build_bm25(texts): # used to t
    tokenized = [tokenize(t) for t in texts]
    return BM25Okapi(tokenized), tokenized

def build_faiss(embs: np.ndarray):
    faiss.normalize_L2(embs) # make vectors unit vectors
    index = faiss.IndexFlatIP(embs.shape[1]) # inner product between unit vecors is cosine similarity
    index.add(embs.astype(np.float32)) 
    return index

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--embedder", default="sentence-transformers/all-MiniLM-L6-v2")
    args = ap.parse_args()

    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(args.in_path)
    doc_ids = df["doc_id"].astype(str).tolist()
    fields = df["text"].astype(str).tolist()

    # BM25
    bm25, tokenized = build_bm25(fields)
    with open(out / "bm25.pkl", "wb") as f:
        pickle.dump({"bm25": bm25, "tokenized": tokenized, "doc_ids": doc_ids}, f)

    # Embeddings
    model = SentenceTransformer(args.embedder)
    embs = model.encode(fields, batch_size=64, show_progress_bar=True, normalize_embeddings=False)
    index = build_faiss(np.asarray(embs))
    faiss.write_index(index, str(out / "faiss.index"))
    with open(out / "faiss_meta.pkl", "wb") as f:
        pickle.dump({"doc_ids": doc_ids, "model": args.embedder}, f)

    print(f"Built indices for {len(doc_ids)} docs -> {out}")

if __name__ == "__main__":
    main()
