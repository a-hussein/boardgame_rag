# bm25 test
import pathlib
import pickle

import numpy as np

p = pathlib.Path("indices")
obj = pickle.loads((p / "bm25.pkl").read_bytes())
bm25 = obj["bm25"]
doc_ids = obj["doc_ids"]

query = "dice trading"
q_toks = query.lower().split()
scores = bm25.get_scores(q_toks)
topk = np.argsort(scores)[-5:][::-1]
print("BM25 top-5 doc_ids:", [doc_ids[i] for i in topk])
print("BM25 top-5 scores:", [float(scores[i]) for i in topk])


# faiss test
# import pickle, numpy as np, faiss
# from pathlib import Path
# from sentence_transformers import SentenceTransformer

# out = Path("indices")
# index = faiss.read_index(str(out/"faiss.index"))
# meta = pickle.loads((out/"faiss_meta.pkl").read_bytes())
# doc_ids = meta["doc_ids"]
# model_name = meta["model"]

# enc = SentenceTransformer(model_name)
# q = "dice trading"
# qv = enc.encode([q], normalize_embeddings=True).astype(np.float32)
# D, I = index.search(qv, k=5)
# print("FAISS top-5 doc_ids:", [doc_ids[i] for i in I[0]])
# print("FAISS sims:", [float(x) for x in D[0]])
