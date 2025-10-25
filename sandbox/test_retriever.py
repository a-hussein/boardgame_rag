from boardgame_rag.retriever import HybridRetriever
retr = HybridRetriever("indices", alpha=0.5)
hits = retr.search("dice trading", k=5)
for h in hits:
    print(h)


# equivalent to runnning: uv run sandbox/test_retriever.py