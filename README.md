# Project: Boardgame RAG POC (for my learning)

# What it does: Demonstrates hybrid retrieval combining lexical search (BM25) and vector search (FAISS) with an evaluation harness.

# Components:
- data_gen: synthetic corpus -> JSONL/Parquet
- index_build: BM25 + FAISS indices
- retriever: hybrid scoring with alpha-fusion, top-k
- eval_harness: Recall@k, sweep α testing 
- api: FastAPI endpoints for /search 

# Run:
- make data 
    - generate synthetic dataset
- make index
    - build BM25 + FAISS
- make eval
    - run metrics
- make api
    - start FastAPI

# Throughout explortation, notes of an (optional) backlog, referencing here: 
- implement the testing suite
- implement reranker, can experiment with these two model types:
    - cross-encoder
    - bi-encoder 
- explore how FAISS clusterings
- add facet filters after bm25
    - filter candidates before fusion using the metadata
- generate gold from actual doc_ids (programmatically), not hand-typed 
- logging
    - alpha, BM25 top-k, FAISS top-k, fused top-k, applied filters, final chosen ids, timings
    - keep in json to track 
- chunking:
    - for longer texts, split into 100–300 tokens with 20–40 token overlap to improve recall (more chances for a query to match a focused passage)
    - include a chunker.py
- agentic implementation
- guardrails
- error analysis
    - write false negatives to reports/errors.csv
- implement OOP
    - dataclasses 
- More metrics
    - nDCG@k (position-aware graded relevance)
    - Precision@k
    - MAP@k 
    - Coverage of gold answers 
- Alpha sweeps
    - Run for alpha ∈ {0.0, 0.25, 0.5, 0.75, 1.0} and print a small table to see best fusion.


