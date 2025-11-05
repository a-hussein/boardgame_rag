from fastapi import FastAPI
from pydantic import BaseModel

# --- import your function (replace with any of your real ones) ---
from boardgame_rag.retriever import HybridRetriever

app = FastAPI()
retriever = HybridRetriever("indices")  # Initialize once at startup

# --- define request schema ---
class Query(BaseModel):
    q: str
    k: int = 5

@app.get("/") # otherwise will have 404 error
def root():
    return {"status": "ok", "try": ["/docs", "/search (POST)"]}

# --- define endpoint ---
@app.post("/search")
def search(query: Query):
    """
    Simple endpoint that calls retriever.search() and returns its results.
    """
    results = retriever.search(query.q, k=query.k)
    return {"query": query.q, "results": results}

# uv run uvicorn boardgame_rag.api:app --reload --port 8000
    # can then run in swagger using /docs
    # can also run using this command:
        # curl -X POST http://127.0.0.1:8000/search \
        #      -H "Content-Type: application/json" \
        #      -d '{"q":"deck building","k":3}'
    
