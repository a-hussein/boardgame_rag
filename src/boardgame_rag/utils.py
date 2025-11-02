import json
import pathlib
import time


def log_query(trace_dir: str, payload: dict):
    d = pathlib.Path(trace_dir)
    d.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    path = d / f"trace_{ts}.jsonl"
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return str(path)
