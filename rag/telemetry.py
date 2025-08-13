import time
import json
import os
from contextlib import contextmanager


@contextmanager
def span(name: str, record: dict):
    t0 = time.time()
    try:
        yield
    finally:
        record[f"{name}_ms"] = int((time.time() - t0) * 1000)


def write_telemetry(record: dict):
    try:
        os.makedirs("logs", exist_ok=True)
        with open("logs/telemetry.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass

