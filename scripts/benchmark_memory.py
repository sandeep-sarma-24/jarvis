#!/usr/bin/env python3
"""Memory benchmark: retrieve_context latency and message write latency.

Builds synthetic corpora (no personal data) of 1k, 10k and 50k rows in a
temporary database and reports P50/P95/P99 via app.contracts.percentiles.
Loads no models.

Usage: python -m scripts.benchmark_memory
"""
from __future__ import annotations

import random
import tempfile
import time
from pathlib import Path

from app.contracts import percentiles
from app.memory.manager import MemoryManager
from app.memory.sqlite_store import SQLiteStore, normalize

SIZES = (1_000, 10_000, 50_000)
QUERIES = 300
COLD_SAMPLES = 10
WRITES = 300
SESSION_SIZE = 100   # messages per synthetic session
MEMORY_SHARE = 0.1   # fraction of rows stored as memories

_SUBJECTS = "the plumber|my sister|the landlord|our team|the dentist|the cat|my manager|the bank|the gym|the neighbour".split("|")
_VERBS = "is coming on|moved the meeting to|asked about|forgot|booked|cancelled|needs help with|recommended|paid for|lost".split("|")
_OBJECTS = ("the kitchen tap|a train ticket|the quarterly report|tennis lessons|the wifi router|a birthday cake|"
            "the car insurance|the garden fence|a flight to Lisbon|the printer|piano practice|the electricity bill").split("|")
_TIMES = "on Monday|next week|tomorrow morning|at 4pm|last night|in March|before lunch|this weekend".split("|")
_QUERIES = ("when is the plumber coming|what did my sister ask about|wifi router|car insurance renewal|"
            "flight to Lisbon|who booked tennis lessons|electricity bill paid|what about the printer|"
            "the garden fence|birthday cake for the weekend|quarterly report meeting|piano practice time").split("|")


def _sentence(rng: random.Random) -> str:
    return f"{rng.choice(_SUBJECTS)} {rng.choice(_VERBS)} {rng.choice(_OBJECTS)} {rng.choice(_TIMES)} (ref {rng.randrange(10**6)})"


def _build(db: Path, rows: int, rng: random.Random) -> None:
    store = SQLiteStore(db)
    conn = store._get_conn()
    now = time.time()
    n_mem = int(rows * MEMORY_SHARE)
    n_msg = rows - n_mem
    sessions = [(f"s{i}", now - (n_msg - i * SESSION_SIZE) * 60) for i in range(0, n_msg, SESSION_SIZE)]
    conn.executemany("INSERT INTO conversation_sessions VALUES (?, ?, ?)", [(s, t, t) for s, t in sessions])
    conn.executemany(
        "INSERT INTO conversation_messages (id, session_id, role, content, timestamp) VALUES (?, ?, ?, ?, ?)",
        [(f"m{i}", f"s{i - i % SESSION_SIZE}", "user" if i % 2 == 0 else "assistant",
          _sentence(rng), now - (n_msg - i) * 60) for i in range(n_msg)],
    )
    mems = [_sentence(rng) for _ in range(n_mem)]
    conn.executemany(
        "INSERT INTO memories (id, type, content, created_at, updated_at, importance, metadata_json, norm_content)"
        " VALUES (?, 'fact', ?, ?, ?, ?, '{}', ?)",
        [(f"k{i}", c, now - i * 300, now - i * 300, rng.random(), normalize(c)) for i, c in enumerate(mems)],
    )
    conn.commit()
    store.close()


def _ms(fn) -> float:
    t = time.perf_counter()
    fn()
    return (time.perf_counter() - t) * 1000


def _fmt(p: dict[str, float]) -> str:
    return " ".join(f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}" for k, v in p.items())


def run() -> None:
    rng = random.Random(7)
    with tempfile.TemporaryDirectory() as tmp:
        for rows in SIZES:
            db = Path(tmp) / f"bench_{rows}.db"
            _build(db, rows, rng)

            cold = []
            for _ in range(COLD_SAMPLES):
                manager = MemoryManager(SQLiteStore(db))
                manager.start_session()
                cold.append(_ms(lambda: manager.retrieve_context(rng.choice(_QUERIES), limit=5)))
                manager.close()

            manager = MemoryManager(SQLiteStore(db))
            conv = manager.start_session()
            warm = [_ms(lambda q=q: manager.retrieve_context(q, limit=5))
                    for q in (rng.choice(_QUERIES) for _ in range(QUERIES))]
            writes = [_ms(lambda i=i: conv.add_user_message(_sentence(rng), f"bench_evt_{i}"))
                      for i in range(WRITES)]
            manager.close()

            print(f"\n{rows:>6} rows")
            print(f"  retrieve cold (first query after open, n={COLD_SAMPLES}): {_fmt(percentiles(cold))}")
            print(f"  retrieve warm (ms):  {_fmt(percentiles(warm))}")
            print(f"  write per message (ms): {_fmt(percentiles(writes))}")
    print("\nTarget: retrieve_context under 50 ms at 10k rows.")


if __name__ == "__main__":
    run()
