"""Shared SQLite connection setup for this app's local cache DBs
(iv_history.db, chain_cache.db) — one place for the PRAGMAs and
connection lifecycle every embedded-cache module should use.

Settings and why:

- **journal_mode=WAL.** Default rollback-journal mode lets a writer
  block every reader (and vice versa) for the duration of a
  transaction. This app now has a background scan thread writing to
  these same files while the interactive Streamlit thread reads/writes
  them too (see background_scan.py) — under the default mode that's a
  real contention path, not a theoretical one. WAL lets readers and a
  single writer proceed concurrently.
- **synchronous=NORMAL.** The safe pairing with WAL per SQLite's own
  docs: the database file itself stays consistent even after a crash
  in NORMAL mode: only a very recent commit could be lost, which is
  fine for a locally-regenerable scan-history cache — not fine for,
  say, a ledger, but that's not what these files are.
- **busy_timeout.** Without one, a writer that can't immediately get
  the lock raises `sqlite3.OperationalError: database is locked`
  right away. A timeout makes a second concurrent writer (background
  scan vs. an interactive scan landing at the same moment) wait and
  retry instead of failing outright.
- **temp_store=MEMORY.** Keeps SQLite's own transient B-trees (e.g.
  the GROUP BY in background_scan._target_universe) in RAM instead of
  spilling to a temp file on disk — cheap win at this data size.
- **auto_vacuum=INCREMENTAL, with a one-time VACUUM to apply it.**
  auto_vacuum only takes effect on a fresh database or immediately
  after a VACUUM, so an existing file (created before this module
  existed) needs that one-time conversion — after which it sticks for
  that file permanently, so this only ever runs once per DB. Once
  incremental, PRAGMA incremental_vacuum reclaims free pages from
  record_scan's per-day DELETE+INSERT churn and chain_cache's
  INSERT...ON CONFLICT UPDATE upserts, without FULL auto_vacuum's
  per-transaction overhead or a full VACUUM's exclusive lock.
- **PRAGMA optimize on close.** SQLite's own recommended replacement
  for manually running ANALYZE (see sqlite.org/pragma.html#pragma_optimize)
  — cheap to call on every connection close since every caller here
  opens a short-lived connection per operation rather than holding a
  pool; it no-ops when statistics are already fresh.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

_BUSY_TIMEOUT_MS = 5_000
_CONNECT_TIMEOUT_S = 30.0
_AUTO_VACUUM_INCREMENTAL = 2


@contextmanager
def connect(path: Path) -> Generator[sqlite3.Connection]:
    """Open a connection to `path` with this app's standard cache-DB
    PRAGMAs applied, and commit/rollback + close it on exit — mirrors
    sqlite3.Connection's own context-manager protocol (commit on clean
    exit, rollback and re-raise on exception) but, unlike the stdlib
    one, actually closes the connection afterward instead of leaving
    that to garbage collection.

    Callers still run their own CREATE TABLE/INDEX/ALTER migration
    logic after this yields — kept in each module next to the schema
    and queries that depend on it, not centralized here.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=_CONNECT_TIMEOUT_S)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute(f"PRAGMA busy_timeout={_BUSY_TIMEOUT_MS}")
    conn.execute("PRAGMA temp_store=MEMORY")
    current_av = conn.execute("PRAGMA auto_vacuum").fetchone()[0]
    if current_av != _AUTO_VACUUM_INCREMENTAL:
        conn.execute("PRAGMA auto_vacuum=INCREMENTAL")
        conn.execute("VACUUM")  # one-time: applies the mode to this file
    else:
        conn.execute("PRAGMA incremental_vacuum")  # reclaim churn from DELETE/UPSERT
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        try:
            conn.execute("PRAGMA optimize")
        except sqlite3.Error:
            pass
        conn.close()
