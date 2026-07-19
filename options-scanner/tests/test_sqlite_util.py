"""Tests for the shared cache-DB connection helper: PRAGMAs actually
take effect, the one-time auto_vacuum migration applies to a legacy
(pre-existing, default-pragma) file, and the connection lifecycle
commits/rolls back/closes correctly.
"""

import sqlite3

import pytest

from options_scanner import sqlite_util


def _pragma(path, name):
    conn = sqlite3.connect(str(path))
    try:
        return conn.execute(f"PRAGMA {name}").fetchone()[0]
    finally:
        conn.close()


def test_connect_sets_wal_journal_mode(tmp_path):
    # journal_mode is persisted in the DB file header, so it's visible
    # from a fresh connection too.
    db = tmp_path / "t.db"
    with sqlite_util.connect(db) as conn:
        conn.execute("CREATE TABLE t (x INTEGER)")
    assert _pragma(db, "journal_mode") == "wal"


def test_connect_sets_normal_synchronous(tmp_path):
    # synchronous is a per-connection setting (not persisted to the file
    # like journal_mode), so it has to be checked within the same
    # connection sqlite_util.connect() set it on.
    db = tmp_path / "t.db"
    with sqlite_util.connect(db) as conn:
        assert conn.execute("PRAGMA synchronous").fetchone()[0] == 1  # NORMAL


def test_connect_sets_busy_timeout(tmp_path):
    db = tmp_path / "t.db"
    with sqlite_util.connect(db) as conn:
        assert conn.execute("PRAGMA busy_timeout").fetchone()[0] == 5000


def test_connect_sets_incremental_auto_vacuum_on_fresh_db(tmp_path):
    db = tmp_path / "t.db"
    with sqlite_util.connect(db) as conn:
        conn.execute("CREATE TABLE t (x INTEGER)")
    assert _pragma(db, "auto_vacuum") == 2  # INCREMENTAL


def test_connect_migrates_legacy_db_to_incremental_auto_vacuum(tmp_path):
    db = tmp_path / "legacy.db"
    # Simulate a pre-existing DB created before this module existed:
    # default rollback-journal, default (NONE) auto_vacuum.
    conn = sqlite3.connect(str(db))
    conn.execute("CREATE TABLE t (x INTEGER)")
    conn.execute("INSERT INTO t VALUES (1), (2), (3)")
    conn.commit()
    conn.close()
    assert _pragma(db, "auto_vacuum") == 0  # NONE, confirms the legacy state

    with sqlite_util.connect(db) as conn:
        rows = conn.execute("SELECT x FROM t ORDER BY x").fetchall()
    assert rows == [(1,), (2,), (3,)]  # data survives the VACUUM migration
    assert _pragma(db, "auto_vacuum") == 2  # now INCREMENTAL
    assert _pragma(db, "journal_mode") == "wal"


def test_connect_commits_on_clean_exit(tmp_path):
    db = tmp_path / "t.db"
    with sqlite_util.connect(db) as conn:
        conn.execute("CREATE TABLE t (x INTEGER)")
        conn.execute("INSERT INTO t VALUES (42)")
    with sqlite_util.connect(db) as conn:
        assert conn.execute("SELECT x FROM t").fetchall() == [(42,)]


def test_connect_rolls_back_on_exception(tmp_path):
    db = tmp_path / "t.db"
    with sqlite_util.connect(db) as conn:
        conn.execute("CREATE TABLE t (x INTEGER)")
        conn.commit()

    with pytest.raises(ValueError):
        with sqlite_util.connect(db) as conn:
            conn.execute("INSERT INTO t VALUES (1)")
            raise ValueError("boom")

    with sqlite_util.connect(db) as conn:
        assert conn.execute("SELECT x FROM t").fetchall() == []  # rolled back


def test_connect_closes_connection_on_exit(tmp_path):
    db = tmp_path / "t.db"
    with sqlite_util.connect(db) as conn:
        captured = conn
    with pytest.raises(sqlite3.ProgrammingError):
        captured.execute("SELECT 1")  # closed connections raise on use


def test_connect_creates_parent_dir(tmp_path):
    db = tmp_path / "nested" / "dir" / "t.db"
    with sqlite_util.connect(db) as conn:
        conn.execute("CREATE TABLE t (x INTEGER)")
    assert db.exists()
