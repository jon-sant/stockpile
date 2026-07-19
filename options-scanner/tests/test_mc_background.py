"""Tests for the background Monte Carlo worker (mc_background.py).

Uses a throwaway DB via OSC_IV_HISTORY_DB. Regression coverage for the
ProcessPoolExecutor -> ThreadPoolExecutor fix: spawning a new OS process
from inside `streamlit run` re-imports/re-bootstraps Streamlit's own
launcher (which isn't guarded by `if __name__ == "__main__":`), raising
`RuntimeError: An attempt has been made to start a new process before
the current process has finished its bootstrapping phase` on Windows on
every tick with pending work.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import date

import pandas as pd
import pytest

from options_scanner import iv_history, mc_background


@pytest.fixture(autouse=True)
def _temp_db(tmp_path, monkeypatch):
    monkeypatch.setenv("OSC_IV_HISTORY_DB", str(tmp_path / "h.db"))
    # Reset module-level state between tests — mc_background holds a
    # process-wide singleton executor/priority ticker.
    mc_background._executor = None
    mc_background._priority_ticker = None
    mc_background._idle_logged = False
    yield
    if mc_background._executor is not None:
        mc_background._executor.shutdown(wait=False, cancel_futures=True)
        mc_background._executor = None


def _snapshot(n: int) -> pd.DataFrame:
    return pd.DataFrame({
        "type": ["call"] * n,
        "strike": [100.0 + i for i in range(n)],
        "expiration": ["2026-09-18"] * n,
        "dte": [61] * n,
        "iv_excess": [0.01 + 0.001 * i for i in range(n)],
        "mid": [5.0 + i for i in range(n)],
        "delta": [0.4] * n,
        "ann_yield_pct": [12.0] * n,
        "iv": [0.40] * n,
        "open_interest": [500] * n,
        "volume": [50] * n,
        "spot": [98.0] * n,
    })


def test_get_executor_is_thread_pool_not_process_pool():
    executor = mc_background._get_executor()
    assert isinstance(executor, ThreadPoolExecutor)
    assert not isinstance(executor, ProcessPoolExecutor)


def test_tick_computes_pending_rows_via_threads():
    iv_history.record_scan("AMD", _snapshot(3), scan_day=date(2026, 7, 19),
                           earnings_next_date=date(2026, 8, 5))
    mc_background._tick()

    mc = iv_history.mc_results_for("AMD", scan_date=date(2026, 7, 19))
    assert (mc["mc_status"] == "done").all()
    assert not iv_history.pending_mc_rows(limit=10).shape[0]


def test_tick_isolates_rows_missing_inputs():
    df = _snapshot(2)
    df.loc[0, "spot"] = None  # simulate a legacy/incomplete row
    iv_history.record_scan("AMD", df, scan_day=date(2026, 7, 19))
    mc_background._tick()

    mc = iv_history.mc_results_for("AMD", scan_date=date(2026, 7, 19))
    statuses = sorted(mc["mc_status"].tolist())
    assert statuses == ["done", "error"]
    errored = mc[mc["mc_status"] == "error"]
    # No more bare TypeError — see test_mc_batch.py for the underlying fix.
    assert "TypeError" not in str(errored)


def test_restart_with_priority_recreates_executor_and_sets_priority():
    first = mc_background._get_executor()
    mc_background.restart_with_priority("TSLA")
    assert mc_background._priority_ticker == "TSLA"
    assert mc_background._wake.is_set()
    mc_background._wake.clear()

    second = mc_background._get_executor()
    assert second is not first
    assert isinstance(second, ThreadPoolExecutor)
