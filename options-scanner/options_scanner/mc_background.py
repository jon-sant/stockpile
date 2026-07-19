"""Background Monte Carlo worker.

Mirrors `background_scan.py`'s singleton-daemon-thread shape (started
once via `start_mc_worker_once()`, guarded by `st.cache_resource` in
run_app.py so it only ever spawns once per server process), but the
thread dispatches CPU-bound Monte Carlo work to a `ProcessPoolExecutor`
instead of doing I/O itself — `mc_batch.compute_mc_for_row` is vectorized
NumPy with no shared state between rows, so it's an embarrassingly
parallel workload across however many CPU cores are available.

Poll interval is short (~20s, vs. background_scan's hourly poll) and can
be woken early via `restart_with_priority()` right after any tab
finishes a scan — that also discards any queued-but-not-yet-started
low-priority batches (recreating the pool) so the just-scanned ticker's
rows get picked up first. In-flight (already-running) tasks finish
naturally; only queued work is dropped.

`mc_batch.py` (the actual compute) has no Streamlit import and no
import-time side effects — required for it to be safely importable in a
spawned worker process. `multiprocessing.get_context("spawn")` is used
explicitly since spawn is the only start method available on Windows.
"""

from __future__ import annotations

import logging
import multiprocessing
import os
import threading
from concurrent.futures import ProcessPoolExecutor, as_completed

from options_scanner import iv_history, mc_batch

log = logging.getLogger(__name__)

_BATCH_SIZE = 25       # rows per ProcessPoolExecutor task — amortizes IPC/pickling overhead
_ROWS_PER_POLL = 200   # pending rows pulled and split into batches per tick
_POLL_SECONDS = 20     # short poll so a fresh scan feels responsive
_MP_CONTEXT = multiprocessing.get_context("spawn")  # Windows requires spawn

_lock = threading.Lock()
_executor: ProcessPoolExecutor | None = None
_priority_ticker: str | None = None
_wake = threading.Event()
_idle_logged = False  # avoid re-logging "backlog cleared" on every idle poll


def _pool_size() -> int:
    """Leave one core free for the Streamlit UI thread itself."""
    return max(1, (os.cpu_count() or 2) - 1)


def _get_executor() -> ProcessPoolExecutor:
    global _executor
    with _lock:
        if _executor is None:
            _executor = ProcessPoolExecutor(
                max_workers=_pool_size(), mp_context=_MP_CONTEXT)
        return _executor


def _chunks(rows: list[dict], size: int):
    for i in range(0, len(rows), size):
        yield rows[i:i + size]


def _tick() -> None:
    """One scheduler iteration — pure enough to unit test without the
    infinite loop around it."""
    global _idle_logged
    pending = iv_history.pending_mc_rows(
        limit=_ROWS_PER_POLL, priority_ticker=_priority_ticker)
    if pending.empty:
        if not _idle_logged:
            log.info("mc_background: backlog cleared, idling")
            _idle_logged = True
        return
    _idle_logged = False

    rows = pending.to_dict("records")
    executor = _get_executor()
    # Compute happens inside spawned worker processes, which don't share
    # this process's logging handlers — so both the start and end lines
    # are logged here in the scheduler thread (start at dispatch time,
    # end once each row's result comes back), not from inside mc_batch.
    for row in rows:
        log.info("mc_background: rowid=%s start (%s %s %s %s)",
                 row["rowid"], row["ticker"], row["type"], row["strike"],
                 row["expiration"])
    futures = [executor.submit(mc_batch.run_batch, batch)
               for batch in _chunks(rows, _BATCH_SIZE)]
    for future in as_completed(futures):
        try:
            results = future.result()
        except Exception:
            # Includes cancellation from restart_with_priority() dropping
            # a queued-but-not-started batch — those rows simply stay
            # pending and get picked up on a later tick.
            log.exception("mc_background: batch failed")
            continue
        for rowid, metrics, error in results:
            if error is not None:
                log.info("mc_background: rowid=%s failed: %s", rowid, error)
                iv_history.record_mc_error(rowid, error)
                continue
            duration_ms = metrics.pop("duration_ms")
            iv_history.record_mc_results(rowid, metrics, duration_ms)
            log.info(
                "mc_background: rowid=%s done in %.0fms "
                "(mc_fair_value=%.2f mc_prob_profit_buy=%.2f "
                "mc_prob_profit_sell=%.2f)",
                rowid, duration_ms, metrics["mc_fair_value"],
                metrics["mc_prob_profit_buy"], metrics["mc_prob_profit_sell"],
            )


def _scheduler_loop() -> None:
    while True:
        try:
            _tick()
        except Exception:
            log.exception("mc_background: scheduler tick failed")
        _wake.wait(timeout=_POLL_SECONDS)
        _wake.clear()


def restart_with_priority(ticker: str) -> None:
    """Call right after any tab's scan completes (after `_enrich()` ->
    `iv_history.record_scan()` returns). Prioritizes `ticker`'s rows on
    the next tick, wakes the scheduler immediately instead of waiting out
    the poll interval, and recreates the process pool so any
    queued-but-not-yet-started low-priority batches are dropped in favor
    of the just-scanned ticker. In-flight (already-running) tasks finish
    naturally — only queued work is cancelled.
    """
    global _executor, _priority_ticker, _idle_logged
    _priority_ticker = ticker.upper()
    _idle_logged = False
    with _lock:
        if _executor is not None:
            _executor.shutdown(wait=False, cancel_futures=True)
            _executor = None
    _wake.set()


def start_mc_worker_once() -> threading.Thread:
    """Spawn the scheduler as a daemon thread and return its handle."""
    thread = threading.Thread(
        target=_scheduler_loop, name="mc-background", daemon=True)
    thread.start()
    return thread
