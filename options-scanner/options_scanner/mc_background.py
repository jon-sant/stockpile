"""Background Monte Carlo worker.

Mirrors `background_scan.py`'s singleton-daemon-thread shape (started
once via `start_mc_worker_once()`, guarded by `st.cache_resource` in
run_app.py so it only ever spawns once per server process), but the
thread dispatches Monte Carlo work to a `ThreadPoolExecutor` instead of
doing I/O itself.

Threads, not processes: `mc_batch.compute_mc_for_row` is vectorized
NumPy (path generation, payoff/metrics — all large-array ufunc/BLAS
calls), and NumPy releases the GIL for those, so a thread pool still
gets real cross-core parallelism for the expensive part of the work
without any of the costs below.

A `ProcessPoolExecutor` was the original design and does NOT work here:
spawning a new process re-imports/re-bootstraps the *parent* process's
entry point, and under `streamlit run options-scanner/run_app.py` that
entry point is Streamlit's own launcher — code this project doesn't
control and can't wrap in the `if __name__ == "__main__":` guard
`multiprocessing` requires on Windows (spawn is the only start method
there; fork isn't available). In production this raised
`RuntimeError: An attempt has been made to start a new process before
the current process has finished its bootstrapping phase` on every
tick with pending work, which — since the tick's except-and-retry never
lets the backlog shrink — fired every ~20s forever, which is also the
likely cause of the flood of "missing ScriptRunContext" warnings
reported alongside it (each failed spawn attempt partially re-imports
the host process). Threads avoid all of this: no new OS process, no
re-import of anything.

Poll interval is short (~20s, vs. background_scan's hourly poll) and can
be woken early via `restart_with_priority()` right after any tab
finishes a scan — that also discards any queued-but-not-yet-started
low-priority batches (recreating the pool) so the just-scanned ticker's
rows get picked up first. In-flight (already-running) tasks finish
naturally; only queued work is dropped.
"""

from __future__ import annotations

import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from options_scanner import iv_history, mc_batch

log = logging.getLogger(__name__)

_BATCH_SIZE = 25       # rows per executor task — amortizes dispatch overhead
_ROWS_PER_POLL = 200   # pending rows pulled and split into batches per tick
_POLL_SECONDS = 20     # short poll so a fresh scan feels responsive

_lock = threading.Lock()
_executor: ThreadPoolExecutor | None = None
_priority_ticker: str | None = None
_wake = threading.Event()
_idle_logged = False  # avoid re-logging "backlog cleared" on every idle poll


def _pool_size() -> int:
    """Leave one core free for the Streamlit UI thread itself. Threads
    are cheap, so this can be generous — NumPy's GIL release means more
    threads than cores still helps rather than just adding overhead."""
    return max(1, (os.cpu_count() or 2) - 1)


def _get_executor() -> ThreadPoolExecutor:
    global _executor
    with _lock:
        if _executor is None:
            _executor = ThreadPoolExecutor(
                max_workers=_pool_size(), thread_name_prefix="mc-worker")
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
    the poll interval, and recreates the thread pool so any
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
