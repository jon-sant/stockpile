"""Automatic daily background re-scan, so IV history keeps accumulating
even on days nobody opens the app.

Runs as a daemon thread inside the Streamlit process (started once via
`start_background_scan_once`, guarded by `st.cache_resource` in
run_app.py so it only ever spawns once per server process). Once per
trading day it re-scans every ticker that's been scanned (via any tab,
any provider) in the trailing 14 days, always through the
"yahoo-headless" provider — the one provider that returns real quotes
any time of day (the plain Yahoo JSON API zeroes bid/ask after hours).

Each ticker gets ONE fetch per day, covering the union of every DTE
range it's been scanned at recently — a background pass with a wide
range satisfies any narrower tab request later (see chain_cache's
range-containment check), and this keeps Selenium load to one browser
session per ticker per day rather than one per distinct historical
range.

Retry behavior: if a day's pass doesn't fully succeed (any ticker
errored or came back empty), the whole pass is retried roughly hourly
until every ticker in that day's universe has a snapshot. Once a
trading day is fully covered, nothing more happens until the next
trading day (checked via chain_cache, not a separate status flag — see
_pass_succeeded_for).

No holiday calendar: "trading day" is a plain Mon-Fri check. A
market-closed weekday just runs a harmless pass (yahoo-headless scrapes
whatever the page shows, doesn't require the market to be open).
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import date, timedelta

from options_scanner import chain_cache, iv_history

log = logging.getLogger(__name__)

_PROVIDER = "yahoo-headless"
_WINDOW_DAYS = 14
_POLL_SECONDS = 3600


def _target_universe(window_days: int = _WINDOW_DAYS
                      ) -> list[tuple[str, int, int | None]]:
    """(ticker, min_dte, max_dte) for every ticker scanned (any tab, any
    provider) in the trailing `window_days`, per iv_history — the union
    of every DTE seen for that ticker in the window. max_dte is always a
    concrete int (the widest DTE actually observed); iv_history never
    records a NULL dte, so there is no "no limit" case to reconstruct
    here — a ticker that was once scanned with no upper limit is simply
    represented by the largest DTE its own history happens to contain.
    """
    cutoff = (date.today() - timedelta(days=window_days)).isoformat()
    try:
        with iv_history._connect() as conn:
            cur = conn.execute(
                "SELECT ticker, MIN(dte), MAX(dte) FROM iv_history "
                "WHERE scan_date >= ? GROUP BY ticker",
                (cutoff,),
            )
            rows = cur.fetchall()
    except Exception:
        log.exception("background_scan: failed to read target universe")
        return []
    return [(ticker, int(lo), int(hi)) for ticker, lo, hi in rows if ticker]


def _is_trading_day(d: date) -> bool:
    """Mon-Fri heuristic — see module docstring for the holiday gap."""
    return d.weekday() < 5


def _run_pass(scan_day: date) -> bool:
    """One pass over _target_universe(): fetch + cache each ticker via
    yahoo-headless. Returns True iff every ticker in the universe ends
    the pass with a snapshot for `scan_day` (freshly fetched or already
    there from an earlier attempt today). A single ticker's failure is
    logged and skipped, not raised, so it can't abort the rest."""
    from options_scanner.chain import fetch_chain

    universe = _target_universe()
    if not universe:
        return True
    all_ok = True
    for ticker, min_dte, max_dte in universe:
        if chain_cache.has_fresh_snapshot(ticker, min_dte, max_dte,
                                          _PROVIDER, scan_day=scan_day):
            continue
        try:
            df = fetch_chain(ticker, opt_type="both", min_dte=min_dte,
                             max_dte=max_dte, provider=_PROVIDER)
        except Exception:
            log.exception("background_scan: fetch failed for %s", ticker)
            all_ok = False
            continue
        if df is None or df.empty:
            log.warning("background_scan: empty chain for %s", ticker)
            all_ok = False
            continue
        chain_cache.save_snapshot(ticker, df, min_dte, max_dte, _PROVIDER,
                                  scan_day=scan_day)
    return all_ok


def _pass_succeeded_for(day: date) -> bool:
    """True iff every ticker in today's target universe already has a
    chain_cache snapshot for `day`. Derived from chain_cache's actual
    contents rather than a separate status flag, so it can't drift out
    of sync with what's really cached."""
    universe = _target_universe()
    if not universe:
        return True
    return all(
        chain_cache.has_fresh_snapshot(ticker, min_dte, max_dte, _PROVIDER,
                                       scan_day=day)
        for ticker, min_dte, max_dte in universe
    )


def _tick() -> None:
    """One scheduler iteration — pure enough to unit test without the
    infinite loop around it."""
    today = date.today()
    if not _is_trading_day(today):
        return
    if _pass_succeeded_for(today):
        return
    ok = _run_pass(today)
    if not ok:
        log.info("background_scan: pass for %s incomplete, will retry", today)


def _scheduler_loop() -> None:
    while True:
        try:
            _tick()
        except Exception:
            log.exception("background_scan: scheduler tick failed")
        time.sleep(_POLL_SECONDS)


def start_background_scan_once() -> threading.Thread:
    """Spawn the scheduler as a daemon thread and return its handle."""
    thread = threading.Thread(
        target=_scheduler_loop, name="background-scan", daemon=True)
    thread.start()
    return thread
