"""Batch Monte Carlo computation for the background worker.

Pure-compute module — deliberately has NO `streamlit` import and no
import-time side effects, since it must be safe to import inside a
spawned worker process (mc_background.py's ProcessPoolExecutor uses the
"spawn" start method, required on Windows).

`compute_mc_for_row` builds a long AND a short single-leg Position from
one persisted chain row (a plain dict, as read back from
`iv_history.pending_mc_rows`) and evaluates both against ONE shared
simulated path set — vol/jump-sigma resolution and path generation never
depend on `Leg.side` (see `montecarlo.engine.prepare_paths`), so sharing
paths is exact, not an approximation, and avoids doubling the expensive
part of the simulation for what would otherwise be two independent runs.
"""

from __future__ import annotations

import hashlib
import time
from datetime import date
from typing import Literal

from options_scanner.montecarlo import Leg, Position, SimulationConfig
from options_scanner.montecarlo.engine import metrics_for_position, prepare_paths


def _deterministic_seed(*parts: object) -> int:
    """Stable seed from a row's identity, so re-running the worker (e.g.
    after a crash, or backlog re-processing) reproduces the same result
    rather than a fresh random draw each time."""
    key = "|".join(str(p) for p in parts)
    return int(hashlib.sha256(key.encode()).hexdigest(), 16) % (2**32)


def _position_for_row(row: dict, side: Literal["long", "short"]) -> Position:
    """Build a single-leg Position from one iv_history row for the given
    side. Mirrors `mc_ui.position_from_chain_row`'s open_cost convention
    (long pays the mid as a positive debit; short receives it, encoded as
    a negative debit) but reads iv_history's column names/shapes rather
    than the UI chain-table's."""
    strike = float(row["strike"])
    expiration = date.fromisoformat(row["expiration"])
    mid = float(row["mid"])
    iv = float(row["iv"]) if row.get("iv") is not None else None

    side_sign = 1 if side == "long" else -1
    open_cost = side_sign * mid * 100.0

    leg = Leg(opt_type=row["type"], strike=strike, expiration=expiration,
             side=side, qty=1, open_cost=open_cost, iv=iv)

    earnings_dates = ()
    if row.get("earnings_next_date"):
        earnings_dates = (date.fromisoformat(row["earnings_next_date"]),)

    return Position(underlying=str(row["ticker"]), spot=float(row["spot"]),
                    legs=(leg,), earnings_dates=earnings_dates)


def compute_mc_for_row(row: dict) -> dict:
    """Compute the 14 Monte Carlo metric columns (see
    `iv_history._MC_METRIC_COLS`) for one chain row, both buy and sell
    sides, plus `duration_ms`.

    `row` must have: ticker, scan_date, type, strike, expiration, mid,
    iv, spot, earnings_next_date (all as read back from
    `iv_history.pending_mc_rows`).

    Raises whatever the underlying engine raises (e.g. no positive IV to
    resolve a vol from) — the caller (`run_batch`) isolates one row's
    failure from the rest of the batch.
    """
    start = time.perf_counter()
    today = date.fromisoformat(row["scan_date"])
    seed = _deterministic_seed(row["ticker"], row["scan_date"], row["type"],
                              row["strike"], row["expiration"])
    config = SimulationConfig(seed=seed)

    long_position = _position_for_row(row, "long")
    short_position = _position_for_row(row, "short")

    paths, days, horizon, n_days, today = prepare_paths(long_position, config, today)
    long_metrics, _, _ = metrics_for_position(
        long_position, paths, days, horizon, today, config, n_days)
    short_metrics, _, _ = metrics_for_position(
        short_position, paths, days, horizon, today, config, n_days)

    duration_ms = (time.perf_counter() - start) * 1000.0

    return {
        # Side-invariant — mc_fair_value comes out non-negative from the
        # long-side evaluation by construction (mean of an always->=0
        # intrinsic payoff); breakeven spot is identical either side
        # (same zero-crossing threshold for long vs. short of one leg).
        "mc_fair_value": long_metrics["mc_fair_value"],
        "mc_breakeven_move_pct": long_metrics["breakeven_move_pct"],
        # Side-asymmetric.
        "mc_prob_profit_buy": long_metrics["prob_profit"],
        "mc_prob_profit_sell": short_metrics["prob_profit"],
        "mc_expected_pnl_buy": long_metrics["expected_pnl"],
        "mc_expected_pnl_sell": short_metrics["expected_pnl"],
        "mc_cvar5_buy": long_metrics["cvar_5pct"],
        "mc_cvar5_sell": short_metrics["cvar_5pct"],
        "mc_var5_buy": long_metrics["var_5pct"],
        "mc_var5_sell": short_metrics["var_5pct"],
        "mc_sortino_buy": long_metrics["sortino"],
        "mc_sortino_sell": short_metrics["sortino"],
        "mc_edge_vs_market_buy": long_metrics["edge_vs_market"],
        "mc_edge_vs_market_sell": short_metrics["edge_vs_market"],
        "duration_ms": duration_ms,
    }


def run_batch(rows: list[dict]) -> list[tuple[int, dict | None, str | None]]:
    """Compute Monte Carlo metrics for a batch of pending rows — the unit
    of work handed to `ProcessPoolExecutor.submit`, batched (~20-50 rows)
    to amortize per-dispatch IPC/pickling overhead. One row's failure
    never aborts the rest of the batch.

    Returns (rowid, results-or-None, error-or-None) per row.
    """
    out: list[tuple[int, dict | None, str | None]] = []
    for row in rows:
        rowid = int(row["rowid"])
        try:
            result = compute_mc_for_row(row)
        except Exception as exc:  # noqa: BLE001 — isolate one row's failure
            out.append((rowid, None, f"{type(exc).__name__}: {exc}"))
            continue
        out.append((rowid, result, None))
    return out
