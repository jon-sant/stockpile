"""Shared helpers for the Yahoo (`chain.py`) and Schwab
(`schwab_chain.py`) chain fetchers.

Both providers walk a different raw shape — Yahoo gives us pandas
DataFrames from yfinance, Schwab gives us nested JSON dicts — but
once each call site has parsed out the per-contract fields, the
downstream work is identical: validate the quote, compute
log-moneyness and annualized yield, and emit a row in the
canonical 17-column schema the rest of the scanner consumes.

`build_option_row` is that downstream half. Each provider parses
its own raw structure and calls this with normalized floats/ints;
the helper applies the shared filters and returns either a dict
ready to append, or None when the quote is too sparse to keep.

Greeks (delta, gamma) come from the caller: Yahoo computes them
via Black-Scholes in chain.py; Schwab takes them straight from the
broker. The helper is Greek-agnostic.
"""

from __future__ import annotations

import math


def safe_float(val, default: float = 0.0) -> float:
    """float(val) that returns `default` for None, '', NaN, or other junk."""
    try:
        f = float(val)
        return f if math.isfinite(f) else default
    except (TypeError, ValueError):
        return default


def safe_int(val, default: int = 0) -> int:
    """int-via-float that returns `default` for None, '', NaN, or other junk."""
    try:
        f = float(val)
        return int(f) if math.isfinite(f) else default
    except (TypeError, ValueError):
        return default


def build_option_row(
    *,
    side: str,
    strike: float,
    expiration: str,
    dte: int,
    spot: float,
    bid: float,
    ask: float,
    mid: float,
    last: float,
    iv: float,
    delta: float,
    gamma: float,
    open_interest: int,
    volume: int,
    theta: float = 0.0,
    vega: float = 0.0,
    last_trade_days: float = float("nan"),
    require_quote: bool = True,
) -> dict | None:
    """Apply quote-quality filters and assemble a canonical chain row.

    Returns None when the quote is too sparse to keep:
      - both bid and ask are zero/missing (skipped when
        require_quote=False — the headless scraper passes that so an
        after-hours chain whose bid/asks Yahoo has zeroed out still
        yields rows priced off the last trade; last_trade_days carries
        the staleness downstream)
      - even after the bid/ask → last fallback, mid is still <= 0
      - IV is below the 1% noise floor
      - strike is non-positive

    When `mid` is missing or zero, falls back to (bid+ask)/2 if both
    sides are positive, otherwise to `last`. Pass mid=0 if the
    provider doesn't supply one directly.

    Returned dict matches the canonical scanner schema. `iv_fitted`
    starts equal to `iv`, `iv_excess`/`signal_score` start at 0, and
    `hv_20`/`vr_ratio` start as NaN; all are overwritten downstream by
    `iv_surface.compute_iv_excess` and `fetch._enrich`.
    """
    if require_quote and bid <= 0 and ask <= 0:
        return None
    if mid <= 0:
        mid = (bid + ask) / 2 if bid > 0 and ask > 0 else last
    if mid <= 0 or iv < 0.01 or strike <= 0:
        return None

    log_m = math.log(strike / spot)
    capital = spot if side == "call" else strike
    # 0DTE rows (same-day expiry) get clamped to 1 day for the
    # annualization — yield is meaningless at this scale anyway, but
    # the row's gamma/OI are still useful (GEX).
    ann_yield = (mid / capital) * (365.0 / max(dte, 1)) * 100.0

    return {
        "type":           side,
        "strike":         strike,
        "expiration":     expiration,
        "dte":            dte,
        "spot":           spot,
        "log_moneyness":  log_m,
        "bid":            bid,
        "ask":            ask,
        "mid":            mid,
        "last":           last,
        "iv":             iv,
        "iv_fitted":      iv,
        "iv_excess":      0.0,
        "signal_score":   0.0,
        "signal_kind":    "IV+pp",
        "delta":          delta,
        "gamma":          gamma,
        "theta":          theta,
        "vega":           vega,
        "ann_yield_pct":  ann_yield,
        "open_interest":  open_interest,
        "volume":         volume,
        # Days since the contract last traded (NaN when the provider
        # doesn't say) — feeds the fresh_quotes surface filter.
        "last_trade_days": last_trade_days,
        "earnings_count": 0,
        "hv_20":          float("nan"),
        "vr_ratio":       float("nan"),
    }
