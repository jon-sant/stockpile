"""Cached chain-fetch helpers used by the scanner tabs.

Wraps `chain.fetch_chain` with the earnings-annotation, IV-surface,
and realized-vol post-processing every tab needs before display. Both
helpers are decorated with `@st.cache_data` so repeated reruns within
a scan session (sidebar tweaks, filter changes) don't refetch.

Pipeline order matters: earnings are annotated *before* the surface
fit so the `exclude_earnings` filter can see `earnings_count`. The
surface is then fit and scored via the pluggable filter / algorithm /
score configs, and the scan snapshot is recorded for the percentile
score's history.

Two flavors:

- `fetch_and_enrich` — caller picks opt_type ("calls", "puts", or
  "both") and an optional max_dte. Used by the single-ticker, GEX,
  and spreads tabs.
- `fetch_position` — calls-only, no max_dte; the portfolio tab calls
  this once per open position so the signature stays narrow.

Both return `(df, earnings_dates, error_msg | None)`.

Imports of `chain`, `iv_surface`, and `earnings` are kept inline
inside the function bodies to preserve cold-start latency — the
established convention in this codebase.
"""

from __future__ import annotations

import logging
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

from options_scanner.iv_algorithms import DEFAULT_CONFIG as ALGO_DEFAULT, AlgorithmConfig
from options_scanner.iv_filters import DEFAULT_CONFIG, SurfaceFilterConfig
from options_scanner.iv_scores import (
    DEFAULT_CONFIG as SCORE_DEFAULT, STAR_DEFAULT, ScoreConfig,
)
from stocks_shared.yahoo import RateLimitError, is_rate_limit_error

log = logging.getLogger(__name__)


def _enrich(df: pd.DataFrame, ticker: str,
            surface_filters: SurfaceFilterConfig,
            algo_config: AlgorithmConfig,
            score_config: ScoreConfig,
            star_score_config: ScoreConfig = STAR_DEFAULT,
            market_view: str | None = None) -> pd.DataFrame:
    """Annotate earnings, fit + score the surface, attach realized vol,
    and record the snapshot. Shared by both fetch helpers.

    star_score_config picks the ranking key behind the ★ star rating —
    independent of score_config (which drives table ranking/sort). Run
    as a second pass over iv_scores.score() reusing the same fit_mask/
    ctx compute_iv_excess already built, rather than duplicating the
    surface fit.

    market_view is the outlook-card stance string (see
    options_scanner.market_view.stance_for) for whichever tab/CLI
    invocation is scanning — persisted per row so the background worker
    can derive Monte Carlo drift from it later. Callers with no
    buy/sell x calls/puts/both concept (GEX, Spreads) simply don't pass
    it, leaving it None/NULL — resolves to no drift."""
    from options_scanner.iv_surface import compute_iv_excess
    from options_scanner.iv_scores import ScoreContext, score as _score_fn
    from options_scanner.earnings import fetch_earnings_dates, annotate_earnings
    from options_scanner import iv_history
    from stocks_shared.yahoo import realized_vol

    earnings = fetch_earnings_dates(ticker)
    df = annotate_earnings(df, earnings)

    hv = realized_vol(ticker)
    ctx = ScoreContext(ticker=ticker, hv_20=hv, history=iv_history)
    df = compute_iv_excess(
        df, surface_filters=surface_filters, algo_config=algo_config,
        score_config=score_config, ctx=ctx,
    )

    star_signal, star_kind = _score_fn(
        df, df["in_fit"].to_numpy(dtype=bool), ctx, star_score_config)
    df["star_score"] = star_signal
    df["star_kind"] = star_kind

    df["hv_20"] = hv
    df["vr_ratio"] = (df["iv"] / hv) if (np.isfinite(hv) and hv > 0) \
        else float("nan")

    if not df.empty and "gamma" in df.columns and "spot" in df.columns:
        from options_scanner.compute import gex_summary
        df["gex_alignment"] = gex_summary.gex_alignment(df, float(df["spot"].iloc[0]))
    else:
        df["gex_alignment"] = float("nan")

    # Percentile-family columns, computed unconditionally (regardless of
    # which score is active) — iv_scores.py's registry only ever
    # materializes ONE score as signal_score, but filtering (min_percentile
    # below) and PR6's composite score both need these present as plain
    # columns no matter what's driving the ranking.
    if not df.empty and {"delta", "dte"} <= set(df.columns):
        df["iv_percentile"] = iv_history.percentile_for(
            ticker, df["iv_excess"], deltas=df["delta"], dtes=df["dte"])
        if "ann_yield_pct" in df.columns:
            df["ann_delta_percentile"] = iv_history.ann_delta_percentile_for(
                ticker, df["ann_yield_pct"], df["delta"], df["dte"])
        else:
            df["ann_delta_percentile"] = float("nan")
    else:
        df["iv_percentile"] = float("nan")
        df["ann_delta_percentile"] = float("nan")

    # IV Rank: where today's representative IV (median across the whole
    # scanned chain) sits vs. this ticker's own trailing scan history
    # (up to 365 days — see iv_history.iv_rank_for). Ticker-level, not
    # per-contract, so every row gets the same value; the date range
    # actually used is carried alongside for the display layer's tooltip
    # (history can be much shorter than 365 days for a newly-scanned
    # ticker, and the UI should say so rather than imply a full year).
    if not df.empty and "iv" in df.columns:
        _today_iv_s = df["iv"].dropna()
        _today_iv = float(_today_iv_s.median()) if not _today_iv_s.empty else float("nan")
    else:
        _today_iv = float("nan")
    _iv_rank, _iv_rank_from, _iv_rank_to = iv_history.iv_rank_for(ticker, _today_iv)
    df["iv_rank"] = _iv_rank if _iv_rank is not None else float("nan")
    df["iv_rank_date_from"] = _iv_rank_from.isoformat() if _iv_rank_from else None
    df["iv_rank_date_to"] = _iv_rank_to.isoformat() if _iv_rank_to else None

    # `earnings` is already fetch_earnings_dates()'s 0-or-1-element nearest-
    # future list — no need to re-derive "next" from it.
    earnings_next_date = earnings[0] if earnings else None
    iv_history.record_scan(ticker, df, earnings_next_date=earnings_next_date,
                           market_view=market_view)

    # Prioritize this ticker's rows in the background MC worker — one
    # choke point for every tab's scan (single/watchlist/portfolio/gex/
    # spreads all funnel through _enrich), so no per-tab wiring needed.
    # Never let a worker hiccup break a live scan.
    try:
        from options_scanner import mc_background
        mc_background.restart_with_priority(ticker)
    except Exception:
        log.exception("mc_background: restart_with_priority failed for %s", ticker)

    # Attach today's Monte Carlo columns (mc_status + the 14 mc_* metric
    # columns) back onto the freshly-recorded rows. On a brand-new scan
    # every row will be mc_status IS NULL ("TBD" in the UI) since the
    # background worker hasn't caught up yet; a same-day rescan may
    # already carry real values if the worker ran in between.
    mc = iv_history.mc_results_for(ticker)
    if not mc.empty:
        df = df.merge(
            mc, on=["type", "strike", "expiration"], how="left", validate="m:1")
    else:
        df["mc_status"] = None
        for col in iv_history._MC_METRIC_COLS:
            df[col] = float("nan")

    return df, earnings


@st.cache_data(ttl=300, show_spinner=False)
def fetch_and_enrich(ticker: str, opt_type: str, min_dte: int,
                     max_dte: int | None, provider: str = "yahoo",
                     schwab_config: dict | None = None,
                     surface_filters: SurfaceFilterConfig = DEFAULT_CONFIG,
                     algo_config: AlgorithmConfig = ALGO_DEFAULT,
                     score_config: ScoreConfig = SCORE_DEFAULT,
                     star_score_config: ScoreConfig = STAR_DEFAULT,
                     moomoo_config: dict | None = None,
                     fit_both_sides: bool = True,
                     market_view: str | None = None):
    """Fetch + enrich a chain. With fit_both_sides (the default), a
    one-sided request ("calls"/"puts") still fetches BOTH sides so the IV
    surface is anchored on both wings of the smile, and the full two-sided
    chain is returned — the caller filters to the side it displays.

    The IV surface is a property of the underlying, not of calls vs puts:
    OTM puts trace the left wing, OTM calls the right. Fitting one wing
    alone leaves curvature (and the m²·√T maturity term) badly under-
    determined and prone to wild extrapolation, so the fit always sees both.
    """
    from options_scanner.chain import fetch_chain
    fetch_type = ("both" if (fit_both_sides and opt_type in ("calls", "puts"))
                  else opt_type)
    try:
        df = fetch_chain(ticker, opt_type=fetch_type, min_dte=min_dte,
                         max_dte=max_dte, provider=provider,
                         schwab_config=schwab_config,
                         moomoo_config=moomoo_config)
    except RateLimitError as exc:
        # These tabs have no retry loop — report it as an actionable
        # error. (Raising would crash the tab; returning keeps the
        # result uncached only on the raise path, so accept the 5-min
        # cache here: Yahoo throttles rarely clear faster anyway.)
        return pd.DataFrame(), [], f"{exc}. Wait a minute or two and rescan."
    except (ValueError, OSError, ConnectionRefusedError, RuntimeError) as exc:
        if is_rate_limit_error(exc):
            return pd.DataFrame(), [], (f"{exc}. Wait a minute or two and "
                                        "rescan.")
        return pd.DataFrame(), [], str(exc)
    except Exception as exc:  # noqa: BLE001 — surface Moomoo/Schwab SDK errors
        if is_rate_limit_error(exc):
            return pd.DataFrame(), [], (f"{exc}. Wait a minute or two and "
                                        "rescan.")
        return pd.DataFrame(), [], f"{type(exc).__name__}: {exc}"
    if df.empty:
        return df, [], None
    df, earnings = _enrich(df, ticker, surface_filters, algo_config,
                           score_config, star_score_config, market_view)
    return df, earnings, None


@st.cache_data(ttl=300, show_spinner=False)
def fetch_position(ticker: str, min_dte: int, provider: str = "yahoo",
                   schwab_config: dict | None = None,
                   surface_filters: SurfaceFilterConfig = DEFAULT_CONFIG,
                   algo_config: AlgorithmConfig = ALGO_DEFAULT,
                   score_config: ScoreConfig = SCORE_DEFAULT,
                   star_score_config: ScoreConfig = STAR_DEFAULT,
                   moomoo_config: dict | None = None,
                   fit_both_sides: bool = True,
                   opt_type: str = "calls",
                   max_dte: int | None = 90,
                   market_view: str | None = None):
    """Cached per-ticker chain fetch for portfolio tab.

    opt_type controls which side(s) are returned: "calls", "puts", or "both".
    Regardless of opt_type, both sides are fetched when fit_both_sides is True
    so the IV surface is anchored on both wings — see fetch_and_enrich.
    max_dte mirrors the same parameter on fetch_and_enrich; None = no upper limit.
    """
    from options_scanner.chain import fetch_chain
    fetch_type = "both" if fit_both_sides else opt_type
    try:
        df = fetch_chain(ticker, opt_type=fetch_type, min_dte=min_dte,
                         max_dte=max_dte, provider=provider,
                         schwab_config=schwab_config,
                         moomoo_config=moomoo_config)
    except RateLimitError:
        raise  # propagate uncached so callers can wait and retry
    except (ValueError, OSError, ConnectionRefusedError, RuntimeError) as exc:
        if is_rate_limit_error(exc):
            raise RateLimitError(str(exc)) from exc
        return pd.DataFrame(), [], str(exc)
    except Exception as exc:  # noqa: BLE001 — surface Moomoo/Schwab SDK errors
        if is_rate_limit_error(exc):
            raise RateLimitError(str(exc)) from exc
        return pd.DataFrame(), [], f"{type(exc).__name__}: {exc}"
    if df.empty:
        return df, [], None
    df, earnings = _enrich(df, ticker, surface_filters, algo_config,
                           score_config, star_score_config, market_view)
    if not df.empty and opt_type in ("calls", "puts"):
        side = "call" if opt_type == "calls" else "put"
        df = df[df["type"] == side].reset_index(drop=True)
    return df, earnings, None


def _fetch_chain_with_cache(ticker: str, opt_type: str, min_dte: int,
                            max_dte: int | None, provider: str,
                            schwab_config: dict | None,
                            moomoo_config: dict | None,
                            fit_both_sides: bool,
                            force_live: bool) -> tuple[pd.DataFrame, datetime, bool]:
    """Raw-chain fetch shared by *_cached wrappers below. Checks the
    persistent chain_cache first — only when provider is "yahoo-headless"
    (the only provider the background scan job ever populates; serving a
    cached Yahoo-headless chain for a user who picked Schwab would show
    different bid/ask/OI than what they actually asked for) and the
    caller hasn't forced a live fetch. On a miss (or any other provider),
    fetches live and — if that fetch was yahoo-headless — saves it to the
    cache for next time. Returns (df, fetched_at, from_cache). Exceptions
    from the live fetch propagate to the caller unchanged.
    """
    from options_scanner import chain_cache
    from options_scanner.chain import fetch_chain

    fetch_type = ("both" if (fit_both_sides and opt_type in ("calls", "puts"))
                  else opt_type)

    if not force_live and provider == "yahoo-headless":
        cached = chain_cache.load_snapshot(ticker, min_dte, max_dte, provider)
        if cached is not None:
            df, fetched_at = cached
            log.info("[chain] %s: local cache hit, provider=%s", ticker, provider)
            return df, fetched_at, True

    log.info("[chain] %s: fetching online, provider=%s", ticker, provider)
    df = fetch_chain(ticker, opt_type=fetch_type, min_dte=min_dte,
                     max_dte=max_dte, provider=provider,
                     schwab_config=schwab_config, moomoo_config=moomoo_config)
    fetched_at = datetime.now().astimezone()
    if not df.empty and provider == "yahoo-headless":
        chain_cache.save_snapshot(ticker, df, min_dte, max_dte, provider)
    return df, fetched_at, False


def fetch_and_enrich_cached(ticker: str, opt_type: str, min_dte: int,
                            max_dte: int | None, provider: str = "yahoo",
                            schwab_config: dict | None = None,
                            surface_filters: SurfaceFilterConfig = DEFAULT_CONFIG,
                            algo_config: AlgorithmConfig = ALGO_DEFAULT,
                            score_config: ScoreConfig = SCORE_DEFAULT,
                            star_score_config: ScoreConfig = STAR_DEFAULT,
                            moomoo_config: dict | None = None,
                            fit_both_sides: bool = True,
                            force_live: bool = False,
                            market_view: str | None = None):
    """Like fetch_and_enrich, but transparently serves a same-day
    background-scanned chain (see chain_cache.py / background_scan.py)
    instead of hitting the network when one covers the request — the
    caller can't tell the difference except via the extra return values.
    Deliberately NOT @st.cache_data: the persistent chain_cache below is
    the real cache here, and force_live needs a clean way to bypass it
    on every call, not just per Streamlit-session TTL.

    Returns (df, earnings_dates, err, from_cache, fetched_at).
    """
    try:
        df, fetched_at, from_cache = _fetch_chain_with_cache(
            ticker, opt_type, min_dte, max_dte, provider, schwab_config,
            moomoo_config, fit_both_sides, force_live)
    except RateLimitError as exc:
        return pd.DataFrame(), [], f"{exc}. Wait a minute or two and rescan.", False, None
    except (ValueError, OSError, ConnectionRefusedError, RuntimeError) as exc:
        if is_rate_limit_error(exc):
            return (pd.DataFrame(), [],
                    f"{exc}. Wait a minute or two and rescan.", False, None)
        return pd.DataFrame(), [], str(exc), False, None
    except Exception as exc:  # noqa: BLE001 — surface Moomoo/Schwab SDK errors
        if is_rate_limit_error(exc):
            return (pd.DataFrame(), [],
                    f"{exc}. Wait a minute or two and rescan.", False, None)
        return pd.DataFrame(), [], f"{type(exc).__name__}: {exc}", False, None
    if df.empty:
        return df, [], None, from_cache, fetched_at
    df, earnings = _enrich(df, ticker, surface_filters, algo_config,
                           score_config, star_score_config, market_view)
    return df, earnings, None, from_cache, fetched_at


def fetch_position_cached(ticker: str, min_dte: int, provider: str = "yahoo",
                          schwab_config: dict | None = None,
                          surface_filters: SurfaceFilterConfig = DEFAULT_CONFIG,
                          algo_config: AlgorithmConfig = ALGO_DEFAULT,
                          score_config: ScoreConfig = SCORE_DEFAULT,
                          star_score_config: ScoreConfig = STAR_DEFAULT,
                          moomoo_config: dict | None = None,
                          fit_both_sides: bool = True,
                          opt_type: str = "calls",
                          max_dte: int | None = 90,
                          force_live: bool = False,
                          market_view: str | None = None):
    """Like fetch_position, but backed by the same persistent chain_cache
    as fetch_and_enrich_cached (see there for the caching rules).
    RateLimitError propagates uncached, same as fetch_position, so
    callers can wait and retry.

    Returns (df, earnings_dates, err, from_cache, fetched_at).
    """
    try:
        df, fetched_at, from_cache = _fetch_chain_with_cache(
            ticker, opt_type, min_dte, max_dte, provider, schwab_config,
            moomoo_config, fit_both_sides, force_live)
    except RateLimitError:
        raise  # propagate uncached so callers can wait and retry
    except (ValueError, OSError, ConnectionRefusedError, RuntimeError) as exc:
        if is_rate_limit_error(exc):
            raise RateLimitError(str(exc)) from exc
        return pd.DataFrame(), [], str(exc), False, None
    except Exception as exc:  # noqa: BLE001 — surface Moomoo/Schwab SDK errors
        if is_rate_limit_error(exc):
            raise RateLimitError(str(exc)) from exc
        return pd.DataFrame(), [], f"{type(exc).__name__}: {exc}", False, None
    if df.empty:
        return df, [], None, from_cache, fetched_at
    df, earnings = _enrich(df, ticker, surface_filters, algo_config,
                           score_config, star_score_config, market_view)
    if not df.empty and opt_type in ("calls", "puts"):
        side = "call" if opt_type == "calls" else "put"
        df = df[df["type"] == side].reset_index(drop=True)
    return df, earnings, None, from_cache, fetched_at
