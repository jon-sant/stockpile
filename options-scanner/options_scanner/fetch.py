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

import numpy as np
import pandas as pd
import streamlit as st

from options_scanner.iv_algorithms import DEFAULT_CONFIG as ALGO_DEFAULT, AlgorithmConfig
from options_scanner.iv_filters import DEFAULT_CONFIG, SurfaceFilterConfig
from options_scanner.iv_scores import DEFAULT_CONFIG as SCORE_DEFAULT, ScoreConfig
from stocks_shared.yahoo import RateLimitError, is_rate_limit_error


def _enrich(df: pd.DataFrame, ticker: str,
            surface_filters: SurfaceFilterConfig,
            algo_config: AlgorithmConfig,
            score_config: ScoreConfig) -> pd.DataFrame:
    """Annotate earnings, fit + score the surface, attach realized vol,
    and record the snapshot. Shared by both fetch helpers."""
    from options_scanner.iv_surface import compute_iv_excess
    from options_scanner.iv_scores import ScoreContext
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

    df["hv_20"] = hv
    df["vr_ratio"] = (df["iv"] / hv) if (np.isfinite(hv) and hv > 0) \
        else float("nan")

    if not df.empty and "gamma" in df.columns and "spot" in df.columns:
        from options_scanner.compute import gex_summary
        df["gex_alignment"] = gex_summary.gex_alignment(df, float(df["spot"].iloc[0]))
    else:
        df["gex_alignment"] = float("nan")

    iv_history.record_scan(ticker, df)
    return df, earnings


@st.cache_data(ttl=300, show_spinner=False)
def fetch_and_enrich(ticker: str, opt_type: str, min_dte: int,
                     max_dte: int | None, provider: str = "yahoo",
                     schwab_config: dict | None = None,
                     surface_filters: SurfaceFilterConfig = DEFAULT_CONFIG,
                     algo_config: AlgorithmConfig = ALGO_DEFAULT,
                     score_config: ScoreConfig = SCORE_DEFAULT,
                     moomoo_config: dict | None = None,
                     fit_both_sides: bool = True):
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
                           score_config)
    return df, earnings, None


@st.cache_data(ttl=300, show_spinner=False)
def fetch_position(ticker: str, min_dte: int, provider: str = "yahoo",
                   schwab_config: dict | None = None,
                   surface_filters: SurfaceFilterConfig = DEFAULT_CONFIG,
                   algo_config: AlgorithmConfig = ALGO_DEFAULT,
                   score_config: ScoreConfig = SCORE_DEFAULT,
                   moomoo_config: dict | None = None,
                   fit_both_sides: bool = True,
                   opt_type: str = "calls",
                   max_dte: int | None = 90):
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
                           score_config)
    if not df.empty and opt_type in ("calls", "puts"):
        side = "call" if opt_type == "calls" else "put"
        df = df[df["type"] == side].reset_index(drop=True)
    return df, earnings, None
